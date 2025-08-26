from flask import Flask, request, jsonify
import pandas as pd
import json
import logging
import requests
import re
import traceback
from collections import Counter
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('medical_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Hardcoded CSV file paths
MEDICALLY_SIGNIFICANT_PATH = "data/Medically significant.csv"
SIGNIFICANT_DISABILITY_PATH = "data/Significant disability.csv"
CONGENITAL_ANOMALY_PATH = "data/Congenital Anomaly.csv"

def safe_json_loads(text):
    """
    Safely parse JSON with multiple fallback strategies
    """
    text = text.strip()
    # Remove code block markers
    text = re.sub(r'```(json)?|```', '', text).strip()
    
    parsing_strategies = [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group(0)),
        lambda t: eval(t)  # Fallback strategy, use with caution
    ]
    
    for strategy in parsing_strategies:
        try:
            return strategy(text)
        except Exception as e:
            logger.warning(f"JSON parsing strategy failed: {str(e)}")
            continue
    
    raise ValueError("Could not parse JSON after trying multiple strategies")

def call_ollama(prompt, model="phi4:latest", max_retries=3):
    """
    Call Ollama API with single deterministic call for consistency
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0,  # Keep at 0 for consistency
                    "top_p": 0.1,      # Very low for deterministic results
                    "repeat_penalty": 1.0,
                    "seed": 42,        # Fixed seed for reproducibility
                    "max_tokens": 500,
                    "format": "json"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                if response_text:
                    return response_text
            else:
                logger.error(f"Ollama API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama failed (Attempt {attempt + 1}): {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama call: {str(e)}")
    
    raise Exception("Failed to get response from Ollama after multiple attempts")

def call_diagnostic_ollama(text, model="phi4:latest", max_retries=3):
    """
    Specialized call to Ollama to extract diagnosed conditions with single consistent call
    """
    prompt = """
    Extract ONLY NEW diagnosed conditions that occurred AFTER taking the medication. Follow these steps carefully:
    
    STEP 1: Identify the TIMELINE - what happened BEFORE vs AFTER the medication
    STEP 2: Identify the medication that was prescribed
    STEP 3: Look for NEW conditions that were diagnosed AFTER starting the medication
    
    CRITICAL RULES:
    1. DO NOT extract conditions that existed BEFORE taking the medication (these are indications/reasons for treatment)
    2. DO NOT extract the original condition that led to prescribing the medication
    3. ONLY extract conditions that were newly diagnosed AFTER the patient started taking the medication
    4. Look for temporal indicators like "after taking", "following medication", "subsequently diagnosed"
    5. Look for phrases like "diagnosed with", "diagnosis of", "diagnosed as having" that occur AFTER medication use
    6. If a diagnosed condition is drug-induced (e.g., "drug-induced hepatitis"), include the complete diagnosis
    
    Examples of what NOT to extract:
    - "Patient had depression, so was prescribed antidepressant" (depression is the indication)
    - "Doctor diagnosed panic attacks and prescribed medication" (panic attacks is the reason for treatment)
    - "Patient complained of headaches, doctor prescribed painkillers" (headaches is the indication)
    
    Examples of what TO extract:
    - "After taking the medication, patient was diagnosed with liver damage"
    - "Following treatment, doctor diagnosed drug-induced rash"
    - "Subsequently diagnosed with medication-induced tremors"
    
    Text:
    {text}
    
    Return in this JSON format:
    {{
        "Diagnosed_Condition": "the exact NEW diagnosed condition that occurred AFTER medication, or empty string if none"
    }}
    """
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": prompt.format(text=text),
                    "stream": False,
                    "temperature": 0,
                    "top_p": 0.1,
                    "repeat_penalty": 1.0,
                    "seed": 42,
                    "max_tokens": 300,
                    "format": "json"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                if response_text:
                    try:
                        parsed = safe_json_loads(response_text)
                        return parsed.get("Diagnosed_Condition", "")
                    except:
                        return ""
            else:
                logger.error(f"Diagnosis API Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Unexpected error in diagnosis extraction: {str(e)}")
    
    return ""

def load_ae_terms_from_csv():
    """
    Load adverse event terms from CSV files with error handling
    """
    try:
        def load_terms(filepath):
            try:
                df = pd.read_csv(filepath, header=None, encoding='utf-8')
                return set(term.lower().strip() for term in df[0].dropna())
            except FileNotFoundError:
                logger.warning(f"CSV file not found: {filepath}")
                return set()
                
        medically_significant_terms = load_terms(MEDICALLY_SIGNIFICANT_PATH)
        significant_disability_terms = load_terms(SIGNIFICANT_DISABILITY_PATH)
        congenital_anomaly_terms = load_terms(CONGENITAL_ANOMALY_PATH)
        
        return medically_significant_terms, significant_disability_terms, congenital_anomaly_terms
    except Exception as e:
        logger.error(f"Error loading CSV files: {str(e)}")
        raise

def check_csv_severity(ae_term, medically_significant_terms, significant_disability_terms, congenital_anomaly_terms):
    """
    Check if the adverse event matches any CSV terms for the three specific categories
    Returns the matched category or None if no match
    """
    ae_term_lower = ae_term.lower().strip()
    
    # Only check CSV terms for these three categories - exact matching required
    if any(term in ae_term_lower for term in significant_disability_terms):
        return "Significant disability or incapacity"
    
    if any(term in ae_term_lower for term in congenital_anomaly_terms):
        return "Congenital anomaly or birth defect"
    
    if any(term in ae_term_lower for term in medically_significant_terms):
        return "Medically significant"
    
    return None

def determine_adr_severity_for_diagnosed(ae_term, medically_significant_terms, significant_disability_terms, congenital_anomaly_terms):
    """
    Determine severity of diagnosed conditions using the same logic as the old function
    """
    ae_term_lower = ae_term.lower().strip()
    
    # Severity matching with multiple term checks
    severity_rules = [
        (["death", "fatal", "mortality", "died", "passed away", "deceased", "expired", "demise"], "Fatal or death"),
        (["life-threatening", "ventilator", "emergency intervention", "critical condition", "near death", "almost died", "intensive care"], "Life threatening"),
        (["hospitalization", "hospital", "admitted", "admission", "emergency room", "ER visit"], "Inpatient hospitalization or prolongation of hospitalization"),
        (significant_disability_terms, "Significant disability or incapacity"),
        (congenital_anomaly_terms, "Congenital anomaly or birth defect"),
        (medically_significant_terms, "Medically significant")
    ]
    
    for keywords, severity in severity_rules:
        if any(keyword in ae_term_lower for keyword in keywords):
            return severity
    
    return "None of the above"

def extract_medical_information(text):
    """
    Extract medical information from text with prioritization of diagnosed conditions
    """
    # First, try to extract any diagnosed conditions
    diagnosed_condition = call_diagnostic_ollama(text)
    
    # If a diagnosed condition is found, use that as the primary adverse event
    if diagnosed_condition and diagnosed_condition.strip():
        logger.info(f"Diagnosed condition found: {diagnosed_condition}")
        
        # Split diagnosed condition by comma if present
        split_conditions = []
        if ',' in diagnosed_condition:
            conditions = [condition.strip() for condition in diagnosed_condition.split(',') if condition.strip()]
            for condition in conditions:
                split_conditions.append({
                    "Term": condition,
                    "Severity": "To be determined",  # Will be determined later
                    "Is_Diagnosis": True
                })
        else:
            split_conditions.append({
                "Term": diagnosed_condition.strip(),
                "Severity": "To be determined",  # Will be determined later
                "Is_Diagnosis": True
            })
        
        return {
            "Adverse_Events": split_conditions
        }
    
    # If no diagnosis found, extract adverse events using the regular approach
    prompt = """Extract adverse events from the medical text and classify their severity.

EXTRACTION RULES:
1. Extract ONLY adverse events that occurred AFTER taking the medication
2. Use ONLY the exact medical terms that appear in the original text
3. Do NOT convert, normalize, or change the terminology
4. Extract the core symptom/condition name as it appears in the text
5. Each adverse event should be extracted only ONCE
6. Extract a maximum of 5 unique adverse events

SEVERITY CLASSIFICATION (choose ONE category for each adverse event):

**Fatal or death**: When patient actually died - use terms like "died", "passed away", "death", "deceased", "expired"
**Life threatening**: Immediate risk of death requiring emergency intervention - use ONLY when text explicitly mentions  ventilator support, resuscitation, or phrases like "nearly died", "critical condition "

**Inpatient hospitalization or prolongation of hospitalization**: Hospital admission, emergency room visits, prolonged hospital stays

**None of the above**: All other adverse events and symptoms like nausea, drowsiness, headache, fatigue, dry mouth, etc.

IMPORTANT SEVERITY RULES:
- ONLY use "Fatal or death" if the patient actually DIED (words like died, passed away, death, deceased)
- ONLY use "Life threatening" if there was immediate risk of death but patient survived
- ONLY use "Hospitalization" if hospital admission is explicitly mentioned
- Use "None of the above" for common side effects

Text: {text}

Return adverse events using exact terms from the original text:
{{
    "Adverse_Events": [
        {{
            "Term": "exact term from original text",
            "Severity": "appropriate severity classification"
        }}
    ]
}}

Be consistent and use the same severity classification logic every time."""

    try:
        # Call Ollama with single deterministic call
        response_text = call_ollama(prompt.format(text=text))
        
        # Parse JSON with safe parsing
        extracted_info = safe_json_loads(response_text)
        
        # Validate extracted information
        if not isinstance(extracted_info, dict) or 'Adverse_Events' not in extracted_info:
            raise ValueError("Invalid response format")
        
        # Process events without complex deduplication to maintain consistency
        events = []
        seen_terms = set()
        
        for event in extracted_info.get("Adverse_Events", []):
            event_term = event.get("Term", "").strip()
            event_severity = event.get("Severity", "None of the above")
            
            if not event_term:
                continue
            
            # Simple duplicate check - case insensitive
            term_lower = event_term.lower().strip()
            if term_lower not in seen_terms:
                seen_terms.add(term_lower)
                events.append({
                    "Term": event_term,
                    "Severity": event_severity,
                    "Is_Diagnosis": False
                })
        
        return {
            "Adverse_Events": events[:5]  # Limit to 5 events
        }
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Load terms once when the app starts
try:
    medically_significant_terms, significant_disability_terms, congenital_anomaly_terms = load_ae_terms_from_csv()
except Exception as e:
    logger.critical("Failed to load CSV terms. Server cannot start.")
    raise

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided for analysis"}), 400
        
        # Extract medical information
        result = extract_medical_information(text)
        
        # Process adverse events
        response_data = {}
        severity_priority = [
            "Fatal or death",
            "Life threatening",
            "Inpatient hospitalization or prolongation of hospitalization",
            "Congenital anomaly or birth defect",
            "Significant disability or incapacity",
            "Medically significant",
            "None of the above"
        ]
        
        # Process each event with specific numbering
        events = result.get("Adverse_Events", [])[:5]  # Limit to 5 events
        
        for i, event in enumerate(events, 1):
            ae_term = event.get("Term", "").strip()
            if not ae_term:
                continue
            
            # Get AI-determined severity first, default to "None of the above" if empty
            ai_severity = event.get("Severity", "None of the above")
            if not ai_severity or ai_severity.strip() == "" or ai_severity == "To be determined":
                # For diagnosed conditions, use the determine function
                if event.get("Is_Diagnosis", False):
                    ai_severity = determine_adr_severity_for_diagnosed(
                        ae_term,
                        medically_significant_terms,
                        significant_disability_terms,
                        congenital_anomaly_terms
                    )
                else:
                    ai_severity = "None of the above"
            
            # Check if it matches any CSV terms for the three specific categories
            csv_severity = check_csv_severity(
                ae_term,
                medically_significant_terms,
                significant_disability_terms,
                congenital_anomaly_terms
            )
            
            # Use CSV severity if found, otherwise use AI severity
            final_severity = csv_severity if csv_severity else ai_severity
            
            # Determine individual Side_Effect_Seriousness
            seriousness = "Non-Serious"
            if final_severity in severity_priority[:-1]:  # Exclude "None of the above"
                seriousness = "Serious"
            
            # Dynamic key naming based on number of adverse events
            if i == 1:
                response_data['Adverse_Event_Terms'] = ae_term
                response_data['Side_Effect_Severity'] = final_severity
                response_data['Side_Effect_Seriousness'] = seriousness
            else:
                response_data[f'Adverse_Event_Terms{i}'] = ae_term
                response_data[f'Side_Effect_Severity{i}'] = final_severity
                response_data[f'Side_Effect_Seriousness{i}'] = seriousness
        
        return jsonify(response_data)
    except ValueError as ve:
        logger.error(f"Validation Error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Medical Information Extractor",
        "version": "0.9.8"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6302, debug=False)
