import json
import re
import ast
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='patient_medical_history_extraction.log'
)

# Field name mappings for standardization
FIELD_MAPPINGS = {
    'patient_medical_history': {
        'past_medical_history_completed': 'Past_Medical_History_Completed',
        'past_medications': 'Past_Medications',
        'past_current_conditions_before_subject_drug': 'Past_Current_Conditions_Before_Subject_Drug',
        'concomitant_medications': 'Concomitant_Medications'
    }
}

def standardize_field_names(data, category):
    """
    Standardize field names in the extracted data according to the mapping
    """
    if not isinstance(data, dict):
        return data

    standardized_data = {}
    mapping = FIELD_MAPPINGS.get(category, {})
    
    for key, value in data.items():
        # Convert key to standard format (lowercase with underscores)
        std_key = key.lower().replace(' ', '_')
        
        # Look up the standardized name in the mapping
        if std_key in mapping:
            standardized_key = mapping[std_key]
        else:
            standardized_key = key  # Keep original if no mapping exists
            
        standardized_data[standardized_key] = value
        
    return standardized_data

# Ollama API call function
def call_ollama(prompt, model="phi4:latest"):
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1500,
            "format": "json"
        })
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            logging.error(f"Ollama API Error: {response.status_code} - {response.text}")
            return ''
    except Exception as e:
        logging.error(f"Ollama API request failed: {str(e)}")
        return ''

# JSON parsing function
def parse_json(text):
    text = re.sub(r'```(json)?|```', '', text).strip()
    
    parsing_strategies = [
        lambda t: json.loads(t),
        lambda t: ast.literal_eval(t),
        lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group(0))
    ]
    
    for strategy in parsing_strategies:
        try:
            return strategy(text)
        except Exception as e:
            logging.warning(f"Parsing failed with strategy: {strategy.__name__}")
    
    raise ValueError("Could not parse JSON")

PROMPT = {
    "patient_medical_history": """You are a medical expert specializing in pharmacovigilance and information extraction.

IMPORTANT INSTRUCTIONS:
- Respond ONLY with a valid JSON object
- Do NOT include any explanatory text outside the JSON
- If no information is found, use empty strings for values
- Use the exact field names specified below

Extract the following patient medical history information from the given medical text. If any information is not available or not provided, return an empty string for that field.

Text:
{text}

Please extract:
1. Past_Medical_History_Completed(Medical_condition)
   - Include ONLY:  
     * include the name surgeries/procedures from the past
     * Medical conditions that are:(Added in the Medical_condition)
       - Completely resolved AND
       - No longer requiring any medication
     * if there is more Medical_condition add ,
   - DO NOT include:
     * Any condition still requiring medication
     * Any ongoing conditions
     * Subject drug-related conditions
     * Conditions without clear resolution status
 
2. Past_Medications(Medication_Name_Past)
   - Include ONLY medications specifically used for Past_Medical_History_Completed
   - DO NOT include:
     * Subject drug
     * Current medications
     * Medications for ongoing conditions
     * ADR treatments
     * if there is more Medical_condition add ,
 
3. Past_Current_Conditions_Before_Subject_Drug(Medical_condition)
   - Include ONLY:
     * Medical conditions that existed before Subject Drug
     * Ongoing conditions which is takeing from the past
   - DO NOT include:
     * Resolved conditions (these go in Past_Medical_History_Completed)
     * Subject drug-related conditions
     * Post-subject drug conditions
     * ADR-related conditions

 
4. Concomitant_Medications:
   - Include up to 3 current medications taken alongside the Subject Drug that are specifically for conditions listed in *Past_Current_Conditions_Before_Subject_Drug*.
   - For each medication, include its name and the start date, formatted as (Medication_Name, Start_Date).
   - If a start date is not mentioned for a medication, leave it as an empty string, like.
   - If more than 3 such medications are mentioned, include only the first 3 in the text.  
- DO NOT include:
   - DO NOT include the Subject Drug itself.
   - DO NOT include medications that are only for treating adverse events or side effects (ADR treatment).
   - DO NOT include any medications that were specifically for conditions that have already been resolved.
   - DO NOT include new medications that were started after the introduction of the Subject Drug.
   - should not give the ADR treatment which is taken for the ADR
   **CRITICAL LOGIC:**
   If a condition is in Past_Current_Conditions_Before_Subject_Drug AND the patient is currently taking medication for it, that medication MUST be in Concomitant_Medications.

    MANDATORY VALIDATION CHECKLIST (COMPLETE BEFORE ANSWERING):
    1. ✓ Subject drug is correctly identified (usually the newly prescribed or problematic medication, NOT maintenance meds)
    2. ✓ Subject drug and its indication are NOT included in Past_Current_Conditions_Before_Subject_Drug or Concomitant_Medications  
    3. ✓ Each Past_Medication MUST be for a condition in Past_Medical_History_Completed  
    4. ✓ Each Concomitant_Medication MUST be for a condition in Past_Current_Conditions_Before_Subject_Drug  
    5. ✓ CRITICAL: Each condition in Past_Current_Conditions_Before_Subject_Drug MUST have its corresponding current medication listed in Concomitant_Medications  
    6. ✓ CRITICAL: NO condition can appear in BOTH Past_Medical_History_Completed AND Past_Current_Conditions_Before_Subject_Drug  
    7. ✓ Present tense medications ("I am on", "I take", "takes") are NOT in Past_Medications  
    8. ✓ Past tense medications ("I took", "was on") are NOT in Concomitant_Medications  
    9. ✓ Conditions without current active treatment are NOT in Past_Current_Conditions_Before_Subject_Drug  
    10. ✓ If Concomitant_Medications is empty but Past_Current_Conditions_Before_Subject_Drug is not empty, re-examine the text for current medications  
    11. ✓ If a condition is resolved and no longer treated, it belongs ONLY in Past_Medical_History_Completed  
    12. ✓ If Reporting_for_someone is "yes", Patient_Name should be empty string
    13. ✓ Concomitant_Medications contains ONLY medication names (no dosage, frequency, or format details)

5. Indication  
   Objective: Extract the **medical condition** for which the **subject drug** was prescribed.  
   Extraction Rules:
   * Must correspond to the subject drug only  
   * Return the specific medical condition only — no surrounding explanation  
   * If the indication is unclear or not related to the subject drug, return an empty string ""
   * The indication should match why the subject drug was newly prescribed or is being reported about

Provide the extracted information in JSON format."""
}

# Flask App
app = Flask(__name__)
CORS(app)

def extract_info(text, category):
    try:
        logging.info(f"Extracting {category} information")
        
        full_prompt = (
            f"Current year is {datetime.now().year}. "
            "IMPORTANT: Respond ONLY in valid JSON format. "
            "No explanatory text. "
            f"{PROMPT[category].format(text=text)}"
        )
        
        response_text = call_ollama(full_prompt)
        
        try:
            extracted_info = parse_json(response_text)
            # Standardize field names
            extracted_info = standardize_field_names(extracted_info, category)
            return extracted_info
        except Exception as e:
            logging.error(f"JSON parsing failed: {e}")
            return {"error": f"Failed to parse JSON. Raw response: {response_text}"}

    except Exception as e:
        logging.error(f"An error occurred during extraction: {str(e)}")
        return {"error": f"An error occurred during extraction: {str(e)}"}

@app.route('/extract/patient-medical-history', methods=['POST'])
def patient_medical_history_extraction():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        result = extract_info(text, 'patient_medical_history')
        return jsonify(result)
    except Exception as e:
        logging.error(f"Patient medical history extraction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5008, debug=True)