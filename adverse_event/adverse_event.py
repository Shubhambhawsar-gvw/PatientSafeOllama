import json
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='adverse_event_extraction.log'
)

app = Flask(__name__)
CORS(app)

# Field name mappings for standardization
FIELD_MAPPINGS = {
    'adverse_event': {
        'route_of_administration': 'Route_of_administration',
        'action_taken': 'Action_Taken',
        'side_effect_resolved': 'side_effect_resolved',
        'side_effect_reappear': 'side_effect_reappear',
        'adr_medications': 'ADR_medications',
        'name_of_adr_treatment_medication': 'Name_of_ADR_Treatment_Medication',
        'adr_current_status': 'ADR_current_status'
    }
}

def fix_side_effect_resolved(data):
    """
    If ADR_medications is "Yes", set side_effect_resolved to "N/A"
    
    Args:
        data (dict): The extracted adverse event data
    
    Returns:
        dict: Modified data with corrected side_effect_resolved
    """
    if data.get("ADR_medications") == "Yes":
        data["side_effect_resolved"] = "N/A"
    
    return data

def post_process_adverse_event(extracted_info):
    """Apply business rules to ensure consistency in adverse event data"""
    processed_info = extracted_info.copy()
    
    # Apply the fix_side_effect_resolved function
    processed_info = fix_side_effect_resolved(processed_info)
    
    return processed_info

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
            return '{}'
    except Exception as e:
        logging.error(f"Ollama API request failed: {str(e)}")
        return '{}'

def parse_json_response(response_text):
    """Parse JSON from the response text, handling different formats"""
    try:
        # Strip any markdown formatting
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
        return {}

# Main extraction route
PROMPT = """You are a medical expert specializing in pharmacovigilance and information extraction.

IMPORTANT INSTRUCTIONS:
- Respond ONLY with a valid JSON object
- Do NOT include any explanatory text outside the JSON
- If no information is found, use empty strings for values
- Use the exact field names specified below

Extract the following adverse event information from the given medical text. If any information is not available or not provided, return an empty string for that field.

Text:
{text}

Please extract:
1. Route_of_administration:
   Must be EXACTLY ONE of: 
   "Oral", "Topical", "Intravenous", "Intramuscular", "Intrathecal", 
   "Nasal", "Subcutaneous", "Rectal", "Vaginal", "Ocular", "Otic"
   Or empty string if route unclear

2. Action_Taken
   VALIDATION RULES:
   - Extract ONLY actions taken with the SUBJECT DRUG in response to adverse events
   - Return exactly ONE of:
     * "Drug withdrawn" - if subject drug was stopped due to adverse event
     * "Dose reduced" - if subject drug dose was decreased due to adverse event
     * "Dose increased" - if subject drug dose was increased due to adverse event
     * "Dose not changed" - if subject drug dose remained same despite adverse event
     * "Not Changed" - if no changes made to subject drug
     * "Unknown" - if action taken is unclear but text discusses adverse event
     * "" (empty string) - if no adverse event or no subject drug mentioned
   - Do NOT consider actions taken with other medications

3. side_effect_resolved (must be EXACTLY one of: "Yes", "No", "N/A", "Unknown")
   Rules for determination:
   - "Yes" if:
     * Action taken was "Drug withdrawn" or "Dose reduced" AND
     * Outcome is "recovered/resolved" or "recovering/resolving" or "recovered/resolved with sequelae"
   - "No" if:
     * Action taken was "Drug withdrawn" or "Dose reduced" AND
     * Outcome is "not recovered/not resolved/ongoing"
     * it should follow the above rules else "N/A"
   - "N/A" if ANY of these are true:
     * Action taken was "Dose increased" or "Dose not changed" or "Not Changed"
     * Outcome is "fatal" or "Unknown"
     * ADR medications were given (ADR_medications = "Yes")
     
   - "Unknown" if:
       * any of the above three not there its "Unknown"

       
4. side_effect_reappear (must be EXACTLY one of: "Unknown", "N/A")
   Rules for determination:
   - "N/A" is all the case
   - "Unknown" should come only if side_effect_resolved is "Unknown"

5. ADR_medications
   VALIDATION RULES:
   - Must return exactly ONE of:
     * "Yes" - ONLY if medications were specifically given to treat an adverse event
     * "No" - ONLY if text explicitly states no medications given for adverse event
     * "" (empty string) - if no clear adverse event mentioned or unclear if medications given
   - Regular medications for other conditions do NOT count
   - Medications taken for symptoms unrelated to subject drug do NOT count

6. Name_of_ADR_Treatment_Medication
   VALIDATION RULES:
   - Must return an array of strings
   - Include ONLY medications explicitly given to treat adverse events from subject drug
   - Format: ["medication_name strength/dose"]
   - Return empty array [] if:
     * No adverse events mentioned
     * No treatment medications given
     * Medications are for conditions unrelated to subject drug
   - Do NOT include:
     * The subject drug itself
     * Regular medications for other conditions
     * Medications taken for symptoms unrelated to subject drug

7. ADR_current_status
   VALIDATION RULES:
   - Must return exactly ONE of:
     * "recovered/resolved" - ONLY if adverse event completely resolved
     * "recovering/resolving" - ONLY if adverse event showing improvement
     * "not recovered/not resolved/ongoing" - ONLY if adverse event still present
     * "recovered/resolved with sequelae" - ONLY if resolved but with lasting effects
     * "fatal" - ONLY if adverse event resulted in death
     * "unknown" - if outcome is unclear but adverse event occurred
     * "" (empty string) - if no adverse event mentioned
   - Default to "" if no clear adverse event described

CRITICAL OVERALL RULES:
1. Only extract information about adverse reactions to the subject drug
2. Regular symptoms or conditions NOT caused by subject drug should be ignored
3. When in doubt, return empty values rather than making assumptions
4. All fields should use consistent logic - if no adverse event is found, all related fields should indicate this
5. Cross-validate responses - e.g., if ADR_medications is "No", Name_of_ADR_Treatment_Medication must be empty

Provide the extracted information in JSON format."""

@app.route('/extract/adverse-event', methods=['POST'])
def adverse_event_extraction():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        # Create the prompt with the provided text
        prompt = PROMPT.format(text=text)
        
        # Call Ollama API
        response_text = call_ollama(prompt)
        
        # Parse the JSON response
        extracted_info = parse_json_response(response_text)
        
        # Apply post-processing rules (includes fix_side_effect_resolved)
        processed_info = post_process_adverse_event(extracted_info)
        
        # Return the processed information directly as JSON
        return jsonify(processed_info)
    except Exception as e:
        logging.error(f"Error in adverse event extraction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)