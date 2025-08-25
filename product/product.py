import json
import logging
import requests
import re
import ast
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='product_extraction.log'
)

app = Flask(__name__)
CORS(app)

# Field name mappings for standardization
FIELD_MAPPINGS = {
    'product': {
        'subject_drug_name': 'Subject_Drug_Name',
        'generic_name': 'Generic_Name',
        'batch_id': 'Batch_ID',
        'product_form': 'Product_Form',
        'indication': 'Indication',
        'dosage_strength': 'Dosage_Strength',
        'manufacturer_name': 'Manufacturer_Name',
        'marketer_name': 'Marketer_Name',
        'expiry_date': 'Expiry_Date',
        'product_dosage_frequency': 'Product_Dosage_Frequency'
    }
}

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

# Standardizing function
def standardize_field_names(data, category):
    if not isinstance(data, dict):
        return data
        
    standardized_data = {}
    mapping = FIELD_MAPPINGS.get(category, {})
    
    for key, value in data.items():
        std_key = key.lower().replace(' ', '_')
        standardized_key = mapping.get(std_key, key)
        standardized_data[standardized_key] = value
        
    return standardized_data

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

# Main extraction prompt
PROMPT = {
    "product": """You are a medical expert specializing in pharmacovigilance and information extraction.

IMPORTANT INSTRUCTIONS:
- Respond ONLY with a valid JSON object
- Do NOT include any explanatory text outside the JSON
- If no information is found, use empty strings for values
- Use the exact field names specified below

Extract the following product information from the given medical text. If any information is not available or not provided, return an empty string for that field.

Text:
{text}

Please extract:
1. Subject_Drug_Name
   - Extract only the specific name of the drug associated with the adverse event
   - Do not include generic terms like "tablet", "medicine", "pill"
   - Do not include dosage or form information
   - Return empty string if no specific drug name is mentioned

2. Generic_Name (if not present in the text return empty string)
3. Batch_ID
4. Product_Form 
   - Use EXACTLY ONE of the following case-sensitive options, or an empty string:
     Tablet, Capsule, Injection, Infusion, Suspension, Syrup, Pill, Gel, Powder, Solution, Ointment, Emulsion, Cream, Lotion, Eye_drops, Ear_drops, Spray, Inhaler, Paste, Plaster, Shampoo, Enema, Lozenge, Aerosol, Suppository, Elixir, Granules, Magma, Cachet, Chewing gum, Foam, Douch

5. Indication 
    * The medical condition for which the subject drug was prescribed
    * only include the medical condition no extra contents
    * If it is other than medical condition return empty

6. Dosage_Strength
   - Only consider the following units: mg, g, mcg, µg, ng, mL, L, mg/mL, g/mL, µg/L, mEq, tsp, tbsp, gtt, kg, mmol/L, U/L, ng/mL, %
   - If a valid dosage strength is found, return in format <number> <unit>
   - If text mentions number with units, extract it
   - Prioritize <number>/<unit> combinations if found

7. Manufacturer_Name
8. Marketer_Name
9. Expiry_Date
   - Extract only if text explicitly mentions "Expiry Date" for subject drug
   - Return an empty string if the date is ambiguous or incomplete

10.Product_Dosage_Frequency Use EXACTLY ONE of the following case-sensitive options, or an empty string if not applicable:
   QD, BID, TID, QID, 5X day, QHS, QAM, QPM, QOD, QH, Q2H, Q3H, Q4H, Q6H, Q8H, Q1W, Q2W, Q3W, Q4W, Q6W, Q8W, PRN, "" (empty string)
   Only it should return in short form
   Meanings:
   QD - Once daily
   BID - Twice daily
   TID - Three times daily
   QID - Four times daily
   5X day - Five times per day
   QHS - Every night at bedtime
   QAM - Every morning
   QPM - Every evening
   QOD - Every other day
   QH - Every hour
   Q2H - Every 2 hours
   Q3H - Every 3 hours
   Q4H - Every 4 hours
   Q6H - Every 6 hours
   Q8H - Every 8 hours
   Q1W - Once weekly
   Q2W - Every 2 weeks
   Q3W - Every 3 weeks
   Q4W - Every 4 weeks
   Q6W - Every 6 weeks
   Q8W - Every 8 weeks
   PRN - As needed
   "" - Empty string (when not applicable)
Provide the extracted information in JSON format."""
}

@app.route('/extract/product', methods=['POST'])
def product_extraction():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        # Get the current year for context
        current_year = datetime.now().year
        
        # Format the prompt with text
        full_prompt = (
            f"Current year is {current_year}. "
            "IMPORTANT: Respond ONLY in valid JSON format. "
            "No explanatory text. "
            f"{PROMPT['product'].format(text=text)}"
        )
        
        # Call Ollama
        response_text = call_ollama(full_prompt)
        
        # Parse the JSON string response
        try:
            extracted_info = parse_json(response_text)
            # Standardize field names
            extracted_info = standardize_field_names(extracted_info, 'product')
        except Exception as e:
            logging.error(f"JSON parsing failed: {e}")
            return jsonify({"error": f"Failed to parse JSON. Raw response: {response_text}"}), 500
        
        return jsonify(extracted_info)
    except Exception as e:
        logging.error(f"Product extraction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5010, debug=True)