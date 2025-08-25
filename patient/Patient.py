import json
import re
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='patient_extraction.log'
)

app = Flask(__name__)
CORS(app)

# Field name mappings for standardization
FIELD_MAPPINGS = {
    'patient': {
        'reporting_for_someone': 'Reporting_for_someone',
        'patient_name': 'Patient_Name',
        'patient_age': 'Patient_Age',
        'dob': 'DOB',
        'gender': 'Gender',
        'pregnancy': 'Pregnancy',
        'address': 'Address',
        'city': 'City',
        'zip_code': 'Zip_Code',
        'country': 'Country',
        'phone_number': 'Phone_number'
    }
}

# Standardizer for date handling
class DateStandardizer:
    @staticmethod
    def get_current_year():
        return datetime.now().year

    @staticmethod
    def standardize_date(date_str):
        if not date_str:
            return ""
        try:
            # Handle month name format like "September 15, 1988"
            for fmt in ["%Y-%m-%d", "%B %d, %Y", "%d %B %Y"]:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    continue
                    
            # Try to parse the date from pieces
            month_names = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            
            # Look for patterns like "September 15, 1988"
            match = re.search(r'(\w+)\s+(\d{1,2})(?:,|\s)+(\d{4})', date_str, re.IGNORECASE)
            if match:
                month_name, day, year = match.groups()
                month = month_names.get(month_name.lower())
                if month:
                    return f"{year}-{month:02d}-{int(day):02d}"
                    
            # Try other date formats or return empty if can't parse
            return ""
        except Exception:
            return ""

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

# Parse JSON response
def parse_json(text):
    # Remove markdown backticks if present
    text = re.sub(r'```(json)?|```', '', text).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try to extract JSON object if it's surrounded by other text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
    
    logging.error(f"Failed to parse JSON: {text}")
    return {}

# Standardizing function
def standardize_field_names(data, category):
    if not isinstance(data, dict):
        return data
        
    standardized_data = {}
    mapping = FIELD_MAPPINGS.get(category, {})
    
    for key, value in data.items():
        std_key = key.lower().replace(' ', '_')
        standardized_key = mapping.get(std_key, key)
        
        # Special handling for DOB to standardize date format
        if standardized_key == 'DOB' and value:
            value = DateStandardizer.standardize_date(value)
            
        standardized_data[standardized_key] = value
        
    return standardized_data

# Main extraction route
PROMPT = {
    "patient": """You are a medical expert specializing in pharmacovigilance and information extraction.

IMPORTANT INSTRUCTIONS:
- Respond ONLY with a valid JSON object
- Do NOT include any explanatory text outside the JSON
- If no information is found, use empty strings for values
- Use the exact field names specified below

Extract the following patient information from the given medical text. If any information is not available or not provided, return an empty string for that field.

Text:
{text}

Please extract:
0. Reporting_for_someone: yes or no
   - "yes" if the report is about a patient from a third party (someone else reporting for the patient), otherwise "no"
   - Look for indicators like "my patient", "patient [name]", third-person references
   - If the text uses first-person language ("I took", "I experienced", "my symptoms") then it's "no"

1. Patient_Name:
   - CRITICAL RULES TO FOLLOW:
   
   **Rule 1: Self-reporting with no specific name mentioned**
   - If Reporting_for_someone is "no" AND no specific patient name is found in the text
   - Return exactly: "Reporter_Name"
   
   **Rule 2: Self-reporting with specific name mentioned**
   - If Reporting_for_someone is "no" AND a specific patient name is mentioned in the text
   - Extract the actual name mentioned (without titles)
   
   **Rule 3: Third-party reporting**
   - If Reporting_for_someone is "yes", extract the patient's name if mentioned
   - Look for patterns like: third-person references, family relationships, patient identifiers
   - If no specific patient name is mentioned, return empty string
   
   **Name Extraction Guidelines:**
   - ALWAYS remove titles: Mr., Mrs., Ms., Miss, Dr., etc.
   - Extract full names when available (first and last names)
   - Look for explicit name indicators in the text
   - Handle punctuation carefully around names
   
2. Patient_Age
3. DOB (only if DOB is there else empty)
4. Gender 
    - should be exactly one of: "Male", "Female" or empty string
   - Look for explicit mentions of gender (e.g., "male", "female", "woman", "man").
   - Check for gender-specific titles (e.g., "Mr.", "Mrs.", "Ms.", "Sir", "Madam").
   - Analyze names that might indicate gender.
   - Look for pronouns (he/him, she/her, they/them).
   - Consider context clues such as mentions of pregnancy or gender-specific medical conditions.
   - If the gender is clearly female (e.g., mentions of pregnancy), use "Female".
   - If no clear indication is given or if there's any ambiguity, use an empty string.
   - Recognize familial roles or relational terms (e.g., "father," "mother," "husband," "wife") that can imply gender.
   - from name it should'not take the gender it should be in those cases

5. Pregnancy (should be exactly one of: "Yes", "No", or empty string)
   - if patient have Pregnancy in current then only "Yes" else "No"
   - if Patient have Pregnancy in the past its should "No"
6. Address (if present, extract entire address)
7. City
8. Zip_Code
9. Country
10. Phone_number

Provide the extracted information in JSON format."""
}

@app.route('/extract/patient', methods=['POST'])
def patient_extraction():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        logging.info("Extracting patient information")
        
        current_year = DateStandardizer.get_current_year()
        prompt = f"Current year is {current_year}. {PROMPT['patient'].format(text=text)}"
        
        response_text = call_ollama(prompt)
        
        # Parse JSON from the response text
        extracted_info = parse_json(response_text)
        
        # Standardize field names and formats
        standardized_info = standardize_field_names(extracted_info, 'patient')
        
        return jsonify(standardized_info)
    except Exception as e:
        logging.error(f"Error during extraction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5007, debug=True)