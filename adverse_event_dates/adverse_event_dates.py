import json
import re
import ast
import logging
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='adverse_event_dates_extraction.log'
)

# Field name mappings for standardization
FIELD_MAPPINGS = {
    'adverse_event_dates': {
        'subject_drug_name': 'Subject_Drug_Name',
        'product_start_date': 'Product_Start_Date',
        'product_end_date': 'Product_End_Date',
        'duration_product': 'Duration_Product',
        'adr_start_date': 'ADR_Start_Date',
        'adr_end_date': 'ADR_End_Date',
        'duration_adr': 'Duration_ADR'
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

# Main extraction prompt
PROMPT = """You are a medical expert specializing in pharmacovigilance and information extraction.

CRITICAL: You MUST follow these rules EXACTLY - NO EXCEPTIONS WHATSOEVER:

RESPONSE FORMAT:
- Respond ONLY with a valid JSON object
- Do NOT include any explanatory text outside the JSON
- Use the exact field names specified below

ABSOLUTE DATE EXTRACTION RULES - ZERO TOLERANCE FOR VIOLATIONS:

1. ONLY extract dates in these TWO specific cases:
   a) Dates with EXPLICIT YEAR mentioned in the text (e.g., "15-07-2025", "July 2025", "2025")
   b) ONLY these EXACT relative terms: "today", "yesterday", "day before"

2. FORBIDDEN - NEVER EXTRACT these date references:
   - "Recently" → EMPTY STRING
   - "After the first dose" → EMPTY STRING  
   - "Following" → EMPTY STRING
   - "Later" → EMPTY STRING
   - "Soon after" → EMPTY STRING
   - "Subsequently" → EMPTY STRING
   - Month names without year (e.g., "July", "January", "March") → EMPTY STRING
   - Day-month combinations without year (e.g., "15 July", "March 10") → EMPTY STRING
   - Vague time references (e.g., "last week", "few days ago", "recently", "following days", "over the days") → EMPTY STRING
   - Duration terms (e.g., "3 days", "2 weeks", "few weeks") → EMPTY STRING
   - Past tense references (e.g., "subsided", "resolved", "recovered") → EMPTY STRING
   - ANY inference or assumption about timing → EMPTY STRING
   - Temporal sequence words (e.g., "after", "before", "then", "next") → EMPTY STRING

3. DO NOT auto-populate current date UNLESS text explicitly says exactly "today"

4. Handle ONLY these relative dates based on current receipt date: {current_date}
   - "today" = current receipt date: {current_date}
   - "yesterday" = current receipt date - 1 day: {yesterday_date}
   - "day before" = current receipt date - 2 days: {day_before_date}

EXAMPLES OF WHAT TO EXTRACT VS NOT EXTRACT:

EXTRACT (return actual date):
- "15-07-2025" → "2025-07-15"
- "July 2025" → "--07-2025"  
- "2025" → "--2025"
- "today" → {current_date}
- "yesterday" → {yesterday_date}
- "day before" → {day_before_date}

DO NOT EXTRACT (return empty string ""):
- "Recently" → ""
- "After the first dose" → ""
- "Following the medication" → ""
- "July" (no year) → ""
- "15 July" (no year) → ""
- "last week" → ""
- "few days ago" → ""
- "subsequently" → ""
- "later that day" → ""
- "the next morning" → ""

DATE FORMAT REQUIREMENTS:
- Full date with year: DD-MM-YYYY format → convert to YYYY-MM-DD
- Month-Year only: MM-YYYY format → convert to --MM-YYYY
- Year only: YYYY format → convert to --YYYY
- If YEAR is NOT explicitly mentioned AND it's not exactly "today/yesterday/day before", return empty string ""

Text to analyze:
{text}

Extract the following information:

1. Subject_Drug_Name
   - Extract only the specific name of the drug associated with the adverse event
   - Do not include generic terms like "tablet", "medicine", "pill"
   - Do not include dosage or form information
   - Return empty string if no specific drug name is mentioned

2. Product_Start_Date
   - Start date when the subject drug was first taken
   - STRICT RULE: ONLY extract if text contains:
     * Complete date with year explicitly mentioned (DD-MM-YYYY) → return as YYYY-MM-DD
     * Month-Year explicitly mentioned (MM-YYYY) → return as --MM-YYYY
     * Year only explicitly mentioned (YYYY) → return as --YYYY
     * Exactly the word "today" → {current_date}
     * Exactly the word "yesterday" → {yesterday_date}
     * Exactly the phrase "day before" → {day_before_date}
   - FORBIDDEN EXTRACTIONS (return empty string):
     * "Recently" → ""
     * "After diagnosis" → ""
     * "Following prescription" → ""
     * "Started treatment" → ""
     * Any vague timing reference → ""

3. Product_End_Date
   - End date when the subject drug was stopped
   - STRICT RULE: ONLY extract if text contains:
     * Complete date with year explicitly mentioned (DD-MM-YYYY) → return as YYYY-MM-DD
     * Month-Year explicitly mentioned (MM-YYYY) → return as --MM-YYYY
     * Year only explicitly mentioned (YYYY) → return as --YYYY
     * Exactly the word "today" → {current_date}
     * Exactly the word "yesterday" → {yesterday_date}
     * Exactly the phrase "day before" → {day_before_date}
   - FORBIDDEN EXTRACTIONS (return empty string):
     * "Doctor advised to stop" → ""
     * "Discontinued" → ""
     * Any indication drug was stopped without explicit date with year → ""

4. Duration_Product
   - Calculate duration ONLY if both Product_Start_Date and Product_End_Date are complete dates (YYYY-MM-DD format)
   - Return empty string if either date is incomplete or missing

5. ADR_Start_Date
   - Start date when adverse reaction first appeared
   - STRICT RULE: ONLY extract if text contains:
     * Complete date with year explicitly mentioned (DD-MM-YYYY) → return as YYYY-MM-DD
     * Month-Year explicitly mentioned (MM-YYYY) → return as --MM-YYYY
     * Year only explicitly mentioned (YYYY) → return as --YYYY
     * Exactly the word "today" → {current_date}
     * Exactly the word "yesterday" → {yesterday_date}
     * Exactly the phrase "day before" → {day_before_date}
   - FORBIDDEN EXTRACTIONS (return empty string):
     * "After the first dose" → ""
     * "Following medication" → ""
     * "Soon after taking" → ""
     * "Subsequently developed" → ""
     * "Later experienced" → ""
     * Any sequence-based timing → ""

6. ADR_End_Date
   - End date when adverse reaction resolved completely
   - STRICT RULE: ONLY extract if text contains:
     * Complete date with year explicitly mentioned (DD-MM-YYYY) → return as YYYY-MM-DD
     * Month-Year explicitly mentioned (MM-YYYY) → return as --MM-YYYY
     * Year only explicitly mentioned (YYYY) → return as --YYYY
     * Exactly the word "today" → {current_date}
     * Exactly the word "yesterday" → {yesterday_date}
     * Exactly the phrase "day before" → {day_before_date}
   - FORBIDDEN EXTRACTIONS (return empty string):
     * "Resolved" → ""
     * "Subsided" → ""
     * "Recovered" → ""
     * "Improved" → ""
     * Any resolution indication without explicit date with year → ""

7. Duration_ADR
   - Calculate duration ONLY if both ADR_Start_Date and ADR_End_Date are complete dates (YYYY-MM-DD format)
   - Return empty string if either date is incomplete or missing

FINAL VALIDATION RULES:
- If you cannot find an explicit year OR the exact words "today"/"yesterday"/"day before" for any date field, that field MUST be empty string ""
- NEVER auto-populate today's date for vague terms
- NEVER infer dates from context or sequence
- When in doubt, ALWAYS return empty string ""
- Current date for reference only: {current_date}

REMEMBER: The text "Recently, she was diagnosed with EDS and prescribed Voclosporin. After the first dose, she developed a fever" contains NO extractable dates because:
- "Recently" has no year → ""
- "After the first dose" has no year → ""

Return ONLY valid JSON with the 7 fields above."""

def extract_info(text):
    current_date = datetime.now()
    yesterday_date = current_date - timedelta(days=1)
    day_before_date = current_date - timedelta(days=2)
    
    current_date_str = current_date.strftime("%Y-%m-%d")
    yesterday_date_str = yesterday_date.strftime("%Y-%m-%d")
    day_before_date_str = day_before_date.strftime("%Y-%m-%d")
    
    category = 'adverse_event_dates'
    
    try:
        logging.info(f"Extracting {category} information")
        
        # Enhanced prompt with even stricter instructions
        full_prompt = (
            "CRITICAL INSTRUCTION: You are STRICTLY FORBIDDEN from auto-populating today's date. "
            "The text 'Recently' and 'After the first dose' do NOT contain years and must result in EMPTY STRINGS. "
            "ONLY extract dates with explicit years or exactly 'today'/'yesterday'/'day before'. "
            "NEVER EVER extract dates from vague timing words. "
            "Respond ONLY in valid JSON format with NO explanatory text. "
            f"{PROMPT.format(text=text, current_date=current_date_str, yesterday_date=yesterday_date_str, day_before_date=day_before_date_str)}"
        )
        
        response_text = call_ollama(full_prompt)
        
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

        try:
            extracted_info = parse_json(response_text)
            # Standardize field names before processing
            extracted_info = standardize_field_names(extracted_info, category)
        except Exception as e:
            logging.error(f"JSON parsing failed: {e}")
            return {"error": f"Failed to parse JSON. Raw response: {response_text}"}

        # POST-PROCESSING VALIDATION: Force empty strings for invalid dates
        def validate_date_field(date_value, field_name):
            """Validate that date field follows the strict rules"""
            if not date_value or date_value == "":
                return ""
            
            # Check if it's a valid format we allow
            valid_formats = [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^--\d{2}-\d{4}$',      # --MM-YYYY  
                r'^--\d{4}$'             # --YYYY
            ]
            
            # Check if it matches our current/yesterday/day before dates
            allowed_relative_dates = [current_date_str, yesterday_date_str, day_before_date_str]
            
            if date_value in allowed_relative_dates:
                return date_value
                
            for pattern in valid_formats:
                if re.match(pattern, date_value):
                    return date_value
            
            # If we get here, it's an invalid date format
            logging.warning(f"Invalid date format detected in {field_name}: {date_value}. Setting to empty string.")
            return ""

        # Validate all date fields
        extracted_info["Product_Start_Date"] = validate_date_field(
            extracted_info.get("Product_Start_Date", ""), "Product_Start_Date"
        )
        extracted_info["Product_End_Date"] = validate_date_field(
            extracted_info.get("Product_End_Date", ""), "Product_End_Date"
        )
        extracted_info["ADR_Start_Date"] = validate_date_field(
            extracted_info.get("ADR_Start_Date", ""), "ADR_Start_Date"
        )
        extracted_info["ADR_End_Date"] = validate_date_field(
            extracted_info.get("ADR_End_Date", ""), "ADR_End_Date"
        )

        # Additional validation: If date equals current date but text doesn't say "today", clear it
        if (extracted_info.get("ADR_Start_Date") == current_date_str and 
            "today" not in text.lower()):
            logging.warning("ADR_Start_Date was auto-populated incorrectly. Setting to empty string.")
            extracted_info["ADR_Start_Date"] = ""
            
        if (extracted_info.get("Product_Start_Date") == current_date_str and 
            "today" not in text.lower()):
            logging.warning("Product_Start_Date was auto-populated incorrectly. Setting to empty string.")
            extracted_info["Product_Start_Date"] = ""

        # Simple duration calculation only for complete dates
        if (extracted_info.get("Product_Start_Date", "").count("-") == 2 and 
            extracted_info.get("Product_End_Date", "").count("-") == 2 and
            not extracted_info.get("Product_Start_Date", "").startswith("--") and
            not extracted_info.get("Product_End_Date", "").startswith("--")):
            try:
                start_date = datetime.strptime(extracted_info["Product_Start_Date"], "%Y-%m-%d")
                end_date = datetime.strptime(extracted_info["Product_End_Date"], "%Y-%m-%d")
                duration = (end_date - start_date).days + 1
                extracted_info["Duration_Product"] = str(duration) if duration > 0 else ""
            except:
                extracted_info["Duration_Product"] = ""
        else:
            extracted_info["Duration_Product"] = ""

        # Calculate ADR Duration
        if (extracted_info.get("ADR_Start_Date", "").count("-") == 2 and 
            extracted_info.get("ADR_End_Date", "").count("-") == 2 and
            not extracted_info.get("ADR_Start_Date", "").startswith("--") and
            not extracted_info.get("ADR_End_Date", "").startswith("--")):
            try:
                start_date = datetime.strptime(extracted_info["ADR_Start_Date"], "%Y-%m-%d")
                end_date = datetime.strptime(extracted_info["ADR_End_Date"], "%Y-%m-%d")
                duration = (end_date - start_date).days + 1
                extracted_info["Duration_ADR"] = str(duration) if duration > 0 else ""
            except:
                extracted_info["Duration_ADR"] = ""
        else:
            extracted_info["Duration_ADR"] = ""

        return extracted_info

    except Exception as e:
        logging.error(f"An error occurred during extraction: {str(e)}")
        return {"error": f"An error occurred during extraction: {str(e)}"}

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/extract/adverse-event-dates', methods=['POST'])
def adverse_event_dates_extraction():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        result = extract_info(text)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Adverse event dates extraction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)