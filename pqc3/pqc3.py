import json
import re
import ast
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='medical_pqc_extraction.log'
)

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

        logging.info(f"Ollama Raw Response: {response.text}")  # Log raw response

        if response.status_code == 200:
            result = response.json()
            logging.info(f"Parsed JSON: {result}")  # Log parsed response
            return result.get('response', '')  
        else:
            logging.error(f"Ollama API Error: {response.status_code} - {response.text}")
            return ''
    except Exception as e:
        logging.error(f"Ollama API request failed: {str(e)}")
        return ''




# Prompt for PQC extraction
# Updated Prompt for PQC extraction
PQC_PROMPT = """Extract the product quality complaint from the given text. A product quality complaint refers to any **physical abnormality, contamination, packaging defect, discoloration, breakage, or any other manufacturing-related problem** observed in the product.

Do **NOT** consider **side effects, adverse drug reactions (ADRs), lack of improvement, or treatment failure** as product quality issues.

Return the extracted issue as **'Product_Quality_Complaint_Term'**. If an issue is found, set **'Product_Quality_Issue'** to 'Yes'. If no issue is found, return **'Product_Quality_Complaint_Term'** as an empty string and **'Product_Quality_Issue'** as 'No'.

Match the extracted issue with one of the following predefined product quality complaint terms if applicable:
"Damaged package", "Empty Container", "Blister pack hard to open", "Empty blister pack",
"Broken seal", "Cartons exposed to moisture", "Missing product label/labelling",
"Incorrect product label/labelling", "Inaccurate product labels/labelling",
"Unreadable/illegible product labels/labelling", "Damaged package insert", "Dissolution",
"Leakage", "Mixed strength", "Mixed product", "Product defect", "Chipped, cracked, or splitting tablets",
"Inconsistent tablet sizes", "Tablet or capsule discolorations", "Incorrect Strength (10mg tablets found in 20mg bottle)",
"Unusual taste or odour", "Colour/appearance is abnormal", "Holes in the tablet", "Deformed and melted capsules",
"Capsules contents spilled in the bottle/cavity", "spots on the tablets", "Hard outer layer of soft gel capsules",
"uneven coating and pitting", "Missing/Shortage tablets or capsules", "Medicine/product not effective",
"low effectiveness", "Reduced efficacy", "worsening of symptoms", "drug resistance",
"Bacterial Contamination", "Foreign material within the product container", "Fiber embedded in the tablets",
"Fungal/mould contamination", "Expired product", "Accidental intake of medicine due to resemblance with other medicine",
"Counterfeit products", "Disappointment with product quality".

If the complaint in the input text closely matches one of these terms, return that exact terms in 'Product_Quality_Complaint_Term'. If **no matching quality issue** is found, return **'Product_Quality_Complaint_Term' as an empty string and 'Product_Quality_Issue' as 'No'.**
"""

def parse_json(text):
    try:
        text = re.sub(r'```(json)?|```', '', text).strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)  
            logging.info(f"Successfully Parsed JSON: {parsed_json}")
            return parsed_json  # Correctly extracted JSON
        logging.warning("No valid JSON found in response")
        return {}  
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}")
        return {}


def group_similar_responses(responses, similarity_threshold=0.8):
    """
    Group similar non-empty responses based on similarity threshold.
    Return most frequent term from the largest group.
    """
    if not responses:
        return []

    groups = []

    for response in responses:
        response_clean = response.strip().lower()
        if not response_clean:
            continue

        placed = False
        for group in groups:
            if SequenceMatcher(None, response_clean, group[0].strip().lower()).ratio() >= similarity_threshold:
                group.append(response)
                placed = True
                break

        if not placed:
            groups.append([response])

    # Log grouped content
    logging.info(f"Grouped Responses: {groups}")

    # Return representative term from each group (optional)
    # return [group[0] for group in groups]

    # ✅ Better: pick the term from the largest group (most common idea)
    largest_group = max(groups, key=len, default=[None])
    return [largest_group[0]] if largest_group[0] else []


# def extract_classification_terms(text):
#     try:
#         full_prompt = f"""{PQC_PROMPT}
        
#         Input: {text}
#         Output (JSON format expected):"""

#         logging.info(f"Generated Prompt: {full_prompt}")  # Log prompt

#         response_text = call_ollama(full_prompt)
#         logging.info(f"Raw Response Before Parsing: {response_text}")  # Log raw response

#         extracted_info = parse_json(response_text)
#         logging.info(f"Parsed Extracted Info: {extracted_info}")  # Log parsed output

#         result = {
#             'Product_Quality_Issue': extracted_info.get('Product_Quality_Issue', 'no').lower(),
#             'Product_Quality_Complaint_Term': extracted_info.get('Product_Quality_Complaint_Term', '').strip()
#         }

#         if result['Product_Quality_Complaint_Term'] and result['Product_Quality_Issue'] == 'no':
#             result['Product_Quality_Issue'] = 'yes'  

#         logging.info(f"Final Extracted Info: {result}")
#         return result

#     except Exception as e:
#         logging.error(f"Error during extraction: {e}")
#         return {
#             'Product_Quality_Issue': 'no',
#             'Product_Quality_Complaint_Term': ''
#         }


def extract_classification_terms(text):
    try:
        full_prompt = f"""{PQC_PROMPT}
        
        Input: {text}
        Output (JSON format expected):"""

        logging.info(f"Generated Prompt: {full_prompt}")  # Log prompt

        term_list = []

        for i in range(5):
            logging.info(f"Calling Ollama - Attempt {i+1}")
            response_text = call_ollama(full_prompt)
            logging.info(f"Raw Response Attempt {i+1}: {response_text}")
            extracted_info = parse_json(response_text)

            term = extracted_info.get('Product_Quality_Complaint_Term', '').strip()
            term_list.append(term)
        
        grouped_terms = group_similar_responses(term_list)

        # Majority vote on Product_Quality_Complaint_Term
        most_common_term = Counter(grouped_terms).most_common(1)
        final_term = most_common_term[0][0] if most_common_term and most_common_term[0][0] else ''

        final_result = {
            'Product_Quality_Complaint_Term': final_term,
            'Product_Quality_Issue': 'yes' if final_term else 'no'
        }

        logging.info(f"Final Extracted Info (with majority vote on term only): {final_result}")
        return final_result

    except Exception as e:
        logging.error(f"Error during extraction: {e}")
        return {
            'Product_Quality_Issue': 'no',
            'Product_Quality_Complaint_Term': ''
        }




# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/medical-terms', methods=['POST'])
def medical_terms_extractor():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                "error": "No text provided",
                "status": 400
            }), 400

        # Extract text
        text = data['text']
        
        # Extract classification terms
        classification_info = extract_classification_terms(text)
        
        # Return only the required fields
        return jsonify({
            "Product_Quality_Complaint_Term": classification_info.get('Product_Quality_Complaint_Term', ''),
            "Product_Quality_Issue": classification_info.get('Product_Quality_Issue', 'no')
        }), 200

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "status": 500
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Medical Terms Extractor",
        "version": "1.0.0"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, debug=True)