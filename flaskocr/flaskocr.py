import json
import re
import ast
import logging
import requests
import io
import base64
from urllib.request import urlopen
from google.cloud import vision
from google.oauth2 import service_account
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='medicine_extraction.log'
)

# Google Cloud Vision API credentials
credentials_dict = {
    "type": "service_account",
    "project_id": "patientsafe",
    "private_key_id": "1e208137295c1d5aa21f16f002d144fa8257bbd9",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCpgWmriU2UgTdv\nDec2Jp4odlVTDB3a8lhgIWukWwvkYG470yP+DQyfWOyKvf6JcwmzWVS9UR2WCz7n\ni99dzWqF5IrqzgjSmOpyWHZKw+/MqsFzB3Q+Updn6FOT9Wc/PfICkvn6dcS0Hzup\nRSmxYj/3PofVmtohlKsh+y3mCihZeXGAkonQGeAhy+dvFQvCAkSa2wpMIyImVeJv\nPpi6/SGy+rwIaDmVuDJTE87IQTHfWx/BjTMNYNkSiSFmVuJlVFOmjVnezQWrwRRh\nz+3j4IIZpVglwZjj6nNJ3V0/xuplCG9qiAbxO30j/vkjP3Km/fEd+gea12Y/qy5x\nrWPsQsdNAgMBAAECggEAC615vTKZYKjCS+lWS3m/naQBXtZP3Dyj8AN9afZHzHUE\nQyTZru/xNa0npqtONOPzACMgXmaPNj4SGFufiNPVCeJj26dUHkYgqL4FcEMtaxk8\neJND4+NGwQbVczUN5uJ6oMjFZgblBU+9iXzVUX4mT/9D1mhzrKqUW6P1VorOmtNR\nNcIgoIz7jAIYubPIQhhM57pLiQ5VI501JdPYPxdwQt8X5P+PfD0e8aAcIz6eWicW\n85pZl+OzlkkLv4PRdpFyjAaZ5VVrUjxAwusxf4S0atOxazWskM7RsvUBYCsVYWWt\ne62d0Xg1w8ml1O9dFfwLxllFpwGJk7CrrBoJOjEu4QKBgQDkpOon+ZSPf+OA4DCW\nDonITykmsSb+4hqdhceIgRGdhNJNV82hGWZP0czwgmQw46uoSv2uo6mg8EcELQcW\nFVch1UfI/2Flxrf9O820cJEA4n4laDpurWuSy6PfhmW8I5pTMqW6YD29vmYrsGYD\nj/FbkEdWoynza88vItVepsD/RQKBgQC9ySeJxd+E2oO3evZECfaHhUplmS2Rqgt8\nFDecrOzbbIH5iu3qxL95rmgqHgRgXYY9a3yuhwNIk9lUs4ZJtfDFUuEZIFLsSfxG\nxEK5qkxGyM8SmNaAMwPdujrVQkinvHcwYh6oBLbNg+huRe6U8iFLFYxcshiUCnKD\n1BroZfkEaQKBgG+eutlNFGOBsZTm3ZFEA6uQSayj6z+fLhMcji5rpCfcAbbUWIR/\nG913tK7tWPAtFU4RXgr1xwfUhTbarIzxWhogPu03D1taSdQMb/3YvlFKQP8OBQin\nDM8bLyMeP2g7kUlwfkugVEPfQY2ujf7LNK7YnpsDCKXXScvfG38btThpAoGBAJRE\nkp8P94/TJxQw/DQrlG5Ls9Or1+306wQEx22fq2vBWcmawESpcO2fU8GTsdeXOUjC\nsKWo1OfemeuSVVdAzNlb5n+6wQ3Yvz6KFZb3dJ/YQe5FeU6ujFatJ0l0f06L7pHe\nFhUtmaL1aqC8AahbgacnLoE8ofcGMnDLRDpLcEABAoGAZzESWG15Mu5Gew/5BL/u\nmX7z6IluCC/XpwaEKJ8IZm+KWgS/usLt2YfOWgpoDNbKToJfXlre8CgRDMpZ1IJq\nVmohxRYJjza/BdsEsYyKL9lUfx3nE1hV++2sNLR70HmJslD588rXIAmmFw07mC1f\nV7Uz2E0EySg1OASpmRdjZWg=\n-----END PRIVATE KEY-----\n",
    "client_email": "cloudvision@patientsafe.iam.gserviceaccount.com",
    "client_id": "101118436177002714467",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/cloudvision%40patientsafe.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Updated Product extraction prompt
PRODUCT_PROMPT = """Analyze the given medicine text and extract specific fields into a JSON object. Use the following guidelines:

Text to analyze:
{text}

Extract these fields (use empty string if information is not found):
**give empty if there is no related data**

1. Subject_Drug_Name:
   - Extract ONLY the main brand/product name, excluding descriptive words
   - The main brand/product name be identified with ™, ®, © at the end.
   - Remove words like: "FORTE", "HAIR RESTORE", "FORMULA", "XL", "PLUS", "SR", "ER", "CR", "LA", "MR"
   - Remove dosage information, strength, and form
   - Remove special symbols like ™, ®, ©
   - Should be the core brand name only

2. Generic_Name:
   - Extract ONLY the main active pharmaceutical ingredient name
   - Look for the actual chemical name of the therapeutic drug, not brand names
   - Check composition sections for the true pharmaceutical ingredient
   - Exclude alcohols, excipients, and inactive ingredients completely
   - Most medicines have only one active ingredient - extract that single ingredient name
   - Only if the medicine is clearly a combination drug with multiple therapeutic ingredients, include up to 2-3 active ingredients separated by spaces
   - Do not include percentages, dosages, or formulation details
   - Do not use commas between ingredient names
   - Focus on the scientifically correct pharmaceutical name
   - Ignore concentration details and extraction methods

3. Batch_ID:
   - Look for patterns like "B.No:", "Batch No:", "B.No.", "No.", "BATCH", "LOT"
   - Can appear anywhere in the text, often near manufacturing or expiry dates
   - Look for complete alphanumeric codes that follow these patterns
   - May be written as separate words or connected
   - Return the full batch identifier including all alphanumeric characters
   - Check for codes that appear after manufacturing information
   - Exclude prefix words like "BNO", "B.No", "BATCH" from the extracted value
   - Extract only the actual batch code numbers and letters

4. Product_Form:
   - Use EXACTLY ONE of these case-sensitive options, or empty string:
     Tablet, Capsule, Injection, Infusion, Suspension, Syrup, Pill, Gel, Powder, Solution, Ointment, Emulsion, Cream, Lotion, Eye_drops, Ear_drops, Spray, Inhaler, Paste, Plaster, Shampoo, Enema, Lozenge, Aerosol, Suppository, Elixir, Granules, Magma, Cachet, Chewing gum, Foam, Douch

5. Dosage_Strength:
   - Extract the main therapeutic concentration mentioned in the product description
   - Look for the primary strength of the active pharmaceutical ingredient
   - Ignore excipient concentrations and inactive ingredient percentages
   - Extract ONLY the numerical values for the main therapeutic drug
   - For multiple active ingredients, use comma separation in the same order as the generic names
   - No units, just numbers
   - Focus on the clinically relevant therapeutic dosage

6. Strength_Unit:
   - Extract the unit corresponding to the main therapeutic strength
   - Valid units: mg, g, mcg, µg, ng, mL, L, mg/mL, g/mL, µg/L, mEq, tsp, tbsp, gtt, kg, mmol/L, U/L, ng/mL, %
   - If the strength is in the form x/ numerical y ( where x and y are units), then take x only
   - Should be a single unit only
   - Choose the unit that applies to the primary active ingredient concentration
   - Ignore units related to excipients or inactive ingredients

7. Manufacturer_Name:
   - Full company name including Pvt Ltd/Limited if present

8. Marketer_Name:
   - Same as Manufacturer_Name unless explicitly different

9. Expiry_Date:
   - Extract the expiry date from the given text with accuracy
   - Format as MM/YYYY if possible

Return a JSON object with exactly these fields. Use empty string for missing information."""

app = Flask(__name__)
CORS(app)

def authenticate_vision_api():
    try:
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        logging.error(f"Authentication failed: {str(e)}")
        raise

def perform_ocr(client, image_bytes):
    try:
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f'Vision API error: {response.error.message}')
            
        if not response.text_annotations:
            return ""
            
        return response.text_annotations[0].description
        
    except Exception as e:
        logging.error(f"OCR error: {str(e)}")
        raise

def call_ollama(prompt, model="phi4", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 1500,
                    "format": "json"
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
            
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            continue
            
    return ''

def parse_json(text):
    if not text:
        raise ValueError("Empty text provided for parsing")
        
    text = re.sub(r'```(json)?|```', '', text).strip()
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {str(e)}")
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as e:
            logging.error(f"AST parsing failed: {str(e)}")
            raise ValueError(f"Could not parse JSON: {str(e)}")

def extract_product_information(text):
    if not text:
        return {}
        
    try:
        logging.info("Starting product information extraction")
        
        full_prompt = (
            "IMPORTANT: Respond ONLY in valid JSON format. "
            "No explanatory text. "
            f"{PRODUCT_PROMPT}"
        ).format(text=text)
        
        response_text = call_ollama(full_prompt)
        
        if not response_text:
            logging.error("Empty response from Ollama API")
            return {}
            
        extracted_info = parse_json(response_text)
        
        cleaned_info = {
            'Subject_Drug_Name': str(extracted_info.get('Subject_Drug_Name', '')).strip().capitalize(),
            'Generic_Name': str(extracted_info.get('Generic_Name', '')).strip().capitalize(),
            'Batch_ID': str(extracted_info.get('Batch_ID', '')).strip().upper(),
            'Product_Form': str(extracted_info.get('Product_Form', '')).strip().capitalize(),
            'Dosage_Strength': str(extracted_info.get('Dosage_Strength', '')).strip(),
            'Strength_Unit': str(extracted_info.get('Strength_Unit', '')).strip(),
            'Manufacturer_Name': str(extracted_info.get('Manufacturer_Name', '')).strip().capitalize(),
            'Marketer_Name': str(extracted_info.get('Marketer_Name', '')).strip().capitalize(),
            'Expiry_Date': str(extracted_info.get('Expiry_Date', '')).strip()
        }
        
        logging.info(f"Successfully extracted information: {json.dumps(cleaned_info)}")
        return cleaned_info
        
    except Exception as e:
        logging.error(f"Error during extraction: {str(e)}")
        return {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
           
        data = request.get_json()
        data['text'] = data['text'].lower()
        if not data:
            return jsonify({
                "status": 404
            }), 404
            
        # Extract product information
        product_info = extract_product_information(data)
        
        if not product_info:
            return jsonify({
                "error": "Could not extract product information from the text",
                "raw_text": data,
                "status": 404
            }), 404
            
        # Return successful response
        return jsonify({
            "data": {
                "extracted_info": product_info,
                "raw_text": data
            },
            "status": 200
        }), 200
        
    except Exception as e:
        logging.error(f"Unexpected error in process-image endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "status": 500
        }), 500

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5005, debug=True, threaded=True)