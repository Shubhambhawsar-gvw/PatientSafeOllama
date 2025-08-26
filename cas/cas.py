
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
    filename='causality_assessment.log'
)

def call_ollama(prompt, model="phi4:latest"):
    """
    Call Ollama API with a given prompt
    
    Args:
        prompt (str): The prompt to send to the Ollama model
        model (str, optional): The Ollama model to use. Defaults to "phi4:latestt".
    
    Returns:
        str: The response from the Ollama model
    """
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "max_tokens": 50,
            "format": "json"  # Request JSON format
        })
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            logging.error(f"Ollama API Error: {response.status_code} - {response.text}")
            return 'Unknown/Not assessable'
    except Exception as e:
        logging.error(f"Ollama API request failed: {str(e)}")
        return 'Unknown/Not assessable'

def extract_causality(text):
    """
    Extract causality assessment from text using Ollama
    
    Args:
        text (str): The text to analyze for causality
    
    Returns:
        str: Causality assessment ('Yes/Suspected', 'No/Not suspected', or 'Unknown/Not assessable')
    """
    prompt = """You are an AI assistant trained to assess the causality of adverse drug reactions (ADRs) with EXTREME STRICTNESS and PRECISION. Your primary goal is to identify statements about causality between medications and adverse events.

Response Instructions:
- Respond ONLY with a valid JSON object
- STRICTLY return ONE of these EXACT values:
  * "Yes/Suspected"
  * "No/Not suspected" 
  * "Unknown/Not assessable"

Evaluation Criteria:
1. Classify as "Yes/Suspected" if AND ONLY if:
   - ANY adverse event occurs at or involves an application site, injection site, infusion site, or vaccination site
     (Examples include but are not limited to: redness, pain, swelling, irritation, rash, burning, dermatitis, 
     erosion, exfoliation, pruritus, warmth, dryness, abscess, bruising, cellulitis, hemorrhage, induration, 
     mass, urticaria, or any other reaction occurring at these sites)
   - Text EXPLICITLY contains attribution phrases like "caused by [medication]", "directly caused by [medication]", 
     "is responsible for", "due to [medication]", "because of [medication]", "resulted from [medication]"
   - Text EXPLICITLY states "had [symptom] due to [medication]" or "[symptom] from [medication]"
   - Patient EXPLICITLY expresses suspicion about medication causing the adverse event (e.g., "I suspect it might be due to...")
   - ANY statement that EXPLICITLY attributes the adverse effect to the medication by name, even with uncertain language

2. Classify as "No/Not suspected" if:
   - Patient or healthcare provider explicitly states the adverse event is NOT related to the medication
   - Includes phrases like "not related to", "not due to", "not connected with", "does not believe is related", 
     "do not believe is related", "is not related", "definitely did not cause", "not the medication"
   - Any clear statement indicating the provider does not suspect a connection between the drug and adverse event

3. Classify as "Unknown/Not assessable" for:
   - Temporal relationships without explicit causation (e.g., "After taking X, I experienced Y")
   - Statements that ONLY mention "after taking the medicine/medication" without naming the specific drug in relation to the adverse effect
   - Statements that ONLY describe a sequence of events without attribution
   - ANY case where the patient simply states they took medication and then experienced symptoms
   - ANY case where the causal relationship is not EXPLICITLY stated
   - ANY statement that doesn't contain direct attribution language linking the specific medication to the specific symptom
   
IMPORTANT RULES:
1. ONLY reactions at application sites, injection sites, infusion sites, or vaccination sites are automatically "Yes/Suspected".
2. Simple temporal statements like "After taking [medication], I experienced [symptom]" MUST be "Unknown/Not assessable".
3. The phrase "I started to develop [symptom]" after medication mention WITHOUT explicit attribution MUST be "Unknown/Not assessable".
4. To qualify as "Yes/Suspected", there MUST be EXPLICIT attribution connecting the specific drug to the specific symptom.
5. ANY statement where a healthcare provider explicitly indicates NO relationship between the drug and adverse event MUST be "No/Not suspected".
6. The default classification for most reports should be "Unknown/Not assessable" unless there is EXPLICIT attribution language.
7. Just mentioning the medication name and then symptoms is NOT enough for "Yes/Suspected" - there must be EXPLICIT connection with attribution words.

Analyze this text for causality:
{text}

Respond with ONLY the classification in JSON format:
{{
    "causality": "Your classification here"
}}"""
    try:
        # Prepare the prompt with the text
        full_prompt = prompt.format(text=text)
        
        # Call Ollama
        response_text = call_ollama(full_prompt)
        
        # Parse the JSON response
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
                except Exception:
                    logging.warning("JSON parsing strategy failed")
            
            raise ValueError("Could not parse JSON")

        try:
            parsed_response = parse_json(response_text)
            causality = parsed_response.get('causality', 'Unknown/Not assessable')
            
            # Validate the response
            valid_responses = {"Yes/Suspected", "No/Not suspected", "Unknown/Not assessable"}
            return causality if causality in valid_responses else "Unknown/Not assessable"
        
        except Exception as e:
            logging.error(f"JSON parsing failed: {e}")
            return "Unknown/Not assessable"

    except Exception as e:
        logging.error(f"An error occurred during causality assessment: {str(e)}")
        return "Unknown/Not assessable"

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict_causality', methods=['POST'])
def predict_causality():
    """Endpoint to assess ADR causality using POST method"""
    try:
        # Get text from JSON body
        data = request.get_json()
        text = data.get('text', '')
        
        # Check if text is empty
        if not text.strip():
            return jsonify({
                "error": "Text cannot be empty",
                "status": "error"
            }), 400
        
        # Process the text
        causality = extract_causality(text)
        
        # Return the result
        return jsonify({
            "result": causality
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1111, debug=True)