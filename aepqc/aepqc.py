from flask import Flask, request, jsonify
import requests
import logging

app = Flask(__name__)

# Define the function to call the Ollama API
def call_ollama(prompt, model="phi4:latest"):
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1500,
            "format": "json"  # Request JSON format
        })
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            logging.error(f"Ollama API Error: {response.status_code} - {response.text}")
            return ''
    except Exception as e:
        logging.error(f"Ollama API request failed: {str(e)}")
        return ''

# Function to classify a medical description
def classify_medical_description(description):
    prompt = f"""
You are a medical expert specializing in categorizing medical text.

IMPORTANT INSTRUCTIONS:
- Respond ONLY with a single category name as the output.
- Do NOT include any explanatory text outside the category name.

Classify the following description into one of the categories:
- 'Product Quality Complaint'
- 'Adverse Reactions'
- 'Both'
- 'Medical Feedback - Positive'
- 'Medical Feedback - Negative'

Text:
{description}

Focus on:
1. **Adverse Reactions:** If the person has described side effects or symptoms caused by taking a medicine.
2. **Product Quality Complaint:** If the issue is related to the quality, packaging, or appearance of the medicine.
3. **Both:** If there are product quality issues **and** adverse reactions, especially when the product issue seems to have caused the reaction.
4. **Medical Feedback - Positive/Negative:** If it's general feedback about the experience with medicine, such as effectiveness or dissatisfaction.
"""
    raw_result = call_ollama(prompt)
    
    # Post-process the result to ensure it matches the specified categories
    valid_categories = [
        'Product Quality Complaint',
        'Adverse Reactions',
        'Both',
        'Medical Feedback - Positive',
        'Medical Feedback - Negative'
    ]
    
    # Handle cases where multiple categories are returned
    if 'Adverse Reactions' in raw_result and 'Product Quality Complaint' in raw_result:
        return 'Both'
    
    # Ensure the response is one of the valid categories
    for category in valid_categories:
        if category in raw_result:
            return category
    
    # Custom logic for partial matches or indirect feedback
    if "partial effectiveness" in description.lower() or "not fully effective" in description.lower():
        return 'Medical Feedback - Negative'
    if "effective" in description.lower() or "worked well" in description.lower():
        return 'Medical Feedback - Positive'
    
    # Default to an empty string if no valid category is matched
    return ''

# Flask route for classification
@app.route('/classify/medical', methods=['POST'])
def classify_medical_route():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        result = classify_medical_description(text)
        return jsonify({"classification": result})
    except Exception as e:
        logging.error(f"Medical classification error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)








# # #cod changed by Gokul for MI prompt checking

# from flask import Flask, request, jsonify
# import ollama  # Ollama Python client

# app = Flask(__name__)

# # Initialize Ollama client
# # Note: Ollama server should be running with GPU support (usually automatic if GPU is available)
# # Make sure you've run: ollama pull phi4

# def classify_question(question: str, route_type: str = "process_questions") -> str:
#     """
#     Classifies the question into medical categories.
#     Uses Ollama's phi4 model with GPU acceleration.
#     route_type: "process_questions" or "classify_medical" to handle different fallback behaviors
#     """
#     try:
#         response = ollama.generate(
#             model='phi4',
#             system="""
#             You are a medical expert specializing in categorizing medical text.
#             IMPORTANT INSTRUCTIONS:
#             - Respond ONLY with a single category name as the output.
#             - Do NOT include any explanatory text outside the category name.
            
#             Classify the following question into one of the categories:
#             - 'Product Quality Complaint'
#             - 'Adverse Reactions'
#             - 'Both'
#             - 'Medical Feedback - Positive'
#             - 'Medical Feedback - Negative'
#             - 'General'
            
#             Focus on:
#             1. **Adverse Reactions:** If the person has described side effects or symptoms caused by taking a medicine.
#             2. **Product Quality Complaint:** If the issue is related to the quality, packaging, or appearance of the medicine.
#             3. **Both:** If there are product quality issues **and** adverse reactions, especially when the product issue seems to have caused the reaction.
#             4. **Medical Feedback - Positive/Negative:** If it's general feedback about the experience with medicine, such as effectiveness or dissatisfaction.
#             5. **General** – If the question is a medical inquiry about a medicine, including:
#                - How to store the medicine
#                - Its usage, dosage, or administration
#                - Possible side effects or interactions (general information)
#                - General information about the medicine's purpose and effects
#                - Preventive or informational questions about potential risks
#                - **Questions seeking understanding about treatment decisions or changes made by healthcare providers**
#                - **Questions about reasons for medication discontinuation when no adverse effects are mentioned**
            
#             Carefully analyze the question's intent and classify it accordingly.
#             """,
#             prompt=question,
#             options={
#                 'temperature': 0,
#                 'num_predict': 50,
#                 'gpu': True  # Enable GPU acceleration
#             }
#         )
#         classification = response['response'].strip()
        
#         # Validate the classification result
#         valid_categories = [
#             'Product Quality Complaint',
#             'Adverse Reactions',
#             'Both',
#             'Medical Feedback - Positive',
#             'Medical Feedback - Negative',
#             'General'
#         ]
        
#         # Check if the response contains any valid category
#         for category in valid_categories:
#             if category in classification:
#                 return category
        
#         # Different fallback behavior based on route
#         if route_type == "classify_medical":
#             return ""  # Return empty string for /classify/medical route
#         else:
#             return "General"  # Return "General" for /process_questions route
        
#     except Exception as e:
#         print(f"Ollama API error: {e}")
#         # Different fallback behavior based on route
#         if route_type == "classify_medical":
#             return ""  # Return empty string for /classify/medical route
#         else:
#             return "General"  # Return "General" for /process_questions route

# @app.route('/process_questions', methods=['GET'])
# def handle_questions():
#     try:
#         # Get the text input
#         text = request.args.get('text', '')
#         questions = [q.strip() for q in text.split('\n') if q.strip()]
        
#         # Categories that should trigger a "yes" response
#         target_categories = [
#             'Product Quality Complaint',
#             'Adverse Reactions',
#             'Both',
#             'Medical Feedback - Positive',
#             'Medical Feedback - Negative'
#         ]
        
#         is_target_category_present = False  # Flag to check if there's a question in target categories
        
#         for question in questions:
#             question_type = classify_question(question, "process_questions")
#             print(f"Question: '{question}' classified as: '{question_type}'")
#             if question_type in target_categories:
#                 is_target_category_present = True
#                 break 
        
#         if is_target_category_present:
#             response_data = {
#                 "result": "yes",
#                 "questions": "\n".join(questions)  # Single string with newline separation
#             }
#         else:
#             response_data = {
#                 "result": "no",
#                 "questions": "no complaints"
#             }
        
#         print("response from llm", response_data)
#         return jsonify(response_data)
    
#     except Exception as e:
#         print(f"Error processing questions: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/classify/medical', methods=['POST'])
# def classify_medical_route():
#     try:
#         data = request.json
#         text = data.get('text', '')
#         if not text:
#             return jsonify({"error": "Missing text"}), 400
        
#         result = classify_question(text, "classify_medical")
#         return jsonify({"classification": result})
    
#     except Exception as e:
#         print(f"Medical classification error: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)