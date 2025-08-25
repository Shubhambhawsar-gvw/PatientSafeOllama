import re
import difflib
import requests
import json
import logging

# Create and configure the logger
log_file = 'word_suggestion.log'  # This will be your separate log file
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Set the minimum level of messages to log
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # Append mode ('w' for overwrite)
)
# Ollama API call function
def call_ollama(prompt, model="phi4:latest"):
    try:
        print(prompt)
        response = requests.post('http://host.docker.internal:11434/api/generate', json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1500,
            "format": "json"
        })
        logging.info(response)
        if response.status_code == 200:
            result = response.json()
            print(result)
            return result.get('response', '')
        else:
            logging.error(f"Ollama API Error: {response.status_code} - {response.text}")
            return ''
    except Exception as e:
        logging.error(f"Ollama API request failed: {str(e)}")
        return ''
def keyword_search_ranked(query, database, max_results=None, threshold=0.70):
    """
    Performs a similarity-based keyword search and ranks results by keyword match density.
    
    Parameters:
        query (str): The search query containing one or more keywords.
        database (list): A list of records (strings) to search within.
        max_results (int, optional): Maximum number of ranked results to return.
        threshold (float): Similarity threshold (0 to 1) for considering a word as matched.
        
    Returns:
        list of tuples: (record, rank) sorted by rank in descending order.
    """
    query_tokens = query.lower().split()
    ranked_records = []

    for record in database:
        record_lower = record.lower()
        words = re.findall(r'\w+', record_lower)
        total_words = len(words)

        if total_words == 0:
            continue

        matched_keywords = 0
        for word in words:
            for token in query_tokens:
                if token == word:
                    matched_keywords += total_words
                else:
                    similarity = difflib.SequenceMatcher(None, token, word).ratio()
                    if similarity >= threshold:
                        matched_keywords += 1
                        break  # Avoid double-counting the same word for multiple tokens

        if matched_keywords > 0:
            rank = matched_keywords / total_words
            ranked_records.append((record, rank))

    ranked_records.sort(key=lambda x: x[1], reverse=True)

    if max_results is not None:
        ranked_records = ranked_records[:min(max_results, len(ranked_records))]

    return list(set(ranked_records))
def search_names_in_database(product_name, generic_names,max_results=3):
    results = keyword_search_ranked(product_name, generic_names, max_results)
    if results:
        print(f"Matched records with generic_names:  {product_name}")
        text = """"""
        for i, (rec, rank) in enumerate(results):
            print(f"{i+1}. name : {rec} rank : {rank}")
            text += f"""{i+1}. {rec}
"""
    
        text += f"""
- Return a valid json has key "drug_names".
- Return all names in a list except dosage related terms.
- Remove the unwanted informations like dosage forms and strengths, return only the names in a list. 
    """
        #print(text)
        response = call_ollama(text)
        logging.info(response)
        res = json.loads(response)
        #res['drug_names']\
        list_values = list(set(res['drug_names']))
        if product_name not in list_values:
            list_values.insert(0,product_name)
        else:
            list_values.insert(0, list_values.pop(list_values.index(product_name)))
        return list_values
    else:
        logging.info("no matched records for ",product_name)
        return None