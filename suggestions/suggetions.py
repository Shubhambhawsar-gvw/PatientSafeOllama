from symspellpy import SymSpell, Verbosity
from flask import Flask, request, jsonify
import logging

# Create and configure the logger
log_file = 'word_suggestion.log'  # This will be your separate log file
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Set the minimum level of messages to log
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # Append mode ('w' for overwrite)
)
app = Flask(__name__)

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load the dictionary
dictionary_path = "Suggestions/filtered_drug_names.txt"
if sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='UTF-8'):
    print("Dictionary loaded successfully!")
else:
    print("Failed to load dictionary.")

# Custom substring search function
def substring_search(input_word, dictionary_path):
    results = []
    with open(dictionary_path, "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip().split()[0]  # Extract the term (first part of each line)
            if input_word.lower() in word.lower():  # Substring search (case-insensitive)
                results.append(word)
    return results

# Function to get SymSpell suggestions and substring matches

def correct_and_suggest(sentence):
    corrected_words = []
    misspelled_words = {}
    word_suggestions = {}
    
    for word in sentence.split():
        # SymSpell suggestions
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        symspell_results = [s.term for s in suggestions]
        
        # Substring search results
        substring_results = substring_search(word, dictionary_path)
        
        # Combine and deduplicate results
        combined_results = list(set(symspell_results + substring_results))
        
        if combined_results:
            misspelled_words[word] = combined_results[0]  # First suggestion as correction
            corrected_words.append(combined_results[0])  # Corrected word
            word_suggestions[word] = combined_results    # All suggestions
        else:
            corrected_words.append(word)  # Keep original word if no suggestion found
    
    return " ".join(corrected_words), misspelled_words, word_suggestions

# Example usage
@app.route('/PV_Drug_Name', methods=['GET'])
def Drug_name():
    word = request.args.get('Product_Name')
    corrected_sentence, misspelled_words, word_suggessions = correct_and_suggest(word)
    print(f"Original Sentence: {word}")
    print(f"Corrected Sentence: {corrected_sentence}")
    print(f"Misseplled words: {misspelled_words}")
    print(f"Word suggestions: {word_suggessions}")
    if bool(word_suggessions):
        result = list(word_suggessions.values())[0]
    else:
        result = [value for value in misspelled_words.values()]
    
    if word not in result:
        result.insert(0, word)
    return result
import pandas as pd
import re
import json
from search_name_in_db import search_names_in_database
def get_drug_details():
    data = pd.read_csv("C:/Users/Shubham.b/OneDrive - Globalvalueweb/PatientSafeOllama/Suggestions/data/Drug details.csv")
    brand_names = data['Brand Name/ Product Name'].values
    generic_names = data['Composition/Generic Name'].values
    return data, list(set(brand_names)), list(set(generic_names))
tata_data, brand_names, generic_names = get_drug_details()
# Example usage
@app.route('/PV_Generic_Name', methods=['GET'])
def Generic_name_suggestion():
    word = request.args.get('Generic_Name')
    logging.info("generic name = ",word)
    # corrected_sentence, misspelled_words, word_suggessions = correct_and_suggest(word)
    generic_name_suggestions = search_names_in_database(word, generic_names)
    logging.info(f"Original Sentence: {word}")
    logging.info(f"suggestions generic name: {generic_name_suggestions}")
    
    return generic_name_suggestions if generic_name_suggestions else word
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5011)
