import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warnings

import nltk
nltk.download('stopwords')
nltk.download('punkt')  # Ensure stopwords and tokenizer are available
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, pipeline
import language_tool_python
import re
import textwrap  # Import textwrap for beautifying answers

# Initialize the spell checker
tool = language_tool_python.LanguageTool('en-US')

# Initialize the tokenizer and TensorFlow T5 model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

# Initialize the summarization pipeline with TensorFlow T5 model
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")

# Load your DataFrame (adjust the path to your actual CSV file)
df = pd.read_csv(r"path to csv file")

# Initialize stemmer, lemmatizer, and stop words
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess the question
def preprocess_question(question):
    words = nltk.word_tokenize(question.lower())
    keywords = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word.isalpha() and word not in stop_words]
    return keywords

# Apply the preprocessing to the "question" column and create a new "keywords" column
df['keywords'] = df['question'].apply(preprocess_question)

# Function to correct spelling mistakes
def correct_spelling(input_text):
    matches = tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    return corrected_text

# Function to preprocess user input and return corrected input
def preprocess_input(user_input):
    corrected_input = correct_spelling(user_input)
    words = nltk.word_tokenize(corrected_input.lower())
    words = [word for word in words if word.isalpha()]
    keywords = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    return keywords, corrected_input  # Return both keywords and corrected input


# Function to find the exact match and return answer, source, and corrected input
def find_exact_match(user_input):
    input_keywords, corrected_input = preprocess_input(user_input)
    for index, row in df.iterrows():
        if set(input_keywords) == set(row['keywords']):
            return row['answer'], row['source'], corrected_input
    return "Sorry, I couldn't find an exact match.", None, corrected_input

# Function to clean up unnecessary punctuation and fix summary artifacts
def clean_generated_summary(summary):
    # Remove excessive punctuation and spaces
    summary = re.sub(r'[^\w\s\.\,\!\?]', '', summary)  # Remove unwanted characters
    summary = re.sub(r'\s+', ' ', summary).strip()  # Remove extra spaces
    summary = re.sub(r'\.+', '.', summary)  # Normalize multiple dots to a single dot
    summary = re.sub(r'\?+', '?', summary)  # Normalize multiple question marks
    summary = re.sub(r'!+', '!', summary)  # Normalize multiple exclamation marks
    
    # Split sentences and remove incomplete ones
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    if not sentences[-1].endswith(('.', '!', '?')):
        sentences.pop()  # Remove last sentence if it's incomplete
    
    # Join the sentences back into a coherent summary
    return " ".join(sentences)

# Updated summarization function
def summarize_answer(answer):
    try:
        cleaned_answer = " ".join(answer.split())
        inputs = tokenizer(cleaned_answer, return_tensors="tf", max_length=512, truncation=True)
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=500,
            min_length=200,
            length_penalty=2.0,
            num_beams=4,
            no_repeat_ngram_size=3,  # Avoid repetition of n-grams
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean the generated summary
        summary = clean_generated_summary(summary)
        return summary
    except Exception as e:
        print(f"Error summarizing answer: {e}")
        return answer


# Function to beautify the answer with proper formatting
def beautify_answer(answer, width=80):
    # Use textwrap to break the answer into lines of specified width
    wrapped_answer = textwrap.fill(answer, width=width)
    return wrapped_answer

# Main function for user interaction
if __name__ == "__main__":
    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        answer, source, corrected_input = find_exact_match(user_input)
        print(f"Corrected Input: {corrected_input}")  # Display the corrected input
        
        if answer != "Sorry, I couldn't find an exact match." and len(answer) > 500:
            user_choice = input("The answer is too big. Do you want a summarized answer? (y/n): ")
            if user_choice.lower() == 'y':
                answer = summarize_answer(answer)
        
        # Beautify and print the formatted answer and source
        formatted_answer = beautify_answer(answer)
        print(f"Answer:\n{formatted_answer}")
        
        if source:  # If a source is found, print it
            print(f"Source: {source}")
