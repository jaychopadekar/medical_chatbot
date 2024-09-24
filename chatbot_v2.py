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
df = pd.read_csv(r"C:\Users\Jay\Documents\Sem 5\NLP\Chatbot\Chatbot\cleaned_medical_qa.csv")

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

# Function to preprocess user input
def preprocess_input(user_input):
    corrected_input = correct_spelling(user_input)
    words = nltk.word_tokenize(corrected_input.lower())
    words = [word for word in words if word.isalpha()]
    keywords = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    return keywords

# Function to find the exact match and return additional information
def find_exact_match(user_input):
    input_keywords = preprocess_input(user_input)
    for index, row in df.iterrows():
        if set(input_keywords) == set(row['keywords']):
            return row['answer']
    return "Sorry, I couldn't find an exact match."

def clean_summary(summary):
    summary = re.sub(r'[\n\r]+', ' ', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'\.+', '.', summary)
    summary = re.sub(r'\?+', '?', summary)
    summary = re.sub(r'!+', '!', summary)
    return summary

# Function to summarize the answer if it's too long
def summarize_answer(answer):
    try:
        cleaned_answer = " ".join(answer.split())
        inputs = tokenizer(cleaned_answer, return_tensors="tf", max_length=512, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=500, min_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return clean_summary(summary)
    except Exception as e:
        print(f"Error summarizing answer: {e}")
        return answer

# Function to beautify the answer with proper formatting
def beautify_answer(answer, width=80):
    # Use textwrap to break the answer into lines of specified width
    wrapped_answer = textwrap.fill(answer, width=width)
    return wrapped_answer

# Main function for command line interaction
if __name__ == "__main__":
    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        answer = find_exact_match(user_input)
        if answer != "Sorry, I couldn't find an exact match." and len(answer) > 500:
            user_choice = input("The answer is too big. Do you want a summarized answer? (y/n): ")
            if user_choice.lower() == 'y':
                answer = summarize_answer(answer)
        
        # Beautify and print the formatted answer
        formatted_answer = beautify_answer(answer)
        print(f"Answer:\n{formatted_answer}")
