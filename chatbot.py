import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warnings

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from transformers import pipeline
import language_tool_python

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

# Load your DataFrame (make sure to adjust the path to your actual CSV file)
df = pd.read_csv(r"C:\Users\Jay\Documents\Sem 5\NLP\Chatbot\Data\medquad_cleaned.csv")

# Initialize stemmer, lemmatizer, and stop words
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize LanguageTool for grammar checking
tool = language_tool_python.LanguageTool("en-US")

# Function to check and correct grammar/spelling in the input
def grammar_check(text):
    corrected_text = tool.correct(text)
    return corrected_text

# Function to preprocess the question
def preprocess_question(question):
    # Tokenize the question
    words = nltk.word_tokenize(question.lower())
    
    # Remove stop words and apply stemming and lemmatization
    keywords = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word.isalpha() and word not in stop_words]
    
    return keywords

# Apply the preprocessing to the "question" column and create a new "keywords" column
df['keywords'] = df['question'].apply(preprocess_question)

# Function to preprocess user input
def preprocess_input(user_input):
    words = nltk.word_tokenize(user_input.lower())
    words = [word for word in words if word.isalpha()]
    keywords = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    return keywords

# Function to find the exact match
def find_exact_match(user_input):
    input_keywords = preprocess_input(user_input)
    
    for index, row in df.iterrows():
        if set(input_keywords) == set(row['keywords']):
            return row['answer']
    
    return "Sorry, I couldn't find an exact match. Check for any spelling error !"

# Function to split and summarize the answer if it's too long
def summarize_answer(answer):
    try:
        summary = summarizer(answer, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
        
    except Exception as e:
        # In case summarization fails, return the original answer and print the error
        print(f"Error summarizing answer: {e}")
        return answer

# Main function for command line interaction
if __name__ == "__main__":
    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        # Correct the grammar and spelling in the user's input
        corrected_input = grammar_check(user_input)
        print(f"Corrected Input: {corrected_input}")

        answer = find_exact_match(corrected_input)
        
        if len(answer) > 500:
            user_choice = input("The answer is too big. Do you want a summarized answer? (y/n): ")
            if user_choice.lower() == 'y':
                answer = summarize_answer(answer)
                
        print(f"Answer: {answer}")
