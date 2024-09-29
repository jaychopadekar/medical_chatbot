import warnings
import os
import tkinter as tk
from tkinter import scrolledtext
import datetime
import nltk
import pandas as pd
import language_tool_python
import re
import textwrap
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download necessary resources for nltk
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Initialize language tool spell checker
tool = language_tool_python.LanguageTool('en-US')

# Load the dataset
df = pd.read_csv(r"C:\Users\Jay\Documents\Sem 5\NLP\Chatbot\Chatbot\cleaned_medical_qa1.csv")

# Initialize stemmer, lemmatizer, and stop words
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Preprocess question
def preprocess_question(question):
    words = nltk.word_tokenize(question.lower())
    keywords = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word.isalpha() and word not in stop_words]
    return " ".join(keywords)

# Extract keywords
def extract_keywords(text):
    return preprocess_question(text).split()

# Correct spelling mistakes
def correct_spelling(input_text):
    matches = tool.check(input_text)
    return language_tool_python.utils.correct(input_text, matches)

# Summarize an answer using Sumy
def summarize_answer(answer):
    try:
        sentences = re.split(r'(?<=[.!?]) +', answer)
        first_three_sentences = " ".join(sentences[:3])
        remaining_text = " ".join(sentences[3:])
        if remaining_text:
            parser = PlaintextParser.from_string(remaining_text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 5)
            summarized_content = " ".join([str(sentence) for sentence in summary])
            final_summary = f"{first_three_sentences} {summarized_content}"
        else:
            final_summary = first_three_sentences
        return re.sub(r'[^\w\s\.\,\!\?]', '', final_summary)
    except Exception as e:
        print(f"Error summarizing answer: {e}")
        return answer

# Format the answer
def beautify_answer(answer, width=80):
    return textwrap.fill(answer, width=width)

# Group rows by 'focus_area' and create a dictionary
focus_area_dict = {}

# Create dictionary mapping 'focus_area' to corresponding rows in the dataframe
for _, row in df.iterrows():
    focus_area = row['focus_area']
    if focus_area not in focus_area_dict:
        focus_area_dict[focus_area] = []
    focus_area_dict[focus_area].append(row)

# Log unanswered questions
def log_unanswered_question(question):
    log_file_path = r"C:\Users\Jay\Desktop\chatbot_log.txt" 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {question}\n")
    print(f"Logged unanswered question: {question}")

# Modify process_input to log unanswered questions
def process_input():
    user_input = input_box.get()
    
    if user_input.lower() in ['bye', 'exit', 'q']:
        chat_display.insert(tk.END, "\nGoodbye! The session is closed.\n")
        root.quit()

    corrected_input = correct_spelling(user_input)
    input_keywords = extract_keywords(corrected_input)

    focus_area_match = None
    max_match_count = 0

    # Step 2: Identify the relevant focus area based on keywords
    for focus_area in focus_area_dict:
        focus_area_keywords = extract_keywords(focus_area)
        match_count = len(set(input_keywords) & set(focus_area_keywords))

        if match_count > max_match_count:
            max_match_count = match_count
            focus_area_match = focus_area

    # Step 3: If a relevant focus area is found, search within its rows
    if focus_area_match:
        answer = ""
        source = ""
        max_match_count = 0

        for row in focus_area_dict[focus_area_match]:
            row_keywords = extract_keywords(row['question'])
            match_count = len(set(input_keywords) & set(row_keywords))

            if match_count > max_match_count:
                max_match_count = match_count
                answer, source = row['answer'], row['source']

        if not answer:
            answer = "Sorry, I do not have the answer to your question."
            log_unanswered_question(user_input)  # Log unanswered question
        else:
            if len(answer) > 500 and ask_summarize.get() == 1:
                answer = summarize_answer(answer)

        # Display the result
        display_answer = f"\nYour Question: {user_input}\nCorrected Input: {corrected_input}"
        if answer:
            display_answer += f"\nAnswer: {beautify_answer(answer)}"
        if source:
            display_answer += f"\nSource: {source}"

        chat_display.insert(tk.END, display_answer + "\n\n")
    else:
        display_answer = f"\nYour Question: {user_input}\nSorry, I do not have the answer to your question."
        chat_display.insert(tk.END, display_answer + "\n")
        log_unanswered_question(user_input)
    
    input_box.delete(0, tk.END)


# Initialize Tkinter window
root = tk.Tk()
root.title("Medical Chatbot")
root.minsize(800, 600)

# Create scrollable chat display
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30, font=("Arial", 10))
chat_display.grid(row=0, column=0, columnspan=2)

# User input field
input_box = tk.Entry(root, width=40, font=("Arial", 14))
input_box.grid(row=1, column=0)

# Submit button
submit_button = tk.Button(root, text="Submit", command=process_input)
submit_button.grid(row=1, column=1)

# Checkbox for summarization option
ask_summarize = tk.IntVar()
summarize_check = tk.Checkbutton(root, text="Summarize long answers", variable=ask_summarize)
summarize_check.grid(row=2, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()
