from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import random
import nltk
from nltk.corpus import wordnet as wn
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from PyPDF2 import PdfReader
import google.generativeai as genai


# Initialize Flask app
app = Flask(__name__)
Bootstrap(app)

# Download necessary NLTK data
nltk.download("wordnet")
nltk.download("punkt")

# Configure Gemini AI API key
genai.configure(api_key="AIzaSyAVCzje26V8PjYsdcGNTZtDuDLsOJSzDyE")  # Replace with actual key

def extract_keywords(text, num_keywords=10):
    """Extracts important keywords using TextRank."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, num_keywords)
    
    keywords = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
    return keywords[:num_keywords]

def refine_content_gemini(prompt):
    """Refine text using Gemini AI."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_mcqs(text, num_questions=5, bloom_level="Understanding"):
    if not text:
        return []
    
    keywords = extract_keywords(text, num_keywords=num_questions + 5)
    if len(keywords) < 2:
        return []
    
    mcqs = []

    for _ in range(min(num_questions, len(keywords))):
        subject = random.choice(keywords)

        # Generate question based on Bloom's level
        prompt_q = f"""
        Generate a multiple-choice question based on Bloom's Taxonomy level: {bloom_level}.
        Ensure that the question aligns with the complexity of {bloom_level}.
        Topic: {subject}
        Generate only the question without explanation and options.
        """
        refined_question = refine_content_gemini(prompt_q)

        # Generate answer choices
        prompt_a = f"""
        Generate exactly four multiple-choice answer options for the following question.
        Ensure one is correct and the others are plausible distractors.
        
        Question: {refined_question}
        
        Provide the options in this format:
        A) correct option
        B) incorrect option 
        C) incorrect option 
        D) incorrect option 
        """
        refined_options = refine_content_gemini(prompt_a).split("\n")

        # Ensure all options exist and are properly formatted
        if len(refined_options) < 4:
            continue

        formatted_options = []
        correct_answer = ""

        for option in refined_options:
            if option.startswith(("A)", "B)", "C)", "D)")):
                formatted_options.append(option.strip())

        if len(formatted_options) == 4:
            correct_answer_text = formatted_options[0][3:].strip()  # Extract correct answer before shuffling
            
            # Find the new position of the correct answer after shuffling
            for option in formatted_options:
                if correct_answer_text in option:
                    correct_answer = option
                    break
            
            mcqs.append((refined_question, formatted_options, correct_answer))

    return mcqs

def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
        
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
        else:
            text = request.form.get('text', '').strip()

        if not text:
            return render_template('index.html', error="No valid text extracted from file.")
        
        try:
            num_questions = int(request.form.get('num_questions', 5))
            if num_questions <= 0:
                raise ValueError
        except ValueError:
            return render_template('index.html', error="Invalid number of questions.")

        bloom_level = request.form.get('bloom_level', 'Understanding')
        mcqs = generate_mcqs(text, num_questions=num_questions, bloom_level=bloom_level)
        
        if not mcqs:
            return render_template('index.html', error="No MCQs could be generated.")

        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        return render_template('mcqs.html', mcqs=mcqs_with_index)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)