from flask import Flask, request, jsonify
import pickle
import re
import PyPDF2
from flask_cors import CORS

# Load the trained model and vectorizer
with open('resume_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your frontend

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def clean_text(text):
    return " ".join(re.findall(r'\b\w+\b', text.lower()))

def calculate_ats_score(job_description, resume_text):
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
    matched_keywords = job_keywords.intersection(resume_keywords)
    ats_score = (len(matched_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    return round(ats_score, 2)

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume uploaded'}), 400

    job_description = request.form.get('job_description', '')
    resume_text = extract_text_from_pdf(request.files['resume'])
    cleaned_resume = clean_text(resume_text)

    # Predict category
    vectorized_resume = tfidf.transform([cleaned_resume])
    predicted_category = model.predict(vectorized_resume)[0]

    # Calculate ATS Score
    ats_score = calculate_ats_score(job_description, cleaned_resume)

    return jsonify({
        'category': predicted_category,
        'ats_score': ats_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
