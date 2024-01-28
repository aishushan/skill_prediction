import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
from PyPDF2 import PdfReader
import io

# Load the saved model, vectorizer, and label encoder
with open('skillmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder (1).pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Load the skills dataset
skills_data = pd.read_csv('C:/Users/Aiswarya/OneDrive/Desktop/DEPLOYMENT/intern prjct/skillsdata.csv')
skills_data['SKILLS'] = skills_data['SKILLS'].str.lower()  # Optional

def preprocess_text(text):
    """Preprocesses text for skill extraction."""

    # Lowercasing
    text = text.lower()

    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Handling contractions (example)
    contractions = {
        "n't": "not",
        "'s": "is",
        "'re": "are",
        "'m": "am",
        "'ll": "will",
        "'ve": "have",
        "'d": "would"
    }
    tokens = [contractions.get(word, word) for word in tokens]

    # Joining tokens back into a sentence
    processed_text = ' '.join(tokens)

    return processed_text

def extract_skills_from_text(preprocessed_text, skills_data):
    """Extracts skills from preprocessed text based on a skills dataset, removing duplicates."""

    extracted_skills = []

    for skill in skills_data:
        if skill in preprocessed_text:
            extracted_skills.append(skill)

    # Remove duplicates while preserving order
    extracted_skills = list(OrderedDict.fromkeys(extracted_skills))

    return extracted_skills

# Create the Streamlit app
st.title("Resume Skills Extractor")

# Upload PDF file option
uploaded_file = st.file_uploader("Upload your resume (PDF):", type=['pdf'])

if uploaded_file:
    try:
        # Extract text from the uploaded PDF
        with uploaded_file as f:
            pdf_reader = PdfReader(f)
            page_text = ''
            for page in pdf_reader.pages:
                page_text += page.extractText()

        # Preprocess the extracted text
        processed_text = preprocess_text(page_text)

        # Vectorize the text
        X_new_data = vectorizer.transform([processed_text])

        # Make predictions
        predictions = model.predict(X_new_data)

        # Extract skills from the text based on predictions
        extracted_skills = extract_skills_from_text(processed_text, skills_data['SKILLS'])

        # Display the extracted skills
        st.write("Extracted Skills:")
        st.write(", ".join(extracted_skills))

    except Exception as e:
        st.error("Error processing PDF: {}".format(e))

else:
    # Text area input for manual resume text entry
    resume_text = st.text_area("Enter resume text:")

    if st.button("Extract Skills"):
        # Preprocess the text
        processed_text = preprocess_text(resume_text)

        # Vectorize the text
        X_new_data = vectorizer.transform([processed_text])

        # Make predictions
        predictions = model.predict(X_new_data)

        # Extract skills from the text based on predictions
        extracted_skills = extract_skills_from_text(processed_text, skills_data['SKILLS'])

        # Display the extracted skills
        st.write("Extracted Skills:")
        st.write(", ".join(extracted_skills))
