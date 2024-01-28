import streamlit as st
import pickle
import pandas as pd
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the model, vectorizer, and label encoder
with open('skillmodel.pkl', 'rb') as model_file:
  model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
  vectorizer = pickle.load(vectorizer_file)
with open('label_encoder.pkl', 'rb') as label_encoder_file:
  label_encoder = pickle.load(label_encoder_file)

# Preprocessing function
def preprocess_text(text):
    # ... (your preprocessing code)
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

    # Handling contractions (you may need a more comprehensive list)
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

    # Removing HTML tags
    tags_removed = re.sub(r'<.*?>', '', ' '.join(tokens))

    # Joining tokens back into a sentence
    processed_text = ' '.join(tokens)

    return processed_text
# **Corrected function to extract skills from text**
def extract_skills_from_text(preprocessed_text, label_encoder):
    # Use label_encoder directly for prediction-based skill extraction
    predicted_skills = label_encoder.inverse_transform([model.predict(vectorizer.transform([preprocessed_text]))[0]])
    return predicted_skills

st.title("Resume Skill Extractor")

# User input: resume text
user_input = st.text_area("Paste your resume text here:")

if st.button("Extract skills"):
  if user_input:
    preprocessed_text = preprocess_text(user_input)

    # Extract skills using the corrected function
    predicted_skills = extract_skills_from_text(preprocessed_text, label_encoder)

    # Convert skill names to strings for display
    predicted_skills_str = ', '.join(str(skill) for skill in predicted_skills)

    # Display the extracted skills
    st.write("Predicted Skills:", predicted_skills_str)
