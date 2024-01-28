import streamlit as st
import pickle
import logging
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
logging.basicConfig(level=logging.DEBUG)  # Enable logging

# Load the model, vectorizer, and label encoder
with open('skillmodel.pkl', 'rb') as model_file:
  model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
  vectorizer = pickle.load(vectorizer_file)
with open('label_encoder.pkl', 'rb') as label_encoder_file:
  label_encoder = pickle.load(label_encoder_file)

# Preprocessing function (replace with your implementation)
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
def extract_skills_from_text(preprocessed_text, label_encoder):
    predicted_indices = model.predict(vectorizer.transform([preprocessed_text]))[0]
    logging.debug("Predicted indices: %s", predicted_indices)

    predicted_skills = label_encoder.inverse_transform(predicted_indices)
    logging.debug("Predicted skills (before joining): %s", predicted_skills)

    skill_list = []
    for skill_index in predicted_skills:
        skill_name = label_encoder.inverse_transform([skill_index])[0]
        skill_list.append(str(skill_name))

    skill_str = ', '.join(skill_list)
    return skill_str

st.title("Resume Skill Extractor")

# User input: resume text
user_input = st.text_area("Paste your resume text here:")

if st.button("Extract skills"):
  if user_input:
    preprocessed_text = preprocess_text(user_input)
    predicted_skills = extract_skills_from_text(preprocessed_text, label_encoder)

    st.write("Predicted Skills:", predicted_skills)
