import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np  # Import numpy for array operations
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer

st.title("Resume Skill Classifier")

with open('skillmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer used during training
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# User input: resume text
user_input = st.text_area("Paste your resume text here:")

if st.button("Extract skills"):
    if user_input:
        def preprocess_text(text):
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

            # Joining tokens back into a list
            processed_text = ' '.join(tokens)

            return processed_text

        def extract_skills_from_text(preprocessed_text, skills_data):
            # Initialize an empty list to store extracted skills
            extracted_skills = []

            # Check for each skill in the preprocessed text
            for skill in skills_data:
                if skill in preprocessed_text.lower():
                    extracted_skills.append(skill)

            return extracted_skills

        def remove_duplicates(lst):
            return list(set(lst))

        prep = preprocess_text(user_input)
        
        # Vectorize the preprocessed text using the same vectorizer
        prep_array = vectorizer.transform([prep])
        
        predictions = model.predict(prep_array)
        result = remove_duplicates(predictions.tolist())
        # Display output
        st.write("Predicted Skills:", result)
