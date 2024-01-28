import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

st.title("Resume Skill Classifier")

with open('skillmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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

            # Handling contractions
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

        # Recreate the vectorizer
        vectorizer = CountVectorizer()

        prep = preprocess_text(user_input)
        prep_array = vectorizer.transform([prep])
        
        predictions = model.predict(prep_array)
        result = remove_duplicates(predictions.tolist())
        # Display output
        st.write("Predicted Skills:", result)
