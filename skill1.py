import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.title("Resume Skill Classifier")

# Load the model and vectorizer
with open('skillmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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
            processed_text = tokens

            return processed_text

        def extract_skills_from_text(preprocessed_text, model, vectorizer):
            # Vectorize the preprocessed text using the loaded vectorizer
            prep_array = vectorizer.transform([' '.join(preprocessed_text)])
            # Make predictions on the new data using the loaded model
            probabilities = model.predict_proba(prep_array)
            predicted_class = np.argmax(probabilities)
            predicted_skill = f"Skill {predicted_class}"  # You can customize this based on your actual skill labels
            return [predicted_skill]

        def remove_duplicates(lst):
            return list(set(lst))

        prep = preprocess_text(user_input)
        predictions = extract_skills_from_text(prep, model, vectorizer)
        result = remove_duplicates(predictions)
        # Display output
        st.write("Predicted Skills:", ', '.join(result))
