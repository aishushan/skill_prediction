import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.title("Resume Skill Classifier")

# Load the model, vectorizer, and LabelEncoder
with open('skillmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

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

        def remove_duplicates(lst):
            return list(set(lst))

        prep = preprocess_text(user_input)
        prep_array = np.array([' '.join(prep)])  # Convert the list to a 1D array
        prep_vectorized = vectorizer.transform(prep_array)

        # Make predictions on the new data
        predictions = model.predict(prep_vectorized)

        # Post-process predictions to extract predicted skills
        predicted_skills = label_encoder.inverse_transform(predictions)
        result = remove_duplicates(predicted_skills.tolist())

        # Display output
        st.write("Predicted Skills:", ', '.join(result))
