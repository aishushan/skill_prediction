import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the model
with open('skillmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

st.title("Resume Skill Extractor")

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

        def extract_skills_from_text(preprocessed_text):
            prep_array = vectorizer.transform([preprocessed_text])
            print("Preprocessed Array Shape:", prep_array.shape)  # Debugging
            predictions = model.predict(prep_array)
            print("Raw Predictions:", predictions)  # Debugging
            # Map class indices to skill labels
            result = label_encoder.inverse_transform(predictions)
            return result

        prep = preprocess_text(user_input)
        print("Preprocessed Text:", prep)  # Debugging
        predicted_skills = extract_skills_from_text(prep)

        # Convert elements in predicted_skills to strings
        predicted_skills = [str(skill) for skill in predicted_skills]

        # Display output
        st.write("Predicted Skills:", ', '.join(predicted_skills))
