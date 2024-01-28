import streamlit as st
import pandas as pd
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved model, vectorizer, and label encoder
with open('skillmodel.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    loaded_label_encoder = pickle.load(label_encoder_file)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    contractions = {"n't": "not", "'s": "is", "'re": "are", "'m": "am", "'ll": "will", "'ve": "have", "'d": "would"}
    tokens = [contractions.get(word, word) for word in tokens]
    tags_removed = re.sub(r'<.*?>', '', ' '.join(tokens))
    processed_text = ' '.join(tokens)
    return processed_text

# Function to extract skills from preprocessed text
def extract_skills_from_text(preprocessed_text):
    X_new_data = loaded_vectorizer.transform([preprocessed_text])
    predictions_new_data = loaded_model.predict(X_new_data)
    predicted_skills = loaded_label_encoder.inverse_transform(predictions_new_data)
    return predicted_skills[0] if predicted_skills else None

# Streamlit app
def main():
    st.title("Skill Extraction App")

    # User input for resume text
    resume_text = st.text_area("Paste your resume here:")

    if st.button("Extract Skills"):
        # Preprocess the input text
        processed_resume = preprocess_text(resume_text)

        # Extract skills
        extracted_skills = extract_skills_from_text(processed_resume)

        # Display extracted skills
        if extracted_skills:
            st.success(f"Extracted Skills: {extracted_skills}")
        else:
            st.warning("No skills extracted.")

if __name__ == "__main__":
    main()
