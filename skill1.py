import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the model
with open('skillmodel.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Pass skills_data to the Streamlit app
st.session_state.skills_data = None  # Initialize the skills_data in session_state

# Streamlit app
st.title("Resume Skill Extractor")

# Input text area for user to enter resume text
user_input = st.text_area("Enter your resume text here:", "")

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
# Check if skills_data is not initialized in session_state
if st.session_state.skills_data is None:
    # Load skills_data
    st.session_state.skills_data = pd.read_csv('/content/drive/MyDrive/kaggle/skillsdata.csv')

# Process the user input
if st.button("Extract Skills"):
    # Preprocess the text
    processed_text = preprocess_text(user_input)

    # Vectorize the preprocessed text
    X_new_data = vectorizer.transform([processed_text])

    # Make predictions on the new data
    predictions_new_data = loaded_model.predict(X_new_data)

    # Post-process predictions to extract predicted skills
    predicted_skills = extract_skills_from_text(processed_text, st.session_state.skills_data['SKILLS'])

    # Display the extracted skills
    st.subheader("Extracted Skills:")
    st.write(', '.join(predicted_skills))


