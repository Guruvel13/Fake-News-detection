import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (use st.cache_data for caching data)
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")   # Make sure this file exists in the same folder
    real = pd.read_csv("True.csv")   # Same here
    fake['label'] = 0
    real['label'] = 1
    data = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)
    return data

# Preprocess text
def preprocess_text(text):
    return text.lower()  # Basic preprocessing; expand if needed

# Train model (use st.cache_resource for heavy model training)
@st.cache_resource
def train_model(data):
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    X = data['cleaned_text']
    y = data['label']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, accuracy

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter a news article to classify it as Real or Fake.")

user_input = st.text_area("News Article", "Type your news article here...")

if st.button("Classify"):
    if user_input:
        try:
            data = load_data()
        except FileNotFoundError:
            st.error("Dataset not found. Make sure 'Fake.csv' and 'True.csv' are in the app folder.")
        else:
            model, vectorizer, accuracy = train_model(data)
            user_input_vec = vectorizer.transform([user_input])
            prediction = model.predict(user_input_vec)
            result = "Real" if prediction[0] == 1 else "Fake"
            st.success(f"Prediction: {result}")
            st.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    else:
        st.warning("Please enter a news article to classify.")

