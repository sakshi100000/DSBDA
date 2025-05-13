import streamlit as st
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import joblib

# Load model and vectorizer from same folder
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define text cleaner
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit App UI
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬")
st.title("💬 Tweet Sentiment Analyzer")
st.write("Enter a tweet and predict whether it's Positive, Negative, or Neutral.")

user_input = st.text_area("Type your tweet here:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a tweet!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success("😊 Sentiment: Positive")
        elif prediction == "negative":
            st.error("😠 Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")

        st.caption("🧼 Cleaned Tweet: " + cleaned)
