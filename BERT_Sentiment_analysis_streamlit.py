import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page Configuration
st.set_page_config(page_title="IMDB Sentiment Analysis using BERT", layout="centered")

# Load Model and Tokenizer
MODEL_PATH = "BERT_Model_Final_Sentiment_Analysis"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# Title
st.title(" IMDB Movie Review Sentiment Analysis using BERT model ")
st.write("Enter a movie review and predict whether it is Positive or Negative.")

# User Input
review = st.text_area("Movie Review", height=200, placeholder="Type your movie review here...")

# Prediction Button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=256)

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()

        if prediction == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")