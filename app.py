
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import unicodedata
import re
from pyngrok import ngrok
from PIL import Image

import time

# Load pre-trained SBERT model
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Function to convert text to SBERT embeddings
def encode_text_sbert(text_list):
    embeddings = sbert_model.encode(text_list, convert_to_numpy=True)
    return embeddings

# Preprocessing function
def clean_text(text):
    # Normalize the text to remove special Unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove text in square brackets, such as footnotes
    text = re.sub(r'\[.*?\]', '', text)

    # Remove text between angle brackets, such as HTML
    text = re.sub(r'<.*?>+', '', text)

    # Preserve punctuation, keep alphanumeric characters and some basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'\"-]', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

from tensorflow.keras.models import model_from_json

json_file = open('nlp_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load the weights into the model
loaded_model.load_weights('nlp_model.weights.h5')

# Streamlit UI

# Home Section
st.title("Welcome, Amazing People!")
st.markdown("""
Hello, my name is Alvin Kurniawan Hanafie.
""")
st.markdown("""
I am pleased to serve Indonesian sentiment analysis prediction app. Hope you enjoy it!
""")

st.title('Sentiment Analysis in Indonesian')
img = Image.open("Sentiment Analysis Picture.png")
st.image(img, width=500)
st.write('Please enter a text below in Indonesian and click "Predict Sentiment" to predict its sentiment:')

user_input = st.text_area("Text Input", "Type your text here...")

if st.button('Predict Sentiment'):
    if user_input:
        # Clean and preprocess the text
        processed_text = clean_text(user_input)

        # Encode the processed text using SBERT
        text_embedding = encode_text_sbert([processed_text])

        # Predict the sentiment
        prediction = loaded_model.predict(text_embedding)
        sentiment_label = np.argmax(prediction, axis=1)

        # Map the predicted label to the sentiment
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_result = sentiment_map[sentiment_label[0]]

        st.subheader('Prediction:')
        st.write(f"The predicted sentiment is: {sentiment_result}")

# Links Section
st.write("Let's Connect!")
# Project folder
project = "https://drive.google.com/drive/folders/1Y6QpMh7T2VIoGKKopcpOn0BjwTu2x_kD?usp=sharing"
st.markdown(f"- [Google Drive]({project})")
# LinkedIn
linkedin = "https://www.linkedin.com/in/alvinhanafie"
st.markdown(f"- [LinkedIn]({linkedin})")
# GitHub
github = "https://github.com/alvinhanafie"
st.markdown(f"- [GitHub]({github})")
