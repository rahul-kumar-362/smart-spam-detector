import logging
import pickle
from text_preprocess import clean_text
import streamlit as st

logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


@st.cache_resource
def load_model():
    model = pickle.load(open("models/model.pkl","rb"))
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    return model, vectorizer

model, vectorizer = load_model()

def predict_mail(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    logging.info(f"Prediction={pred} | Confidence={prob:.4f} | Text={text}")

    return pred, prob
