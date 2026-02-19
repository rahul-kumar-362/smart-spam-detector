import os
import sys
import pickle
import logging
import streamlit as st

# allow import from parent directory
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from text_preprocess import clean_text


# ---------------- LOGGING ----------------
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(BASE_DIR, "model.pkl")
        vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))

        logging.info("Model + Vectorizer loaded successfully")

        return model, vectorizer

    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e


# load once
model, vectorizer = load_model()


# ---------------- PREDICTION FUNCTION ----------------
def predict_mail(text):

    try:
        cleaned = clean_text(text)

        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()

        logging.info(f"Prediction={pred} | Confidence={prob:.4f} | Text={cleaned}")

        return pred, prob

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "error", 0.0
