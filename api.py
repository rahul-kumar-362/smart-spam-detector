import os
import logging
import pickle
import re
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

from text_preprocess import clean_text

# ---------------- LOGGING ----------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ---------------- FLASK APP ----------------
app = Flask(__name__)
CORS(app)  # Allow Chrome extension requests

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))
    logging.info("Model + Vectorizer loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise e

# ---------------- SPAM TRIGGER WORDS ----------------
SPAM_WORDS = [
    "free", "win", "winner", "prize", "claim", "click", "offer", "urgent", "money", "reward",
    "lottery", "bonus", "cash", "deal", "discount", "limited", "exclusive", "act now", "buy now",
    "guaranteed", "gift", "congratulations", "earn", "income", "profit", "cheap", "credit", "loan",
    "approved", "extra", "amazing", "vip", "secret", "priority", "alert", "final notice",
    "last chance", "hurry", "expire", "deadline", "today only", "instant", "fast", "quick",
    "limited time", "best price", "jackpot", "million", "earn money", "work from home",
    "financial freedom", "investment", "crypto", "bitcoin", "forex", "verify account",
    "confirm details", "update account", "account suspended", "login now", "reset password",
    "click link", "download now", "install", "limited stock", "free trial", "free access",
    "free membership", "instant approval", "selected user", "exclusive invitation"
]

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Smart Spam Detector API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        text = data["text"].strip()

        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        # Extra signals from Chrome extension
        gmail_spam = data.get("gmail_spam", False)
        sender = data.get("sender", "")

        # Preprocess
        cleaned = clean_text(text)

        # Predict
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]  # [ham_prob, spam_prob]
        classes = list(model.classes_)  # e.g. ['ham', 'spam']

        ham_idx = classes.index("ham") if "ham" in classes else 0
        spam_idx = classes.index("spam") if "spam" in classes else 1

        ham_prob = float(proba[ham_idx])
        spam_prob = float(proba[spam_idx])

        pred = "spam" if spam_prob > ham_prob else "ham"
        confidence = max(ham_prob, spam_prob)

        # Find trigger words (whole-word matching only)
        text_lower = text.lower()
        trigger_words = []
        for word in SPAM_WORDS:
            # Use word boundaries to avoid substring matches
            # e.g., "free" should NOT match "freedom", "win" should NOT match "following"
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text_lower):
                trigger_words.append(word)
        trigger_count = len(trigger_words)

        # ---------- HYBRID SCORING ----------
        original_pred = pred

        # Signal 1: Gmail itself flagged it as spam (very strong signal)
        if gmail_spam and pred == "ham" and ham_prob < 0.92:
            pred = "spam"
            confidence = max(spam_prob + 0.30, 0.75)
            confidence = min(confidence, 0.98)

        # Signal 2: Trigger word boosting
        if pred == "ham" and trigger_count >= 2:
            word_count = max(len(text.split()), 1)
            trigger_density = (trigger_count / word_count) * 100

            boost = min(trigger_count * 0.08, 0.45)
            adjusted_spam_prob = spam_prob + boost

            if (trigger_count >= 4 or
                (trigger_count >= 3 and spam_prob > 0.10) or
                (trigger_count >= 2 and spam_prob > 0.25) or
                trigger_density > 2.5):
                pred = "spam"
                confidence = min(adjusted_spam_prob, 0.98)

        logging.info(
            f"Prediction={pred} (model={original_pred}) | "
            f"spam_prob={spam_prob:.4f} | ham_prob={ham_prob:.4f} | "
            f"triggers={trigger_count} | gmail_spam={gmail_spam} | "
            f"sender={sender[:30]} | Text={cleaned[:50]}..."
        )

        return jsonify({
            "prediction": pred,
            "confidence": round(confidence * 100, 2),
            "trigger_words": trigger_words
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
