# Smart Spam Detector

An AI-powered, multi-platform spam detection system that combines Machine Learning, a Chrome Extension with live Gmail integration, and a Streamlit web dashboard -- all backed by a Hybrid Scoring Engine that goes far beyond simple classification.

---

## Key Features

- **Hybrid Scoring Engine** -- Not just a model wrapper. Combines ML probabilities, trigger-word density analysis, and Gmail contextual signals for superior accuracy.
- **Chrome Extension (Manifest V3)** -- Injects directly into Gmail's UI. One-click spam scan with an in-page overlay -- no need to leave your inbox.
- **Streamlit Dashboard** -- A standalone web interface with real-time predictions, confidence scores, trigger word highlighting, and live model metrics.
- **Lightweight NLP Pipeline** -- Custom-built text preprocessor with zero dependency on NLTK. Faster cold-starts, smaller deployment footprint.
- **Explainable AI** -- Every prediction highlights the exact spam trigger words found, giving users full transparency into why a message was flagged.
- **Production Ready** -- Gunicorn + Flask backend, automated build.sh training pipeline, and deployment configs for Render/Heroku out of the box.

---

## Architecture

```
+-----------------------------------------------------------------------------+
|                        SMART SPAM DETECTOR                                  |
+-----------------------------------------------------------------------------+
|                                                                             |
|   +------------------+      +--------------------------------------+       |
|   |   DATA LAYER     |      |         ML PIPELINE                  |       |
|   |                  |      |                                      |       |
|   |  dataset.csv     |----->|  merge_data.py  -->  text_preprocess |       |
|   |  enron_emails.csv|      |        .py               .py         |       |
|   |                  |      |         |                   |         |       |
|   |  (~15K+ emails)  |      |         v                   v         |       |
|   |                  |      |      train.py  -->  TF-IDF + MultinomialNB    |
|   +------------------+      |                                      |       |
|                             |         +---------------------+      |       |
|                             |         v                     v      |       |
|                             |  models/model.pkl + vectorizer.pkl    |       |
|                             +----------+---------------------------+       |
|                                        |                                    |
|                                        v                                    |
|   +------------------------------------------------------------------+      |
|   |                 FLASK API (api.py)                                 |      |
|   |                                                                    |      |
|   |   POST /predict  -->  +-----------------------------------------+ |      |
|   |                       |      HYBRID SCORING ENGINE              | |      |
|   |                       |                                         | |      |
|   |                       |   Signal 1: ML Probability (Naive Bayes)| |      |
|   |                       |   Signal 2: Gmail Context (spam folder?)| |      |
|   |                       |   Signal 3: Trigger Word Density (~67 wrds| |      |
|   |                       |                                         | |      |
|   |                       |   -->  Fusion Logic                     | |      |
|   |                       |                                         | |      |
|   |                       |   -->  Final Verdict + Confidence Score | |      |
|   |                       +-----------------------------------------+ |      |
|   +---------------------------+-----------------------+------------+      |
|                               |                       |                    |
|                               v                       v                    |
|   +----------------------+  +------------------------+                     |
|   |  CHROME EXTENSION    |  |  STREAMLIT DASHBOARD   |                     |
|   |   (Manifest V3)      |  |      (app.py)          |                     |
|   |                      |  |                        |                     |
|   |   * Gmail DOM inject |  |   * Web-based scanner  |                     |
|   |   * In-page overlay  |  |   * Confidence meter   |                     |
|   |   * Scan history     |  |   * Model metrics      |                     |
|   |   * Dark glass UI    |  |   * Trigger highlights |                     |
|   +----------------------+  +------------------------+                     |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## Hybrid Scoring Engine

> The core differentiator -- our backend doesn't just return raw ML predictions. It fuses **three independent signals** to dramatically reduce false negatives.

```
+----------------------+
|   Input Email Text   |
+----------+-----------+
           |
      +----+----+---------------+
      v         v               v
+---------------+ +-----------+ +-----------------+
| Signal 1:     | | Signal 2: | | Signal 3:       |
| ML Model      | | Gmail     | | Trigger Word    |
| Probability   | | Context   | | Density         |
|               | |           | |                 |
| Naive Bayes   | | Is email  | | ~67 curated     |
| predict_proba | | in spam   | | spam keywords     |
| via TF-IDF    | | folder?   | | with word-        |
|               | | Phishing  | | boundary regex    |
|               | | warning?  | |                 |
+-------+-------+ +-----+-----+ +--------+--------+
        |               |                |
        +---------------+----------------+
                        |
                        v
             +------------------+
             |   Fusion Logic   |
             |                  |
             | * Gmail override |  --> Only in Chrome Extension
             | * Density boost  |  --> In both Chrome + Streamlit
             | * Confidence cap |  --> Always (max 98%)
             +--------+---------+
                      |
                      v
             +--------------------+
             |  Final Prediction  |
             |   + Confidence %   |
             |   + Trigger Words  |
             +--------------------+
```

| Signal | Source | When It Fires |
|--------|--------|---------------|
| **ML Probability** | `MultinomialNB.predict_proba()` | Always -- base prediction |
| **Gmail Context** | Chrome extension reads Gmail UI state | When email is in spam folder or has phishing banner |
| **Trigger Density** | Regex scan against ~67 keywords | When >=2 trigger words found with sufficient density |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Model** | Scikit-Learn (MultinomialNB) | Text classification via TF-IDF features |
| **NLP** | Custom Regex Pipeline | Lightweight preprocessing without NLTK |
| **Backend API** | Flask + Flask-CORS + Gunicorn | RESTful prediction endpoint |
| **Chrome Extension** | Manifest V3 + Vanilla JS | Gmail DOM injection & in-page UI overlay |
| **Web Dashboard** | Streamlit | Standalone web-based scanner & metrics |
| **Data** | Pandas | Dataset merging, cleaning & augmentation |
| **Deployment** | Render / Heroku compatible | `Procfile` + `build.sh` + `runtime.txt` |

---

## Project Structure

```
smart-spam-detector/
|
+-- ML & Backend
|   +-- merge_data.py          # Merges SMS + Enron datasets
|   +-- text_preprocess.py      # Custom NLP pipeline (no NLTK)
|   +-- train.py               # Model training script
|   +-- api.py                 # Flask API + Hybrid Scoring Engine
|   +-- models/
|   |   +-- model.pkl          # Serialized Naive Bayes model
|   |   +-- vectorizer.pkl   # Serialized TF-IDF vectorizer
|   |   +-- model.py         # Prediction helper module
|   +-- dataset.csv           # Combined training dataset (~15K+)
|   +-- enron_emails.csv      # Enron email corpus
|
+-- Chrome Extension
|   +-- chrome-extension/
|       +-- manifest.json     # Manifest V3 configuration
|       +-- background.js     # Service worker (API relay)
|       +-- content.js        # Gmail DOM injection engine
|       +-- content.css       # In-page overlay styles
|       +-- popup.html        # Extension popup UI
|       +-- popup.css         # Dark glassmorphism styles
|       +-- popup.js          # Scan / History / Settings logic
|       +-- icons/            # Extension icons (16/48/128px)
|
+-- Streamlit Dashboard
|   +-- app.py               # Web-based scanner + metrics
|
+-- Deployment
|   +-- requirements.txt     # Python dependencies
|   +-- build.sh             # Automated build + retrain script
|   +-- Procfile             # Gunicorn config for production
|   +-- runtime.txt          # Python version lock (3.10)
|   +-- .gitignore           # Ignore rules
|
+-- README.md
```

---

## Getting Started

### Prerequisites
- **Python 3.10+**
- **Google Chrome** (for the extension)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/rahul-kumar-362/smart-spam-detector.git
cd smart-spam-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train.py
```
> This generates `models/model.pkl` and `models/vectorizer.pkl`.

### 4. Start the API Server
```bash
# Development
python api.py

# Production
gunicorn api:app --bind 0.0.0.0:5000
```
The API will be live at `http://localhost:5000`.

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 6. Install the Chrome Extension
1. Open `chrome://extensions/` in Google Chrome
2. Enable **Developer Mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `chrome-extension/` folder
5. Open Gmail -> Click the extension icon -> Go to **Settings** -> Enter your API URL
6. Open any email -> Click the floating **"Check Spam"** button

---

## API Reference

### `GET /`
Health check endpoint.
```json
{
  "status": "ok",
  "message": "Smart Spam Detector API is running"
}
```

### `POST /predict`
Analyze text for spam.

**Request:**
```json
{
  "text": "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
  "gmail_spam": false,
  "sender": "unknown@example.com"
}
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 94.57,
  "trigger_words": ["congratulations", "free", "click", "claim", "prize"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | **(Required)** The email/message text to analyze |
| `gmail_spam` | `boolean` | Whether Gmail itself flagged this email as spam |
| `sender` | `string` | Sender address (logged for analysis) |

---

## Model Performance

- 96.68% Accuracy
- 97% Precision
- 96% Recall
- 96.5% F1 Score

| Detail | Specification |
|--------|--------------|
| **Algorithm** | Multinomial Naive Bayes |
| **Feature Extraction** | TF-IDF (top 5,000 features) |
| **Training Data** | ~15,000+ emails (SMS Spam + Enron Corpus) |
| **Train/Test Split** | 80/20 with `random_state=42` |
| **Preprocessing** | Custom regex pipeline, 45 stopwords, no NLTK |

---

## Chrome Extension Highlights

The Gmail integration is the most technically complex part of this project:
- **SPA-Aware Polling** -- Gmail is a Single Page Application. The content script uses `setInterval` to monitor URL hash changes and detect when a user opens an email.
- **4-Strategy Extraction Engine** -- Gmail's DOM is heavily obfuscated. The extension tries four different CSS selector strategies as fallbacks to reliably extract email body text.
- **Noise Reduction** -- Automatically strips Gmail UI text ("Reply", "Forward", "Report spam") that gets accidentally included during DOM scraping.
- **Dark Glassmorphism UI** -- The popup features animated gradients, backdrop blur effects, floating orbs, and a custom tab navigator -- all in vanilla CSS.
- **Persistent History** -- The last 20 scans are saved via `chrome.storage.local`, surviving browser restarts.

---

## Deployment

The project ships with production-ready deployment configs:

```bash
# Automated build + retrain
chmod +x build.sh && ./build.sh
```

| File | Purpose |
|------|---------|
| `Procfile` | `web: gunicorn api:app` -- Gunicorn process config |
| `build.sh` | Installs deps + retrains model on deploy |
| `runtime.txt` | Locks Python version to 3.10 |
| `requirements.txt` | Pinned dependency versions |

> **Compatible with:** Render, Heroku, Railway, and any Docker-based platform.

---

## Contributors

| Role | Responsibility |
|------|---------------|
| **Role 1** | ML & Backend Architect: Data pipeline, NLP preprocessing, model training, Flask API, & Hybrid Scoring Engine |
| **Role 2** | Frontend & Extension Architect: Chrome Extension (MV3), Gmail DOM integration, Streamlit dashboard & UI/UX design |

---

## License

This project is licensed under the **MIT License** -- see the [LICENSE](LICENSE) file for details.

Built with Python, Flask, Scikit-Learn, and pure JavaScript.
