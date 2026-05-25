<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Chrome-Extension%20MV3-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">📧 Smart Spam Detector</h1>

<p align="center">
  <strong>An AI-powered, multi-platform spam detection system that combines Machine Learning, a Chrome Extension with live Gmail integration, and a Streamlit web dashboard — all backed by a Hybrid Scoring Engine that goes far beyond simple classification.</strong>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-hybrid-scoring-engine">Hybrid Engine</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-getting-started">Setup</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-model-performance">Performance</a> •
  <a href="#-contributors">Contributors</a>
</p>

---

## 🎯 Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Hybrid Scoring Engine</h3>
      <p>Not just a model wrapper — combines <strong>ML probabilities</strong>, <strong>trigger-word density analysis</strong>, and <strong>Gmail contextual signals</strong> for superior accuracy.</p>
    </td>
    <td width="50%">
      <h3>🔌 Chrome Extension (Manifest V3)</h3>
      <p>Injects directly into Gmail's UI. One-click spam scan with an in-page overlay — no need to leave your inbox.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📊 Streamlit Dashboard</h3>
      <p>A standalone web interface with real-time predictions, confidence scores, trigger word highlighting, and live model metrics.</p>
    </td>
    <td width="50%">
      <h3>⚡ Lightweight NLP Pipeline</h3>
      <p>Custom-built text preprocessor with <strong>zero dependency on NLTK</strong>. Faster cold-starts, smaller deployment footprint.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔍 Explainable AI</h3>
      <p>Every prediction highlights the exact <strong>spam trigger words</strong> found, giving users full transparency into <em>why</em> a message was flagged.</p>
    </td>
    <td width="50%">
      <h3>☁️ Production Ready</h3>
      <p>Gunicorn + Flask backend, automated <code>build.sh</code> training pipeline, and deployment configs for Render/Heroku out of the box.</p>
    </td>
  </tr>
</table>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SMART SPAM DETECTOR                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────────────────────────────────┐  │
│  │   DATA LAYER     │     │              ML PIPELINE                     │  │
│  │                  │     │                                              │  │
│  │  dataset.csv     │────▶│  merge_data.py ──▶ text_preprocess.py       │  │
│  │  enron_emails.csv│     │       │                   │                  │  │
│  │                  │     │       ▼                   ▼                  │  │
│  │  (~15K+ emails)  │     │   train.py ──▶ TF-IDF + Naive Bayes        │  │
│  └──────────────────┘     │       │                                      │  │
│                           │       ▼                                      │  │
│                           │  models/model.pkl + vectorizer.pkl           │  │
│                           └──────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     FLASK API  (api.py)                               │  │
│  │                                                                       │  │
│  │  POST /predict ──▶ ┌─────────────────────────────────────────────┐   │  │
│  │                     │        HYBRID SCORING ENGINE                │   │  │
│  │                     │                                             │   │  │
│  │                     │  Signal 1: ML Probability (Naive Bayes)     │   │  │
│  │                     │  Signal 2: Gmail Context (spam folder?)     │   │  │
│  │                     │  Signal 3: Trigger Word Density (~70 words) │   │  │
│  │                     │                                             │   │  │
│  │                     │  ──▶ Final Verdict + Confidence Score       │   │  │
│  │                     └─────────────────────────────────────────────┘   │  │
│  └────────────────────────┬──────────────────────────┬───────────────────┘  │
│                           │                          │                      │
│              ┌────────────▼────────────┐  ┌─────────▼────────────┐         │
│              │   CHROME EXTENSION      │  │  STREAMLIT DASHBOARD │         │
│              │   (Manifest V3)         │  │  (app.py)            │         │
│              │                         │  │                      │         │
│              │  • Gmail DOM injection  │  │  • Web-based scanner │         │
│              │  • In-page overlay      │  │  • Confidence meter  │         │
│              │  • Scan history         │  │  • Model metrics     │         │
│              │  • Dark glass UI        │  │  • Trigger highlights │         │
│              └─────────────────────────┘  └──────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Hybrid Scoring Engine

> The core differentiator — our backend doesn't just return raw ML predictions. It fuses **three independent signals** to dramatically reduce false negatives.

```
                    ┌──────────────────────┐
                    │   Input Email Text    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌───────────┐ ┌─────────────────┐
     │  Signal 1:     │ │ Signal 2: │ │   Signal 3:     │
     │  ML Model      │ │ Gmail     │ │   Trigger Word  │
     │  Probability   │ │ Context   │ │   Density       │
     │                │ │           │ │                 │
     │ Naive Bayes    │ │ Is email  │ │ ~70 curated     │
     │ predict_proba  │ │ in spam   │ │ spam keywords   │
     │ via TF-IDF     │ │ folder?   │ │ with word-      │
     │ vectorization  │ │ Phishing  │ │ boundary regex  │
     │                │ │ warning?  │ │                 │
     └───────┬────────┘ └─────┬─────┘ └───────┬─────────┘
             │                │               │
             └────────────────┼───────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Fusion Logic    │
                    │                  │
                    │ • Gmail override │
                    │ • Density boost  │
                    │ • Confidence cap │
                    └────────┬─────────┘
                             ▼
                  ┌────────────────────┐
                  │  Final Prediction  │
                  │  + Confidence %    │
                  │  + Trigger Words   │
                  └────────────────────┘
```

| Signal | Source | When It Fires |
|--------|--------|---------------|
| **ML Probability** | `MultinomialNB.predict_proba()` | Always — base prediction |
| **Gmail Context** | Chrome extension reads Gmail UI state | When email is in spam folder or has phishing banner |
| **Trigger Density** | Regex scan against 70+ keywords | When ≥2 trigger words found with sufficient density |

---

## 🛠 Tech Stack

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

## 📁 Project Structure

```
smart-spam-detector/
│
├── 🧠 ML & Backend
│   ├── merge_data.py           # Merges SMS + Enron datasets
│   ├── text_preprocess.py      # Custom NLP pipeline (no NLTK)
│   ├── train.py                # Model training script
│   ├── api.py                  # Flask API + Hybrid Scoring Engine
│   ├── models/
│   │   ├── model.pkl           # Serialized Naive Bayes model
│   │   ├── vectorizer.pkl      # Serialized TF-IDF vectorizer
│   │   └── model.py            # Prediction helper module
│   ├── dataset.csv             # Combined training dataset (~15K+)
│   └── enron_emails.csv        # Enron email corpus
│
├── 🔌 Chrome Extension
│   └── chrome-extension/
│       ├── manifest.json       # Manifest V3 configuration
│       ├── background.js       # Service worker (API relay)
│       ├── content.js          # Gmail DOM injection engine
│       ├── content.css         # In-page overlay styles
│       ├── popup.html          # Extension popup UI
│       ├── popup.css           # Dark glassmorphism styles
│       ├── popup.js            # Scan / History / Settings logic
│       └── icons/              # Extension icons (16/48/128px)
│
├── 📊 Streamlit Dashboard
│   └── app.py                  # Web-based scanner + metrics
│
├── ⚙️ Deployment
│   ├── requirements.txt        # Python dependencies
│   ├── build.sh                # Automated build + retrain script
│   ├── Procfile                # Gunicorn config for production
│   ├── runtime.txt             # Python version lock (3.10)
│   └── .gitignore              # Ignore rules
│
└── 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Google Chrome** (for the extension)
- **Git**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rahul-kumar-362/smart-spam-detector.git
cd smart-spam-detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train.py
```

> This generates `models/model.pkl` and `models/vectorizer.pkl`.

### 4️⃣ Start the API Server

```bash
# Development
python api.py

# Production
gunicorn api:app --bind 0.0.0.0:5000
```

The API will be live at `http://localhost:5000`.

### 5️⃣ Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

### 6️⃣ Install the Chrome Extension

1. Open `chrome://extensions/` in Google Chrome
2. Enable **Developer Mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `chrome-extension/` folder
5. Open Gmail → Click the extension icon → Go to **Settings** → Enter your API URL
6. Open any email → Click the floating **"Check Spam"** button

---

## 📡 API Reference

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

## 📈 Model Performance

<table>
  <tr>
    <td align="center">
      <h3>96.68%</h3>
      <p><strong>Accuracy</strong></p>
    </td>
    <td align="center">
      <h3>97%</h3>
      <p><strong>Precision</strong></p>
    </td>
    <td align="center">
      <h3>96%</h3>
      <p><strong>Recall</strong></p>
    </td>
    <td align="center">
      <h3>96.5%</h3>
      <p><strong>F1 Score</strong></p>
    </td>
  </tr>
</table>

| Detail | Specification |
|--------|--------------|
| **Algorithm** | Multinomial Naive Bayes |
| **Feature Extraction** | TF-IDF (top 5,000 features) |
| **Training Data** | ~15,000+ emails (SMS Spam + Enron Corpus) |
| **Train/Test Split** | 80/20 with `random_state=42` |
| **Preprocessing** | Custom regex pipeline, 74 stopwords, no NLTK |

---

## 🧩 Chrome Extension Highlights

The Gmail integration is the most technically complex part of this project:

- **🔄 SPA-Aware Polling** — Gmail is a Single Page Application. The content script uses `setInterval` to monitor URL hash changes and detect when a user opens an email.

- **📝 4-Strategy Extraction Engine** — Gmail's DOM is heavily obfuscated. The extension tries four different CSS selector strategies as fallbacks to reliably extract email body text.

- **🧹 Noise Reduction** — Automatically strips Gmail UI text ("Reply", "Forward", "Report spam") that gets accidentally included during DOM scraping.

- **🎨 Dark Glassmorphism UI** — The popup features animated gradients, backdrop blur effects, floating orbs, and a custom tab navigator — all in vanilla CSS.

- **💾 Persistent History** — The last 20 scans are saved via `chrome.storage.local`, surviving browser restarts.

---

## 🚢 Deployment

The project ships with production-ready deployment configs:

```bash
# Automated build + retrain
chmod +x build.sh && ./build.sh
```

| File | Purpose |
|------|---------|
| `Procfile` | `web: gunicorn api:app` — Gunicorn process config |
| `build.sh` | Installs deps + retrains model on deploy |
| `runtime.txt` | Locks Python version to 3.10 |
| `requirements.txt` | Pinned dependency versions |

> **Compatible with:** Render, Heroku, Railway, and any Docker-based platform.

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <strong>Role 1</strong><br/>
      <sub>ML & Backend Architect</sub><br/><br/>
      <em>Data pipeline, NLP preprocessing,<br/>model training, Flask API,<br/>& Hybrid Scoring Engine</em>
    </td>
    <td align="center">
      <strong>Role 2</strong><br/>
      <sub>Frontend & Extension Architect</sub><br/><br/>
      <em>Chrome Extension (MV3), Gmail<br/>DOM integration, Streamlit<br/>dashboard & UI/UX design</em>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ using Python, Flask, Scikit-Learn, and pure JavaScript</sub>
</p>
