import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from text_preprocess import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
logging.info("Loading dataset...")
try:
    df = pd.read_csv("dataset.csv", encoding="latin-1")
except FileNotFoundError:
    logging.error("dataset.csv not found! Training aborted.")
    exit(1)

# Ensure necessary columns exist (renaming if old format is still present, just in case)
if 'v1' in df.columns:
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})

# Keep only required columns
df = df[["label", "text"]]

logging.info(f"Dataset loaded: {len(df)} records")

# Apply cleaning
logging.info("Preprocessing text...")
df["text"] = df["text"].apply(clean_text)

# Split features + labels
X = df["text"]
y = df["label"]

# Convert text -> numbers
logging.info("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
X_vec = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train Model
logging.info("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
logging.info(f"Model Training Complete. Accuracy: {accuracy:.4f}")

# Save model
logging.info("Saving model artifacts...")
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
logging.info("All done!")
