import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from text_preprocess import clean_text

# Load dataset
df = pd.read_csv("dataset.csv", encoding="latin-1")

# Keep only required columns
df = df[["v1", "v2"]]

# Rename columns for clarity (optional but clean)
df.columns = ["label", "text"]

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# Split features + labels
X = df["text"]
y = df["label"]

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save model
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
