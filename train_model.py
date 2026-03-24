import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_dataset():
    fake_path = BASE_DIR / "dataset" / "Fake.csv"
    true_path = BASE_DIR / "dataset" / "True.csv"

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise Exception("Required dataset files not found: dataset/Fake.csv and dataset/True.csv")

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    required_columns = {"title", "text"}
    if not required_columns.issubset(fake.columns) or not required_columns.issubset(true.columns):
        raise Exception("Dataset files must contain at least 'title' and 'text' columns")

    fake["label"] = "fake"
    true["label"] = "real"

    data = pd.concat([fake, true], ignore_index=True)
    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")
    data["content"] = (data["title"] + " " + data["text"]).str.strip()
    data = data[data["content"].str.len() > 0]

    return data[["content", "label"]]


data = load_dataset()

# Features
X = data["content"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Accuracy
accuracy = model.score(vectorizer.transform(X_test), y_test)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model
pickle.dump(model, open(BASE_DIR / "model.pkl", "wb"))
pickle.dump(vectorizer, open(BASE_DIR / "vectorizer.pkl", "wb"))

print("Model saved successfully!")