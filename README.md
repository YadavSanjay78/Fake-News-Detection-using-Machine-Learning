# Fake News Detection Project

A Flask-based fake news classifier trained on two datasets:
- `dataset/Fake.csv`
- `dataset/True.csv`

The app predicts whether input news text is **fake** or **real**, and includes a **graph dashboard** with dataset and prediction analytics.

## Features

- News classification using `TfidfVectorizer` + `MultinomialNB`
- Training from `Fake.csv` and `True.csv`
- Confidence score for each prediction
- Dashboard charts:
  - Class distribution (fake vs real)
  - Average text length
  - Top subjects
  - Prediction history totals
  - Prediction trend (last 25 predictions)
- Mobile responsive UI

## Project Structure

```text
app.py
train_model.py
requirements.txt
dataset/
  Fake.csv
  True.csv
templates/
  index.html
  dashboard.html
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train_model.py
```

This generates:
- `model.pkl`
- `vectorizer.pkl`

## Run the Flask App

Run:

```bash
python app.py
```

Open in browser:
- Home: `http://127.0.0.1:5000/`
- Dashboard: `http://127.0.0.1:5000/dashboard`

## Notes

- The model is trained using combined `title + text` content from the dataset.
- Prediction history on dashboard is currently stored in memory and resets when the app restarts.
