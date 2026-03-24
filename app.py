from flask import Flask, render_template, request, Response
import pickle
import os
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, (
            "Model artifacts are missing. Run 'python train_model.py' from the project root "
            "to generate model.pkl and vectorizer.pkl."
        )

    try:
        loaded_model = pickle.load(open(MODEL_PATH, "rb"))
        loaded_vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
        return loaded_model, loaded_vectorizer, None
    except Exception as exc:
        return None, None, (
            "Model artifacts could not be loaded. Re-run 'python train_model.py'. "
            f"Details: {exc}"
        )


model, vectorizer, startup_error = load_artifacts()

prediction_history = {
    "fake": 0,
    "real": 0,
    "timeline": []
}


def load_dashboard_metrics():
    fake_path = BASE_DIR / "dataset" / "Fake.csv"
    true_path = BASE_DIR / "dataset" / "True.csv"

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        return {
            "error": "Required dataset files (Fake.csv and True.csv) were not found.",
            "labels": [],
            "counts": [],
            "avg_text_length": [],
            "subjects": [],
            "subject_counts": []
        }

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = "fake"
    true_df["label"] = "real"

    data = pd.concat([fake_df, true_df], ignore_index=True)
    if "text" not in data.columns:
        data["text"] = ""
    else:
        data["text"] = data["text"].fillna("")

    if "subject" not in data.columns:
        data["subject"] = "unknown"
    else:
        data["subject"] = data["subject"].fillna("unknown")

    label_counts = data["label"].value_counts()
    avg_text_length = data.groupby("label")["text"].apply(lambda series: series.str.split().str.len().mean())
    subject_counts = data["subject"].value_counts().head(8)

    return {
        "error": None,
        "labels": label_counts.index.tolist(),
        "counts": label_counts.values.tolist(),
        "avg_text_length": [round(avg_text_length.get("fake", 0), 2), round(avg_text_length.get("real", 0), 2)],
        "subjects": subject_counts.index.tolist(),
        "subject_counts": subject_counts.values.tolist(),
        "prediction_labels": ["fake", "real"],
        "prediction_counts": [prediction_history["fake"], prediction_history["real"]],
        "prediction_timeline_labels": [item["index"] for item in prediction_history["timeline"]],
        "prediction_timeline_fake": [item["fake_total"] for item in prediction_history["timeline"]],
        "prediction_timeline_real": [item["real_total"] for item in prediction_history["timeline"]]
    }

@app.route("/")
def home():
    return render_template("index.html", startup_error=startup_error)


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/dashboard")
def dashboard():
    metrics = load_dashboard_metrics()
    return render_template("dashboard.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    if startup_error:
        return render_template("index.html", startup_error=startup_error), 500

    text = request.form["news"]

    # Transform
    vector = vectorizer.transform([text])

    # Predict
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0]) * 100

    if prediction == "real":
        prediction_history["real"] += 1
    else:
        prediction_history["fake"] += 1

    prediction_history["timeline"].append(
        {
            "index": len(prediction_history["timeline"]) + 1,
            "fake_total": prediction_history["fake"],
            "real_total": prediction_history["real"]
        }
    )

    if len(prediction_history["timeline"]) > 25:
        prediction_history["timeline"] = prediction_history["timeline"][-25:]

    return render_template(
        "index.html",
        result=prediction,
        confidence=round(confidence, 2),
        startup_error=startup_error
    )

if __name__ == "__main__":
    app.run(debug=True)