from pathlib import Path

import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# Load model and vectorizer
vectorizer = joblib.load(MODELS_DIR / "tfidf.joblib")
model = joblib.load(MODELS_DIR / "logreg.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text", "").strip()

    if not text:
        return render_template(
            "index.html",
            error="Please enter a news article or headline."
        )

    try:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vec)[0]
            confidence = round(max(probabilities) * 100, 2)

        # Adjust this mapping if your dataset labels are reversed
        predicted_label = "FAKE" if prediction == 1 else "REAL"

        return render_template(
            "index.html",
            result=predicted_label,
            confidence=confidence,
            input_text=text
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction failed: {str(e)}",
            input_text=text
        )


if __name__ == "__main__":
    app.run(debug=True)