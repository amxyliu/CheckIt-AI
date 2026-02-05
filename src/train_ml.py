# src/train_ml.py
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}. Run preprocess first:\n"
            "  python src/preprocess.py"
        )

    df = pd.read_csv(DATA_PATH)
    if "combined_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'combined_text' and 'label' in processed.csv")

    #Split data into training and testing
    X = df["combined_text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    #train Logistic Regression model
    model = LogisticRegression(max_iter=2000, n_jobs=None)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)


    #output
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    #save files
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf.joblib")
    joblib.dump(model, MODELS_DIR / "logreg.joblib")
    print("\nSaved model + vectorizer to:", MODELS_DIR)


if __name__ == "__main__":
    main()
