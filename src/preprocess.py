# src/preprocess.py
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


#path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "processed.csv"


def clean_text(s: str) -> str:
    """Basic text cleaning suitable for ML baselines."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # URLs
    s = re.sub(r"<.*?>", " ", s)                    # HTML tags
    s = re.sub(r"[^a-z0-9\s]+", " ", s)             # punctuation/symbols
    s = re.sub(r"\s+", " ", s).strip()              # extra whitespace
    return s


def main() -> None:
    fake_path = RAW_DIR / "Fake.csv"
    true_path = RAW_DIR / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Expected files not found. Make sure you have:\n"
            f"  {fake_path}\n"
            f"  {true_path}\n"
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Create binary labels: fake=1, real=0
    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Safety: handle common column names
    # Kaggle dataset has: title, text, subject, date
    if "text" not in df.columns:
        raise ValueError("Could not find a 'text' column in the CSVs.")
    if "title" not in df.columns:
        df["title"] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Combine title + body 
    df["combined_text"] = (df["title"] + " " + df["text"]).str.strip()
    df["combined_text"] = df["combined_text"].apply(clean_text)

    # Drop empty texts
    df = df[df["combined_text"].str.len() > 0].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Keep optional columns if they exist (useful for later analysis)
    keep_cols = ["combined_text", "label"]
    for optional in ["title", "text", "subject", "date"]:
        if optional in df.columns:
            keep_cols.append(optional)

    df_out = df[keep_cols]
    df_out.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(df_out))
    print("Label counts:\n", df_out["label"].value_counts())


if __name__ == "__main__":
    main()
