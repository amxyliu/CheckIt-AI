# src/train_lstm.py
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def build_model(vocab_size: int, max_len: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=128,
                mask_zero=True,  # helps LSTM ignore padding
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.0)
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}. Run preprocess first:\n"
            "  python src/preprocess.py"
        )

    df = pd.read_csv(DATA_PATH)
    if "combined_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'combined_text' and 'label' in processed.csv")

    # Data
    X = df["combined_text"].astype(str).fillna("")
    y = df["label"].astype(int).to_numpy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize
    vocab_size_target = 50000  # cap vocab
    max_len = 250  # cap sequence length (tune this)

    tokenizer = Tokenizer(
        num_words=vocab_size_target,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(X_train.tolist())

    train_seq = tokenizer.texts_to_sequences(X_train.tolist())
    test_seq = tokenizer.texts_to_sequences(X_test.tolist())

    X_train_pad = pad_sequences(
        train_seq, maxlen=max_len, padding="post", truncating="post"
    )
    X_test_pad = pad_sequences(
        test_seq, maxlen=max_len, padding="post", truncating="post"
    )

    # Effective vocab size for Embedding:
    # Keras Tokenizer indexes start at 1; index 0 is padding.
    # If num_words is set, only words < num_words are kept in sequences.
    vocab_size_effective = min(vocab_size_target, len(tokenizer.word_index) + 1)

    # Build + train
    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_model(vocab_size_effective, max_len)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    probs = model.predict(X_test_pad, batch_size=256).ravel()
    y_pred = (probs >= 0.5).astype(int)

    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lstm.keras"
    tok_path = MODELS_DIR / "tokenizer.joblib"
    cfg_path = MODELS_DIR / "lstm_config.joblib"

    model.save(model_path)
    joblib.dump(tokenizer, tok_path)
    joblib.dump(
        {
            "max_len": max_len,
            "vocab_size_target": vocab_size_target,
            "vocab_size_effective": vocab_size_effective,
            "threshold": 0.5,
        },
        cfg_path,
    )

    print("\nSaved LSTM model to:", model_path)
    print("Saved tokenizer to:", tok_path)
    print("Saved config to:", cfg_path)


if __name__ == "__main__":
    main()