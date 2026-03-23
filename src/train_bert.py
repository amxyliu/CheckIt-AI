from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_checkpoints"



def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    # 1. Load Data
    df = pd.read_csv(DATA_PATH).dropna(subset=['combined_text', 'label'])
    
    # Select ONLY what we need to avoid "Duplicate Column" errors
    # We take your combined_text and label, then rename them for the model
    df = df[['combined_text', 'label']].copy()
    df.columns = ['text', 'label'] 
    
    # Split the data
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Convert to Hugging Face format
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    # 2. Tokenize
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 3. Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5, # BERT needs very small learning rates
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 5. Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    # 6. Train
    print("Starting BERT training...")
    trainer.train()

    # 7. Final Evaluation
    preds = trainer.predict(tokenized_test)
    y_pred = np.argmax(preds.predictions, axis=1)
    
    print("\nBERT Classification Report:\n")
    print(classification_report(test_df['label'], y_pred, digits=4))

    # Save the final model
    model.save_pretrained(PROJECT_ROOT / "models" / "bert_final")
    tokenizer.save_pretrained(PROJECT_ROOT / "models" / "bert_final")

if __name__ == "__main__":
    main()