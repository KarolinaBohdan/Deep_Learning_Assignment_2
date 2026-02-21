import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- Configuration ---
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
PARQUET_FILE = "patents_50k_green.parquet"
GOLD_CSV = "hitl_green_100_gold.csv"
OUTPUT_DIR = "patentsberta_green_ft"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 1

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    print(">>> Loading Data...")
    # 1. Load the full dataset
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(f"Could not find {PARQUET_FILE}")
    
    full_df = pd.read_parquet(PARQUET_FILE)
    
    # 2. Load the human labeled gold data
    # Note: Your notebook indicated this CSV uses ';' separator
    if not os.path.exists(GOLD_CSV):
        raise FileNotFoundError(f"Could not find {GOLD_CSV}")
        
    hitl_gold = pd.read_csv(GOLD_CSV, sep=";")
    
    # 3. Merge Logic (Replicating notebook cells 70-71)
    if "doc_id" in hitl_gold.columns and "id" not in hitl_gold.columns:
        hitl_gold = hitl_gold.rename(columns={"doc_id": "id"})
    
    # Keep only relevant columns for merge
    hitl_gold_small = hitl_gold[["id", "is_green_human"]].copy()
    
    # Merge into full dataset
    df = full_df.merge(hitl_gold_small, on="id", how="left")
    
    # Create is_green_gold column
    df["is_green_gold"] = df["is_green_silver"]
    mask = df["is_green_human"].notna()
    
    # Override silver labels with human labels where available
    df.loc[mask, "is_green_gold"] = df.loc[mask, "is_green_human"].astype(int)
    df["is_green_gold"] = df["is_green_gold"].astype(int)
    
    print(f">>> Merged Gold Labels. {mask.sum()} human labels found.")

    # 4. Create Splits
    # Train uses gold labels (HITL requirement)
    train_df = df[df["split"] == "train_silver"].copy()
    
    # Eval uses silver labels (Assignment requirement)
    eval_df = df[df["split"] == "eval_silver"].copy()
    
    # Optional: Separate gold set for reporting
    gold_100_df = df[df["is_green_human"].notna()].copy()

    # 5. Convert to HF Datasets
    train_hf = Dataset.from_pandas(
        train_df[["text", "is_green_gold"]].rename(columns={"is_green_gold": "label"}),
        preserve_index=False
    )
    eval_hf = Dataset.from_pandas(
        eval_df[["text", "is_green_silver"]].rename(columns={"is_green_silver": "label"}),
        preserve_index=False
    )
    gold_hf = Dataset.from_pandas(
        gold_100_df[["text", "is_green_gold"]].rename(columns={"is_green_gold": "label"}),
        preserve_index=False
    )

    print(f">>> Train size: {len(train_hf)} | Eval size: {len(eval_hf)}")

    # 6. Tokenization
    print(">>> Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
    
    train_ds = train_hf.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_hf.map(tokenize_fn, batched=True, remove_columns=["text"])
    gold_ds = gold_hf.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. Model Setup
    print(">>> Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # 8. Trainer Setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        save_total_limit=1,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 9. Train
    print(">>> Starting Training...")
    trainer.train()

    # 10. Final Evaluation
    print("\n>>> Evaluating on Eval Set (Silver Labels):")
    eval_metrics = trainer.evaluate(eval_dataset=eval_ds)
    print(eval_metrics)

    print("\n>>> Evaluating on Gold 100 Set (Human Labels):")
    gold_metrics = trainer.evaluate(eval_dataset=gold_ds)
    print(gold_metrics)

    # 11. Save Model
    print(f">>> Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()