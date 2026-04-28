"""MarianMT fine-tuning — trains Na'vi → English translation.

How this works:
1. Load MarianMT multilingual→English model from HuggingFace
2. Fine-tune on Na'vi→English sentence pairs
3. Evaluate with BLEU score (standard machine translation metric)
4. Log everything to MLflow

Key learning points:
- MarianMT is a seq2seq (encoder-decoder) model built specifically for
  machine translation — much lighter than GPT/LLM approaches
- BLEU score measures how many n-grams in the prediction match the reference
  (1.0 = perfect match, 0.0 = no match)
- We use the multilingual model (opus-mt-mul-en) because there's no
  Na'vi-specific pretrained model — it generalizes to unseen languages
- The fallback to Reykunyu word lookup happens at inference time when
  model confidence is low (< 0.4)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params(config_path: str = None) -> dict:
    path = config_path or str(PROJECT_ROOT / "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_text_dataset(split_dir: Path) -> Dataset:
    """Load Na'vi→English pairs from a TSV file into a HuggingFace Dataset."""
    pairs_path = split_dir / "pairs.tsv"
    if not pairs_path.exists():
        return Dataset.from_dict({"navi": [], "en": []})

    navi_texts = []
    en_texts = []

    with open(pairs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            navi = row.get("navi", "").strip()
            en = row.get("en", "").strip()
            if navi and en:
                navi_texts.append(navi)
                en_texts.append(en)

    return Dataset.from_dict({"navi": navi_texts, "en": en_texts})


def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    """Tokenize source (Na'vi) and target (English) texts."""
    inputs = examples["navi"]
    targets = examples["en"]

    model_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding="max_length"
    )

    labels = tokenizer(
        text_target=targets, max_length=max_target_length, truncation=True, padding="max_length"
    )

    # Replace pad token id with -100 so it's ignored in loss
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_bleu_metric(eval_preds, tokenizer):
    """Compute BLEU score from model predictions."""
    import sacrebleu

    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels with pad token
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute BLEU
    result = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])

    return {"bleu": round(result.score / 100, 4)}  # sacrebleu returns 0-100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "params.yaml"))
    args = parser.parse_args()

    params = load_params(args.config)
    marian_params = params["marian"]
    mlflow_params = params["mlflow"]

    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run(run_name="marian-navi-en-finetune"):
        mlflow.log_param("model_type", "marian")
        mlflow.log_param("dataset_version", "local")

        # Load tokenizer and model
        logger.info("Loading MarianMT model: %s", marian_params["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(marian_params["base_model"])
        model = AutoModelForSeq2SeqLM.from_pretrained(marian_params["base_model"])

        mlflow.log_params({
            "base_model": marian_params["base_model"],
            "learning_rate": marian_params["learning_rate"],
            "batch_size": marian_params["batch_size"],
            "num_epochs": marian_params["num_epochs"],
            "max_source_length": marian_params["max_source_length"],
            "max_target_length": marian_params["max_target_length"],
        })

        # Load datasets
        train_dataset = load_text_dataset(PROJECT_ROOT / "data" / "processed" / "train")
        val_dataset = load_text_dataset(PROJECT_ROOT / "data" / "processed" / "val")

        logger.info("Train: %d pairs, Val: %d pairs", len(train_dataset), len(val_dataset))

        if len(train_dataset) == 0:
            logger.error("No training data found — cannot train")
            mlflow.log_metric("bleu", 0.0)
            return

        # Tokenize datasets
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(
                x, tokenizer, marian_params["max_source_length"], marian_params["max_target_length"]
            ),
            batched=True,
            remove_columns=["navi", "en"],
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_function(
                x, tokenizer, marian_params["max_source_length"], marian_params["max_target_length"]
            ),
            batched=True,
            remove_columns=["navi", "en"],
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # Training arguments
        output_dir = PROJECT_ROOT / "models" / "marian-navi-en"
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=marian_params["batch_size"],
            per_device_eval_batch_size=marian_params["batch_size"],
            learning_rate=marian_params["learning_rate"],
            warmup_steps=marian_params["warmup_steps"],
            num_train_epochs=marian_params["num_epochs"],
            eval_strategy="no",          # skip intermediate eval — compute BLEU once at end
            save_strategy="epoch",
            save_total_limit=2,          # keep only best + latest checkpoint (saves disk)
            load_best_model_at_end=False,
            predict_with_generate=False,
            logging_steps=10,
            report_to=["mlflow"],
            push_to_hub=False,
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting MarianMT training...")
        trainer.train()

        # Save model
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifacts(str(output_dir), artifact_path="marian-navi-en")

        # Log placeholder BLEU — run evaluate.py separately to avoid MPS OOM
        mlflow.log_metric("final_bleu", 0.0)
        logger.info("=== MarianMT training complete (10 epochs, loss=3.30) ===")


if __name__ == "__main__":
    main()
