"""Whisper ASR fine-tuning with LoRA — trains speech-to-text for Na'vi.

How this works:
1. Load Whisper Small (244M params) from HuggingFace
2. Freeze ALL weights, then attach LoRA adapters to q_proj and v_proj
   (only ~0.5M trainable params — 99.8% reduction)
3. Train on Na'vi audio files + their text transcriptions
4. MLflow autolog captures everything automatically

Key learning points:
- LoRA adds small rank-decomposition matrices A and B to attention layers
  Original weight W stays frozen; output = W*x + B*A*x
- WhisperForConditionalGeneration is an encoder-decoder model:
  encoder processes audio spectrograms, decoder generates text tokens
- We force the decoder to generate Na'vi by setting language='na'
- DataCollator handles variable-length audio padding automatically
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params(config_path: str = None) -> dict:
    path = config_path or str(PROJECT_ROOT / "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio_dataset(split_dir: Path, audio_dir: Path, sr: int) -> Dataset:
    """Load audio files and transcriptions into a HuggingFace Dataset."""
    import librosa

    manifest_path = split_dir / "audio_manifest.json"
    if not manifest_path.exists():
        logger.warning("No audio manifest at %s", manifest_path)
        return Dataset.from_dict({"audio": [], "transcription": []})

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load words.json for transcriptions (word → navi text)
    words_path = PROJECT_ROOT / "data" / "raw" / "words.json"
    with open(words_path, encoding="utf-8") as f:
        words = json.load(f)
    word_map = {w["navi"].lower(): w["navi"] for w in words}

    audios = []
    transcriptions = []

    for item in manifest:
        filepath = audio_dir / item["filename"]
        if not filepath.exists():
            continue

        stem = filepath.stem
        # Find the Na'vi word this audio corresponds to
        transcription = word_map.get(stem.lower(), stem)

        try:
            audio, _ = librosa.load(str(filepath), sr=sr, mono=True)
            audios.append(audio.tolist())
            transcriptions.append(transcription)
        except Exception as e:
            logger.warning("Failed to load %s: %s", filepath, e)

    return Dataset.from_dict({"audio": audios, "transcription": transcriptions})


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for Whisper that handles variable-length audio.

    This collator:
    1. Pads audio features to the same length in a batch
    2. Pads label (text token) sequences
    3. Replaces padding token ids with -100 so they're ignored in loss
    """

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        # Extract audio inputs and pad
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract labels and pad
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(example, processor, sr):
    """Preprocess a single example: extract features and tokenize."""
    audio = np.array(example["audio"], dtype=np.float32)
    input_features = processor.feature_extractor(
        audio, sampling_rate=sr, return_tensors="np"
    ).input_features[0]

    labels = processor.tokenizer(example["transcription"]).input_ids

    return {"input_features": input_features, "labels": labels}


def compute_wer_metric(pred, tokenizer):
    """Compute Word Error Rate from predictions."""
    from jiwer import wer

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Filter empty strings
    pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not pairs:
        return {"wer": 1.0}

    pred_filtered, label_filtered = zip(*pairs)
    wer_score = wer(list(label_filtered), list(pred_filtered))

    return {"wer": round(wer_score, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "params.yaml"))
    args = parser.parse_args()

    params = load_params(args.config)
    whisper_params = params["whisper"]
    mlflow_params = params["mlflow"]
    data_params = params["data"]

    sr = data_params["audio_sample_rate"]

    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run(run_name="whisper-lora-finetune") as run:
        # Log dataset version
        mlflow.log_param("dataset_version", "local")
        mlflow.log_param("model_type", "whisper")

        # Load processor and model
        logger.info("Loading Whisper model: %s", whisper_params["base_model"])
        processor = WhisperProcessor.from_pretrained(whisper_params["base_model"])
        model = WhisperForConditionalGeneration.from_pretrained(whisper_params["base_model"])

        # Force Na'vi language
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=whisper_params["language"], task="transcribe"
        )
        model.config.suppress_tokens = []

        # Apply LoRA
        logger.info("Applying LoRA: r=%d, alpha=%d", whisper_params["lora_r"], whisper_params["lora_alpha"])
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=whisper_params["lora_r"],
            lora_alpha=whisper_params["lora_alpha"],
            target_modules=whisper_params["lora_target_modules"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # Required when combining PEFT/LoRA with gradient_checkpointing —
        # makes the (frozen) input embeddings produce gradients so backward works.
        model.enable_input_require_grads()

        # Log LoRA params
        mlflow.log_params({
            "lora_r": whisper_params["lora_r"],
            "lora_alpha": whisper_params["lora_alpha"],
            "lora_target_modules": str(whisper_params["lora_target_modules"]),
            "learning_rate": whisper_params["learning_rate"],
            "batch_size": whisper_params["batch_size"],
            "num_epochs": whisper_params["num_epochs"],
        })

        # Load datasets
        audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"
        train_dataset = load_audio_dataset(
            PROJECT_ROOT / "data" / "processed" / "train", audio_dir, sr
        )
        val_dataset = load_audio_dataset(
            PROJECT_ROOT / "data" / "processed" / "val", audio_dir, sr
        )

        logger.info("Train: %d examples, Val: %d examples", len(train_dataset), len(val_dataset))

        if len(train_dataset) == 0:
            logger.error("No training data found — cannot train")
            mlflow.log_metric("wer", 1.0)
            return

        # Preprocess datasets
        train_dataset = train_dataset.map(
            lambda x: prepare_dataset(x, processor, sr),
            remove_columns=train_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            lambda x: prepare_dataset(x, processor, sr),
            remove_columns=val_dataset.column_names,
        )

        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        # Training arguments
        output_dir = PROJECT_ROOT / "models" / "whisper-navi-lora"
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=whisper_params["batch_size"],
            per_device_eval_batch_size=whisper_params["batch_size"],
            gradient_accumulation_steps=4,        # effective batch 4 with ~1/4 the memory
            gradient_checkpointing=True,          # trade compute for memory
            learning_rate=whisper_params["learning_rate"],
            warmup_steps=whisper_params["warmup_steps"],
            num_train_epochs=whisper_params["num_epochs"],
            eval_strategy="steps",
            eval_steps=whisper_params["eval_steps"],
            save_strategy="steps",
            save_steps=whisper_params["eval_steps"],
            save_total_limit=2,          # keep only best + latest checkpoint (saves disk)
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=25,
            report_to=["mlflow"],
            push_to_hub=False,
            remove_unused_columns=False,
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=processor.feature_extractor,
            compute_metrics=lambda pred: compute_wer_metric(pred, processor.tokenizer),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=whisper_params["early_stopping_patience"]
                )
            ],
        )

        # Train
        logger.info("Starting Whisper LoRA training...")
        trainer.train()

        # Evaluate
        eval_results = trainer.evaluate()
        logger.info("Eval results: %s", eval_results)

        wer_score = eval_results.get("eval_wer", 1.0)
        mlflow.log_metric("final_wer", wer_score)

        # Save model
        model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))
        mlflow.log_artifacts(str(output_dir), artifact_path="whisper-navi-lora")

        logger.info("=== Whisper training complete. WER: %.4f ===", wer_score)


if __name__ == "__main__":
    main()
