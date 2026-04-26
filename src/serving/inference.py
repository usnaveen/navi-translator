"""Inference engine — loads models and runs the translation chain.

The inference chain for audio:
  Audio → Whisper (ASR) → Na'vi text → MarianMT (NMT) → English

For text input, we skip the Whisper step.

Key learning points:
- Models are loaded ONCE at startup, not per-request (expensive!)
- We load from MLflow registry with stage='Production'
- The fallback mechanism: if MarianMT confidence < 0.4, we fall back
  to word-by-word Reykunyu dictionary lookup
- torch.no_grad() disables gradient computation during inference (saves memory)
"""

import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TranslationEngine:
    """Manages model loading and inference for Na'vi translation."""

    def __init__(self, params_file: str = None):
        self.params = self._load_params(params_file)
        self.whisper_model = None
        self.whisper_processor = None
        self.marian_model = None
        self.marian_tokenizer = None
        self.words = []
        self.word_lookup = {}
        self.model_version = "none"
        self._loaded = False

    def _load_params(self, params_file: str = None) -> dict:
        path = params_file or str(PROJECT_ROOT / "params.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def whisper_loaded(self) -> bool:
        return self.whisper_model is not None

    @property
    def marian_loaded(self) -> bool:
        return self.marian_model is not None

    def load_models(self):
        """Load all models. Called once at startup."""
        self._load_dictionary()
        self._load_whisper()
        self._load_marian()
        self._loaded = self.whisper_loaded or self.marian_loaded
        logger.info("Models loaded. Ready: %s", self._loaded)

    def _load_dictionary(self):
        """Load Na'vi dictionary for word-by-word fallback."""
        words_path = PROJECT_ROOT / "data" / "raw" / "words.json"
        try:
            with open(words_path, encoding="utf-8") as f:
                self.words = json.load(f)
            self.word_lookup = {w["navi"].lower(): w["en"] for w in self.words}
            logger.info("Loaded %d dictionary entries", len(self.words))
        except FileNotFoundError:
            logger.warning("No dictionary found at %s", words_path)

    def _load_whisper(self):
        """Load Whisper model from MLflow registry or local path."""
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            # Try local model directory first
            model_dir = PROJECT_ROOT / "models" / "whisper-navi-lora"
            if model_dir.exists():
                self.whisper_processor = WhisperProcessor.from_pretrained(str(model_dir))
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
                self.whisper_model.eval()
                logger.info("Whisper loaded from %s", model_dir)
            else:
                # Load base model as fallback
                base_model = self.params["whisper"]["base_model"]
                self.whisper_processor = WhisperProcessor.from_pretrained(base_model)
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(base_model)
                self.whisper_model.eval()
                logger.info("Whisper loaded from base: %s", base_model)
        except Exception as e:
            logger.error("Failed to load Whisper: %s", e)

    def _load_marian(self):
        """Load MarianMT model from local path or base model."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_dir = PROJECT_ROOT / "models" / "marian-navi-en"
            if model_dir.exists():
                self.marian_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.marian_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
                self.marian_model.eval()
                logger.info("MarianMT loaded from %s", model_dir)
            else:
                base_model = self.params["marian"]["base_model"]
                self.marian_tokenizer = AutoTokenizer.from_pretrained(base_model)
                self.marian_model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
                self.marian_model.eval()
                logger.info("MarianMT loaded from base: %s", base_model)
        except Exception as e:
            logger.error("Failed to load MarianMT: %s", e)

    def transcribe_audio(self, audio_bytes: bytes) -> tuple[str, float]:
        """Transcribe Na'vi audio to text using Whisper.

        Returns (transcription, confidence).
        """
        if not self.whisper_loaded:
            raise RuntimeError("Whisper model not loaded")

        sr = self.params["data"]["audio_sample_rate"]

        # Load audio from bytes
        import io
        import soundfile as sf

        audio_io = io.BytesIO(audio_bytes)
        try:
            audio, file_sr = sf.read(audio_io)
        except Exception:
            # Try librosa for more format support
            audio_io.seek(0)
            audio, file_sr = librosa.load(audio_io, sr=sr, mono=True)

        # Resample if needed
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Extract features and generate
        input_features = self.whisper_processor.feature_extractor(
            audio.astype(np.float32), sampling_rate=sr, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            output = self.whisper_model.generate(
                input_features, return_dict_in_generate=True, output_scores=True
            )

        transcription = self.whisper_processor.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0].strip()

        # Estimate confidence from scores
        if hasattr(output, "scores") and output.scores:
            log_probs = [torch.max(s.log_softmax(dim=-1)).item() for s in output.scores]
            confidence = min(1.0, max(0.0, np.exp(np.mean(log_probs))))
        else:
            confidence = 0.5

        return transcription, round(confidence, 3)

    def translate_text(self, navi_text: str) -> tuple[str, float, list[dict]]:
        """Translate Na'vi text to English.

        Returns (english, confidence, word_breakdown).
        If model confidence < threshold, falls back to dictionary lookup.
        """
        threshold = self.params["marian"]["fallback_confidence_threshold"]
        english = ""
        confidence = 0.0
        word_breakdown = []

        # Try MarianMT first
        if self.marian_loaded:
            english, confidence = self._translate_with_marian(navi_text)

        # Fallback to dictionary if confidence too low
        if confidence < threshold or not self.marian_loaded:
            english_fb, breakdown = self._translate_with_dictionary(navi_text)
            word_breakdown = breakdown
            if confidence < threshold:
                english = english_fb
                confidence = 0.3  # dictionary lookup confidence

        return english, round(confidence, 3), word_breakdown

    def _translate_with_marian(self, text: str) -> tuple[str, float]:
        """Translate using MarianMT model."""
        inputs = self.marian_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )

        with torch.no_grad():
            output = self.marian_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )

        translation = self.marian_tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0].strip()

        # Confidence from sequence scores
        if hasattr(output, "sequences_scores") and output.sequences_scores is not None:
            confidence = min(1.0, max(0.0, float(torch.sigmoid(output.sequences_scores[0]))))
        else:
            confidence = 0.5

        return translation, confidence

    def _translate_with_dictionary(self, text: str) -> tuple[str, list[dict]]:
        """Fallback: word-by-word dictionary lookup via Reykunyu."""
        tokens = text.strip().split()
        translations = []
        breakdown = []

        for token in tokens:
            en = self.word_lookup.get(token.lower(), f"[{token}]")
            first_meaning = en.split(";")[0].strip() if ";" in en else en
            translations.append(first_meaning)
            breakdown.append({
                "navi": token,
                "en": first_meaning,
                "found": token.lower() in self.word_lookup,
            })

        return " ".join(translations), breakdown
