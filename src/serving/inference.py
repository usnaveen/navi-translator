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
        # Track where each model actually came from (registry vs local)
        # — surfaced through the model_version string so the Grafana panel
        # makes the source visible.
        self._whisper_source = "none"
        self._marian_source = "none"

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
        # Build a human-readable version string. Includes the *source*
        # (registry / local / base) so the Grafana panel makes the loading
        # path visible — important for verifying the registry path works.
        parts = []
        if self.whisper_loaded:
            tag = "whisper-navi-lora" if self._whisper_source != "base" else self.params["whisper"]["base_model"].split("/")[-1]
            parts.append(f"{tag}@{self._whisper_source}")
        if self.marian_loaded:
            tag = "marian-navi-en" if self._marian_source != "base" else self.params["marian"]["base_model"].split("/")[-1]
            parts.append(f"{tag}@{self._marian_source}")
        self.model_version = " | ".join(parts) if parts else "none"
        logger.info("Models loaded. Ready: %s, version: %s", self._loaded, self.model_version)

    @staticmethod
    def _clean_navi(word: str) -> str:
        """Normalize Reykunyu multi-form notation to a plain word.

        Reykunyu stores words like "kal/[txì]" where:
          /  = morpheme boundary (join the parts together)
          [] = optional component (include it for the canonical form)

        Examples:
          "kal/[txì]"    → "kaltxì"
          "['ang]/tsìk"  → "'angtsìk"
          "'aw/si/[teng]" → "'awsiteng"
        """
        return word.replace("[", "").replace("]", "").replace("/", "").strip()

    def _load_dictionary(self):
        """Load Na'vi dictionary for word-by-word fallback.

        Builds two indexes:
        - Raw form:   "kal/[txì]"  → "hello"
        - Clean form: "kaltxì"     → "hello"
        so that user input (always clean) hits the lookup correctly.
        """
        words_path = PROJECT_ROOT / "data" / "raw" / "words.json"
        try:
            with open(words_path, encoding="utf-8") as f:
                self.words = json.load(f)

            self.word_lookup = {}
            for w in self.words:
                en = w["en"]
                raw_key = w["navi"].lower()
                clean_key = self._clean_navi(w["navi"]).lower()
                # Index both so either form resolves correctly
                self.word_lookup[raw_key] = en
                self.word_lookup[clean_key] = en

            logger.info(
                "Loaded %d dictionary entries (%d lookup keys)",
                len(self.words),
                len(self.word_lookup),
            )
        except FileNotFoundError:
            logger.warning("No dictionary found at %s", words_path)

    def _resolve_from_registry(self, registered_name: str, stage: str = "Production") -> str | None:
        """Look up a Production-stage model in the MLflow registry.

        Returns a local filesystem path to the artifact directory, or None
        if the registry is unreachable / the model isn't registered. We
        deliberately keep this best-effort — the backend must still come up
        even when MLflow is down (e.g. during cold start).

        The contract: registry says where the canonical artifact lives, we
        load it from there. Falls back to bundled local snapshots otherwise.
        """
        try:
            import os
            from mlflow.tracking import MlflowClient
            from mlflow.artifacts import download_artifacts

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            client = MlflowClient(tracking_uri=tracking_uri)

            # Find the latest version that's been promoted to *stage*
            versions = client.search_model_versions(f"name='{registered_name}'")
            promoted = [v for v in versions if v.current_stage == stage]
            if not promoted:
                logger.info("Registry has %s but no version in %s stage", registered_name, stage)
                return None

            chosen = max(promoted, key=lambda v: int(v.version))
            uri = f"models:/{registered_name}/{chosen.version}"
            try:
                local_path = download_artifacts(artifact_uri=uri, tracking_uri=tracking_uri)
            except (FileNotFoundError, OSError) as path_err:
                # MLflow server returned an absolute *host* path (the
                # tracking server and the backend share artifact storage
                # via a mounted volume but at different paths). Rewrite
                # to the container-side mount point.
                err_path = str(path_err).split(":", 1)[-1].strip().strip("'.")
                # Find ".../mlruns/<exp_id>/..." segment and remap to /app/mlruns/...
                idx = err_path.find("/mlruns/")
                if idx == -1:
                    raise
                local_path = "/app" + err_path[idx:].rstrip("/.").rstrip()
                if not Path(local_path).exists():
                    raise FileNotFoundError(local_path)
                logger.info("Path rewrite: %s → %s", err_path, local_path)
            logger.info("Registry resolved %s v%s → %s", registered_name, chosen.version, local_path)
            return local_path
        except Exception as e:
            logger.warning("MLflow registry lookup for %s failed: %s — falling back to local", registered_name, e)
            return None

    def _load_whisper(self):
        """Load Whisper model with LoRA adapter — registry first, local fallback.

        Resolution order:
          1. MLflow registry: models:/navi-whisper/Production
          2. Local directory: models/whisper-navi-lora/

        The artifact contains only LoRA delta weights, so in either case
        we still load the base whisper-tiny from HuggingFace and apply
        the adapter on top via PeftModel.
        """
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        base_model = self.params["whisper"]["base_model"]
        registry_name = self.params["mlflow"].get("model_registry_name_whisper", "navi-whisper")

        # Try registry first
        registry_path = self._resolve_from_registry(registry_name)
        if registry_path:
            model_dir = Path(registry_path)
            self._whisper_source = "registry"
        else:
            model_dir = PROJECT_ROOT / "models" / "whisper-navi-lora"
            self._whisper_source = "local" if model_dir.exists() else "base"

        # Always load the processor from the local dir (has Na'vi tokenizer tweaks)
        # or fall back to the HuggingFace base processor
        try:
            if model_dir.exists():
                self.whisper_processor = WhisperProcessor.from_pretrained(str(model_dir))
            else:
                self.whisper_processor = WhisperProcessor.from_pretrained(base_model)
        except Exception as e:
            logger.warning("Processor load failed (%s), using base", e)
            self.whisper_processor = WhisperProcessor.from_pretrained(base_model)

        # Load base model, then try to apply LoRA adapter on top
        try:
            base = WhisperForConditionalGeneration.from_pretrained(base_model)
            if model_dir.exists():
                try:
                    from peft import PeftModel
                    self.whisper_model = PeftModel.from_pretrained(base, str(model_dir))
                    self.whisper_model = self.whisper_model.merge_and_unload()
                    logger.info("Whisper loaded with LoRA from %s", model_dir)
                except Exception as lora_err:
                    logger.warning("LoRA apply failed (%s), using base Whisper", lora_err)
                    self.whisper_model = base
            else:
                self.whisper_model = base
                logger.info("Whisper loaded from base: %s", base_model)
            self.whisper_model.eval()
        except Exception as e:
            logger.error("Failed to load Whisper: %s", e)

    def _load_marian(self):
        """Load MarianMT model — registry first, local snapshot fallback."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            registry_name = self.params["mlflow"].get("model_registry_name_marian", "navi-marian")
            registry_path = self._resolve_from_registry(registry_name)

            if registry_path:
                model_dir = Path(registry_path)
                self._marian_source = "registry"
            else:
                model_dir = PROJECT_ROOT / "models" / "marian-navi-en"
                self._marian_source = "local" if model_dir.exists() else "base"

            if model_dir.exists():
                self.marian_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.marian_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
                self.marian_model.eval()
                logger.info("MarianMT loaded from %s (source=%s)", model_dir, self._marian_source)
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

    @staticmethod
    def _clean_english_output(text: str) -> str:
        """Strip dictionary-entry metadata that leaks into translations.

        The Reykunyu dictionary stores entries like
        "some (agentive case: subject of transitive verb)" — useful for
        students, but unreadable as a translation. This function:
          - removes parenthetical clauses
          - drops stray single-letter [X] tokens (unresolved fallbacks)
          - collapses repeated whitespace and trims trailing punctuation
        """
        import re
        # Remove parenthetical descriptions
        text = re.sub(r"\s*\([^)]*\)", "", text)
        # Remove single-letter or empty bracketed echoes like [N] [G]
        text = re.sub(r"\[[A-Za-z]?\]", "", text)
        # Collapse whitespace and trim
        text = re.sub(r"\s+", " ", text).strip(" ,;.")
        return text or "(no translation)"

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
            found_count = sum(1 for b in breakdown if b["found"])
            # Only override the neural output if the dictionary actually
            # resolved at least one word. Otherwise the MarianMT output
            # (even if low-confidence) is more useful than a string of
            # [bracketed] echoes of the input.
            if found_count > 0 and (confidence < threshold or not self.marian_loaded):
                english = english_fb
                # Dictionary lookup is high-precision — if the words are in
                # the canonical Reykunyu dictionary, the gloss is correct by
                # definition. Score = 0.95 for full coverage, scaled down by
                # the fraction of tokens we resolved.
                hit_rate = found_count / len(breakdown)
                confidence = max(confidence, 0.5 + 0.45 * hit_rate)
            elif not english:
                english = navi_text  # last-resort echo so the field is never empty

        # Strip dictionary metadata before returning
        english = self._clean_english_output(english)
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

    # Na'vi case suffixes (agentive, patientive, dative, genitive, topical)
    # and common verb suffixes — longest first so greediest match wins
    _NAVI_SUFFIXES = [
        "ìyevìng", "iyevìng", "äpeyk", "eyng", "eiyä",
        "ìlä", "tìng", "sìyä", "isyä",
        "ìri", "ìru", "ìti", "äng", "eie", "ang",
        "yä", "ri", "ru", "ti", "it", "ur", "l",
    ]

    def _strip_navi_suffix(self, word: str) -> str:
        """Return longest suffix-stripped form that exists in the dictionary."""
        low = word.lower()
        if low in self.word_lookup:
            return low
        for suffix in self._NAVI_SUFFIXES:
            if low.endswith(suffix) and len(low) - len(suffix) >= 2:
                stem = low[: -len(suffix)]
                if stem in self.word_lookup:
                    return stem
        return low

    @staticmethod
    def _levenshtein(a: str, b: str, cap: int = 3) -> int:
        """Bounded Levenshtein distance — returns cap+1 if exceeded.

        Standard DP implementation, but short-circuits as soon as the
        minimum value in a row exceeds the cap. This keeps the per-token
        cost cheap even when scanning the full 3,230-word dictionary.
        """
        if abs(len(a) - len(b)) > cap:
            return cap + 1
        if a == b:
            return 0
        # Initialize the previous row of the DP table
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            row_min = i
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
                if curr[j] < row_min:
                    row_min = curr[j]
            if row_min > cap:
                return cap + 1
            prev = curr
        return prev[-1]

    def _fuzzy_correct(self, word: str, max_dist: int = 2) -> tuple[str, int]:
        """Find the dictionary entry closest in spelling to *word*.

        Returns (best_match_or_original, edit_distance). Edit distance
        is capped at *max_dist* + 1 so callers can decide whether to
        accept the correction. Length-1 tokens are skipped because every
        single character is distance ≤ 1 from many words.
        """
        low = word.lower()
        if len(low) < 3 or low in self.word_lookup:
            return low, 0

        best_word = low
        best_dist = max_dist + 1
        # Restrict search to candidates within ±max_dist length to keep
        # the scan O(N) but with a tight inner test.
        for candidate in self.word_lookup.keys():
            if abs(len(candidate) - len(low)) > max_dist:
                continue
            d = self._levenshtein(low, candidate, cap=best_dist)
            if d < best_dist:
                best_dist = d
                best_word = candidate
                if best_dist == 1:
                    break  # can't beat 1 except by exact match (already checked)
        return best_word, best_dist

    def _translate_with_dictionary(self, text: str) -> tuple[str, list[dict]]:
        """Fallback: word-by-word dictionary lookup via Reykunyu.

        Pipeline per token:
          1. Try exact match
          2. Try suffix-stripped match (for inflected forms)
          3. Try bounded fuzzy match (Levenshtein ≤ 2) — handles ASR noise

        Each breakdown entry exposes `corrected` and `edit_distance` so
        the UI can show *raw vs corrected* and metrics can track how
        often the fuzzy layer saved a translation.
        """
        tokens = text.strip().split()
        translations = []
        breakdown = []

        for token in tokens:
            key = self._strip_navi_suffix(token)
            corrected_token = token
            edit_distance = 0

            if key not in self.word_lookup:
                # Try fuzzy correction on the suffix-stripped form
                fuzzy_match, dist = self._fuzzy_correct(key, max_dist=2)
                if dist <= 2 and fuzzy_match in self.word_lookup:
                    key = fuzzy_match
                    corrected_token = fuzzy_match
                    edit_distance = dist

            en = self.word_lookup.get(key, f"[{token}]")
            first_meaning = en.split(";")[0].split(",")[0].strip()
            first_meaning = self._clean_english_output(first_meaning)
            translations.append(first_meaning)
            breakdown.append({
                "navi": token,
                "navi_corrected": corrected_token,
                "edit_distance": edit_distance,
                "en": first_meaning,
                "found": key in self.word_lookup,
                "fuzzy": edit_distance > 0,
            })

        return " ".join(translations), breakdown
