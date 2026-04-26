# Na'vi Translator — Low-Level Design (LLD)

## 1. API Endpoint Specifications

### POST /translate/audio

**Input:** `multipart/form-data` with `file` field (WAV/MP3/OGG, max 30s)

**Output:**
```json
{
    "navi_text": "kaltxì",
    "english": "hello",
    "confidence": 0.85,
    "latency_ms": 1200
}
```

**Error responses:**
- `415` — Unsupported audio format
- `413` — Audio exceeds 30 seconds
- `503` — Models not loaded

**Processing steps:**
1. Read uploaded file bytes
2. Validate content type (WAV/MP3/OGG)
3. Check file size against 30s threshold
4. Resample to 16kHz mono using librosa
5. Extract Whisper input features via `WhisperProcessor.feature_extractor`
6. Generate transcription with `model.generate()` under `torch.no_grad()`
7. Estimate ASR confidence from output log-probabilities
8. Pass Na'vi text to MarianMT for translation
9. If NMT confidence < 0.4, fall back to dictionary lookup
10. Record metrics (latency, OOV rate, fallback usage)
11. Return JSON response

---

### POST /translate/text

**Input:**
```json
{ "text": "oel ngati kameie" }
```

**Output:**
```json
{
    "english": "I see you",
    "confidence": 0.72,
    "word_breakdown": [
        { "navi": "oel", "en": "I (ergative)", "found": true },
        { "navi": "ngati", "en": "you (accusative)", "found": true },
        { "navi": "kameie", "en": "see (spiritual)", "found": true }
    ],
    "latency_ms": 85
}
```

**Validation:** Text must be 1–500 characters (Pydantic enforced).

---

### POST /vocab/submit

**Input:**
```json
{
    "navi_word": "fya'o",
    "english_meaning": "path, way",
    "audio_b64": null
}
```

**Output:**
```json
{ "accepted": true, "message": "Thank you! 'fya'o' has been submitted for review." }
```

**Validation:** Na'vi word checked against allowed character set. Invalid characters return `accepted: false`.

---

### GET /health

```json
{ "status": "ok", "model_version": "v3", "uptime_s": 3421.5 }
```

### GET /ready

```json
{ "ready": true, "whisper_loaded": true, "marian_loaded": true }
```

### GET /metrics

Returns Prometheus text exposition format with all 8 registered metrics.

---

## 2. Data Models

### words.json entry
```json
{ "navi": "kaltxì", "en": "hello", "type": "intj" }
```

### pairs.tsv row
```
navi\ten
kaltxì\thello
oel ngati kameie\tI see you
```

### baselines.json
```json
{
    "mfcc": { "mean": [float x 13], "std": [float x 13], "count": 1500 },
    "oov": { "oov_rate": 0.032, "total_tokens": 5000, "oov_tokens": 160, "vocab_size": 2600 },
    "audio_duration": { "mean": 1.2, "std": 0.4, "p25": 0.9, "p75": 1.5 }
}
```

---

## 3. Module Dependency Graph

```
src/ingestion/
    data_ingest.py ──► reykunyu_client.py  (downloads words.json)
                   ──► tts_generate.py     (generates synthetic audio)

src/preprocessing/
    build_pairs.py       (words.json ──► pairs.tsv)
    audio_features.py    (raw audio ──► processed 16kHz WAV)
    split_data.py        (processed ──► train/val/test splits)
    baseline_stats.py    (train data ──► baselines.json)

src/training/
    train_whisper.py     (train audio + LoRA ──► MLflow run)
    train_marian.py      (train pairs ──► MLflow run)
    evaluate.py          (test data + model ──► promote to registry)

src/serving/
    fastapi_app.py       (HTTP endpoints, imports inference + schemas)
    inference.py         (TranslationEngine: loads models, runs chain)
    schemas.py           (Pydantic request/response models)

src/monitoring/
    prometheus_exporter.py  (metric definitions + helpers)
    drift.py                (EvidentlyAI report + OOV comparison)
```

---

## 4. DVC Pipeline DAG

```
ingest ──► preprocess_text ──┐
                              ├──► split ──► baseline_stats
       ──► preprocess_audio ─┘
```

Each stage tracks:
- **deps:** source code files
- **params:** values from params.yaml
- **outs:** data artifacts (cached by DVC)
- **metrics:** baselines.json (not cached, tracked in Git)

---

## 5. Airflow Task Graph

```
ingest_data ──► preprocess ──► compute_baselines ──┐
                                                    ├──► evaluate_and_promote ──► notify
                              train_whisper ────────┘
                              train_marian ─────────┘
```

`train_whisper` and `train_marian` run in parallel after baselines are computed.

---

## 6. Docker Compose Network

```
navi-net (bridge)
    ├── backend    :8000  (FastAPI)
    ├── frontend   :3000  (nginx, proxies /api/* → backend:8000)
    ├── mlflow     :5000  (tracking server)
    ├── prometheus :9090  (scrapes backend:8000/metrics)
    └── grafana    :3001  (queries prometheus:9090)
```

---

## 7. Prometheus Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `navi_translation_requests_total` | Counter | input_type, status |
| `navi_translation_latency_ms` | Histogram | — |
| `navi_whisper_wer_live` | Gauge | — |
| `navi_marian_bleu_live` | Gauge | — |
| `navi_oov_rate` | Gauge | — |
| `navi_fallback_rate` | Gauge | — |
| `navi_model_version` | Info | version |
| `navi_vocab_submissions_total` | Counter | — |

---

## 8. Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighErrorRate | error rate > 5% over 5min | critical |
| OOVDriftDetected | navi_oov_rate > 0.15 | warning → triggers retrain |
| HighLatency | p95 latency > 5000ms over 5min | warning |
