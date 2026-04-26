# Na'vi Translator — High-Level Design (HLD)

## 1. Purpose

Build a production-grade MLOps system that translates Na'vi language input (audio or text) into English, with automated retraining triggered by data drift.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MONITORING LAYER                         │
│  EvidentlyAI drift reports │ Prometheus metrics │ Grafana UI    │
│  Alert: OOV drift > 15% ──► triggers retrain DAG               │
└───────────────────────────────┬─────────────────────────────────┘
                                │ scrapes /metrics
┌───────────────────────────────┴─────────────────────────────────┐
│                        SERVING LAYER                            │
│  ┌──────────┐     /api/*      ┌──────────────────────────────┐  │
│  │ Frontend ├────────────────►│ FastAPI Backend               │  │
│  │ (nginx)  │   reverse proxy │ POST /translate/audio         │  │
│  │ :3000    │                 │ POST /translate/text          │  │
│  └──────────┘                 │ POST /vocab/submit            │  │
│                               │ GET  /health, /ready, /metrics│  │
│                               │ Loads Production model from   │  │
│                               │ MLflow registry at startup    │  │
│                               └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                            │
│  Airflow DAG (weekly + drift-triggered):                        │
│  ingest ► preprocess ► baselines ► [Whisper, MarianMT] ► eval  │
│                                                                 │
│  MLflow tracks: params, metrics, artifacts, model registry      │
│  Auto-promote if WER/BLEU improves ≥ 0.01                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  Sources: Reykunyu API │ learnnavi corpus │ Synthetic TTS       │
│  DVC pipeline: ingest ► preprocess_text ► preprocess_audio      │
│                        ► split ► baseline_stats                 │
│  Reproducibility: Git SHA + DVC lock file                       │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Design Decisions

### Why LoRA over full Whisper fine-tuning?
LoRA reduces trainable parameters by ~99% (from 244M to ~0.5M). This makes training feasible on local CPU/GPU within the no-cloud constraint. Full fine-tuning would require cloud GPUs.

### Why MarianMT and not a larger LLM?
MarianMT is a lightweight seq2seq model specifically designed for machine translation. It outperforms general-purpose LLMs on domain-specific fine-tuning with small datasets, which is exactly the Na'vi scenario (~2,600 word pairs).

### Why Airflow and not cron?
Airflow provides DAG visualization, task-level retry, a web UI, and programmatic triggering — all required by the evaluation rubric. A cron job cannot express task dependencies or provide observability.

### Why DVC for data?
Audio files are large binaries. Git degrades with large files. DVC stores files in a content-addressed cache and commits only small pointer files to Git. A dataset snapshot is reproducible via Git SHA + DVC lock file.

### Why the Reykunyu dictionary fallback?
Na'vi has flexible word order and infixes inside verb roots. For short inputs with common words, dictionary lookup is more reliable than a neural model trained on limited data. The fallback is a safety net for low-confidence outputs.

### Why Docker Compose with 5 services?
Isolation ensures the frontend, backend, MLflow, Prometheus, and Grafana can be built and deployed independently. No shared code dependencies between services — configuration via environment variables only.

## 4. Data Flow

### Audio Translation
```
User speaks ─► Browser MediaRecorder ─► POST /api/translate/audio
    ─► Resample to 16kHz mono ─► Whisper LoRA (ASR) ─► Na'vi text
    ─► MarianMT (NMT) ─► English translation
    ─► If confidence < 0.4: Reykunyu word-by-word fallback
    ─► JSON response with translation + confidence + latency
```

### Drift Detection Loop
```
Prometheus scrapes /metrics every 10s
    ─► navi_oov_rate gauge tracks OOV rate over last 100 requests
    ─► Alert fires if OOV > baseline * 1.15
    ─► Webhook triggers Airflow navi_retrain_dag
    ─► New models trained, evaluated, auto-promoted if better
    ─► Backend restarts and loads new Production model
```

## 5. Technology Stack

| Layer | Tools |
|-------|-------|
| ASR | Whisper Small + PEFT LoRA |
| NMT | MarianMT (opus-mt-mul-en) |
| Orchestration | Apache Airflow 2.8+ |
| Experiment Tracking | MLflow 2.10+ |
| Data Versioning | DVC 3.x |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana + EvidentlyAI |
| Containers | Docker Compose |
| CI | GitHub Actions |
| Testing | pytest + pytest-cov |
| Linting | ruff |

## 6. Security Considerations

- No secrets stored in Git (environment variables via docker-compose)
- Na'vi character set validation on community submissions
- Audio file size limits (30s max) prevent resource exhaustion
- Structured error responses with request IDs for traceability
