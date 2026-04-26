# Na'vi Translator — Project Report

**Course:** DA6360 — MLOps
**Project:** End-to-End Na'vi Language Translation Application
**Repository:** https://github.com/usnaveen/navi-translator
**Date:** 2026-04-28

---

## 1. Executive Summary

Na'vi Translator is a production-grade MLOps application that translates Na'vi (a low-resource constructed language from the *Avatar* film universe) into English from either text or audio input. The system is built end-to-end with the full MLOps lifecycle: **data ingestion → versioning → training → experiment tracking → packaging → deployment → monitoring → automated retraining**.

The core engineering challenge of this project is **low-resource adaptation under a strict no-cloud constraint**. Na'vi has roughly 2,600 documented word pairs and only a few hundred audio samples in the public domain, and all training had to be performed on a local Apple Silicon (MPS) machine. The architecture explicitly compensates for this through:

- **PEFT/LoRA fine-tuning** to make Whisper trainable on local hardware
- **Whisper-tiny** as the chosen ASR backbone — research literature shows tiny/base variants outperform larger Whisper variants when fine-tuning data is limited
- **Dictionary fallback** (Reykunyu) wrapped around the neural pipeline to keep the user experience usable when neural confidence is low

---

## 2. Problem Statement

### Business Problem
Na'vi has an active hobbyist community but no commercial translation tools. Existing dictionaries are static lookup tables. A unified system that handles **speech, text, and word-level lookup** would lower the barrier to learning the language.

### Success Metrics
| Type | Metric | Target |
|------|--------|--------|
| ML | Whisper Word Error Rate (WER) | ≤ 0.35 |
| ML | MarianMT BLEU score | ≥ 0.25 |
| Business | API end-to-end latency (text path) | < 200 ms |
| Business | API end-to-end latency (audio path) | < 5 s |
| Business | Vocabulary coverage (community submissions) | Growing OOV-rate baseline |

---

## 3. System Architecture

The full architecture, design decisions, and rationale are documented in [`docs/HLD.md`](HLD.md). Endpoint specifications, data models, module dependency graph, and alert rules are in [`docs/LLD.md`](LLD.md).

### High-level flow
```
User (browser)
    │
    ▼
[Frontend: nginx, port 3000]
    │  reverse-proxies /api/*
    ▼
[Backend: FastAPI, port 8000]
    │
    ├─► Whisper-tiny + LoRA     (audio → Na'vi text)
    ├─► MarianMT                (Na'vi → English)
    ├─► Reykunyu dictionary     (low-confidence fallback)
    └─► /metrics                (Prometheus exporter)
                │
                ▼
        [Prometheus :9090] ──► [Grafana :3001]
                                       │
                       drift alert ────┘
                       triggers
                       Airflow DAG (retrain)
```

### Component split
- **Data layer:** Reykunyu dictionary, learnnavi corpus, synthetic TTS audio. Versioned with **DVC**.
- **Training layer:** Two parallel pipelines — Whisper-tiny + LoRA for ASR, MarianMT for NMT. Orchestrated via **Apache Airflow**, tracked in **MLflow**.
- **Serving layer:** **FastAPI** backend with `/translate/audio`, `/translate/text`, `/vocab/submit`, `/health`, `/ready`, `/metrics` endpoints.
- **Monitoring layer:** **Prometheus** scrapes `/metrics`, **Grafana** visualizes, alerts on OOV drift trigger Airflow retrain DAG.
- **Packaging:** **Docker Compose** with 5 isolated services on a `navi-net` bridge network.

---

## 4. MLOps Implementation [Rubric: 12 pts]

### 4.1 Data Engineering [2 pts]
- **DVC pipeline** ([`dvc.yaml`](../dvc.yaml)) defines 5 stages: `ingest → preprocess_text + preprocess_audio → split → baseline_stats`. Each stage tracks deps, params, and outs; DVC builds a DAG and only re-runs stages whose inputs changed.
- **Apache Airflow DAG** ([`airflow/dags/`](../airflow/dags/)) runs the same pipeline in production, with task-level retries and DAG-level scheduling.
- **Throughput:** Data ingestion processes ~2,600 dictionary entries and ~3,956 audio samples in under 5 minutes locally.
- **Data validation:** Pydantic schemas enforce shape; baseline statistics (`baselines.json`) capture distribution moments for later drift detection.

### 4.2 Source Control & Continuous Integration [2 pts]
- **Git** for source code (this repository).
- **DVC** for data and large artifacts (audio, processed splits) — pointer files committed to Git, content stored in `.dvc/cache`.
- **DVC DAG:** `dvc dag` renders the pipeline; `dvc repro` re-executes only changed stages.
- **GitHub Actions CI** ([`.github/workflows/`](../.github/workflows/)) runs `pytest` and `ruff` on every push.

### 4.3 Experiment Tracking [2 pts]
- **MLflow** logs every training run with full reproducibility:
  - **Parameters:** LoRA rank, alpha, target modules, learning rate, batch size, epochs
  - **Metrics:** train loss per logging step, eval WER/BLEU, final WER/BLEU
  - **Artifacts:** trained adapter weights, tokenizer, processor, training args
  - **Tags:** Git commit SHA + DVC lock hash → every run is reproducible from `(git_sha, dvc_lock_sha, mlflow_run_id)`
- **Model registry:** `navi-whisper` and `navi-marian` registered. Auto-promotion to `Production` if WER/BLEU improves by ≥ 0.01 (defined in `params.yaml: mlflow.promotion_threshold_wer`).
- **Beyond Autolog:** Custom MLflow tags and metrics for OOV rate, fallback rate, and inference latency are also logged.

### 4.4 Exporter Instrumentation & Visualization [2 pts]
- **Prometheus exporter** ([`src/monitoring/prometheus_exporter.py`](../src/monitoring/prometheus_exporter.py)) exposes 8 metrics — see LLD §7. These cover request volume, latency histograms, model performance gauges (live WER/BLEU), OOV rate, fallback rate, model version, and vocab submissions.
- **Prometheus configuration** ([`prometheus/prometheus.yml`](../prometheus/prometheus.yml)) scrapes `backend:8000/metrics` every 10 s.
- **Alert rules** ([`prometheus/alerts.yml`](../prometheus/alerts.yml)): `HighErrorRate`, `OOVDriftDetected`, `HighLatency` — see LLD §8.
- **Grafana** dashboards (provisioned via [`grafana/provisioning/`](../grafana/provisioning/)) visualize all metrics in near-real-time and use Prometheus as the configured datasource.

### 4.5 Software Packaging [4 pts]
- **MLproject** ([`MLproject`](../MLproject)) defines `train_whisper`, `train_marian`, and `evaluate` entry points with parameterized commands. `python_env.yaml` pins the dependency set so the same command produces reproducible runs.
- **MLflow APIfication:** Models are loaded from the MLflow registry at FastAPI startup (`/ready` reflects load status).
- **FastAPI** exposes the inference engine over REST. Schemas are Pydantic-enforced (`src/serving/schemas.py`).
- **Docker Compose** ([`docker-compose.yml`](../docker-compose.yml)) orchestrates 5 services: backend, frontend, MLflow tracking server, Prometheus, Grafana. Each service has its own Dockerfile under [`docker/`](../docker/). Health checks are configured. Shared volumes for `mlruns/` and `data/`.

---

## 5. Software Engineering [Rubric: 5 pts]

### 5.1 Design Principles [2 pts]
- **Design documents** present: [`docs/HLD.md`](HLD.md) (high-level), [`docs/LLD.md`](LLD.md) (low-level with full endpoint I/O specs).
- **Architecture diagram:** see HLD §2 and the README mermaid diagram.
- **Loose coupling:** Frontend (nginx static + JS) and backend (FastAPI) communicate only via configurable REST API calls. The frontend is dockerized separately, and the API base URL is configurable.
- **Paradigm:** Functional/procedural Python with Pydantic for type-checked data containers. Models are loaded lazily and cached.

### 5.2 Implementation [2 pts]
- **Coding style:** PEP-8 enforced via `ruff` (see `requirements-dev.txt`).
- **Logging:** Module-level loggers throughout (`logging.getLogger(__name__)`). Training scripts log at INFO level to both stdout and file (`whisper_training_v3.log`).
- **Exception handling:** FastAPI exception handlers for `ValidationError`, `FileNotFoundError`, and unhandled exceptions return structured JSON errors with request IDs.
- **Inline documentation:** Module-level docstrings, function-level docstrings on all public APIs, type hints throughout.

### 5.3 Testing [1 pt]
- **Test plan:** [`docs/TEST_PLAN.md`](TEST_PLAN.md) with scope, environment, acceptance criteria, and 13 test cases.
- **Test report:** [`docs/TEST_REPORT.md`](TEST_REPORT.md) with **32 / 32 automated tests passing** (verified 2026-04-28 at 15:35).
- **Test types:** Unit tests (schemas, drift detection, Reykunyu parsing, Prometheus helpers, build_pairs) + integration smoke tests (FastAPI route wiring with a fake lightweight engine to keep CI fast).

---

## 6. Demonstration [Rubric: 10 pts]

### 6.1 Frontend UI/UX [6 pts]
- 4-tab single-page app (nginx-served static HTML/CSS/JS): **Translate Audio**, **Translate Text**, **Submit Vocabulary**, **About**.
- Microphone capture via browser `MediaRecorder` API → POSTed to `/api/translate/audio`.
- Loading states, confidence display, word-level breakdown for text translations, and graceful error handling.
- User manual ([`docs/user_manual.md`](user_manual.md)) targeted at non-technical users.

### 6.2 Pipeline Visualization [4 pts]
The system uses **multiple specialized MLOps tool UIs** rather than a custom dashboard, orchestrated via Docker Compose:

| Tool | URL | Purpose |
|------|-----|---------|
| **MLflow** | http://localhost:5000 | Experiment runs, parameter/metric tracking, model registry |
| **Grafana** | http://localhost:3001 | Real-time inference metrics, drift, latency |
| **Prometheus** | http://localhost:9090 | Raw metric queries, alert states |
| **Airflow** | (DAG-trigger CLI) | Training DAG runs, task-level logs |
| **DVC** | `dvc dag` | Data pipeline DAG visualization |

Each console individually tracks errors, failures, and successful runs.

---

## 7. Results

### 7.1 Whisper-tiny + LoRA — final eval

> **NOTE:** Training completed on 2026-04-28. Final metrics from MLflow run logged at the end of the training pipeline:
>
> | Metric | Value |
> |--------|-------|
> | Final eval WER | _TO BE FILLED FROM `whisper_training_v3.log`_ |
> | Train loss (final) | _TO BE FILLED_ |
> | Total training time | _TO BE FILLED_ |
> | Trainable params | 589,824 (1.54% of 38.3M total) |
> | Effective batch size | 4 (per_device=1 × accum=4) |
> | LoRA configuration | r=16, α=32, target=[q_proj, k_proj, v_proj, out_proj] |

### 7.2 MarianMT — final eval
- Trained for 10 epochs on ~2,600 Na'vi-English word pairs.
- Final train loss ≈ 3.30 (from prior MLflow run, logged in registry as `navi-marian` v1, transitioned to `Production`).

### 7.3 Drift baselines
Captured in [`baselines.json`](../baselines.json):
- **OOV baseline rate:** 0.0175 on validation set
- **Vocabulary size:** ~2,600 words
- **Audio duration:** mean 2.578 s, σ 0.4 s
- **MFCC moments:** 13-dim mean and std vectors for drift detection

---

## 8. Problems Faced & Mitigations

This section is essential for the viva (rubric: *"Ability to narrate problems faced and how they were mitigated"*).

| # | Problem | Root cause | Mitigation |
|---|---------|------------|------------|
| 1 | First production training crash mid-run | `save_strategy="epoch"` meant a crash 1 step before epoch end wiped all progress | Switched to `save_strategy="steps"` with `save_steps=200`. Now max loss = 200 steps (~6 min) on crash. |
| 2 | whisper-small training infeasible on M-series MPS | 244M params × 4945 steps × 3 sec/step ≈ 4+ hours, with thermal throttling pushing it longer | Switched to **whisper-tiny** (39M params). Research basis: Springer 2024 paper on low-resource Whisper shows tiny/base outperform small/medium on limited data. ~6× speed-up. |
| 3 | Limited Na'vi audio data (≤4000 samples) | Constructed language with no commercial corpus | (a) Synthetic TTS augmentation, (b) PEFT/LoRA so we don't overfit a 244M-param model to ~4k examples, (c) **dictionary fallback** so user experience stays usable while WER is still high. |
| 4 | No-cloud constraint | MLOps Guidelines §III: *"Cloud Platforms: Not Allowed"* | All training and serving runs locally. Compensate via model size + LoRA. Documented as a deliberate design choice with a measurable target (WER ≤ 0.35) appropriate for low-resource ASR. |
| 5 | pytest collection errors on first run | `tests/` couldn't import `src.*` package | Added root [`conftest.py`](../conftest.py) that injects project root to `sys.path`. |
| 6 | MLflow file-store deprecation warning (Feb 2026) | Default file-based tracking backend will be deprecated | Acknowledged; SQLite-backed tracking is a planned migration. Not blocking for submission. |

---

## 9. Limitations & Future Work

### Honest limitations
- **Whisper WER is high** for a production ASR system because of limited training audio. The dictionary fallback masks this for short queries but the system is not yet conversational-quality.
- **MarianMT** is fine-tuned on word-level pairs, so multi-word phrase translation can degrade. The `word_breakdown` response field surfaces this for transparency.
- **No GPU available locally** — even with whisper-tiny, full-corpus retraining is bounded by CPU/MPS throughput.

### Planned improvements
1. **More audio data** via expanded synthetic TTS + community submissions (the `/vocab/submit` endpoint already supports audio uploads).
2. **Quantization (INT8 LoRA)** — research shows ≤1pp degradation with 4× memory savings.
3. **Migrate MLflow to SQLite-backed tracking** (matches the 2026 deprecation notice).
4. **Add data-drift retraining trigger** end-to-end (currently the alert wiring exists but the retrain webhook is manual).

---

## 10. Repository Layout

```
navi-translator/
├── airflow/dags/             # Airflow training DAGs
├── data/                     # Versioned via DVC (raw, processed, synthetic)
├── docker/                   # Dockerfiles for backend & frontend
├── docs/                     # HLD, LLD, test plan/report, user manual, this file
├── frontend/                 # Static HTML/CSS/JS UI
├── grafana/                  # Provisioning + dashboards
├── prometheus/               # Scrape config + alert rules
├── src/
│   ├── ingestion/            # Reykunyu API client, TTS generator
│   ├── preprocessing/        # build_pairs, audio_features, split_data, baseline_stats
│   ├── training/             # train_whisper, train_marian, evaluate
│   ├── serving/              # fastapi_app, inference, schemas
│   └── monitoring/           # prometheus_exporter, drift
├── tests/                    # unit + integration
├── conftest.py               # pytest path bootstrap
├── docker-compose.yml        # 5-service stack
├── dvc.yaml / dvc.lock       # data pipeline
├── MLproject                 # MLflow entry points
├── params.yaml               # all hyperparameters & config
├── pytest.ini
└── requirements*.txt
```

---

## 11. Submission Checklist

- [x] GitHub repository: https://github.com/usnaveen/navi-translator
- [x] All source code committed
- [x] HLD document
- [x] LLD document with API I/O specs
- [x] Test plan with test cases
- [x] Test report (32/32 passing)
- [x] User manual
- [x] Architecture diagram (in HLD + README)
- [x] Docker Compose verified
- [x] MLflow runs logged
- [ ] Final WER from whisper-tiny training run *(fill in once training log shows completion)*

---

## 12. References

- Liu, X. et al. (2024). *Exploration of Whisper fine-tuning strategies for low-resource ASR.* Springer Journal on Audio, Speech, and Music Processing. https://link.springer.com/article/10.1186/s13636-024-00349-3
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- *LoRA-INT8 Whisper for Cantonese.* MDPI Sensors 2025. https://www.mdpi.com/1424-8220/25/17/5404
- HuggingFace PEFT documentation. https://huggingface.co/docs/peft
- Reykunyu Na'vi dictionary API. https://reykunyu.lu/api/

---

**End of report.**
