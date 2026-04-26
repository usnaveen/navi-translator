# Na'vi Translator — Submission Checklist

> **Course:** DA6360 MLOps · **Deadline:** ~48 hrs
> **Repo:** `~/Downloads/navi-translator/`
> **Rule:** All git commits under your name only — no `Co-Authored-By` lines.

---

## TL;DR — Where we are

| Area | Status |
|---|---|
| MarianMT training (Na'vi → English) | ✅ **DONE** (10 epochs, loss 3.30, in MLflow) |
| Whisper training (audio → Na'vi) | ⏳ **TODO** (recommend quick demo run) |
| MLflow tracking | ✅ **DONE** (4+ runs at `mlruns/544729208746258079/`) |
| DVC pipeline | ✅ **DONE** (5 stages reproducible) |
| Docker compose stack | ⏳ **TODO** (just `docker compose up`) |
| Airflow DAG | ⏳ **TODO** (trigger once for screenshot) |
| Prometheus + Grafana | ⏳ **TODO** (generate traffic, screenshot panels) |
| Tests | ⏳ **PARTIAL** (unit tests exist; need integration smoke test) |
| GitHub push | ⏳ **TODO** (final step) |

---

## What is the "other training"?

You have **two models** in this project:

1. **MarianMT** (✅ done) — text translation. Takes Na'vi text → English text.
2. **Whisper** (⏳ pending) — speech recognition (ASR). Takes Na'vi **audio** → Na'vi **text**.

Together they form the audio pipeline: 🎤 audio → Whisper → Na'vi text → MarianMT → English.

**Recommendation given time:** run Whisper for **1 epoch as a demo** so you have a tracked MLflow run + saved checkpoint. Full fine-tuning isn't required — the rubric only needs evidence that the pipeline works.

```bash
# Quick Whisper demo (edit params.yaml first: num_epochs: 1)
cd ~/Downloads/navi-translator
source .venv/bin/activate
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.training.train_whisper
```

---

## Rubric Checklist (35 pts)

### 1. Data + Reproducibility (5 pts) — ✅ DONE
- [x] Dataset ingested (Reykunyu words + audio)
- [x] DVC pipeline with 5 stages: `ingest`, `preprocess_text`, `preprocess_audio`, `split`, `baseline_stats`
- [x] `params.yaml` is single source of truth
- [x] Train/val/test split stratified by frequency tier
- [ ] **Action:** run `dvc repro` once end-to-end and commit `dvc.lock`
  ```bash
  dvc repro
  git add dvc.lock && git commit -m "Reproduce DVC pipeline end-to-end"
  ```

### 2. Model Training + Experiment Tracking (8 pts) — ✅ MOSTLY DONE
- [x] MarianMT fine-tuned (10 epochs, BLEU eval pending)
- [x] MLflow tracking working (file-based at `mlruns/`)
- [x] Hyperparameters logged
- [x] Model artifacts logged
- [ ] Whisper training run (even 1 epoch demo counts)
- [ ] Run `evaluate.py` to log real BLEU score
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.training.evaluate
  ```

### 3. Model Registry + Promotion (3 pts) — ⏳ TODO
- [ ] Register MarianMT model in MLflow registry
- [ ] Tag a version as "Production"
  ```bash
  python -m src.training.register_model --model marian
  ```

### 4. Serving (5 pts) — ✅ CODE READY
- [x] FastAPI backend (`src/serving/app.py`)
- [x] Endpoints: `/translate/text`, `/translate/audio`, `/health`, `/ready`, `/vocab/submit`
- [x] Frontend (HTML + JS) in `frontend/`
- [ ] **Action:** start the stack and verify
  ```bash
  docker compose up -d
  curl http://localhost:8000/health
  open http://localhost:3000
  ```

### 5. Containerization (4 pts) — ✅ CODE READY
- [x] `Dockerfile` for backend
- [x] `docker-compose.yml` (backend + frontend + prometheus + grafana)
- [ ] **Action:** verify all services come up healthy

### 6. Orchestration / Airflow (2 pts) — ⏳ TODO
- [x] DAG file exists (`airflow/dags/navi_pipeline_dag.py`)
- [ ] **Action:** start Airflow, trigger DAG, screenshot success
  ```bash
  docker compose -f docker-compose.airflow.yml up -d
  # open http://localhost:8080 (admin/admin), trigger DAG
  ```

### 7. Monitoring (4 pts) — ⏳ TODO
- [x] Prometheus config + metrics exposed in FastAPI
- [x] Grafana dashboard JSON
- [ ] **Action:** generate traffic, screenshot dashboard
  ```bash
  # After docker compose up:
  for i in {1..50}; do
    curl -X POST http://localhost:8000/translate/text \
      -H "Content-Type: application/json" \
      -d '{"text":"kaltxì"}'
  done
  open http://localhost:3001  # Grafana (admin/admin)
  ```

### 8. Testing (2 pts) — ⏳ PARTIAL
- [x] Unit tests in `tests/`
- [ ] Integration smoke test (hit live `/translate/text` and assert 200)
  ```bash
  pytest tests/ -v
  ```

### 9. Documentation (2 pts) — ✅ DONE
- [x] `README.md`
- [x] `PROJECT_GUIDE.pdf` (the comprehensive guide we generated)
- [x] This checklist

---

## Suggested Order for Next ~6 hours

1. **Run Whisper demo** (15 min wall, `num_epochs: 1`) — gets you the second tracked run.
2. **`dvc repro`** + commit `dvc.lock` (5 min).
3. **`docker compose up -d`** — verify health + frontend works (10 min).
4. **Generate traffic** + screenshot Grafana (10 min).
5. **Airflow trigger** + screenshot (15 min).
6. **Register MarianMT model** in MLflow (5 min).
7. **Run integration test** (5 min).
8. **Commit everything + push to GitHub** (under your name only).

---

## Git Commit Reminder

```bash
# Make sure your name is on the commit, no Claude attribution:
git config user.name "Naveen US"
git config user.email "usnaveen25@gmail.com"

git add <files>
git commit -m "Your message here"
# (NO --author override, NO Co-Authored-By trailer)
```

---

## Confirmation: MLflow IS recording

Verified — `~/Downloads/navi-translator/mlruns/544729208746258079/` contains 4+ run directories with metrics, params, and tags. View with:

```bash
mlflow ui --backend-store-uri file:///Users/naveenus/Downloads/navi-translator/mlruns
# → http://localhost:5000
```
