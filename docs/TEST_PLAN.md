# Test Plan

## Scope

This plan verifies that the Na'vi Translator is ready for an MLOps project demonstration. It covers API schema validation, data-processing helpers, monitoring helpers, dictionary parsing, FastAPI endpoint wiring, and basic operational checks.

## Test Types

| Type | Purpose | Location / Tool |
| --- | --- | --- |
| Unit tests | Validate pure functions, schemas, metrics helpers, drift logic, and Reykunyu parsing | `pytest tests/unit/` |
| Integration smoke tests | Verify FastAPI routing, request/response schemas, health, and readiness without loading large models | `pytest tests/integration/` |
| Manual API smoke | Verify the running backend responds to `/health`, `/ready`, and `/translate/text` | `curl`, FastAPI docs |
| Frontend smoke | Verify the UI loads and can call the configured REST API | Browser at `http://localhost:3000` |
| MLOps tool smoke | Verify MLflow, Prometheus, Grafana, and Airflow are accessible for demo evidence | Browser screenshots |
| Reproducibility check | Verify DVC status and baseline artifacts where possible | `dvc status`, `dvc dag` |

## Environment

- Working directory: `/Users/naveenus/Downloads/navi-translator`
- Python environment: `.venv/`
- Python version: 3.13
- Local hardware: Apple Silicon with MPS constraints
- MLflow tracking URI: `file:///Users/naveenus/Downloads/navi-translator/mlruns`
- Docker services: backend, frontend, MLflow, Prometheus, Grafana

## Acceptance Criteria

- Unit tests pass locally.
- Integration smoke tests pass locally and in CI without requiring large model downloads.
- Backend `/health` returns HTTP 200.
- Backend `/ready` reports model readiness when the runtime models are available.
- Frontend can issue a text translation request through `/api/translate/text`.
- Prometheus can scrape backend metrics.
- Grafana dashboard loads with Prometheus as its datasource.
- MLflow UI shows model training runs with parameters and metrics.
- Documentation explains known low-resource model limitations honestly.

## Test Cases

| ID | Area | Test | Expected result |
| --- | --- | --- | --- |
| UT-001 | Schemas | Validate text translation request length constraints | Invalid payloads fail, valid payloads pass |
| UT-002 | Reykunyu ingestion | Parse raw dictionary entries with string, list, dict, and translation-list English fields | Normalized `navi`, `en`, and `type` fields are returned |
| UT-003 | Text preprocessing | Build Na'vi-English pairs and deduplicate them | Empty meanings skipped and duplicates removed |
| UT-004 | Monitoring | Record Prometheus counters, histograms, and gauges | Metrics are exposed in Prometheus text format |
| UT-005 | Drift | Compute OOV rate from recent requests | Correct OOV rate for zero, partial, and full drift examples |
| IT-001 | API | POST `/translate/text` using a fake lightweight engine | HTTP 200 with translation response schema |
| IT-002 | API | GET `/health` and `/ready` using a fake lightweight engine | HTTP 200 and expected status fields |
| MAN-001 | Runtime | `curl -f http://localhost:8000/health` | HTTP 200 |
| MAN-002 | Frontend | Load `http://localhost:3000` | UI renders without visual errors |
| MAN-003 | Monitoring | Load Prometheus targets and Grafana dashboard | Backend target up and dashboard visible |
| MAN-004 | Orchestration | Trigger Airflow DAG | DAG run completes or failure is documented with logs |

## Known Risks

- Whisper quality is limited by only 44 training audio samples. This is handled as a low-resource modeling constraint rather than a data bug.
- The processed audio artifacts survived, but the raw audio source directories are empty. Re-running `preprocess_audio` without restoring raw audio can overwrite useful metadata.
- Docker may need enough memory to load models and run monitoring services at the same time.
