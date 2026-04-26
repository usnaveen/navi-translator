# Test Report

## Summary

Local verification was run from `/Users/naveenus/Downloads/navi-translator` using the project `.venv`.

| Test suite | Command | Result |
| --- | --- | --- |
| Unit tests | `./.venv/bin/python -m pytest tests/unit -q` | Passed: 30, Failed: 0 |
| Integration smoke tests | `./.venv/bin/python -m pytest tests/integration -q` | Passed: 2, Failed: 0 |
| Combined automated tests | `./.venv/bin/python -m pytest tests/unit tests/integration -q` | Passed: 32, Failed: 0 |
| DVC status | `./.venv/bin/python -m dvc status` | Data and pipelines are up to date |
| Runtime health | `curl -sS http://127.0.0.1:8000/health` | HTTP 200, `{"status":"ok"}` |
| Runtime readiness | `curl -sS http://127.0.0.1:8000/ready` | HTTP 200, Whisper and Marian loaded |
| Text translation smoke | `POST /translate/text` with `{"text":"kaltxì"}` | HTTP 200, dictionary fallback response |
| MLflow UI | `curl` against `http://127.0.0.1:5000/` | HTTP 200 |
| Frontend static UI | `curl` against `http://127.0.0.1:3000/` | HTTP 200 |

## Unit Test Coverage

Validated areas:

- Reykunyu dictionary parsing and lookup.
- Text-pair construction and deduplication.
- Pydantic request/response schemas.
- Prometheus metric export helpers.
- OOV drift-rate calculation.

## Integration Smoke Coverage

Validated areas:

- `POST /translate/text` endpoint routing and response schema.
- `GET /health` endpoint routing and response schema.
- `GET /ready` endpoint routing and response schema.

The integration tests use a fake lightweight translation engine so CI can verify API wiring without downloading or loading Whisper/Marian model weights.

## Current Model Evidence

- MarianMT training completed for 10 epochs with final train loss around `3.30`.
- Whisper-Small + LoRA training completed for 10 epochs on the low-resource audio set.
- Whisper validation WER is currently `2.6`, which is a known limitation caused by the tiny audio dataset and is addressed in the project narrative via low-resource LoRA training plus dictionary fallback.
- MarianMT is registered in MLflow as `navi-marian` version `1` and transitioned to `Production`.
- Drift baselines are recorded in `baselines.json`: 44 audio baseline samples, validation OOV rate `0.0175`, and mean audio duration `2.578s`.

## Screenshot Evidence

Screenshots captured in `docs/screenshots/`:

- `frontend-home.png`
- `api-health.png`
- `api-docs.png`
- `mlflow-home.png`
- `mlflow-registry-navi-marian.png`

## Known Issues

- Docker Desktop is installed but its daemon did not become reachable during verification: `docker ps` returned "Cannot connect to the Docker daemon". Because of that, Docker Compose, Grafana, Prometheus, and Airflow browser screenshots could not be captured in this pass.
- Raw audio source directories are empty while processed audio artifacts are present. DVC is clean after committing the current preserved artifact state, but re-running `preprocess_audio` without restoring raw audio should be avoided.
- The frontend is functionally complete, but final visual screenshots should be reviewed before submission.

## Sign-off

The automated Python test suites are currently green. DVC status is clean. Local backend, frontend, and MLflow UI smoke checks pass. Docker daemon recovery is the remaining environment blocker for Compose, Grafana, Prometheus, and Airflow screenshots.
