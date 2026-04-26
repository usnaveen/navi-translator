# Viva Quick-Reference

## Service URLs (all local)

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:3000 | Translator UI |
| Backend | http://localhost:8000 | FastAPI |
| Backend `/health` | http://localhost:8000/health | Liveness + model_version |
| Backend `/metrics` | http://localhost:8000/metrics | Prometheus exposition |
| MLflow | http://localhost:5001 | Experiment + registry UI |
| Grafana | http://localhost:3001 | Dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Query + alert rules |
| Alertmanager | http://localhost:9093 | Alert routing UI |
| **MailHog inbox** | http://localhost:8025 | **Where alert emails appear** |

## Demo flow (15 minutes)

### Act 1 — End-to-end translation (3 min)
1. Open http://localhost:3000 → Translate Text tab
2. Click `kaltxì` example → instant translation, ~95% confidence
3. Try `oel ngati kameie` → walk through how the dictionary fallback resolves each token
4. Switch to Translate Audio tab → upload `demo_assets/audio/irayo.wav`
5. Show the **3-stage card stack**:
   - Stage 1 raw Whisper output
   - Stage 2 phonetic correction
   - Stage 3 NMT/dictionary

### Act 2 — Architecture (3 min)
1. Show http://localhost:5001 — point at registered models, both at Production stage
2. Show `/health` — model_version reads `whisper-navi-lora@registry | marian-navi-en@registry`
3. *"Backend resolves the model URI from MLflow on startup. If MLflow is down it falls back to local snapshots — that's the resilience contract."*
4. Open Grafana http://localhost:3001 → walk through 12 panels:
   - Throughput / latency / errors
   - Confidence distribution / high-conf rate
   - OOV rate / top-10 OOV / fallback rate
   - Fuzzy correction panels (rate + edit-distance breakdown)

### Act 3 — Drift demo (5 min)
1. Open Grafana + MailHog inbox side by side
2. Run drift simulation:
   ```bash
   python scripts/simulate_drift.py
   ```
3. Watch live during the **DRIFT phase** (~middle 90s):
   - Top-10 OOV panel populates with new gibberish tokens
   - Fallback rate climbs from ~30% to >80%
   - Confidence histogram heatmap shifts left
   - High-Confidence Rate panel turns red
4. Within 2 min, MailHog inbox receives alerts:
   - `[Na'vi Translator] FIRING — HighFallbackRate`
   - `[Na'vi Translator] FIRING — LowConfidenceTranslations`
5. After RECOVERY phase, alerts move back to "RESOLVED" and you get follow-up emails

### Act 4 — Closing the loop (4 min)
1. Switch to Contribute tab → submit a vocab word
2. Show `navi_vocab_submissions_total` metric incrementing
3. Walk through retrain loop diagram:
   ```
   Vocab queue → Airflow DAG → train run → MLflow registry → 
   promotion gate (WER/BLEU thresholds) → backend reload
   ```
4. *"The architecture supports the full feedback loop. The nightly DAG trigger
   is the next sprint; everything around it — registry, metrics, alerts,
   queue — is wired."*

## Talking points

- **Why fuzzy match isn't cheating**: every production speech pipeline has a language-model rescoring step after the acoustic model. We added a third pipeline stage with **bounded Levenshtein** (max distance 2), tracked separately in metrics, with raw vs corrected both visible in the UI. Transparent.
- **Why low ASR confidence**: whisper-tiny (39M params) is the smallest variant; demo audio is TTS-generated English phonetic approximations, not authentic Na'vi pronunciation. A Whisper-small or Whisper-base run would help, but the data is the bottleneck. The **system honestly reports its uncertainty** rather than hiding it.
- **Why text confidence is ~95%**: when all input tokens are in the canonical Reykunyu dictionary, the gloss is correct by definition. Confidence is `0.5 + 0.45 × hit_rate`.
- **Drift architecture**: every translation logs (`navi_oov_words_total{word=...}`, `navi_fallback_rate`, `navi_translation_confidence`). The dashboard surfaces all three. Alerts fire when thresholds are breached. Email goes to MailHog (swap to real SMTP for production).

## Files reviewers might ask about

| What they ask | Where to look |
|---|---|
| "How does the registry lookup work?" | `src/serving/inference.py::_resolve_from_registry` |
| "Show me the alert rules" | `prometheus/alerts.yml` |
| "How is fuzzy match implemented?" | `src/serving/inference.py::_fuzzy_correct` |
| "Where do confidence scores come from?" | `src/serving/inference.py::_translate_with_marian` and `translate_text` |
| "How would you scale this?" | Grafana drives autoscaling signals (latency p95, in-flight gauge) |
| "How is the dashboard provisioned?" | `grafana/provisioning/` (datasource auto-bound by uid) |

## If something breaks during demo

| Symptom | Quick fix |
|---|---|
| Backend `/health` returns 503 | `docker restart navi-backend` |
| MLflow page won't load | `docker restart navi-mlflow` |
| Grafana panel shows "No data" | Wait 30s; Prometheus needs to scrape after backend restart |
| Alerts not arriving in MailHog | `docker restart navi-alertmanager`, give it 30s |
| Drift simulation fails | `python scripts/simulate_drift.py --base http://localhost:8000 --duration 60` |
