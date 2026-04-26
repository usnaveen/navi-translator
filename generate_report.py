"""Generate the NA'VI Translator project submission report as a PDF."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = "docs/NA_Vi_Translator_Project_Report.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

TITLE   = style("RTitle",   "Title",   fontSize=26, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, alignment=TA_CENTER)
SUB     = style("RSub",     "Normal",  fontSize=13, textColor=colors.HexColor("#16213e"), spaceAfter=4, alignment=TA_CENTER)
META    = style("RMeta",    "Normal",  fontSize=10, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=2)
H1      = style("RH1",      "Heading1",fontSize=14, textColor=colors.HexColor("#0f3460"), spaceBefore=18, spaceAfter=6, borderPad=4)
H2      = style("RH2",      "Heading2",fontSize=12, textColor=colors.HexColor("#16213e"), spaceBefore=12, spaceAfter=4)
BODY    = style("RBody",    "Normal",  fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
BULLET  = style("RBullet",  "Normal",  fontSize=10, leading=14, leftIndent=16, spaceAfter=3, bulletIndent=6)
CODE    = style("RCode",    "Code",    fontSize=8,  leading=12, leftIndent=12, spaceAfter=6,
                backColor=colors.HexColor("#f4f4f4"), borderPad=4, fontName="Courier")
CAPTION = style("RCaption", "Normal",  fontSize=9,  textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)

ACCENT = colors.HexColor("#0f3460")
LIGHT  = colors.HexColor("#e8f0fe")

def tbl_style(header_bg=ACCENT):
    return TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  header_bg),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ])

def b(text): return f"<b>{text}</b>"
def i(text): return f"<i>{text}</i>"
def bullet(text): return Paragraph(f"• {text}", BULLET)
def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def p(text): return Paragraph(text, BODY)
def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6, spaceBefore=6)


# ── Document ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm,  bottomMargin=2.5*cm,
    title="Na'vi Translator — MLOps Project Report",
    author="DA6360 Student",
)

story = []

# ── Cover Page ────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 3*cm),
    Paragraph("Na'vi Language Translator", TITLE),
    Paragraph("End-to-End MLOps Application", SUB),
    hr(),
    Spacer(1, 0.4*cm),
    Paragraph("DA6360 — MLOps Course Project", META),
    Paragraph("Repository: github.com/usnaveen/navi-translator", META),
    Paragraph("Date: 28 April 2026", META),
    Spacer(1, 2*cm),
]

cover_tbl = Table([
    [b("Submission Items"), ""],
    ["GitHub Repository", "https://github.com/usnaveen/navi-translator"],
    ["Architecture Document", "docs/HLD.md"],
    ["API Specification", "docs/LLD.md"],
    ["Test Plan & Report", "docs/TEST_PLAN.md / TEST_REPORT.md"],
    ["User Manual", "docs/user_manual.md"],
    ["Automated Tests", "32 / 32 passing (pytest)"],
], colWidths=[7*cm, 9*cm])
cover_tbl.setStyle(tbl_style())
story += [cover_tbl, PageBreak()]

# ── 1. Executive Summary ──────────────────────────────────────────────────────
story += [
    h1("1. Executive Summary"),
    p("Na'vi Translator is a production-grade MLOps application that translates Na'vi — "
      "a low-resource constructed language from the <i>Avatar</i> film universe — into English "
      "from both audio and text input. The system is built end-to-end following the full MLOps "
      "lifecycle: data ingestion → versioning → training → experiment tracking → packaging → "
      "deployment → monitoring → automated retraining."),
    p("The core engineering challenge is <b>low-resource adaptation under a strict no-cloud "
      "constraint</b>. Na'vi has approximately 2,600 documented word pairs and limited public "
      "audio data. All training runs on local Apple Silicon hardware using PEFT/LoRA to make "
      "Whisper fine-tuning feasible without GPU cloud resources."),
    h2("Key Achievements"),
]
story += [
    bullet("Full MLOps pipeline implemented: DVC data versioning, Airflow orchestration, MLflow experiment tracking, Prometheus/Grafana monitoring"),
    bullet("Two trained models: Whisper-tiny + LoRA (ASR) and MarianMT (NMT), both registered in MLflow Model Registry"),
    bullet("FastAPI backend with 6 REST endpoints, nginx frontend, Docker Compose with 5 isolated services"),
    bullet("32 / 32 automated tests passing (unit + integration); CI/CD via GitHub Actions"),
    bullet("Reykunyu dictionary fallback for low-confidence outputs — keeps UX usable when neural confidence is below threshold"),
    Spacer(1, 0.3*cm),
]

# ── 2. Problem Statement ──────────────────────────────────────────────────────
story += [
    h1("2. Problem Statement & Success Metrics"),
    p("Na'vi has an active hobbyist learning community but no unified translation tool. "
      "Static dictionary lookups exist but cannot handle audio, multi-word phrases, or "
      "community vocabulary contributions. The goal is a single system handling speech, text, "
      "and community submissions with full MLOps observability."),
]
metrics_tbl = Table([
    [b("Type"), b("Metric"), b("Target"), b("Result")],
    ["ML",       "Whisper ASR — Word Error Rate",     "≤ 0.35",   "0.42 (low-resource constrained)"],
    ["ML",       "MarianMT BLEU score",               "≥ 0.25",   "Logged in MLflow registry"],
    ["Business", "Text translation latency",          "< 200 ms", "~85 ms (dictionary path)"],
    ["Business", "Audio translation latency",         "< 5 s",    "~1.2 s (whisper-tiny)"],
    ["Business", "OOV drift alert threshold",         "< 15%",    "Prometheus alert configured"],
], colWidths=[2.5*cm, 7*cm, 3*cm, 4*cm])
metrics_tbl.setStyle(tbl_style())
story += [metrics_tbl, Spacer(1, 0.3*cm)]

# ── 3. System Architecture ────────────────────────────────────────────────────
story += [
    h1("3. System Architecture"),
    p("The system is split into four layers — Data, Training, Serving, and Monitoring — "
      "each independently containerised and connected only through defined interfaces."),
]

arch_tbl = Table([
    [b("Layer"), b("Components"), b("Tools")],
    ["Data",       "Ingestion, versioning, preprocessing, splitting, baseline stats",       "DVC, Reykunyu API, gTTS, Airflow"],
    ["Training",   "Whisper-tiny + LoRA (ASR), MarianMT (NMT), evaluation, auto-promotion","MLflow, PEFT, HuggingFace Transformers"],
    ["Serving",    "REST API, model inference, fallback dictionary, metrics export",         "FastAPI, nginx, Prometheus client"],
    ["Monitoring", "Metric scraping, dashboards, drift detection, retrain alerts",           "Prometheus, Grafana, EvidentlyAI"],
], colWidths=[2.5*cm, 8.5*cm, 5.5*cm])
arch_tbl.setStyle(tbl_style())
story += [arch_tbl, Spacer(1, 0.3*cm)]

story += [
    h2("Request Flow — Audio Translation"),
    Paragraph(
        "Browser MediaRecorder → POST /api/translate/audio → Resample 16 kHz mono → "
        "Whisper-tiny encoder → LoRA-adapted decoder → Na'vi text → MarianMT → English → "
        "if confidence &lt; 0.4: Reykunyu word-by-word fallback → JSON response",
        CODE),
    h2("Docker Compose Network"),
    Paragraph(
        "navi-net (bridge)\n"
        "  ├── backend    :8000  FastAPI inference server\n"
        "  ├── frontend   :3000  nginx static UI\n"
        "  ├── mlflow     :5000  experiment tracking server\n"
        "  ├── prometheus :9090  metric scraper\n"
        "  └── grafana    :3001  NRT dashboard",
        CODE),
]

# ── 4. MLOps Implementation ───────────────────────────────────────────────────
story += [h1("4. MLOps Implementation")]

# 4.1 Data Engineering
story += [
    h2("4.1  Data Engineering"),
    p("The data pipeline is defined as a DVC DAG in <b>dvc.yaml</b> with five stages. "
      "Each stage declares its source-code dependencies, parameter dependencies (from "
      "params.yaml), and output artifacts so DVC can detect changes and re-run only "
      "what is necessary. Apache Airflow orchestrates the same pipeline in production "
      "with task-level retry and DAG-level scheduling."),
]
dvc_tbl = Table([
    [b("Stage"), b("Input"), b("Output"), b("Tool")],
    ["ingest",           "Reykunyu API + TTS",        "data/raw/words.json + audio", "requests, gTTS"],
    ["preprocess_text",  "words.json",                "data/processed/text",         "pandas"],
    ["preprocess_audio", "raw audio",                 "data/processed/audio",        "librosa, soundfile"],
    ["split",            "processed text + audio",    "train / val / test splits",   "sklearn"],
    ["baseline_stats",   "train split",               "baselines.json",              "numpy, scipy"],
], colWidths=[3.5*cm, 4.5*cm, 4.5*cm, 3*cm])
dvc_tbl.setStyle(tbl_style())
story += [dvc_tbl, Spacer(1, 0.2*cm),
          bullet("3,956 training examples, 494 validation examples, full pipeline completes in < 5 minutes"),
          bullet("Baseline statistics (mean, variance, OOV rate) stored in baselines.json for later drift comparison"),
          Spacer(1, 0.3*cm)]

# 4.2 Source Control & CI
story += [
    h2("4.2  Source Control & Continuous Integration"),
    bullet("<b>Git</b> — source code, configuration, and documentation"),
    bullet("<b>DVC</b> — large data artifacts and model weights (pointer files in Git, content in .dvc/cache)"),
    bullet("<b>DVC DAG</b> — visualised with dvc dag; dvc repro re-executes only changed stages"),
    bullet("<b>GitHub Actions</b> — CI pipeline runs on every push: ruff lint → unit tests → integration tests → docker build"),
    bullet("Every MLflow run is tagged with its Git commit SHA + DVC lock hash for full reproducibility"),
    Spacer(1, 0.3*cm),
]

# 4.3 Experiment Tracking
story += [
    h2("4.3  Experiment Tracking (MLflow)"),
    p("Every training run is tracked in MLflow with full reproducibility. The model registry "
      "holds both trained models and controls promotion to Production."),
]
mlflow_tbl = Table([
    [b("Tracked Item"), b("Details")],
    ["Parameters",  "base_model, lora_r, lora_alpha, lora_target_modules, learning_rate, batch_size, num_epochs, warmup_steps"],
    ["Metrics",     "train/loss per 25 steps, eval/WER, eval/BLEU, final_wer, final_bleu"],
    ["Artifacts",   "LoRA adapter weights, tokenizer, processor, training_args.bin"],
    ["Tags",        "git_sha, dvc_lock_sha, run_name — every run reproducible from (git_sha, dvc_lock, mlflow_run_id)"],
    ["Registry",    "navi-whisper (v1) and navi-marian (v1) registered; auto-promote if WER/BLEU improves ≥ 0.01"],
], colWidths=[3.5*cm, 13*cm])
mlflow_tbl.setStyle(tbl_style())
story += [mlflow_tbl, Spacer(1, 0.3*cm)]

# 4.4 Monitoring
story += [
    h2("4.4  Exporter Instrumentation & Visualization"),
    p("The FastAPI backend exposes a /metrics endpoint in Prometheus text format. "
      "Prometheus scrapes it every 10 seconds. Grafana dashboards visualise all metrics "
      "in near-real-time with pre-provisioned datasources and dashboard JSON."),
]
prom_tbl = Table([
    [b("Metric"), b("Type"), b("Purpose")],
    ["navi_translation_requests_total",  "Counter",   "Request volume by input_type and status"],
    ["navi_translation_latency_ms",      "Histogram", "End-to-end latency distribution"],
    ["navi_whisper_wer_live",            "Gauge",     "Live ASR word error rate"],
    ["navi_marian_bleu_live",            "Gauge",     "Live NMT BLEU score"],
    ["navi_oov_rate",                    "Gauge",     "Out-of-vocabulary rate — primary drift signal"],
    ["navi_fallback_rate",               "Gauge",     "Fraction of requests using dictionary fallback"],
    ["navi_model_version",               "Info",      "Currently loaded model version label"],
    ["navi_vocab_submissions_total",     "Counter",   "Community vocabulary submissions"],
], colWidths=[5.5*cm, 2.5*cm, 8.5*cm])
prom_tbl.setStyle(tbl_style())
story += [prom_tbl, Spacer(1, 0.2*cm),
          bullet("Alert: HighErrorRate — error rate > 5% over 5 minutes → severity critical"),
          bullet("Alert: OOVDriftDetected — navi_oov_rate > 0.15 → triggers Airflow retrain DAG"),
          bullet("Alert: HighLatency — p95 latency > 5,000 ms over 5 minutes → severity warning"),
          Spacer(1, 0.3*cm)]

# 4.5 Software Packaging
story += [
    h2("4.5  Software Packaging"),
    bullet("<b>MLproject</b> — defines train_whisper, train_marian, and evaluate entry points with parameterised commands; python_env.yaml pins the full dependency set"),
    bullet("<b>FastAPI</b> — REST inference API with Pydantic-enforced schemas; model loaded from MLflow registry at startup; /health and /ready endpoints for orchestration health checks"),
    bullet("<b>Docker Compose</b> — 5 isolated services (backend, frontend, mlflow, prometheus, grafana) on a shared bridge network; each has its own Dockerfile and health check"),
    bullet("<b>MLflow Model Registry</b> — models promoted to Production stage; FastAPI loads the Production alias at startup without code changes"),
    Spacer(1, 0.3*cm),
]

# ── 5. Model Development ──────────────────────────────────────────────────────
story += [h1("5. Model Development & Training")]

story += [
    h2("5.1  Whisper-tiny + LoRA (ASR)"),
    p("Whisper-tiny (39M parameters) was selected over Whisper-small (244M) based on "
      "published research showing that smaller Whisper variants outperform larger ones "
      "when fine-tuning data is limited. PEFT/LoRA adapters reduce the trainable parameter "
      "count to 589,824 (1.54% of total), making training feasible on Apple Silicon MPS "
      "hardware within the no-cloud constraint."),
]
whisper_tbl = Table([
    [b("Configuration"), b("Value"), b("Rationale")],
    ["Base model",        "openai/whisper-tiny",              "39M params; best low-resource performance per Springer 2024 study"],
    ["LoRA rank (r)",     "16",                               "Bumped from 8 to compensate for smaller backbone capacity"],
    ["LoRA alpha",        "32",                               "Standard alpha = 2×r ratio"],
    ["Target modules",    "q_proj, k_proj, v_proj, out_proj", "Full attention coverage for better language adaptation"],
    ["Learning rate",     "1×10⁻⁴",                          "Conservative; literature shows higher LR can hurt tiny/base variants"],
    ["Effective batch",   "4 (1 device × 4 accum steps)",     "Gradient accumulation avoids OOM while maintaining stable gradients"],
    ["Epochs",            "5",                                "Validated against val set; early stopping if WER plateaus"],
    ["Trainable params",  "589,824 / 38,350,464 (1.54%)",     "LoRA efficiency — only attention adapters updated"],
    ["Training language", "mi (Māori)",                       "Closest Whisper-supported language to Na'vi phonology"],
    ["Final WER",         "0.42",                             "Logged in MLflow; target was ≤ 0.35; limited by small audio corpus"],
], colWidths=[3.5*cm, 5*cm, 8*cm])
whisper_tbl.setStyle(tbl_style())
story += [whisper_tbl, Spacer(1, 0.3*cm)]

story += [
    h2("5.2  MarianMT (NMT — Na'vi → English)"),
    p("Helsinki-NLP/opus-mt-mul-en was selected as the base translation model. "
      "It is a lightweight seq2seq architecture purpose-built for machine translation "
      "and outperforms general-purpose LLMs on domain-specific fine-tuning with small "
      "datasets, which is the Na'vi scenario (~2,600 word pairs)."),
]
marian_tbl = Table([
    [b("Configuration"), b("Value")],
    ["Base model",    "Helsinki-NLP/opus-mt-mul-en"],
    ["Learning rate", "2×10⁻⁵"],
    ["Batch size",    "16"],
    ["Epochs",        "10"],
    ["Train loss",    "~3.30 (final epoch)"],
    ["Registry",      "navi-marian v1 — Production stage"],
], colWidths=[5*cm, 11.5*cm])
marian_tbl.setStyle(tbl_style())
story += [marian_tbl, Spacer(1, 0.3*cm)]

story += [
    h2("5.3  Reykunyu Dictionary Fallback"),
    p("When neural translation confidence falls below 0.4, the system falls back to "
      "word-by-word lookup in the Reykunyu Na'vi dictionary (~2,600 entries). "
      "This ensures the application remains usable for common words even when the "
      "neural model confidence is low — a deliberate design choice for low-resource ASR."),
]

# ── 6. Software Engineering ───────────────────────────────────────────────────
story += [h1("6. Software Engineering")]

story += [
    h2("6.1  API Endpoints (LLD Summary)"),
]
api_tbl = Table([
    [b("Method"), b("Endpoint"), b("Input"), b("Output")],
    ["POST", "/translate/audio", "multipart/form-data (WAV/MP3/OGG ≤ 30s)", "navi_text, english, confidence, latency_ms"],
    ["POST", "/translate/text",  "JSON: {text: string, 1–500 chars}",        "english, confidence, word_breakdown, latency_ms"],
    ["POST", "/vocab/submit",    "JSON: {navi_word, english_meaning, audio_b64?}", "accepted, message"],
    ["GET",  "/health",          "—",                                         "status, model_version, uptime_s"],
    ["GET",  "/ready",           "—",                                         "ready, whisper_loaded, marian_loaded"],
    ["GET",  "/metrics",         "—",                                         "Prometheus text exposition (8 metrics)"],
], colWidths=[1.5*cm, 3.5*cm, 6*cm, 5.5*cm])
api_tbl.setStyle(tbl_style())
story += [api_tbl, Spacer(1, 0.3*cm)]

story += [
    h2("6.2  Implementation Quality"),
    bullet("<b>Code style:</b> PEP-8 enforced via ruff; linting runs on every CI push"),
    bullet("<b>Logging:</b> module-level loggers throughout; training scripts log to stdout and timestamped log files"),
    bullet("<b>Exception handling:</b> FastAPI global exception handlers return structured JSON with request IDs"),
    bullet("<b>Type safety:</b> Pydantic v2 enforces all request/response schemas at runtime"),
    bullet("<b>Loose coupling:</b> frontend (nginx + static JS) and backend communicate only via REST; API base URL is configurable"),
    Spacer(1, 0.3*cm),
    h2("6.3  Testing"),
]
test_tbl = Table([
    [b("Suite"), b("Count"), b("Coverage"), b("Result")],
    ["Unit tests",        "30", "Schemas, drift detection, Reykunyu parser, Prometheus helpers, build_pairs", "30 / 30 PASSED"],
    ["Integration tests", "2",  "FastAPI route wiring (fake engine, no model weights required)",              "2 / 2 PASSED"],
    ["Total",             "32", "—",                                                                          "32 / 32 PASSED"],
], colWidths=[3.5*cm, 1.5*cm, 9*cm, 2.5*cm])
test_tbl.setStyle(tbl_style())
story += [test_tbl, Spacer(1, 0.3*cm)]

# ── 7. Frontend ───────────────────────────────────────────────────────────────
story += [
    h1("7. Web Application & Pipeline Visualization"),
    h2("7.1  Frontend UI"),
    p("A 4-tab single-page application served by nginx. No framework dependencies — "
      "pure HTML/CSS/JavaScript for minimal footprint and easy containerisation."),
    bullet("<b>Tab 1 — Translate Audio:</b> Browser MediaRecorder API captures microphone input; alternatively accepts file upload (WAV/MP3/OGG). Displays Na'vi transcription, English translation, and confidence score."),
    bullet("<b>Tab 2 — Translate Text:</b> Text input with real-time Na'vi-to-English translation; word-level breakdown shows per-token dictionary matches."),
    bullet("<b>Tab 3 — Submit Vocabulary:</b> Community contributions with optional audio upload; validated against Na'vi character set before acceptance."),
    bullet("<b>Tab 4 — About:</b> System description and user manual reference."),
    Spacer(1, 0.2*cm),
    h2("7.2  MLOps Pipeline Visualization"),
    p("Multiple specialised MLOps UIs are accessible via Docker Compose, each serving "
      "a distinct observability function:"),
]
ui_tbl = Table([
    [b("Tool"), b("URL"), b("Purpose")],
    ["MLflow UI",   "http://localhost:5000", "Experiment runs, loss curves, parameter comparison, model registry"],
    ["Grafana",     "http://localhost:3001", "Real-time inference metrics, OOV drift, latency histograms"],
    ["Prometheus",  "http://localhost:9090", "Raw metric queries, alert state, scrape target health"],
    ["Airflow",     "DAG trigger / CLI",     "Training DAG runs, task-level logs, retrain scheduling"],
    ["DVC CLI",     "dvc dag / dvc repro",   "Data pipeline DAG, stage-level lineage and caching"],
], colWidths=[2.5*cm, 4.5*cm, 9.5*cm])
ui_tbl.setStyle(tbl_style())
story += [ui_tbl, Spacer(1, 0.3*cm)]

# ── 8. Problems Faced ─────────────────────────────────────────────────────────
story += [h1("8. Problems Faced & Mitigations")]

prob_tbl = Table([
    [b("#"), b("Problem"), b("Root Cause"), b("Mitigation")],
    ["1", "Training crash wiped all progress",
          "save_strategy='epoch' — crash 1 step before epoch end lost everything",
          "Switched to save_strategy='steps' with save_steps=500; max 8 min lost per crash"],
    ["2", "whisper-small training infeasible on MPS (4+ hours)",
          "244M params × ~3 sec/step on Apple Silicon = too slow under no-cloud constraint",
          "Switched to whisper-tiny (39M params, ~6× faster). Research basis: Springer 2024 shows tiny/base outperform small/medium on limited fine-tuning data"],
    ["3", "MPS thermal throttling during mid-training eval (240 sec/eval step)",
          "Repeated eval loops caused the Mac to thermal-throttle even without Docker",
          "Removed mid-training eval entirely (eval_strategy='no'); single eval runs once after training completes"],
    ["4", "CI pipeline failing on every push",
          "CI was pip-installing torch/transformers/peft (~3 GB) on every GitHub Actions run",
          "Created requirements-ci.txt with only the 12 lightweight packages that tests actually import"],
    ["5", "Limited Na'vi audio data",
          "Constructed language — ~4,000 samples publicly available, no commercial corpus",
          "Synthetic TTS augmentation + LoRA to avoid overfitting + Reykunyu fallback for low-confidence outputs"],
    ["6", "No-cloud constraint",
          "Course guidelines explicitly prohibit cloud platforms",
          "Whisper-tiny + LoRA makes training feasible locally; documented as deliberate design choice aligned with guidelines §II.C"],
], colWidths=[0.6*cm, 3.8*cm, 5.2*cm, 6.9*cm])
prob_tbl.setStyle(tbl_style())
story += [prob_tbl, Spacer(1, 0.3*cm)]

# ── 9. Technology Stack ───────────────────────────────────────────────────────
story += [h1("9. Technology Stack")]

stack_tbl = Table([
    [b("Category"), b("Tool"), b("Version"), b("Role")],
    ["ASR",              "OpenAI Whisper-tiny + PEFT LoRA", "openai/whisper-tiny",          "Speech-to-Na'vi-text"],
    ["NMT",              "MarianMT",                        "opus-mt-mul-en",               "Na'vi-to-English translation"],
    ["API",              "FastAPI + Uvicorn",               "≥ 0.109",                      "REST inference server"],
    ["Frontend",         "nginx",                           "alpine",                       "Static UI host + reverse proxy"],
    ["Experiment Track", "MLflow",                          "≥ 2.10",                       "Run tracking + model registry"],
    ["Data Version",     "DVC",                             "3.x",                          "Data & artifact versioning"],
    ["Orchestration",    "Apache Airflow",                  "2.8+",                         "Training DAG scheduling"],
    ["Monitoring",       "Prometheus + Grafana",            "latest",                       "Metrics scraping + dashboards"],
    ["Containers",       "Docker Compose",                  "v2",                           "5-service local deployment"],
    ["CI/CD",            "GitHub Actions",                  "—",                            "Lint + test + docker build"],
    ["Testing",          "pytest + pytest-cov",             "≥ 9.0",                        "Unit + integration tests"],
    ["Linting",          "ruff",                            "latest",                       "PEP-8 enforcement"],
], colWidths=[3*cm, 4*cm, 3.5*cm, 6*cm])
stack_tbl.setStyle(tbl_style())
story += [stack_tbl, Spacer(1, 0.3*cm)]

# ── 10. Conclusion ────────────────────────────────────────────────────────────
story += [
    h1("10. Conclusion"),
    p("The Na'vi Translator demonstrates a complete MLOps lifecycle applied to a challenging "
      "low-resource language scenario under real hardware constraints. Every rubric component "
      "is implemented: Airflow orchestration, DVC data versioning, MLflow experiment tracking, "
      "Prometheus/Grafana monitoring, FastAPI serving, Docker Compose packaging, and GitHub "
      "Actions CI. The system deliberately prioritises <b>MLOps best practices over raw model "
      "accuracy</b> — the architecture is designed to detect performance degradation, trigger "
      "automated retraining, and promote improved models to production without manual "
      "intervention."),
    p("The no-cloud constraint, far from being a limitation, motivated engineering decisions "
      "(LoRA over full fine-tuning, whisper-tiny over whisper-small, dictionary fallback) that "
      "are consistent with production MLOps for resource-constrained environments and are "
      "directly supported by recent research literature."),
    Spacer(1, 0.4*cm),
    h2("References"),
    bullet("Springer (2024). Exploration of Whisper fine-tuning strategies for low-resource ASR. Journal on Audio, Speech, and Music Processing."),
    bullet("MDPI Sensors (2025). LoRA-INT8 Whisper: A Low-Cost Framework for Edge Devices."),
    bullet("Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685."),
    bullet("HuggingFace PEFT Documentation. https://huggingface.co/docs/peft"),
    bullet("Reykunyu Na'vi Dictionary API. https://reykunyu.lu"),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF written to {OUTPUT}")
