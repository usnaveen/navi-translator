"""Na'vi Translator — Airflow Training DAG.

This DAG orchestrates the entire training pipeline:
  ingest → preprocess → baselines → [train_whisper, train_marian] → evaluate → notify

Key learning points:
- A DAG (Directed Acyclic Graph) defines the order of tasks
- >> operator sets dependencies: A >> B means B runs after A
- [A, B] >> C means C runs after BOTH A and B complete
- PythonOperator runs a Python callable; BashOperator runs shell commands
- schedule='@weekly' means automatic weekly retraining
- The DAG can also be triggered on-demand by the monitoring drift alert
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "navi-translator",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

PROJECT_DIR = "/opt/airflow"

with DAG(
    dag_id="navi_train_dag",
    default_args=default_args,
    description="End-to-end Na'vi translator training pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["navi", "training"],
) as dag:

    # Task 1: Data ingestion via DVC
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {PROJECT_DIR} && dvc repro ingest",
    )

    # Task 2: Preprocessing (text + audio)
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && dvc repro preprocess_text preprocess_audio split",
    )

    # Task 3: Compute baseline statistics
    compute_baselines = BashOperator(
        task_id="compute_baselines",
        bash_command=f"cd {PROJECT_DIR} && dvc repro baseline_stats",
    )

    # Task 4a: Train Whisper ASR (runs in parallel with 4b)
    train_whisper = BashOperator(
        task_id="train_whisper",
        bash_command=f"cd {PROJECT_DIR} && python src/training/train_whisper.py --config params.yaml",
    )

    # Task 4b: Train MarianMT NMT (runs in parallel with 4a)
    train_marian = BashOperator(
        task_id="train_marian",
        bash_command=f"cd {PROJECT_DIR} && python src/training/train_marian.py --config params.yaml",
    )

    # Task 5: Evaluate and promote best models
    evaluate_and_promote = BashOperator(
        task_id="evaluate_and_promote",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/training/evaluate.py --model-type whisper "
            "--run-id $(mlflow runs list --experiment-name navi-translator "
            "--filter 'tags.model_type=\"whisper\"' --max-results 1 "
            "--output json | python -c \"import sys,json; print(json.load(sys.stdin)[0]['run_id'])\") && "
            "python src/training/evaluate.py --model-type marian "
            "--run-id $(mlflow runs list --experiment-name navi-translator "
            "--filter 'tags.model_type=\"marian\"' --max-results 1 "
            "--output json | python -c \"import sys,json; print(json.load(sys.stdin)[0]['run_id'])\")"
        ),
    )

    # Task 6: Notify completion and restart backend
    notify = BashOperator(
        task_id="notify",
        bash_command=(
            'echo "Training pipeline completed at $(date)" && '
            'echo "Restarting backend service to load new model..." && '
            "docker compose restart backend 2>/dev/null || echo 'Backend restart skipped (not in Docker)'"
        ),
    )

    # DAG dependency chain
    ingest_data >> preprocess >> compute_baselines
    compute_baselines >> [train_whisper, train_marian] >> evaluate_and_promote >> notify


# --- Retrain DAG (triggered by drift detection) ---
with DAG(
    dag_id="navi_retrain_dag",
    default_args=default_args,
    description="Retraining pipeline triggered by data drift detection",
    schedule=None,  # only triggered externally
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["navi", "retraining", "drift"],
) as retrain_dag:

    retrain_ingest = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {PROJECT_DIR} && dvc repro ingest",
    )

    retrain_preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && dvc repro preprocess_text preprocess_audio split",
    )

    retrain_baselines = BashOperator(
        task_id="compute_baselines",
        bash_command=f"cd {PROJECT_DIR} && dvc repro baseline_stats",
    )

    retrain_whisper = BashOperator(
        task_id="train_whisper",
        bash_command=f"cd {PROJECT_DIR} && python src/training/train_whisper.py",
    )

    retrain_marian = BashOperator(
        task_id="train_marian",
        bash_command=f"cd {PROJECT_DIR} && python src/training/train_marian.py",
    )

    retrain_evaluate = BashOperator(
        task_id="evaluate_and_promote",
        bash_command=f"cd {PROJECT_DIR} && echo 'Evaluate and promote after retrain'",
    )

    retrain_notify = BashOperator(
        task_id="notify",
        bash_command='echo "Drift-triggered retraining completed at $(date)"',
    )

    retrain_ingest >> retrain_preprocess >> retrain_baselines
    retrain_baselines >> [retrain_whisper, retrain_marian] >> retrain_evaluate >> retrain_notify
