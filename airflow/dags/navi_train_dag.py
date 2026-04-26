"""Na'vi Translator — Airflow Training DAG (stub for Phase 1)."""

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

with DAG(
    dag_id="navi_train_dag",
    default_args=default_args,
    description="End-to-end Na'vi translator training pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["navi", "training"],
) as dag:

    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command="echo 'stub: dvc repro ingest'",
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="echo 'stub: dvc repro preprocess_text preprocess_audio'",
    )

    compute_baselines = BashOperator(
        task_id="compute_baselines",
        bash_command="echo 'stub: dvc repro baseline_stats'",
    )

    train_whisper = BashOperator(
        task_id="train_whisper",
        bash_command="echo 'stub: mlflow run . -e train_whisper'",
    )

    train_marian = BashOperator(
        task_id="train_marian",
        bash_command="echo 'stub: mlflow run . -e train_marian'",
    )

    evaluate_and_promote = BashOperator(
        task_id="evaluate_and_promote",
        bash_command="echo 'stub: evaluate and promote models'",
    )

    notify = BashOperator(
        task_id="notify",
        bash_command="echo 'stub: notify completion'",
    )

    # DAG dependency chain
    ingest_data >> preprocess >> compute_baselines
    compute_baselines >> [train_whisper, train_marian] >> evaluate_and_promote >> notify
