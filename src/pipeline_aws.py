import os
from pathlib import Path
import boto3
import pandas as pd
import subprocess


print("pipeline_aws.py started")

BASE_PATH = os.environ.get("BASE_PATH", "/app")

BUCKET_NAME = os.environ["S3_BUCKET"]
BATCH_KEY = os.environ["BATCH_KEY"]

LOCAL_BATCH_DIR = f"{BASE_PATH}/data/batches"
LOCAL_RESULTS_DIR = f"{BASE_PATH}/data/results"
LOCAL_MIXED_DIR = f"{BASE_PATH}/data/mixed"

STATE_S3_KEY = "state/state_buffer.pkl"
RESULTS_S3_PREFIX = "results"

s3 = boto3.client("s3")


def setup_dirs():
    Path(LOCAL_BATCH_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_MIXED_DIR).mkdir(parents=True, exist_ok=True)


def clear_local_inputs_outputs():
    for folder in [LOCAL_BATCH_DIR, LOCAL_RESULTS_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)

        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def download_batch():
    local_batch_path = f"{LOCAL_BATCH_DIR}/batch_current.csv"

    print(f"Downloading s3://{BUCKET_NAME}/{BATCH_KEY}")
    s3.download_file(BUCKET_NAME, BATCH_KEY, local_batch_path)
    print(f"Downloaded batch to {local_batch_path}")


def download_state_if_exists():
    local_state_path = f"{LOCAL_MIXED_DIR}/state_buffer.pkl"

    try:
        print(f"Trying to download s3://{BUCKET_NAME}/{STATE_S3_KEY}")
        s3.download_file(BUCKET_NAME, STATE_S3_KEY, local_state_path)
        print("Downloaded existing state buffer.")
    except Exception as e:
        print("No existing state buffer found. Starting fresh.")
        print(str(e))


def run_script(script_name):
    script_path = f"{BASE_PATH}/src/{script_name}"

    print(f"Running {script_name}")

    result = subprocess.run(
        ["python", "-u", script_path],
        capture_output=True,
        text=True,
        env={**os.environ, "BASE_PATH": BASE_PATH}
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed")


def run_inference():
    run_script("inference.py")


def run_llm_review_if_needed():
    anomalies_path = f"{LOCAL_RESULTS_DIR}/inference_anomalies.csv"

    if not os.path.exists(anomalies_path):
        print("No inference_anomalies.csv found. Likely warmup batch. Skipping LLM review.")
        return

    if os.path.getsize(anomalies_path) == 0:
        print("inference_anomalies.csv is empty. Skipping LLM review.")
        return

    try:
        anomalies_df = pd.read_csv(anomalies_path)
    except Exception as e:
        print("Could not read inference_anomalies.csv. Skipping LLM review.")
        print(str(e))
        return

    if len(anomalies_df) == 0:
        print("No anomaly rows found. Skipping LLM review.")
        return

    run_script("llm_review.py")


def download_s3_csv_if_exists(s3_key, local_path):
    try:
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"Downloaded existing cumulative file: s3://{BUCKET_NAME}/{s3_key}")
        return True
    except Exception:
        print(f"No existing cumulative file found: s3://{BUCKET_NAME}/{s3_key}")
        return False


def append_to_cumulative(current_file, cumulative_s3_key, cumulative_local_file):
    if not os.path.exists(current_file):
        print(f"No current file found, skipping cumulative append: {current_file}")
        return

    current_df = pd.read_csv(current_file)

    if len(current_df) == 0:
        print(f"Current file is empty, skipping cumulative append: {current_file}")
        return

    if download_s3_csv_if_exists(cumulative_s3_key, cumulative_local_file):
        existing_df = pd.read_csv(cumulative_local_file)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df.copy()

    combined_df = combined_df.drop_duplicates(
        subset=["simulationrun", "sample", "time_stamp", "stream_index"],
        keep="last"
    )

    combined_df.to_csv(cumulative_local_file, index=False)

    s3.upload_file(cumulative_local_file, BUCKET_NAME, cumulative_s3_key)
    print(f"Uploaded cumulative file: s3://{BUCKET_NAME}/{cumulative_s3_key}")
    print(f"Cumulative rows: {len(combined_df)}")


def update_cumulative_outputs():
    append_to_cumulative(
        current_file=f"{LOCAL_RESULTS_DIR}/inference_results.csv",
        cumulative_s3_key=f"{RESULTS_S3_PREFIX}/inference_results_all.csv",
        cumulative_local_file=f"{LOCAL_RESULTS_DIR}/inference_results_all.csv"
    )

    append_to_cumulative(
        current_file=f"{LOCAL_RESULTS_DIR}/inference_anomalies.csv",
        cumulative_s3_key=f"{RESULTS_S3_PREFIX}/inference_anomalies_all.csv",
        cumulative_local_file=f"{LOCAL_RESULTS_DIR}/inference_anomalies_all.csv"
    )

    append_to_cumulative(
        current_file=f"{LOCAL_RESULTS_DIR}/ollama_reviewed_anomalies.csv",
        cumulative_s3_key=f"{RESULTS_S3_PREFIX}/ollama_reviewed_anomalies_all.csv",
        cumulative_local_file=f"{LOCAL_RESULTS_DIR}/ollama_reviewed_anomalies_all.csv"
    )


def upload_file_if_exists(local_path, s3_key):
    if os.path.exists(local_path):
        print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}")
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
    else:
        print(f"File not found, skipping upload: {local_path}")


def upload_latest_outputs():
    upload_file_if_exists(f"{LOCAL_MIXED_DIR}/state_buffer.pkl", STATE_S3_KEY)

    latest_files = [
        "inference_results.csv",
        "inference_results.pkl",
        "inference_anomalies.csv",
        "inference_anomalies.pkl",
        "ollama_reviewed_anomalies.csv",
        "final_output.csv"
    ]

    for file_name in latest_files:
        local_path = f"{LOCAL_RESULTS_DIR}/{file_name}"
        s3_key = f"{RESULTS_S3_PREFIX}/latest/{file_name}"
        upload_file_if_exists(local_path, s3_key)


def main():
    print("Starting AWS anomaly detection pipeline")

    setup_dirs()
    clear_local_inputs_outputs()
    download_batch()
    download_state_if_exists()

    run_inference()
    run_llm_review_if_needed()

    upload_latest_outputs()
    update_cumulative_outputs()

    print("AWS pipeline completed successfully")


if __name__ == "__main__":
    main()