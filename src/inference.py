import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from attribution import get_top_contributing_variables


BASE_PATH = os.environ.get("BASE_PATH", "/Users/sourabh18/Desktop/Projects")


def load_artifacts():
    artifact_path = f"{BASE_PATH}/artifacts"

    model = load_model(f"{artifact_path}/lstm_autoencoder_model.keras")
    scaler = joblib.load(f"{artifact_path}/scaler.pkl")
    feature_columns = joblib.load(f"{artifact_path}/feature_columns.pkl")

    with open(f"{artifact_path}/config.json", "r") as f:
        config = json.load(f)

    return model, scaler, feature_columns, config


def load_batch(batch_file):
    return pd.read_csv(batch_file)


def load_state_buffer(state_path):
    if os.path.exists(state_path):
        return pd.read_pickle(state_path)
    return None


def save_state_buffer(buffer_df, state_path):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    buffer_df.to_pickle(state_path)


def update_buffer(prev_buffer, new_batch_df, seq_length):
    if prev_buffer is None:
        combined = new_batch_df.copy()
    else:
        combined = pd.concat([prev_buffer, new_batch_df], ignore_index=True)

    return combined.iloc[-seq_length:].copy().reset_index(drop=True)


def align_feature_columns(feature_df, feature_columns):
    incoming_map = {col.lower(): col for col in feature_df.columns}
    aligned_df = pd.DataFrame(index=feature_df.index)

    for col in feature_columns:
        key = col.lower()
        if key not in incoming_map:
            raise ValueError(f"Missing required feature column in batch data: {col}")
        aligned_df[col] = feature_df[incoming_map[key]]

    return aligned_df


def run_inference_on_buffer(buffer_df, model, scaler, feature_columns, threshold):
    meta_cols = ["simulationrun", "sample", "time_stamp", "source_label", "stream_index"]
    meta_df = buffer_df[meta_cols].copy()

    feature_df = buffer_df.drop(columns=meta_cols).copy()
    feature_df = align_feature_columns(feature_df, feature_columns)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)

    scaled = scaler.transform(feature_df)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_seq = np.array([scaled], dtype="float32")

    pred = model.predict(X_seq, verbose=0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    mse = float(np.mean(np.square(X_seq - pred), axis=(1, 2))[0])
    is_anomaly = int(mse > threshold)

    last_row = meta_df.iloc[-1]

    result = {
        "simulationrun": last_row["simulationrun"],
        "sample": last_row["sample"],
        "time_stamp": last_row["time_stamp"],
        "source_label": last_row["source_label"],
        "stream_index": last_row["stream_index"],
        "reconstruction_error": mse,
        "is_anomaly": is_anomaly
    }

    if is_anomaly == 1:
        top_vars = get_top_contributing_variables(
            X_seq,
            pred,
            feature_columns,
            top_n=5
        )

        result["top_contributing_variables"] = json.dumps(top_vars)
        result["top_variable_names"] = ", ".join([v["mapped_feature"] for v in top_vars])
    else:
        result["top_contributing_variables"] = ""
        result["top_variable_names"] = ""

    return result


def main():
    batch_path = f"{BASE_PATH}/data/batches"
    results_path = f"{BASE_PATH}/data/results"
    state_path = f"{BASE_PATH}/data/mixed/state_buffer.pkl"

    os.makedirs(results_path, exist_ok=True)

    model, scaler, feature_columns, config = load_artifacts()

    seq_length = config["seq_length"]
    threshold = config["threshold"]

    print("Loaded artifacts successfully")
    print("Base path:", BASE_PATH)
    print("Sequence length:", seq_length)
    print("Threshold:", threshold)
    print("Number of expected features:", len(feature_columns))

    batch_files = sorted([
        os.path.join(batch_path, f)
        for f in os.listdir(batch_path)
        if f.startswith("batch_") and f.endswith(".csv") and f != "batch_summary.csv"
    ])

    print("Total batch files found:", len(batch_files))

    buffer_df = load_state_buffer(state_path)
    all_results = []

    for batch_file in batch_files:
        batch_df = load_batch(batch_file)
        buffer_df = update_buffer(buffer_df, batch_df, seq_length)

        if len(buffer_df) < seq_length:
            last_row = buffer_df.iloc[-1]

            warmup_result = {
                "simulationrun": last_row.get("simulationrun", None),
                "sample": last_row.get("sample", None),
                "time_stamp": last_row.get("time_stamp", None),
                "source_label": last_row.get("source_label", None),
                "stream_index": last_row.get("stream_index", None),
                "reconstruction_error": None,
                "is_anomaly": None,
                "top_contributing_variables": "",
                "top_variable_names": ""
            }

            all_results.append(warmup_result)
            save_state_buffer(buffer_df, state_path)
            continue

        result = run_inference_on_buffer(
            buffer_df=buffer_df,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            threshold=threshold
        )

        all_results.append(result)
        save_state_buffer(buffer_df, state_path)

    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)

        results_df.to_csv(f"{results_path}/inference_results.csv", index=False)
        results_df.to_pickle(f"{results_path}/inference_results.pkl")

        anomalies_df = results_df[results_df["is_anomaly"] == 1].copy()

        anomalies_df.to_csv(f"{results_path}/inference_anomalies.csv", index=False)
        anomalies_df.to_pickle(f"{results_path}/inference_anomalies.pkl")

        print("\nSaved inference results.")
        print("Total rows:", len(results_df))
        print("Total anomalies:", len(anomalies_df))
    else:
        print("\nNo results saved.")


if __name__ == "__main__":
    main()