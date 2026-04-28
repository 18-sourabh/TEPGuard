import os
import pandas as pd


def main():
    base_path = "/Users/sourabh18/Desktop/Projects"

    mixed_path = f"{base_path}/data/mixed/mixed_stream.pkl"
    batch_path = f"{base_path}/data/batches"

    os.makedirs(batch_path, exist_ok=True)

    if not os.path.exists(mixed_path):
        raise FileNotFoundError(f"{mixed_path} not found. Run create_mixed_stream.py first.")

    df = pd.read_pickle(mixed_path).reset_index(drop=True)

    print("Loaded mixed stream shape:", df.shape)

    batch_size = 5
    num_batches = 2000   # 32 batches × 5 rows × 3 min = 8 hours

    rows_needed = batch_size * num_batches
    df = df.iloc[:rows_needed].copy().reset_index(drop=True)

    print("Using rows for demo:", df.shape)

    batch_summary = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_df = df.iloc[start_idx:end_idx].copy()

        if len(batch_df) < batch_size:
            print(f"Stopping early at batch {i+1}, not enough rows left.")
            break

        batch_id = f"batch_{i+1:05d}"
        batch_file = f"{batch_path}/{batch_id}.csv"

        batch_df.to_csv(batch_file, index=False)

        summary_row = {
            "batch_id": batch_id,
            "start_index": start_idx,
            "end_index": end_idx - 1,
            "num_rows": len(batch_df),
            "start_time": str(batch_df["time_stamp"].iloc[0]),
            "end_time": str(batch_df["time_stamp"].iloc[-1]),
            "has_fault": int(batch_df["source_label"].max()),
            "fault_count": int(batch_df["source_label"].sum())
        }

        batch_summary.append(summary_row)

    summary_df = pd.DataFrame(batch_summary)
    summary_df.to_csv(f"{batch_path}/batch_summary.csv", index=False)

    print("\nBatch creation complete")
    print("Total batches created:", len(summary_df))
    print(summary_df.head(10))


if __name__ == "__main__":
    main()