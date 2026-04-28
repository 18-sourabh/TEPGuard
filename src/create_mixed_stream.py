import os
import pandas as pd

from preprocess import process_tep_data


def split_runs(df):
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    runs = []

    for run in df["simulationrun"].unique():
        run_df = df[df["simulationrun"] == run].reset_index(drop=True)
        runs.append(run_df)

    return runs


def take_continuous_block(run_df, block_size):
    if len(run_df) <= block_size:
        return run_df.copy()

    return run_df.iloc[:block_size].copy()


def build_mixed_stream(ff_runs, f_runs, total_rows=10000):
    mixed_parts = []

    block_size = 200 
    rows_collected = 0

    i = 0

    while rows_collected < total_rows:
        if i < len(ff_runs):
            block = take_continuous_block(ff_runs[i], block_size)
            mixed_parts.append(block)
            rows_collected += len(block)

        if i < len(f_runs):
            block = take_continuous_block(f_runs[i], block_size)
            mixed_parts.append(block)
            rows_collected += len(block)

        i += 1

        if i >= max(len(ff_runs), len(f_runs)):
            break

    mixed_df = pd.concat(mixed_parts, ignore_index=True)

    # Trim to exact size
    mixed_df = mixed_df.iloc[:total_rows].copy()
    mixed_df["stream_index"] = range(len(mixed_df))

    return mixed_df


def main():
    base_path = "/Users/sourabh18/Desktop/Projects"
    raw_path = f"{base_path}/data/raw"
    mixed_path = f"{base_path}/data/mixed"

    os.makedirs(mixed_path, exist_ok=True)

    fault_free_test_file = f"{raw_path}/TEP_FaultFree_Testing.RData"
    faulty_test_file = f"{raw_path}/TEP_Faulty_Testing.RData"

    ff_test = process_tep_data(fault_free_test_file)
    f_test = process_tep_data(faulty_test_file)

    print("Original fault-free shape:", ff_test.shape)
    print("Original faulty shape:", f_test.shape)

    ff_test["source_label"] = 0
    f_test["source_label"] = 1

    # Split into runs (IMPORTANT)
    ff_runs = split_runs(ff_test)
    f_runs = split_runs(f_test)

    print("Fault-free runs:", len(ff_runs))
    print("Faulty runs:", len(f_runs))

    # Build 10,000-row sequence-safe stream
    mixed_df = build_mixed_stream(ff_runs, f_runs, total_rows=10000)

    mixed_df.to_pickle(f"{mixed_path}/mixed_stream.pkl")
    mixed_df.to_csv(f"{mixed_path}/mixed_stream.csv", index=False)

    print("Final mixed shape:", mixed_df.shape)

    print(
        mixed_df[
            ["simulationrun", "sample", "time_stamp", "source_label", "stream_index"]
        ].head(60)
    )


if __name__ == "__main__":
    main()