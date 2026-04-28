import pyreadr
import pandas as pd


def process_tep_data(file_path: str) -> pd.DataFrame:
    result = pyreadr.read_r(file_path)
    df = list(result.values())[0].copy()

    start_time = pd.to_datetime("2025-01-01 00:00:00")
    df["time_stamp"] = start_time + pd.to_timedelta((df["sample"] - 1) * 3, unit="m")

    df = df.sort_values(["simulationRun", "time_stamp"]).reset_index(drop=True)

    exclude_cols = ["simulationRun", "sample", "time_stamp", "faultNumber"]
    features = [col for col in df.columns if col not in exclude_cols]

    window = 5

    rollmean_df = pd.DataFrame(index=df.index)
    diff_df = pd.DataFrame(index=df.index)

    for col in features:
        rollmean_df[f"{col}_rollmean"] = (
            df.groupby("simulationRun")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    for col in features:
        diff_df[f"{col}_diff"] = (
            df.groupby("simulationRun")[col]
            .diff()
        )

    df = pd.concat([df, rollmean_df, diff_df], axis=1)
    df = df.fillna(0).copy()

    return df


if __name__ == "__main__":
    file_path = "/Users/sourabh18/Desktop/Projects/data/raw/TEP_FaultFree_Training.RData"  # replace if needed
    df = process_tep_data(file_path)
    print(df.head())
    print(df.shape)