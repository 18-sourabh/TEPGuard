import os
import json
import requests
import pandas as pd


BASE_PATH = os.environ.get("BASE_PATH", "/Users/sourabh18/Desktop/Projects")
RESULTS_PATH = f"{BASE_PATH}/data/results"
ARTIFACT_PATH = f"{BASE_PATH}/artifacts"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "mistral")


PROCESS_CONTEXT = """
The process is the Tennessee Eastman Process. It contains feed streams, a reactor, condenser, vapor-liquid separator, recycle compressor, purge stream, stripper, and final product stream.

Fresh feeds A, D, and E enter the reactor through feed control valves:
- xmeas_1 / A_feed_stream and xmv_3 / A_feed_flow_valve represent A feed behavior.
- xmeas_2 / D_feed_stream and xmv_1 / D_feed_flow_valve represent D feed behavior.
- xmeas_3 / E_feed_stream and xmv_2 / E_feed_flow_valve represent E feed behavior.
- xmeas_4 / Total_fresh_feed_stripper and xmv_4 / Total_feed_flow_stripper_valve represent total feed flow toward the stripper/feed section.

The reactor is the main reaction section. Important reactor variables include:
- xmeas_6 / Reactor_feed_rate
- xmeas_7 / Reactor_pressure
- xmeas_8 / Reactor_level
- xmeas_9 / Reactor_temp
- xmeas_21 / Reactor_cooling_water_outlet_temp
- xmv_10 / Reactor_cooling_water_flow_valve
- xmv_12 / Agitator_speed

The reactor outlet is cooled in the condenser before entering the separator. Important condenser/cooling variables include:
- xmeas_22 / Condenser_cooling_water_outlet_temp
- xmv_11 / Condenser_cooling_water_flow_valve

The separator splits vapor and liquid after condensation. Important separator variables include:
- xmeas_11 / Separator_temp
- xmeas_12 / Separator_level
- xmeas_13 / Separator_pressure
- xmeas_14 / Separator_underflow
- xmv_7 / Separator_pot_liquid_flow_valve

Separator vapor is recycled back to the reactor through the compressor. A purge stream removes inert/byproduct buildup. Important recycle and purge variables include:
- xmeas_5 / Recycle_flow_into_rxtr
- xmeas_10 / Purge_rate
- xmeas_20 / Compressor_work
- xmv_5 / Compressor_recycle_valve
- xmv_6 / Purge_valve

Separator liquid goes to the stripper. The stripper separates final products from remaining light components. Important stripper variables include:
- xmeas_15 / Stripper_level
- xmeas_16 / Stripper_pressure
- xmeas_17 / Stripper_underflow
- xmeas_18 / Stripper_temperature
- xmeas_19 / Stripper_steam_flow
- xmv_8 / Stripper_liquid_product_flow_valve
- xmv_9 / Stripper_steam_valve

Composition variables indicate material balance and product quality:
- xmeas_23 to xmeas_28 describe reactor feed composition.
- xmeas_29 to xmeas_36 describe purge composition.
- xmeas_37 to xmeas_41 describe product composition.

Rolling mean variables represent sustained deviation over the recent 15-minute window. Rate-of-change variables represent sudden movement or transient shifts.
"""


def load_variable_stats():
    with open(f"{ARTIFACT_PATH}/variable_stats.json", "r") as f:
        return json.load(f)


def load_config():
    with open(f"{ARTIFACT_PATH}/config.json", "r") as f:
        return json.load(f)


def classify_value(value, stats):
    if value < stats["p1"]:
        return "EXTREME_LOW"
    if value > stats["p99"]:
        return "EXTREME_HIGH"
    if value < stats["p5"]:
        return "LOW"
    if value > stats["p95"]:
        return "HIGH"
    return "NORMAL"


def build_variable_range_context(top_vars, variable_stats):
    lines = []

    for item in top_vars:
        raw = item.get("raw_feature")
        mapped = item.get("mapped_feature", raw)
        current = item.get("current_scaled_value")
        contribution = item.get("reconstruction_error_contribution")

        if raw is None or current is None:
            continue

        if raw not in variable_stats:
            lines.append(
                f"- {mapped} ({raw}): current={current:.4f}, "
                f"status=UNKNOWN, normal range unavailable, "
                f"contribution={contribution:.4f}"
            )
            continue

        stats = variable_stats[raw]
        status = classify_value(current, stats)

        lines.append(
            f"- {mapped} ({raw}): current={current:.4f}, status={status}, "
            f"normal p5-p95=({stats['p5']:.4f}, {stats['p95']:.4f}), "
            f"extreme p1-p99=({stats['p1']:.4f}, {stats['p99']:.4f}), "
            f"contribution={contribution:.4f}"
        )

    return "\n".join(lines) if lines else "No top variable details available."


def build_prompt(row, variable_stats, threshold):
    top_vars_raw = row.get("top_contributing_variables", "[]")

    try:
        top_vars = json.loads(top_vars_raw) if isinstance(top_vars_raw, str) else []
    except Exception:
        top_vars = []

    variable_range_context = build_variable_range_context(top_vars, variable_stats)

    reconstruction_error = row.get("reconstruction_error", "not_available")
    time_stamp = row.get("time_stamp", "not_available")
    simulationrun = row.get("simulationrun", "not_available")
    sample = row.get("sample", "not_available")

    return f"""
You are a chemical process monitoring assistant reviewing an autoencoder anomaly alert.

The autoencoder alert is NOT proof of a real fault. Decide whether there is enough process evidence to call it a likely real process anomaly.

PROCESS CONTEXT:
{PROCESS_CONTEXT}

MODEL ALERT:
- Time stamp: {time_stamp}
- Simulation run: {simulationrun}
- Sample: {sample}
- Reconstruction error: {reconstruction_error}
- Autoencoder threshold: {threshold}

TOP VARIABLES WITH CURRENT VALUES AND NORMAL RANGES:
{variable_range_context}

DECISION RULES:
1. Classify as likely_real_anomaly ONLY if there is strong process evidence.
2. A single variable outside p5-p95 is NOT enough.
3. Two or more variables outside p5-p95 are NOT enough unless they belong to the same process area or show a physically related pattern.
4. Values outside p1-p99 are strong evidence, but still check process consistency.
5. If variables are unrelated across different units, classify as uncertain or possible_false_positive.
6. If most variables are close to p5-p95 boundaries, classify as possible_false_positive.
7. Use reconstruction error as supporting evidence only.
8. If evidence is mixed, choose uncertain instead of likely_real_anomaly.

Return ONLY valid JSON:
{{
  "classification": "likely_real_anomaly | possible_false_positive | uncertain",
  "affected_area": "reactor | separator | stripper | recycle_loop | purge_system | cooling_system | product_quality | feed_system | unknown",
  "abnormal_variables": ["variable_1", "variable_2"],
  "reasoning": "brief explanation based only on the provided evidence",
  "recommended_action": "specific next check",
  "confidence": 0.0
}}
"""


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json().get("response", "")


def main():
    input_file = f"{RESULTS_PATH}/inference_anomalies.csv"
    output_file = f"{RESULTS_PATH}/ollama_reviewed_anomalies.csv"

    if not os.path.exists(input_file):
        print("inference_anomalies.csv not found. Skipping LLM review.")
        return

    if os.path.getsize(input_file) == 0:
        print("inference_anomalies.csv is empty. Skipping LLM review.")
        return

    anomalies = pd.read_csv(input_file)

    if len(anomalies) == 0:
        print("No anomalies found. Skipping LLM review.")
        return

    variable_stats = load_variable_stats()
    config = load_config()
    threshold = config["threshold"]

    reviewed_rows = []

    for i, row in anomalies.iterrows():
        print(f"Reviewing anomaly {i + 1}/{len(anomalies)}")

        prompt = build_prompt(row, variable_stats, threshold)

        try:
            review = call_ollama(prompt)
        except Exception as e:
            review = json.dumps({
                "classification": "ollama_error",
                "affected_area": "unknown",
                "abnormal_variables": [],
                "reasoning": str(e),
                "recommended_action": "Check Ollama server or skip LLM review in AWS container.",
                "confidence": 0.0
            })

        reviewed_row = row.to_dict()
        reviewed_row["ollama_review"] = review
        reviewed_rows.append(reviewed_row)

    reviewed_df = pd.DataFrame(reviewed_rows)
    reviewed_df.to_csv(output_file, index=False)

    print("Saved:", output_file)


if __name__ == "__main__":
    main()