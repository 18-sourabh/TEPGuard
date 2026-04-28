import json
import re
from io import StringIO

import boto3
import pandas as pd
import streamlit as st


S3_BUCKET = "tep-anomaly-sourabh"
INFERENCE_KEY = "results/inference_anomalies_all.csv"
OLLAMA_KEY = "results/ollama_reviewed_anomalies_all.csv"
PFD_IMAGE_PATH = "/Users/sourabh18/Desktop/Projects/TE_flow.jpg"


st.set_page_config(
    page_title="Process Monitoring Dashboard",
    layout="wide"
)


def load_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))


@st.cache_data(ttl=60)
def load_data():
    inference_df = load_csv_from_s3(S3_BUCKET, INFERENCE_KEY)

    try:
        ollama_df = load_csv_from_s3(S3_BUCKET, OLLAMA_KEY)
    except Exception:
        ollama_df = pd.DataFrame()

    return inference_df, ollama_df


def extract_json_from_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return {}

    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "classification": "unknown",
        "affected_area": "unknown",
        "reasoning": cleaned,
        "recommended_action": "Review manually.",
        "confidence": 0.0,
        "abnormal_variables": []
    }


def severity_from_error(error):
    if pd.isna(error):
        return "Unknown"
    if error >= 1.20:
        return "High"
    if error >= 0.95:
        return "Medium"
    return "Low"


def format_top_tags(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""

    tags = []

    try:
        parsed = json.loads(value)

        for item in parsed:
            name = item.get("mapped_feature") or item.get("raw_feature")
            if name:
                tags.append(str(name))
    except Exception:
        tags = [x.strip() for x in str(value).split(",") if x.strip()]

    tags = tags[:5]

    return "<br>".join([f"• {tag}" for tag in tags])


def filter_last_10_hours(df):
    if df.empty or "time_stamp" not in df.columns:
        return df, None, None

    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.dropna(subset=["time_stamp"])

    if df.empty:
        return df, None, None

    latest_time = df["time_stamp"].max()
    start_time = latest_time - pd.Timedelta(hours=10)

    filtered_df = df[df["time_stamp"] >= start_time].copy()

    return filtered_df, start_time, latest_time


def clean_affected_area(value):
    if not value:
        return "unknown"

    value = str(value)

    if "|" in value:
        return "multiple areas"

    return value


def short_text(text, max_chars=240):
    if not text:
        return "Not available."

    text = str(text).replace("\n", " ").strip()

    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."

    return text


def prepare_inference_df(df):
    df = df.copy()

    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    if "reconstruction_error" in df.columns:
        df["severity"] = df["reconstruction_error"].apply(severity_from_error)
    else:
        df["severity"] = "Unknown"

    if "top_contributing_variables" in df.columns:
        df["top_tags_display"] = df["top_contributing_variables"].apply(format_top_tags)
    elif "top_variable_names" in df.columns:
        df["top_tags_display"] = df["top_variable_names"].apply(format_top_tags)
    else:
        df["top_tags_display"] = ""

    return df


def render_latest_anomalies_table(df):
    latest = df.head(10).copy()

    rows = []

    for _, row in latest.iterrows():
        time_value = row.get("time_stamp")
        time_display = pd.to_datetime(time_value).strftime("%b %d, %Y %I:%M %p") if pd.notna(time_value) else "N/A"

        rec_error = row.get("reconstruction_error", None)
        rec_error_display = f"{rec_error:.4f}" if pd.notna(rec_error) else "N/A"

        severity = row.get("severity", "Unknown")
        severity_class = str(severity).lower()

        tags_html = row.get("top_tags_display", "")
        tags_html = tags_html.replace("<br>", "</li><li>")
        tags_html = f"<ul><li>{tags_html.replace('• ', '')}</li></ul>" if tags_html else ""

        rows.append(f"""
        <tr>
            <td class="time-col">{time_display}</td>
            <td class="tags-col">{tags_html}</td>
            <td class="error-col">{rec_error_display}</td>
            <td class="severity-col">
                <span class="severity-pill {severity_class}">{severity}</span>
            </td>
        </tr>
        """)

    table_html = f"""
    <div class="table-wrap">
        <table class="anomaly-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Top Contributing Tags</th>
                    <th>Reconstruction Error</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """

    st.html(table_html)


def render_llm_cards(ollama_df):
    if ollama_df.empty or "ollama_review" not in ollama_df.columns:
        st.info("No Ollama review data available in the last 10 hours.")
        return

    ollama_df = ollama_df.copy()

    if "time_stamp" in ollama_df.columns:
        ollama_df["time_stamp"] = pd.to_datetime(ollama_df["time_stamp"], errors="coerce")
        ollama_df = ollama_df.sort_values("time_stamp", ascending=False)

    records = ollama_df.head(6).to_dict("records")

    for i in range(0, len(records), 3):
        cols = st.columns(3)

        for col, row in zip(cols, records[i:i + 3]):
            with col:
                review = extract_json_from_text(row.get("ollama_review", ""))

                classification = review.get("classification", "unknown")
                affected_area = clean_affected_area(review.get("affected_area", "unknown"))
                reasoning = short_text(review.get("reasoning", "No reasoning available."), 250)
                recommended_action = short_text(review.get("recommended_action", "Review manually."), 180)
                confidence = review.get("confidence", 0.0)
                abnormal_variables = review.get("abnormal_variables", [])

                if isinstance(abnormal_variables, list):
                    abnormal_vars_display = "<br>".join([f"• {str(x)}" for x in abnormal_variables[:4]])
                else:
                    abnormal_vars_display = str(abnormal_variables)

                timestamp = row.get("time_stamp", None)
                time_display = pd.to_datetime(timestamp).strftime("%b %d, %Y %I:%M %p") if pd.notna(timestamp) else "Time unavailable"

                error = row.get("reconstruction_error", None)
                severity = severity_from_error(error)
                severity_class = str(severity).lower()

                st.markdown(
                    f"""
                    <div class="llm-card">
                        <div class="card-header-row">
                            <div>
                                <b>{time_display}</b><br>
                                <span class="small-muted">Classification: {classification}</span>
                            </div>
                            <span class="severity-pill {severity_class}">{severity}</span>
                        </div>
                        <p><b>Affected Area:</b> {affected_area}</p>
                        <p><b>Summary:</b><br>{reasoning}</p>
                        <p><b>Top Variables:</b><br>{abnormal_vars_display if abnormal_vars_display else "Not specified"}</p>
                        <p><b>Recommended Action:</b><br>{recommended_action}</p>
                        <p class="small-muted">Confidence: {confidence}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def main():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #fffaf3 0%, #ffffff 45%, #fff3e0 100%);
            color: #111827;
        }

        .block-container {
            padding-top: 3rem;
            padding-bottom: 2rem;
            max-width: 96%;
        }

        .main-title {
            font-size: 34px;
            line-height: 1.3;
            font-weight: 800;
            color: #000000;
            margin-bottom: 4px;
            letter-spacing: -0.3px;
            white-space: normal;
        }

        .subtitle {
            color: #6b7280;
            font-size: 15px;
            margin-bottom: 6px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #fed7aa;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 6px 18px rgba(234, 88, 12, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            color: #7c2d12;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: #111827;
            font-weight: 850;
        }

        h2, h3 {
            color: #111827;
            letter-spacing: -0.3px;
        }

        .small-muted {
            color: #78716c;
            font-size: 13px;
        }

        .table-wrap {
            width: 100%;
            background: #ffffff;
            border: 1px solid #fed7aa;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(234, 88, 12, 0.10);
            margin-top: 10px;
        }

        .anomaly-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .anomaly-table thead {
            background: linear-gradient(90deg, #111827 0%, #1f2937 55%, #9a3412 100%);
        }

        .anomaly-table th {
            color: #ffffff;
            text-align: left;
            padding: 15px 18px;
            font-weight: 800;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .anomaly-table td {
            padding: 15px 18px;
            border-bottom: 1px solid #ffedd5;
            vertical-align: top;
            color: #111827;
        }

        .anomaly-table tr:nth-child(even) {
            background: #fff7ed;
        }

        .anomaly-table tr:hover {
            background: #ffedd5;
        }

        .time-col {
            width: 22%;
            font-weight: 700;
            color: #292524;
        }

        .tags-col {
            width: 48%;
            line-height: 1.5;
        }

        .tags-col ul {
            margin: 0;
            padding-left: 18px;
        }

        .tags-col li {
            margin-bottom: 4px;
        }

        .error-col {
            width: 18%;
            font-weight: 800;
            color: #9a3412;
        }

        .severity-col {
            width: 12%;
        }

        .severity-pill {
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 800;
            display: inline-block;
            border: 1px solid transparent;
        }

        .severity-pill.high {
            background-color: #111827;
            color: #fb923c;
            border-color: #fb923c;
        }

        .severity-pill.medium {
            background-color: #fed7aa;
            color: #9a3412;
            border-color: #fdba74;
        }

        .severity-pill.low {
            background-color: #ffedd5;
            color: #c2410c;
            border-color: #fed7aa;
        }

        .severity-pill.unknown {
            background-color: #e5e7eb;
            color: #374151;
        }

        .llm-card {
            border-left: 6px solid #ea580c;
            background: #ffffff;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 16px;
            border-top: 1px solid #fed7aa;
            border-right: 1px solid #fed7aa;
            border-bottom: 1px solid #fed7aa;
            box-shadow: 0 8px 22px rgba(234, 88, 12, 0.10);
            font-size: 14px;
            min-height: 330px;
        }

        .llm-card p {
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .card-header-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 8px;
        }

        .stButton > button {
            background: #111827;
            color: white;
            border-radius: 12px;
            border: 1px solid #ea580c;
            font-weight: 700;
        }

        .stButton > button:hover {
            background: #ea580c;
            color: white;
            border-color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    header_left, header_right = st.columns([5, 1])

    with header_left:
        st.markdown("<div class='main-title'>Process Monitoring Dashboard</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Real-time anomaly detection using LSTM Autoencoder + Ollama LLM</div>",
            unsafe_allow_html=True
        )

    with header_right:
        st.write("")
        st.write("")
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        inference_df, ollama_df = load_data()
    except Exception as e:
        st.error(f"Could not load data from S3: {e}")
        st.stop()

    if inference_df.empty:
        st.warning("No anomaly data found.")
        st.stop()

    inference_df, start_time, latest_time = filter_last_10_hours(inference_df)
    ollama_df, _, _ = filter_last_10_hours(ollama_df)

    if inference_df.empty:
        st.warning("No anomaly data found in the last 10 hours.")
        st.stop()

    inference_df = prepare_inference_df(inference_df)
    inference_df = inference_df.sort_values("time_stamp", ascending=False)

    time_window = (
        f"{start_time.strftime('%b %d, %Y %I:%M %p')} to {latest_time.strftime('%b %d, %Y %I:%M %p')}"
        if start_time is not None and latest_time is not None
        else "Unavailable"
    )

    st.caption(f"Showing anomalies from latest entry minus 10 hours: {time_window}")

    total_anomalies = len(inference_df)
    avg_error = inference_df["reconstruction_error"].mean() if "reconstruction_error" in inference_df.columns else 0
    high_count = int((inference_df["severity"] == "High").sum())
    medium_count = int((inference_df["severity"] == "Medium").sum())
    low_count = int((inference_df["severity"] == "Low").sum())

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Anomalies", total_anomalies)
    k2.metric("High", high_count)
    k3.metric("Medium", medium_count)
    k4.metric("Low", low_count)
    k5.metric("Avg Error", f"{avg_error:.4f}")

    st.divider()

    st.subheader("Latest Anomalies Detected by Autoencoder")
    render_latest_anomalies_table(inference_df)

    st.divider()

    st.subheader("LLM Root Cause Insights")
    render_llm_cards(ollama_df)

    st.divider()

    with st.expander("Process Flow Diagram (PFD)", expanded=True):
        try:
            st.image(PFD_IMAGE_PATH, width="stretch")
        except Exception:
            st.warning("PFD image not found. Please check PFD_IMAGE_PATH.")


if __name__ == "__main__":
    main()