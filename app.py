"""Streamlit evaluation dashboard for the saved hypothyroid classifiers."""

from pathlib import Path
from html import escape

import joblib
import traceback
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_models import MODEL_FILES, TARGET_COLUMN, encode_target, prepare_features


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
# Keep the bundled holdout scores aligned with the README's fixed-split observations.
README_HOLDOUT_METRICS = {
    "Logistic Regression": [0.9735, 0.9837, 0.7568, 0.9655, 0.8485, 0.8416],
    "Decision Tree": [0.9974, 0.9907, 0.9828, 0.9828, 0.9828, 0.9813],
    "K-Nearest Neighbors": [0.9523, 0.8090, 0.9583, 0.3966, 0.5610, 0.5998],
    "Gaussian Naive Bayes": [0.2040, 0.6223, 0.0868, 0.9828, 0.1594, 0.0966],
    "Support Vector Machine": [0.9709, 0.9625, 0.9737, 0.6379, 0.7708, 0.7753],
}
# Use the README wording only when the uploaded data reproduces its holdout scores.
README_OBSERVATIONS = {
    "Logistic Regression": (
        "High recall (0.9655) and AUC (0.9837), with lower precision (0.7568); "
        "balanced weights catch most positive cases but include more false positives."
    ),
    "Decision Tree": (
        "Best overall holdout results: 0.9974 accuracy, 0.9907 AUC, 0.9828 recall, "
        "and 0.9813 MCC. It balances minority-class detection and precision particularly "
        "well."
    ),
    "K-Nearest Neighbors": (
        "Excellent precision (0.9583) but recall is only 0.3966, so it misses many "
        "hypothyroid cases at the default decision threshold."
    ),
    "Gaussian Naive Bayes": (
        "Recall is high (0.9828), but precision (0.0868) and accuracy (0.2040) are poor, "
        "showing that its predictions produce many false positives on this dataset."
    ),
    "Support Vector Machine": (
        "The calibrated, balanced SVM has high precision (0.9737), but recall is 0.6379; "
        "it misses more positive cases than Logistic Regression and the Decision Tree."
    ),
}
README_WINNER_OBSERVATION = (
    "Decision Tree, which achieved the strongest accuracy, AUC, precision, F1, and MCC "
    "on the fixed stratified holdout while retaining 0.9828 recall. It balances "
    "minority-class detection and precision particularly well."
)

st.set_page_config(page_title="Hypothyroid Classifier Evaluation", page_icon="🧪", layout="wide")
# Apply the dashboard's responsive clinical-lab theme and readable table styles.
st.markdown(
    """
    <style>
    :root {
        --ink: #1d302d;
        --forest: #20463f;
        --sage: #e8efea;
        --paper: #f4f6f3;
        --line: #d7dfd9;
        --rust: #b65335;
        --muted: #64736e;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244, 246, 243, .94); }
    .block-container { padding: 2rem 2.4rem 3rem; max-width: 1440px; animation: page-arrive .35s ease-out both; }
    @keyframes page-arrive { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) { .block-container { animation: none; } }
    .stApp h1 { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-size: 2.65rem; font-weight: 600; line-height: 1.1; letter-spacing: 0; margin: .15rem 0 .4rem; }
    .stApp h2, .stApp h3 { color: var(--forest); font-family: 'Trebuchet MS', sans-serif; letter-spacing: 0; }
    .stApp h3 { font-size: 1.12rem; font-weight: 700; }
    .stCaption { color: var(--muted); }
    [data-testid="stMetric"] { background: #fff; padding: 16px 18px; border: 1px solid var(--line); border-top: 3px solid var(--rust); border-radius: 6px; box-shadow: 0 2px 8px rgba(29, 48, 45, .04); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    div[data-testid="stFileUploader"] { background: #fff; border: 1px dashed #9aada2; border-radius: 7px; padding: 12px; }
    div[data-baseweb="select"] > div { background: #fff; border-color: #bfcac3; border-radius: 6px; }
    .stDownloadButton button { border-radius: 6px; border: 1px solid var(--forest); color: var(--forest); background: #fff; font-weight: 600; }
    .stDownloadButton button:hover { border-color: var(--rust); color: var(--rust); }
    [data-testid="stAlert"] { border-radius: 6px; }
    .observations-table { width: 100%; max-width: 100%; table-layout: fixed; border-collapse: collapse; background: #fff; }
    .observations-table th, .observations-table td {
        border: 1px solid var(--line); padding: 12px 14px; text-align: left; vertical-align: top;
        white-space: normal; overflow-wrap: anywhere; word-break: normal;
    }
    .observations-table th:first-child, .observations-table td:first-child { width: 27%; }
    .observations-table th { background: var(--forest); color: #fff; font: 700 .82rem 'Trebuchet MS', sans-serif; }
    .observations-table tbody tr:nth-child(even) { background: #f7f9f7; }
    .winner-note { margin-top: 14px; padding: 14px 16px; background: var(--sage); border-left: 4px solid var(--rust); border-radius: 4px; color: var(--ink); line-height: 1.55; }
    .winner-note strong { color: var(--forest); }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .stApp h1 { font-size: 2rem; }
        .observations-table th, .observations-table td { padding: 9px 8px; font-size: .88rem; }
        .observations-table th:first-child, .observations-table td:first-child { width: 34%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    # Load saved preprocessing and estimators once, then reuse them across reruns.
    preprocessing = joblib.load(MODEL_DIR / "preprocessor.joblib")
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return preprocessing, models


def score(model, features, labels):
    # Calculate class predictions and probability-based evaluation metrics.
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    try:
        auc = roc_auc_score(labels, probabilities) if np.unique(labels).size == 2 else np.nan
    except ValueError:
        auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }
    return metrics, predictions


def describe_performance(name, metrics):
    # Summarize recall, precision, AUC, and MCC for nonstandard uploaded datasets.
    recall = metrics["Recall"]
    precision = metrics["Precision"]
    mcc = metrics["MCC"]

    if recall >= 0.9:
        recall_note = "catches nearly all positive cases"
    elif recall >= 0.75:
        recall_note = "has strong positive-case detection"
    elif recall >= 0.5:
        recall_note = "has moderate positive-case detection"
    else:
        recall_note = "misses many positive cases"

    if precision >= 0.9:
        precision_note = "positive predictions are highly reliable"
    elif precision >= 0.7:
        precision_note = "positive predictions are reasonably reliable"
    elif precision >= 0.4:
        precision_note = "positive predictions include many false alarms"
    else:
        precision_note = "positive predictions include many false alarms"

    if mcc >= 0.8:
        balance_note = "MCC indicates strong overall balance"
    elif mcc >= 0.5:
        balance_note = "MCC indicates useful but imperfect class balance"
    else:
        balance_note = "MCC indicates weak overall class balance"

    auc = metrics["AUC"]
    auc_text = "not available" if pd.isna(auc) else f"{auc:.3f}"
    return (
        f"Recall {recall:.3f}: {recall_note}; precision {precision:.3f}: "
        f"{precision_note}. AUC {auc_text}; {balance_note} ({mcc:.3f})."
    )


# Present the evaluation workflow and stop early if the training artifacts are unavailable.
st.title("Hypothyroid Classifier Evaluation")
st.caption("Hypothyroid screening | Metrics | Confusion Matrices | Reports for Five classifiers")

if not (MODEL_DIR / "preprocessor.joblib").exists():
    st.error("Model artifacts are missing. Run `py train_models.py` in the project folder first.")
    st.stop()

preprocessing, models = load_artifacts()
# Collect the labeled CSV and model choice in the input panel.
left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Evaluation data")
    uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "Download held-out test CSV",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    model_choice = st.selectbox("Classifier", ["Compare all", *models.keys()])

if uploaded is None:
    # Explain that test labels are required before calculating evaluation metrics.
    with right:
        if model_choice != "Compare all":
            st.warning(
                f"Upload a labeled test CSV before evaluating {model_choice}. "
                "The selected classifier will be applied after upload."
            )
        else:
            st.info(
                "Upload a labeled CSV to calculate test metrics. "
                "The included held-out CSV is available above."
            )
    st.stop()

try:
    # Read the uploaded test set and confirm it has every model input and its target.
    test_data = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

missing_columns = [
    column for column in [*preprocessing["raw_feature_names"], TARGET_COLUMN]
    if column not in test_data.columns
]
if missing_columns:
    st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
    st.stop()
if test_data.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

try:
    # Reuse training-time imputers, encoders, feature order, and scaler for evaluation.
    labels = encode_target(test_data[TARGET_COLUMN])
    
    # FIX: Create an independent mutable copy of the features slice
    features_slice = test_data[preprocessing["raw_feature_names"]].copy()
    
    encoded, _ = prepare_features(features_slice, preprocessing)
    
    # Ensure the matrix buffer is explicitly marked as writable before scaling
    if hasattr(encoded, 'values'):
        encoded.values.flags.writeable = True
    elif isinstance(encoded, np.ndarray):
        encoded.flags.writeable = True
        
    scaled = preprocessing["scaler"].transform(encoded)
except Exception as error:
    st.error(f"Could not preprocess this CSV: {error}")
    with st.expander("See full error logs and stacktrace"):
        st.code(traceback.format_exc(), language="python")
    st.stop()

results = {}
predictions_by_model = {}
# Score every saved classifier so comparison mode can display all five results.
for name, model in models.items():
    results[name], predictions_by_model[name] = score(model, scaled, labels)

winner = max(
    results,
    key=lambda name: (
        results[name]["MCC"],
        results[name]["Recall"],
        results[name]["F1"],
    ),
)
# Detect whether this upload is the documented holdout before using README observations.
matches_readme_holdout = all(
    [round(results[name][metric], 4) for metric in METRIC_COLUMNS]
    == README_HOLDOUT_METRICS[name]
    for name in models
)
display_names = {
    "K-Nearest Neighbors": "KNN",
    "Gaussian Naive Bayes": "Naive Bayes",
}
observation_rows = [
    {
        "ML Model Name": display_names.get(name, name),
        "Observation about model performance": (
            README_OBSERVATIONS[name]
            if matches_readme_holdout
            else describe_performance(name, metrics)
        ),
    }
    for name, metrics in results.items()
]

# Show the uploaded records that are being evaluated.
with right:
    st.subheader("Test set")
    st.write(f"{len(test_data):,} labeled records | {int(labels.sum()):,} positive cases")
    st.dataframe(test_data.head(8), use_container_width=True, hide_index=True)

st.subheader("Model comparison")
# Present all six metrics side by side for the selected test CSV.
result_frame = pd.DataFrame(results).T.reindex(columns=METRIC_COLUMNS)
st.dataframe(
    result_frame.style.format("{:.4f}", na_rep="N/A").highlight_max(
        axis=0, subset=METRIC_COLUMNS, color="#d8e8dc"
    ),
    use_container_width=True,
)

if model_choice != "Compare all":
    metric_values = results[model_choice]
    metric_columns = st.columns(len(METRIC_COLUMNS))
    for column, metric_name in zip(metric_columns, METRIC_COLUMNS):
        value = metric_values[metric_name]
        column.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.3f}")

st.subheader("Model observations")
# Escape dynamic text before placing it into the responsive HTML table.
observation_table = [
    "<table class='observations-table'><thead><tr>"
    "<th>ML Model Name</th><th>Observation about model performance</th>"
    "</tr></thead><tbody>"
]
for row in observation_rows:
    observation_table.append(
        f"<tr><td>{escape(row['ML Model Name'])}</td>"
        f"<td>{escape(row['Observation about model performance'])}</td></tr>"
    )
observation_table.append("</tbody></table>")
st.markdown("".join(observation_table), unsafe_allow_html=True)
winner_observation = (
    README_WINNER_OBSERVATION
    if matches_readme_holdout
    else f"{winner}, selected by highest MCC ({results[winner]['MCC']:.3f}); recall and F1 break ties."
)
# Give the winning model a distinct callout below the observations table.
st.markdown(
    f"<div class='winner-note'><strong>Overall, winner for this dataset:</strong> "
    f"{escape(winner_observation)}</div>",
    unsafe_allow_html=True,
)

st.subheader("Classification detail")
# Open diagnostics for every model in comparison mode, or just the chosen model.
detail_models = list(models) if model_choice == "Compare all" else [model_choice]
for detail_model in detail_models:
    with st.expander(
        f"{detail_model} · Confusion matrix and classification report",
        expanded=True,
    ):
        detail_columns = st.columns([1, 1], gap="large")
        with detail_columns[0]:
            st.markdown("**Confusion matrix**")
            matrix = confusion_matrix(labels, predictions_by_model[detail_model], labels=[0, 1])
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual negative", "Actual positive"],
                    columns=["Predicted negative", "Predicted positive"],
                ),
                use_container_width=True,
            )
        with detail_columns[1]:
            st.markdown("**Classification report**")
            report = classification_report(
                labels,
                predictions_by_model[detail_model],
                labels=[0, 1],
                target_names=["Negative", "Hypothyroid"],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)