"""Train the five classifiers and save artifacts for Streamlit."""

from __future__ import annotations  # Enable modern type annotations without runtime evaluation.

import json  # Serialize the measured model scores for the app and README.
from pathlib import Path  # Build paths relative to this project file.

import joblib  # Save fitted models and preprocessing artifacts.
import numpy as np  # Handle arrays and indexed train/test rows.
import pandas as pd  # Load and transform the tabular dataset.
from sklearn.calibration import CalibratedClassifierCV  # Calibrate SVM scores into probabilities.
from sklearn.metrics import (
    accuracy_score,  # Measure the fraction of correct predictions.
    matthews_corrcoef,  # Measure balanced correlation between labels and predictions.
    precision_score,  # Measure the fraction of positive predictions that are correct.
    recall_score,  # Measure the fraction of actual positive cases detected.
    f1_score,  # Combine precision and recall into one score.
    roc_auc_score,  # Measure ranking quality across classification thresholds.
)
from sklearn.model_selection import train_test_split  # Create a stratified holdout split.
from sklearn.naive_bayes import GaussianNB  # Provide the Gaussian Naive Bayes classifier.
from sklearn.neighbors import KNeighborsClassifier  # Provide the nearest-neighbor classifier.
from sklearn.preprocessing import StandardScaler  # Scale features for distance-based models.
from sklearn.svm import SVC  # Provide the support vector classifier used by calibration.
from sklearn.linear_model import LogisticRegression  # Provide the linear classifier.
from sklearn.tree import DecisionTreeClassifier  # Provide the tree-based classifier.


ROOT = Path(__file__).resolve().parent  # Locate project assets independently of the working directory.
MODEL_DIR = ROOT / "model"  # Keep all serialized estimators and metadata together.
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/"  # Base URL for the public UCI files.
COLUMN_NAMES = [  # Name the fields because the raw UCI files have no header row.
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",  # Demographics and treatment flags.
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",  # Medical history and query flags.
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",  # Additional clinical indicators.
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",  # TSH, T3, and TT4 measurements.
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",  # Remaining laboratory values.
    "referral_source", "class",  # Referral category and target label.
]
NUMERIC_COLUMNS = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]  # Numeric fields to coerce and impute.
DROP_COLUMNS = ["TBG", "TBG_measured"]  # Remove fields that do not provide usable signal.
TARGET_COLUMN = "class"  # Name of the diagnosis label in the source files.
MODEL_FILES = {  # Map display names to saved estimator filenames.
    "Logistic Regression": "logistic_regression.joblib",  # Stored logistic regression model.
    "Decision Tree": "decision_tree.joblib",  # Stored decision tree model.
    "K-Nearest Neighbors": "knn.joblib",  # Stored KNN model.
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",  # Stored Gaussian Naive Bayes model.
    "Support Vector Machine": "support_vector_machine.joblib",  # Stored calibrated SVM model.
}


def load_dataset() -> pd.DataFrame:  # Download, combine, and normalize the raw UCI records.
    """Load the UCI allhypo train and test files used for hypothyroid classification."""
    frames = [  # Read both the training and supplied test portions of the UCI data.
        pd.read_csv(f"{UCI_BASE}{filename}", header=None, names=COLUMN_NAMES)  # Attach the declared column names.
        for filename in ("allhypo.data", "allhypo.test")  # Include both published data files.
    ]
    data = pd.concat(frames, ignore_index=True)  # Combine rows and reset the row index.
    data[TARGET_COLUMN] = (  # Remove record IDs and punctuation attached to diagnosis labels.
        data[TARGET_COLUMN].astype(str).str.split("|").str[0].str.rstrip(".").str.strip()  # Normalize each target value.
    )
    return data  # Return the cleaned source table.


def encode_target(values: pd.Series) -> np.ndarray:  # Convert diagnosis categories into binary labels.
    """Map the three hypothyroid diagnoses to positive."""
    labels = values.astype(str).str.strip().str.lower()  # Normalize string labels for comparison.
    numeric = pd.to_numeric(labels, errors="coerce")  # Also support test files with numeric targets.
    is_numeric = numeric.notna()  # Mark labels that parsed as numbers.
    result = labels.ne("negative").astype(int).to_numpy().copy()  # Treat every hypothyroid subtype as positive.
    if is_numeric.any():  # Override the default mapping for numeric target values.
        result[is_numeric.to_numpy()] = (numeric[is_numeric] != 0).astype(int).to_numpy()  # Map zero to negative and nonzero to positive.
    return result  # Return labels in the order supplied.


def prepare_features(  # Apply the same feature transformations during training and app inference.
    raw_features: pd.DataFrame,  # Unprocessed input columns from the source CSV.
    state: dict | None = None,  # Saved training statistics, or None when fitting preprocessing.
) -> tuple[pd.DataFrame, dict]:  # Return encoded features and reusable preprocessing state.
    """Clean and encode source features with stable columns for model inference."""
    features = raw_features.copy()  # Avoid modifying the caller's dataframe.
    features = features.where(features != "?", np.nan)  # Convert UCI missing-value markers to NaN.
    features = features.drop(columns=DROP_COLUMNS, errors="ignore")  # Remove unusable TBG columns if present.

    if state is None:  # Learn imputation and category settings from training data.
        for column in NUMERIC_COLUMNS:  # Convert each numeric lab/demographic field.
            features[column] = pd.to_numeric(features[column], errors="coerce")  # Turn invalid numbers into missing values.
        features.loc[features["age"] > 120, "age"] = np.nan  # Treat implausible ages as missing.
        numeric_medians = features[NUMERIC_COLUMNS].median().to_dict()  # Store training medians for later imputation.

        tf_columns = [  # Identify binary clinical fields encoded as t/f.
            column  # Keep the current candidate column.
            for column in features.columns  # Examine each remaining source field.
            if column not in ("sex", "referral_source")  # Exclude fields with their own encoding logic.
            and features[column].dropna().size  # Ignore columns without observed values.
            and features[column].dropna().isin(["t", "f"]).all()  # Require all observed values to be binary flags.
        ]
        sex_mode = features["sex"].map({"M": 0, "F": 1}).mode().iloc[0]  # Learn the most frequent sex encoding for imputation.
    else:  # Reuse training-time preprocessing values for uploaded data.
        numeric_medians = state["numeric_medians"]  # Restore learned numeric medians.
        tf_columns = state["tf_columns"]  # Restore the training binary field list.
        sex_mode = state["sex_mode"]  # Restore the training sex imputation value.
        for column in NUMERIC_COLUMNS:  # Convert uploaded numeric fields consistently.
            features[column] = pd.to_numeric(features[column], errors="coerce")  # Coerce invalid entries to missing values.
        features.loc[features["age"] > 120, "age"] = np.nan  # Apply the same implausible-age rule.

    features[NUMERIC_COLUMNS] = features[NUMERIC_COLUMNS].fillna(numeric_medians)  # Fill numeric gaps using saved or learned medians.
    binary_modes = {}  # Hold binary imputation values learned during training.
    for column in tf_columns:  # Encode and impute every binary clinical flag.
        values = features[column].map({"t": 1, "f": 0})  # Convert true/false strings into zero/one.
        if state is None:  # Learn this flag's mode from training rows.
            mode = pd.to_numeric(values, errors="coerce").mode()  # Find the most common encoded value.
            fill_value = mode.iloc[0] if not mode.empty else 0  # Use zero only if the column has no values.
            binary_modes[column] = int(fill_value)  # Save the learned value for future uploads.
        else:  # Reuse the saved mode during inference.
            fill_value = state["binary_modes"].get(column, 0)  # Default safely if a stored mode is absent.
        features[column] = pd.to_numeric(values, errors="coerce").fillna(fill_value).astype(int)  # Encode, impute, and cast the flag.

    features["sex"] = features["sex"].map({"M": 0, "F": 1})  # Encode sex categories as binary values.
    features["sex"] = pd.to_numeric(features["sex"], errors="coerce").fillna(sex_mode).astype(int)  # Impute missing sex values and cast to integers.

    features = pd.get_dummies(features, columns=["referral_source"], drop_first=True, dtype=int)  # One-hot encode referral categories.
    if state is None:  # Save every learned transformation setting.
        dummy_columns = [name for name in features.columns if name.startswith("referral_source_")]  # Record referral indicator columns.
        state = {  # Bundle preprocessing data so training and inference stay aligned.
            "numeric_medians": numeric_medians,  # Save numeric imputation values.
            "tf_columns": tf_columns,  # Save binary input field names.
            "binary_modes": binary_modes,  # Save binary imputation values.
            "sex_mode": int(sex_mode),  # Save the encoded sex imputation value.
            "dummy_columns": dummy_columns,  # Save generated referral fields.
            "feature_names": features.columns.tolist(),  # Save model input order.
            "raw_feature_names": raw_features.columns.tolist(),  # Save required upload columns.
        }
    else:  # Align new data to the exact training feature layout.
        features = features.reindex(columns=state["feature_names"], fill_value=0)  # Add missing dummy columns and discard unknown extras.

    return features.astype(float), state  # Return numeric model inputs and reusable preprocessing metadata.


def build_models() -> dict:  # Create all classifiers with consistent settings.
    return {  # Return model instances keyed by their display names.
        "Logistic Regression": LogisticRegression(  # Configure weighted logistic regression.
            max_iter=2000, class_weight="balanced", random_state=42  # Allow convergence and compensate for class imbalance.
        ),
        "Decision Tree": DecisionTreeClassifier(  # Configure a class-balanced decision tree.
            class_weight="balanced", random_state=42  # Weight rare cases and make tree fitting repeatable.
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),  # Use five neighbors weighted by distance.
        "Gaussian Naive Bayes": GaussianNB(),  # Fit the Gaussian conditional-feature baseline.
        "Support Vector Machine": CalibratedClassifierCV(  # Calibrate SVM outputs for probability-based AUC.
            estimator=SVC(class_weight="balanced", random_state=42),  # Fit a balanced-margin SVM.
            method="sigmoid",  # Use sigmoid calibration to estimate class probabilities.
            cv=5,  # Learn calibration from five stratified folds.
        ),
    }


def evaluate_model(model, features: np.ndarray, labels: np.ndarray) -> dict:  # Calculate holdout metrics for one fitted model.
    predictions = model.predict(features)  # Generate class predictions for metric calculations.
    probabilities = model.predict_proba(features)[:, 1]  # Get positive-class probabilities for AUC.
    return {  # Convert each metric to a JSON-serializable Python float.
        "Accuracy": float(accuracy_score(labels, predictions)),  # Score overall correctness.
        "AUC": float(roc_auc_score(labels, probabilities)),  # Score probability ranking quality.
        "Precision": float(precision_score(labels, predictions, zero_division=0)),  # Guard against no positive predictions.
        "Recall": float(recall_score(labels, predictions, zero_division=0)),  # Guard against no actual positive cases.
        "F1": float(f1_score(labels, predictions, zero_division=0)),  # Summarize precision/recall balance.
        "MCC": float(matthews_corrcoef(labels, predictions)),  # Include an imbalance-aware correlation score.
    }


def main() -> None:  # Train, evaluate, and persist the full model suite.
    MODEL_DIR.mkdir(exist_ok=True)  # Ensure the artifacts directory exists.
    data = load_dataset()  # Retrieve and combine the public UCI records.
    labels = encode_target(data[TARGET_COLUMN])  # Convert diagnosis categories to binary labels.
    raw_features = data.drop(columns=[TARGET_COLUMN])  # Separate inputs from the target.
    features, preprocessing = prepare_features(raw_features)  # Learn data cleaning and category settings.

    train_indices, test_indices = train_test_split(  # Split row indices to preserve raw test rows for export.
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels  # Keep an 80/20 reproducible class-stratified split.
    )
    scaler = StandardScaler()  # Prepare to standardize all model inputs.
    train_features = scaler.fit_transform(features.iloc[train_indices])  # Learn scaling only from training rows.
    test_features = scaler.transform(features.iloc[test_indices])  # Apply training scaling to held-out rows.
    preprocessing["scaler"] = scaler  # Save the fitted scaler for app inference.

    results = {}  # Collect the holdout metrics for every classifier.
    for name, model in build_models().items():  # Fit each model using the same training rows.
        model.fit(train_features, labels[train_indices])  # Train this classifier on the scaled training data.
        results[name] = evaluate_model(model, test_features, labels[test_indices])  # Evaluate it on the unseen split.
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])  # Save the fitted classifier for Streamlit.
        print(f"{name}: {results[name]}")  # Display the model's evaluation metrics.

    joblib.dump(preprocessing, MODEL_DIR / "preprocessor.joblib")  # Save the fitted preprocessing and feature schema.
    data.iloc[test_indices].drop(columns=[TARGET_COLUMN]).assign(  # Keep source-format test features for app upload.
        **{TARGET_COLUMN: labels[test_indices]}  # Add the binary ground-truth labels required for evaluation.
    ).to_csv(ROOT / "test_data.csv", index=False)  # Write the held-out labeled rows as a CSV.
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")  # Persist readable model metrics.

    print(f"\nSaved five models and preprocessing artifacts to {MODEL_DIR}")  # Report where trained artifacts were saved.
    print(f"Saved {len(test_indices)} labeled held-out rows to {ROOT / 'test_data.csv'}")  # Report the generated test CSV.


if __name__ == "__main__":  # Run training only when this file is executed directly.
    main()  # Start the training and artifact-generation workflow.