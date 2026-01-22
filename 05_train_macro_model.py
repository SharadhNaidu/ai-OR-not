import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def load_features():
    try:
        macro_df = pd.read_csv(config.OOF_FEATURES_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{config.OOF_FEATURES_PATH} not found. Run 04_create_macro_features.py first."
        )

    prob_cols = [
        col
        for col in macro_df.columns
        if "prob_mean" in col or "prob_max" in col or "prob_std" in col
    ]
    stylo_cols = ["avg_sent_len", "type_token_ratio", "punct_density"]
    features = prob_cols + stylo_cols
    return macro_df, features


def train_classifier(macro_df, features):
    print("\n--- Training Classifier ---")
    clf_df = macro_df.dropna(subset=features + ["binary_label"])

    if clf_df.empty:
        raise RuntimeError("No data available for classifier training after dropna().")

    X_clf = clf_df[features]
    y_clf = clf_df["binary_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clf,
        y_clf,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y_clf,
    )

    clf_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=config.RANDOM_SEED,
                    max_iter=1000,
                ),
            ),
        ]
    )

    print("Fitting classifier pipeline...")
    clf_pipeline.fit(X_train, y_train)

    preds = clf_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print("\nClassifier Evaluation (on hold-out split of macro data):")
    print(classification_report(y_test, preds))
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    joblib.dump(clf_pipeline, config.MACRO_CLASSIFIER_PATH)
    print(f"Classifier saved to: {config.MACRO_CLASSIFIER_PATH}")

    return acc, f1


def train_regressor(macro_df, features):
    print("\n--- Training Regressor ---")
    reg_df = macro_df[macro_df["specialist_label"] == "synthetic"].dropna(
        subset=features + ["percent_ai"]
    )

    if len(reg_df) < 100:
        print("Warning: Not enough synthetic data to train regressor. Skipping.")
        return None

    X_reg = reg_df[features]
    y_reg = reg_df["percent_ai"]

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_reg,
        y_reg,
        test_size=0.3,
        random_state=config.RANDOM_SEED,
    )

    reg_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(random_state=config.RANDOM_SEED))]
    )

    print("Fitting regressor pipeline...")
    reg_pipeline.fit(X_train, y_train)

    preds = reg_pipeline.predict(X_cal)
    mae = mean_absolute_error(y_cal, preds)
    print(f"\nRegressor MAE (on calibration split): {mae:.4f} (on 0-100 scale)")

    residuals = np.abs(y_cal - preds)

    joblib.dump(reg_pipeline, config.MACRO_REGRESSOR_PATH)
    joblib.dump(residuals, config.CONFORMAL_RESIDUALS_PATH)
    print(f"Regressor saved to: {config.MACRO_REGRESSOR_PATH}")
    print(f"Conformal residuals saved to: {config.CONFORMAL_RESIDUALS_PATH}")

    return mae


def main():
    print("--- [Step 4] START: Training Macro-Models (Classifier & Regressor) ---")

    try:
        macro_df, features = load_features()
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}")
        return

    print(f"Using {len(features)} features: {features}")

    classifier_metrics = train_classifier(macro_df, features)
    regressor_metric = train_regressor(macro_df, features)

    print("\n--- [Step 4] COMPLETE: Macro-models are trained and saved. ---")

    if classifier_metrics is not None:
        acc, f1 = classifier_metrics
        print(f"Classifier Accuracy: {acc:.4f}")
        print(f"Classifier F1: {f1:.4f}")

    if regressor_metric is not None:
        print(f"Regressor MAE: {regressor_metric:.4f}")


if __name__ == "__main__":
    main()
