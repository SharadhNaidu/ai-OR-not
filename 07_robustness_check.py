import importlib.util
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config


def _load_predictor_class():
    module_path = Path(__file__).resolve().parent / "06_evaluate_and_predict.py"
    if not module_path.exists():
        raise FileNotFoundError("06_evaluate_and_predict.py not found; cannot load AiOrNotPredictor.")

    spec = importlib.util.spec_from_file_location("evaluate_and_predict_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for type checkers
    spec.loader.exec_module(module)

    if not hasattr(module, "AiOrNotPredictor"):
        raise AttributeError("AiOrNotPredictor class not found in 06_evaluate_and_predict.py")

    return module.AiOrNotPredictor


def main():
    print("--- [Step 6] START: Final Robustness Check ---")

    print(f"Loading data from disk: {config.TRAIN_DATA_PATH}")
    try:
        full_train_data = load_from_disk(config.TRAIN_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: {config.TRAIN_DATA_PATH} not found. Run 01_ingest_data.py first.")
        return

    unseen_data = full_train_data.filter(
        lambda x: x["specialist_label"] == "omni_unseen",
        desc="Filtering for 'omni_unseen' data",
    )

    if len(unseen_data) == 0:
        print("No 'omni_unseen' data was found in train.arrow. Cannot run robustness test.")
        return

    print(f"Found {len(unseen_data)} 'omni_unseen' samples to test against.")
    unseen_df = unseen_data.to_pandas()

    text_column = getattr(config, "TEXT_COLUMN", "text")
    if text_column not in unseen_df.columns:
        raise KeyError(f"Column '{text_column}' not found in omni_unseen dataframe.")

    AiOrNotPredictor = _load_predictor_class()

    try:
        predictor = AiOrNotPredictor()
    except Exception as exc:
        print(f"Failed to load AiOrNotPredictor: {exc}")
        return

    predictions = []
    for text in tqdm(unseen_df[text_column], desc="Running robustness test"):
        predictions.append(predictor.predict(text))

    pred_df = pd.DataFrame(predictions)
    true_labels = [1] * len(unseen_df)
    pred_labels = pred_df["prediction"].tolist()

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print("\n" + "=" * 50)
    print("     FINAL ROBUSTNESS REPORT (on 'omni_unseen' models)")
    print("=" * 50 + "\n")
    print(f"  - Total 'omni_unseen' samples tested: {len(true_labels)}")
    print(f"  - Accuracy (Correctly ID'd as AI): {accuracy * 100:.2f}%")
    print(f"  - F1-Score: {f1:.4f}")
    print("\n" + "=" * 50)
    print("--- [Step 6] Robustness check complete. ---")


if __name__ == "__main__":
    main()
