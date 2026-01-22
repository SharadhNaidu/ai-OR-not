import json
import math
from pathlib import Path

import config
import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer

from nltk.tokenize import sent_tokenize, word_tokenize


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover
        nltk.download("punkt")


def calculate_stylometric_features(text: str) -> dict[str, float]:
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        if not words:
            return {"avg_sent_len": 0.0, "type_token_ratio": 0.0, "punct_density": 0.0}

        sent_lengths = [len(word_tokenize(sent)) for sent in sentences if word_tokenize(sent)]
        avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        type_token_ratio = float(len(set(words)) / len(words))
        punct_chars = sum(char in ".,!?;:" for char in text)
        punct_density = float(punct_chars / max(len(text), 1))
        return {
            "avg_sent_len": avg_sent_len,
            "type_token_ratio": type_token_ratio,
            "punct_density": punct_density,
        }
    except LookupError:  # pragma: no cover
        return {"avg_sent_len": 0.0, "type_token_ratio": 0.0, "punct_density": 0.0}


def _load_tokenizer():
    tokenizer = None
    errors = []
    for use_fast in (True, False):
        if tokenizer is not None:
            break
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.MICRO_BACKBONE, use_fast=use_fast)
        except Exception as exc:  # pragma: no cover
            errors.append(("AutoTokenizer", use_fast, exc))
            tokenizer = None

    if tokenizer is None and "deberta" in config.MICRO_BACKBONE.lower():
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(config.MICRO_BACKBONE)
        except Exception as exc:  # pragma: no cover
            errors.append(("DebertaV2Tokenizer", None, exc))

    if tokenizer is None:
        raise RuntimeError(f"Failed to construct tokenizer for {config.MICRO_BACKBONE}: {errors}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    return tokenizer


def get_calibrated_probs(logits: torch.Tensor, temperature: float) -> np.ndarray:
    temperature = float(temperature) if float(temperature) > 0 else 1.0
    probs = softmax(logits / temperature, dim=1)
    return probs[:, 1].cpu().numpy()


def get_specialist_predictions_for_doc(
    doc_text: str,
    model: AutoModelForSequenceClassification,
    tokenizer,
    temperature: float,
) -> dict[str, float]:
    tokenized = tokenizer(doc_text, truncation=False, padding=False, add_special_tokens=False)
    input_ids = tokenized.get("input_ids", [])
    if not input_ids:
        return {"prob_mean": 0.0, "prob_max": 0.0, "prob_std": 0.0}

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    chunk_ids: list[list[int]] = []
    chunk_masks: list[list[int]] = []

    start = 0
    while start < len(input_ids):
        end = start + config.CHUNK_SIZE - 2
        chunk = [cls_id] + input_ids[start:end] + [sep_id]
        chunk = chunk[: config.CHUNK_SIZE]
        attention = [1] * len(chunk)

        if len(chunk) < config.CHUNK_SIZE:
            padding_length = config.CHUNK_SIZE - len(chunk)
            chunk += [pad_id] * padding_length
            attention += [0] * padding_length

        chunk_ids.append(chunk)
        chunk_masks.append(attention[: config.CHUNK_SIZE])

        if end >= len(input_ids):
            break
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP

    if not chunk_ids:
        return {"prob_mean": 0.0, "prob_max": 0.0, "prob_std": 0.0}

    logits = []
    batch_size = 16
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(chunk_ids), batch_size):
            batch_input = torch.tensor(chunk_ids[idx : idx + batch_size], dtype=torch.long, device=config.DEVICE)
            batch_mask = torch.tensor(chunk_masks[idx : idx + batch_size], dtype=torch.long, device=config.DEVICE)
            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            logits.append(outputs.logits.cpu())

    logits_tensor = torch.cat(logits, dim=0)
    chunk_probs = get_calibrated_probs(logits_tensor, temperature)

    return {
        "prob_mean": float(np.mean(chunk_probs)),
        "prob_max": float(np.max(chunk_probs)),
        "prob_std": float(np.std(chunk_probs)),
    }


class AiOrNotPredictor:
    def __init__(self):
        print("Loading all models and artifacts...")
        self.tokenizer = _load_tokenizer()
        self.specialist_models: dict[str, AutoModelForSequenceClassification] = {}

        for spec_name in config.FINAL_SPECIALISTS:
            model_path = Path(f"micro_{spec_name}")
            if not model_path.exists():
                raise FileNotFoundError(f"Required specialist model {model_path} not found.")
            print(f"Loading {model_path}...")
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(config.DEVICE)
            model.eval()
            self.specialist_models[spec_name] = model

        with open(config.CALIBRATION_SCALARS_PATH, "r", encoding="utf-8") as f:
            self.temperatures = json.load(f)

        self.macro_classifier = joblib.load(config.MACRO_CLASSIFIER_PATH)
        self.macro_regressor = joblib.load(config.MACRO_REGRESSOR_PATH)
        self.conformal_residuals = joblib.load(config.CONFORMAL_RESIDUALS_PATH)

        tail_prob = 1 - config.PERCENT_AI_CONFIDENCE
        q_level = 1 - tail_prob / 2.0
        self.conformal_quantile = float(np.quantile(self.conformal_residuals, q_level))
        print("All models loaded successfully.")

        self.feature_order = self._build_feature_order()

    def _build_feature_order(self) -> list[str]:
        prob_cols = []
        for spec_name in config.FINAL_SPECIALISTS:
            prob_cols.extend(
                [
                    f"{spec_name}_prob_mean",
                    f"{spec_name}_prob_max",
                    f"{spec_name}_prob_std",
                ]
            )
        stylo_cols = ["avg_sent_len", "type_token_ratio", "punct_density"]
        return prob_cols + stylo_cols

    def _get_features(self, text: str) -> dict[str, float]:
        features: dict[str, float] = {}

        for spec_name, model in self.specialist_models.items():
            temperature = self.temperatures.get(spec_name, 1.0)
            preds = get_specialist_predictions_for_doc(text, model, self.tokenizer, temperature)
            features[f"{spec_name}_prob_mean"] = preds["prob_mean"]
            features[f"{spec_name}_prob_max"] = preds["prob_max"]
            features[f"{spec_name}_prob_std"] = preds["prob_std"]

        features.update(calculate_stylometric_features(text))
        return features

    def predict(self, text: str) -> dict[str, float]:
        features_dict = self._get_features(text)
        features_df = pd.DataFrame([features_dict], columns=self.feature_order)

        clf_proba = self.macro_classifier.predict_proba(features_df)[0, 1]
        clf_pred = int(clf_proba >= 0.5)

        reg_pred = float(self.macro_regressor.predict(features_df)[0])
        reg_pred = float(np.clip(reg_pred, 0.0, 100.0))
        interval = self.conformal_quantile
        ci_low = max(0.0, reg_pred - interval)
        ci_high = min(100.0, reg_pred + interval)

        return {
            "prediction": clf_pred,
            "probability_ai": clf_proba,
            "percent_ai": reg_pred,
            "percent_ai_ci_low": ci_low,
            "percent_ai_ci_high": ci_high,
        }


def main():
    print("--- [Step 5] START: Final Evaluation on Test Set ---")
    _ensure_nltk()

    predictor = AiOrNotPredictor()

    text_column = getattr(config, "TEXT_COLUMN", "text")

    try:
        test_dataset = load_from_disk(config.TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: {config.TEST_DATA_PATH} not found. Run 01_ingest_data.py first.")
        return

    test_df = test_dataset.to_pandas()
    if text_column not in test_df.columns:
        raise KeyError(f"Column '{text_column}' not present in test dataset.")

    predictions = []
    for text in tqdm(test_df[text_column], desc="Running final evaluation"):
        predictions.append(predictor.predict(text))

    pred_df = pd.DataFrame(predictions)
    results_df = pd.concat([test_df.reset_index(drop=True), pred_df], axis=1)

    y_true = results_df["binary_label"]
    y_pred = results_df["prediction"]

    print("\n" + "=" * 50)
    print("     FINAL EVALUATION REPORT (held-out test.arrow)")
    print("=" * 50 + "\n")

    print("--- [A] Overall Classifier Performance ---")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

    print("--- [B] Robustness Check (Unseen AI Model: omni_unseen) ---")
    unseen_df = results_df[results_df["specialist_label"] == "omni_unseen"]
    if not unseen_df.empty:
        unseen_acc = accuracy_score(unseen_df["binary_label"], unseen_df["prediction"])
        unseen_f1 = f1_score(unseen_df["binary_label"], unseen_df["prediction"])
        print(f"  Accuracy: {unseen_acc:.4f}")
        print(f"  F1-Score: {unseen_f1:.4f}")
    else:
        print("  No 'omni_unseen' samples present in the held-out set.")

    print("--- [C] Percent-AI Regressor Performance ---")
    results_df["percent_ai_true"] = results_df["binary_label"] * 100.0
    reg_mae = mean_absolute_error(results_df["percent_ai_true"], results_df["percent_ai"])
    print(f"  MAE on {len(results_df)} samples (0/100 labels): {reg_mae:.4f}")

    print("\n" + "=" * 50)
    print("--- [Step 5] COMPLETE ---")


if __name__ == "__main__":
    main()
