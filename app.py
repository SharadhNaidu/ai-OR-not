import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer

import config


# --- Streamlit configuration -------------------------------------------------
st.set_page_config(
    page_title="Ai-or-NOT – Specialist Ensemble",
    page_icon="🤖",
    layout="wide",
)


# --- Utility helpers ---------------------------------------------------------
def _ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover
        nltk.download("punkt")


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
        raise RuntimeError(f"Failed to load tokenizer for {config.MICRO_BACKBONE}: {errors}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    return tokenizer


def calculate_stylometric_features(text: str) -> Dict[str, float]:
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


def get_calibrated_probs(logits: torch.Tensor, temperature: float) -> np.ndarray:
    temperature = float(temperature) if float(temperature) > 0 else 1.0
    return softmax(logits / temperature, dim=1)[:, 1].cpu().numpy()


def get_specialist_predictions_for_doc(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer,
    temperature: float,
) -> Dict[str, float]:
    tokenized = tokenizer(text, truncation=False, padding=False, add_special_tokens=False)
    input_ids = tokenized.get("input_ids", [])
    if not input_ids:
        return {"prob_mean": 0.0, "prob_max": 0.0, "prob_std": 0.0}

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    chunk_ids: List[List[int]] = []
    chunk_masks: List[List[int]] = []

    start = 0
    while start < len(input_ids):
        end = start + config.CHUNK_SIZE - 2
        chunk = [cls_id] + input_ids[start:end] + [sep_id]
        chunk = chunk[: config.CHUNK_SIZE]
        attention = [1] * len(chunk)

        if len(chunk) < config.CHUNK_SIZE:
            pad_length = config.CHUNK_SIZE - len(chunk)
            chunk += [pad_id] * pad_length
            attention += [0] * pad_length

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


def _build_feature_order() -> List[str]:
    feature_cols: List[str] = []
    for spec in config.FINAL_SPECIALISTS:
        feature_cols.extend(
            [
                f"{spec}_prob_mean",
                f"{spec}_prob_max",
                f"{spec}_prob_std",
            ]
        )
    feature_cols.extend(["avg_sent_len", "type_token_ratio", "punct_density"])
    return feature_cols


# --- Resource loading -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_resources() -> Dict[str, Any]:
    _ensure_nltk()

    tokenizer = _load_tokenizer()

    specialists: Dict[str, AutoModelForSequenceClassification] = {}
    for spec_name in config.FINAL_SPECIALISTS:
        model_path = Path(f"micro_{spec_name}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing specialist weights: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(config.DEVICE)
        model.eval()
        specialists[spec_name] = model

    with open(config.CALIBRATION_SCALARS_PATH, "r", encoding="utf-8") as f:
        temperatures = json.load(f)

    macro_classifier = joblib.load(config.MACRO_CLASSIFIER_PATH)
    macro_regressor = joblib.load(config.MACRO_REGRESSOR_PATH)
    residuals = joblib.load(config.CONFORMAL_RESIDUALS_PATH)

    tail_prob = 1 - config.PERCENT_AI_CONFIDENCE
    q_level = 1 - tail_prob / 2.0
    conformal_quantile = float(np.quantile(residuals, q_level))

    return {
        "tokenizer": tokenizer,
        "specialists": specialists,
        "temperatures": temperatures,
        "macro_classifier": macro_classifier,
        "macro_regressor": macro_regressor,
        "conformal_quantile": conformal_quantile,
        "feature_order": _build_feature_order(),
    }


# --- Prediction core --------------------------------------------------------
def build_feature_vector(text: str, resources: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}

    for spec_name, model in resources["specialists"].items():
        temperature = resources["temperatures"].get(spec_name, 1.0)
        stats = get_specialist_predictions_for_doc(text, model, resources["tokenizer"], temperature)
        features[f"{spec_name}_prob_mean"] = stats["prob_mean"]
        features[f"{spec_name}_prob_max"] = stats["prob_max"]
        features[f"{spec_name}_prob_std"] = stats["prob_std"]

    features.update(calculate_stylometric_features(text))
    return features


def predict_text(
    text: str,
    resources: Dict[str, Any],
    decision_threshold: float,
) -> Dict[str, Any]:
    feature_vector = build_feature_vector(text, resources)
    feature_df = pd.DataFrame([feature_vector])
    feature_df = feature_df.reindex(columns=resources["feature_order"]).fillna(0.0)

    clf_probs = resources["macro_classifier"].predict_proba(feature_df)[0]
    prob_ai = float(clf_probs[1])
    prob_human = float(clf_probs[0])
    prediction = int(prob_ai >= decision_threshold)

    reg_pred = float(resources["macro_regressor"].predict(feature_df)[0])
    reg_pred = float(np.clip(reg_pred, 0.0, 100.0))

    width = resources["conformal_quantile"]
    ci_low = max(0.0, reg_pred - width)
    ci_high = min(100.0, reg_pred + width)

    specialist_view = {spec: feature_vector[f"{spec}_prob_mean"] for spec in config.FINAL_SPECIALISTS}

    return {
        "prediction": prediction,
        "prob_ai": prob_ai,
        "prob_human": prob_human,
        "percent_ai": reg_pred,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "feature_vector": feature_vector,
        "specialist_view": specialist_view,
    }


def batch_predict(
    texts: List[str],
    resources: Dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for idx, text in enumerate(texts):
        result = predict_text(text, resources, threshold)
        row = {
            "index": idx,
            "text": text,
            "prediction": result["prediction"],
            "prob_ai": result["prob_ai"],
            "prob_human": result["prob_human"],
            "percent_ai": result["percent_ai"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
        }
        for spec, score in result["specialist_view"].items():
            row[f"{spec}_prob_mean"] = score
        row.update(calculate_stylometric_features(text))
        rows.append(row)
    return pd.DataFrame(rows)


# --- UI layout ---------------------------------------------------------------
st.title("🤖 Ai-or-NOT Manager")
st.subheader("Ensemble specialists + macro manager for high-confidence AI text detection")

resources = load_resources()

with st.sidebar:
    st.header("Configuration")
    default_threshold = st.slider(
        "Classification threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability above which a document is flagged as AI-generated.",
    )
    show_features = st.checkbox("Show feature vector", value=False)
    st.markdown(
        "---\nModel confidence interval computed with conformal residuals at ``{:.0f}%``.".format(
            config.PERCENT_AI_CONFIDENCE * 100
        )
    )

single_tab, batch_tab = st.tabs(["Single Document", "Batch Evaluation"])

with single_tab:
    sample_text = (
        "This is a test of the Ai-or-NOT system. I am writing this text myself. "
        "I wonder if the model will classify it as human. Let's add another sentence to be sure."
    )
    text_input = st.text_area(
        "Enter text (longer passages improve accuracy):",
        height=260,
        value=sample_text,
    )

    if st.button("Analyze Text", type="primary"):
        if len(text_input.strip()) < 20:
            st.warning("Input is very short; consider adding more context for reliable predictions.")

        result = predict_text(text_input, resources, default_threshold)

        verdict_container = st.container()
        with verdict_container:
            prob_ai_pct = result["prob_ai"] * 100
            prob_human_pct = result["prob_human"] * 100
            percent_ai = result["percent_ai"]

            verdict_cols = st.columns([1.2, 0.9, 0.9])
            verdict_text = "AI-generated" if result["prediction"] else "Human-authored"
            if result["prediction"]:
                verdict_cols[0].error(
                    f"Verdict: {verdict_text}\nAI {prob_ai_pct:.2f}% · Human {prob_human_pct:.2f}%"
                )
            else:
                verdict_cols[0].success(
                    f"Verdict: {verdict_text}\nHuman {prob_human_pct:.2f}% · AI {prob_ai_pct:.2f}%"
                )

            verdict_cols[1].metric("P(AI)", f"{prob_ai_pct:.2f}%")
            verdict_cols[2].metric("P(Human)", f"{prob_human_pct:.2f}%")

            metric_cols = st.columns([1, 1])
            metric_cols[0].metric("Estimated % AI", f"{percent_ai:.2f}%")
            metric_cols[1].metric(
                "Confidence Interval",
                f"{result['ci_low']:.2f}% – {result['ci_high']:.2f}%",
            )

            st.progress(min(int(percent_ai), 100))

        st.subheader("Specialist Panel")
        cols = st.columns(len(config.FINAL_SPECIALISTS))
        for idx, spec_name in enumerate(config.FINAL_SPECIALISTS):
            with cols[idx]:
                st.metric(
                    label=f"{spec_name.title()}",
                    value=f"{result['specialist_view'][spec_name] * 100:.2f}%",
                )

        if show_features:
            st.subheader("Feature Vector (debug view)")
            feature_df = (
                pd.DataFrame.from_dict(result["feature_vector"], orient="index", columns=["value"])
                .sort_index()
            )
            st.dataframe(feature_df)

with batch_tab:
    st.write(
        "Upload a CSV file with a column named `text` to score multiple documents. "
        "Results can be downloaded as CSV for further analysis."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to read CSV: {exc}")
            df = None

        if df is not None:
            if "text" not in df.columns:
                st.error("CSV must include a `text` column.")
            else:
                with st.spinner("Running batch inference..."):
                    results_df = batch_predict(df["text"].astype(str).tolist(), resources, default_threshold)
                st.success(f"Scored {len(results_df)} documents.")
                st.dataframe(results_df)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv_bytes,
                    file_name="ai_or_not_predictions.csv",
                    mime="text/csv",
                )

st.caption("Ai-or-NOT v1.0 — built with DeBERTa-v3 specialists, calibrated ensembles, and conformal prediction.")
