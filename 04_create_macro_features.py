import json
import random
from pathlib import Path

import config
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer

from nltk.tokenize import sent_tokenize, word_tokenize


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


def calculate_stylometric_features(text):
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        if not words:
            return {"avg_sent_len": 0.0, "type_token_ratio": 0.0, "punct_density": 0.0}

        sent_lengths = [len(word_tokenize(s)) for s in sentences if word_tokenize(s)]
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


def get_calibrated_probs(logits, temperature):
    temperature = float(temperature) if float(temperature) > 0 else 1.0
    return softmax(logits / temperature, dim=1)[:, 1].cpu().numpy()


def get_specialist_predictions_for_doc(doc_text, model, tokenizer, temperature):
    tokenized_doc = tokenizer(doc_text, truncation=False, padding=False, add_special_tokens=False)
    input_ids = tokenized_doc["input_ids"]
    if not input_ids:
        return {"prob_mean": 0.0, "prob_max": 0.0, "prob_std": 0.0}

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    chunk_ids = []
    chunk_masks = []
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
        chunk_masks.append(attention)

        if end >= len(input_ids):
            break
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP

    if not chunk_ids:
        return {"prob_mean": 0.0, "prob_max": 0.0, "prob_std": 0.0}

    all_logits = []
    batch_size = 16
    model.eval()

    with torch.no_grad():
        for idx in range(0, len(chunk_ids), batch_size):
            batch_input = torch.tensor(chunk_ids[idx : idx + batch_size], dtype=torch.long, device=config.DEVICE)
            batch_mask = torch.tensor(chunk_masks[idx : idx + batch_size], dtype=torch.long, device=config.DEVICE)
            logits = model(input_ids=batch_input, attention_mask=batch_mask).logits
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    chunk_probs = get_calibrated_probs(logits_tensor, temperature)

    return {
        "prob_mean": float(np.mean(chunk_probs)),
        "prob_max": float(np.max(chunk_probs)),
        "prob_std": float(np.std(chunk_probs)),
    }


def create_synthetic_mixed_docs(human_docs, ai_docs, num_docs=2000, max_chunks=20):
    print(f"Creating {num_docs} synthetic mixed-percent documents...")
    synthetic_data = []

    human_chunks = [doc[i : i + 100] for doc in human_docs for i in range(0, len(doc), 100) if doc[i : i + 100].strip()]
    ai_chunks = [doc[i : i + 100] for doc in ai_docs for i in range(0, len(doc), 100) if doc[i : i + 100].strip()]

    if not human_chunks or not ai_chunks:
        print("Warning: Could not create synthetic docs, chunk pools are empty.")
        return pd.DataFrame()

    for idx in range(num_docs):
        n_chunks = random.randint(5, max_chunks)
        k_ai = random.randint(0, n_chunks)
        percent_ai = 100.0 * k_ai / n_chunks if n_chunks else 0.0

        ai_samples = random.choices(ai_chunks, k=k_ai) if k_ai else []
        human_samples = random.choices(human_chunks, k=n_chunks - k_ai) if n_chunks - k_ai else []

        doc_chunks = ai_samples + human_samples
        random.shuffle(doc_chunks)

        synthetic_data.append(
            {
                "doc_id": f"synthetic_{idx}",
                "text": " ".join(doc_chunks),
                "binary_label": 1 if k_ai > 0 else 0,
                "specialist_label": "synthetic",
                "percent_ai": percent_ai,
            }
        )

    return pd.DataFrame(synthetic_data)


def main():
    print("--- [Step 3.2] START: Creating Macro-Model Features ---")

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    try:
        full_train_dataset = load_from_disk(config.TRAIN_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: {config.TRAIN_DATA_PATH} not found. Run 01_ingest_data.py first.")
        return

    full_train_df = full_train_dataset.to_pandas()

    try:
        with open(config.CALIBRATION_SCALARS_PATH, "r", encoding="utf-8") as f:
            temperatures = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: {config.CALIBRATION_SCALARS_PATH} not found. Run 03_calibrate_models.py first.")
        return

    tokenizer = _load_tokenizer()

    models = {}
    for spec_name in config.FINAL_SPECIALISTS:
        model_path = Path(f"micro_{spec_name}")
        if not model_path.exists():
            print(f"Warning: {model_path} missing. Skipping macro features for {spec_name}.")
            continue
        print(f"Loading specialist model: {spec_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(config.DEVICE)
        model.eval()
        models[spec_name] = model

    if not models:
        print("FATAL: No specialist models available for feature generation.")
        return

    text_column = "text"
    macro_features = []

    for _, row in tqdm(full_train_df.iterrows(), total=len(full_train_df), desc="Generating macro features"):
        doc_text = row[text_column]
        features = {
            "doc_id": row.get("doc_id"),
            "binary_label": row.get("binary_label", 0),
            "percent_ai": float(row.get("binary_label", 0) * 100.0),
            "specialist_label": row.get("specialist_label"),
        }

        for spec_name, model in models.items():
            temperature = temperatures.get(spec_name, 1.0)
            preds = get_specialist_predictions_for_doc(doc_text, model, tokenizer, temperature)
            features[f"{spec_name}_prob_mean"] = preds["prob_mean"]
            features[f"{spec_name}_prob_max"] = preds["prob_max"]
            features[f"{spec_name}_prob_std"] = preds["prob_std"]

        features.update(calculate_stylometric_features(doc_text))
        macro_features.append(features)

    macro_df = pd.DataFrame(macro_features)

    human_docs = full_train_df[full_train_df["binary_label"] == 0][text_column].tolist()
    ai_docs = full_train_df[full_train_df["binary_label"] == 1][text_column].tolist()
    synthetic_df = create_synthetic_mixed_docs(human_docs, ai_docs)

    synthetic_features = []
    if not synthetic_df.empty:
        for _, row in tqdm(synthetic_df.iterrows(), total=len(synthetic_df), desc="Processing synthetic docs"):
            doc_text = row["text"]
            features = {
                "doc_id": row["doc_id"],
                "binary_label": row["binary_label"],
                "percent_ai": float(row["percent_ai"]),
                "specialist_label": row["specialist_label"],
            }

            for spec_name, model in models.items():
                temperature = temperatures.get(spec_name, 1.0)
                preds = get_specialist_predictions_for_doc(doc_text, model, tokenizer, temperature)
                features[f"{spec_name}_prob_mean"] = preds["prob_mean"]
                features[f"{spec_name}_prob_max"] = preds["prob_max"]
                features[f"{spec_name}_prob_std"] = preds["prob_std"]

            features.update(calculate_stylometric_features(doc_text))
            synthetic_features.append(features)

    if synthetic_features:
        synthetic_df_features = pd.DataFrame(synthetic_features)
        final_macro_df = pd.concat([macro_df, synthetic_df_features], ignore_index=True)
    else:
        final_macro_df = macro_df

    final_macro_df.to_csv(config.OOF_FEATURES_PATH, index=False)

    print(f"--- [Step 3.2] COMPLETE: Macro features saved to {config.OOF_FEATURES_PATH} ---")


if __name__ == "__main__":
    main()
