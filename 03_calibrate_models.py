import json
import importlib.util
from pathlib import Path

import config
import numpy as np
import torch
from datasets import load_from_disk
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer


def _load_prepare_specialist_data():
    module_path = Path(__file__).resolve().parent / "02_train_specialist.py"
    if not module_path.exists():
        raise FileNotFoundError("02_train_specialist.py is required for calibration.")

    spec = importlib.util.spec_from_file_location("train_specialist_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # silence type checker
    spec.loader.exec_module(module)
    return module.prepare_specialist_data


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


def find_optimal_temperature(model, val_loader):
    model.eval()
    all_logits = []
    all_labels = []

    print("Getting logits from validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calibrating", leave=False):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    if not all_logits:
        raise RuntimeError("Validation loader produced no batches; cannot calibrate.")

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    best_ece = float("inf")
    best_temp = 1.0
    bin_boundaries = np.linspace(0.0, 1.0, 11)

    for temp in np.linspace(0.5, 3.0, 20):
        scaled_probs = softmax(all_logits / temp, dim=1)
        confidences, preds = torch.max(scaled_probs, dim=1)
        accuracies = preds.eq(all_labels)

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if not torch.any(bin_mask):
                continue

            bin_acc = accuracies[bin_mask].float().mean().item()
            bin_conf = confidences[bin_mask].mean().item()
            bin_prob = bin_mask.float().mean().item()
            ece += abs(bin_acc - bin_conf) * bin_prob

        if ece < best_ece:
            best_ece = ece
            best_temp = float(temp)

    print(f"Best ECE: {best_ece:.4f}, Best Temperature: {best_temp:.3f}")
    return best_temp


def main():
    print("--- [Step 3.1] START: Calibrating Specialists ---")

    try:
        full_train_data = load_from_disk(config.TRAIN_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: {config.TRAIN_DATA_PATH} not found. Run 01_ingest_data.py first.")
        return

    tokenizer = _load_tokenizer()
    prepare_specialist_data = _load_prepare_specialist_data()

    calibration_scalars = {}

    for spec_name in config.FINAL_SPECIALISTS:
        print(f"--- Calibrating: {spec_name} ---")
        model_path = Path(f"micro_{spec_name}")
        if not model_path.exists():
            print(f"Warning: {model_path} missing. Skipping.")
            continue

        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(config.DEVICE)

        specialist_splits = prepare_specialist_data(full_train_data, spec_name, tokenizer)
        val_dataset = specialist_splits["test"]
        if "label" in val_dataset.column_names:
            val_dataset = val_dataset.rename_column("label", "labels")
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.MICRO_TRAIN_PARAMS.get("per_device_eval_batch_size", 16),
        )

        temp = find_optimal_temperature(model, val_loader)
        calibration_scalars[spec_name] = temp

    if not calibration_scalars:
        print("No calibration scalars were computed.")
        return

    with open(config.CALIBRATION_SCALARS_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration_scalars, f, indent=2)

    print(f"--- [Step 3.1] COMPLETE: Scalars saved to {config.CALIBRATION_SCALARS_PATH} ---")


if __name__ == "__main__":
    main()
