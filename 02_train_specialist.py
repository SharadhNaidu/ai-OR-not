import argparse
import os

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Tokenizer,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

import config


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1}


def chunk_dataset(dataset, tokenizer):
    print(f"Chunking {len(dataset)} documents...")
    chunked_inputs = {"input_ids": [], "attention_mask": [], "label": []}

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    for doc in tqdm(dataset, desc="Chunking documents"):
        tokenized_doc = tokenizer(
            doc["text"],
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = tokenized_doc["input_ids"]
        if not input_ids:
            continue

        start = 0
        while start < len(input_ids):
            end = start + config.CHUNK_SIZE - 2
            chunk = [cls_id] + input_ids[start:end] + [sep_id]

            if len(chunk) < config.CHUNK_SIZE:
                padding_length = config.CHUNK_SIZE - len(chunk)
                chunk += [pad_id] * padding_length
                attention_mask = [1] * (config.CHUNK_SIZE - padding_length) + [0] * padding_length
            else:
                chunk = chunk[: config.CHUNK_SIZE]
                attention_mask = [1] * config.CHUNK_SIZE

            chunked_inputs["input_ids"].append(chunk)
            chunked_inputs["attention_mask"].append(attention_mask)
            chunked_inputs["label"].append(doc["binary_label"])

            if end >= len(input_ids):
                break

            start += config.CHUNK_SIZE - config.CHUNK_OVERLAP

    return Dataset.from_dict(chunked_inputs)


def prepare_specialist_data(full_train_data, specialist_name, tokenizer):
    print(f"--- Preparing data for specialist: {specialist_name} ---")

    human_data = full_train_data.filter(
        lambda x: x["specialist_label"] == "human",
        desc="Filtering human data",
    )
    ai_data = full_train_data.filter(
        lambda x: x["specialist_label"] == specialist_name,
        desc=f"Filtering {specialist_name} data",
    )

    print(f"Found {len(human_data)} human samples and {len(ai_data)} AI samples.")

    if len(human_data) > len(ai_data):
        print(f"Downsampling human data from {len(human_data)} to {len(ai_data)}")
        human_data = human_data.shuffle(seed=config.RANDOM_SEED).select(range(len(ai_data)))
    elif len(ai_data) > len(human_data):
        print(f"Downsampling AI data from {len(ai_data)} to {len(human_data)}")
        ai_data = ai_data.shuffle(seed=config.RANDOM_SEED).select(range(len(human_data)))

    binary_data = concatenate_datasets([human_data, ai_data]).shuffle(seed=config.RANDOM_SEED)
    print(f"Final balanced dataset size: {len(binary_data)}")

    chunked_dataset = chunk_dataset(binary_data, tokenizer)
    print(f"Total chunks created: {len(chunked_dataset)}")

    return chunked_dataset.train_test_split(test_size=0.1, seed=config.RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser(description="Train a specialist micro-model.")
    parser.add_argument(
        "--specialist",
        type=str,
        required=True,
        choices=config.FINAL_SPECIALISTS,
        help="The name of the specialist to train (e.g., 'chatgpt').",
    )
    args = parser.parse_args()

    print(f"--- [Step 2] START: Training Specialist: {args.specialist} ---")

    print(f"Loading tokenizer: {config.MICRO_BACKBONE}")
    tokenizer = None
    tokenizer_errors = []
    for use_fast in (True, False):
        if tokenizer is not None:
            break
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.MICRO_BACKBONE, use_fast=use_fast
            )
        except Exception as exc:  # fall back to slow tokenizer explicitly
            tokenizer_errors.append(("AutoTokenizer", use_fast, exc))
            tokenizer = None

    if tokenizer is None and "deberta" in config.MICRO_BACKBONE.lower():
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(config.MICRO_BACKBONE)
        except Exception as exc:
            tokenizer_errors.append(("DebertaV2Tokenizer", None, exc))

    if tokenizer is None:
        raise RuntimeError(
            f"Failed to load tokenizer for {config.MICRO_BACKBONE}: {tokenizer_errors}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    print(f"Loading data from disk: {config.TRAIN_DATA_PATH}")
    try:
        full_train_data = load_from_disk(config.TRAIN_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: '{config.TRAIN_DATA_PATH}' not found.")
        print("Please run 01_ingest_data.py first.")
        return

    specialist_splits = prepare_specialist_data(full_train_data, args.specialist, tokenizer)

    print(f"Loading backbone model: {config.MICRO_BACKBONE}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MICRO_BACKBONE, num_labels=2
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    output_dir = f"micro_{args.specialist}"
    training_args_dict = config.MICRO_TRAIN_PARAMS.copy()
    training_args_dict["output_dir"] = output_dir
    training_args_dict.pop("max_length", None)
    if "evaluation_strategy" in training_args_dict:
        training_args_dict["eval_strategy"] = training_args_dict.pop("evaluation_strategy")

    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=specialist_splits["train"],
        eval_dataset=specialist_splits["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training for {args.specialist}...")
    trainer.train()
    print("Training complete.")

    print(f"Saving model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"--- [Step 2] COMPLETE: Specialist '{args.specialist}' is trained. ---")


if __name__ == "__main__":
    main()
