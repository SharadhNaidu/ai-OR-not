import config
from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main() -> None:
    print("--- [Step 1.3] START: Data Ingestion & Processing ---")

    print(f"Loading source dataset: {config.SOURCE_DATASET_NAME}...")
    try:
        dataset = load_dataset(
            config.SOURCE_DATASET_NAME,
            config.SOURCE_DATASET_CONFIG,
            split="train",
        )
        print("Source dataset loaded successfully.")
    except Exception as exc:
        print(f"FATAL ERROR: Could not load dataset from Hugging Face. {exc}")
        return

    print("Restructuring data from 'wide' to 'long' format...")
    processed_rows = []

    for row in tqdm(dataset, desc="Processing rows"):
        human_text = row.get(config.HUMAN_COLUMN)
        if human_text:
            processed_rows.append(
                {
                    "text": human_text,
                    "generator_label": "human",
                    "binary_label": 0,
                    "specialist_label": "human",
                }
            )

        for specialist_name, model_list in config.SPECIALIST_NAMES_MAP.items():
            for model_column in model_list:
                model_text = row.get(model_column)
                if model_text:
                    processed_rows.append(
                        {
                            "text": model_text,
                            "generator_label": model_column,
                            "binary_label": 1,
                            "specialist_label": specialist_name,
                        }
                    )

    print(f"Restructuring complete. Created {len(processed_rows)} total text samples.")

    if not processed_rows:
        print("FATAL ERROR: No rows were processed. Check config.py column names.")
        return

    full_df = pd.DataFrame(processed_rows)
    full_df["doc_id"] = range(len(full_df))

    print("Splitting into main Train and Test sets...")
    train_df, test_df = train_test_split(
        full_df,
        test_size=config.TEST_SET_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=full_df["specialist_label"],
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    if "__index_level_0__" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("__index_level_0__")
    if "__index_level_0__" in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns("__index_level_0__")

    print(f"Saving train set to: {config.TRAIN_DATA_PATH}")
    train_dataset.save_to_disk(config.TRAIN_DATA_PATH)

    print(f"Saving test set to: {config.TEST_DATA_PATH}")
    test_dataset.save_to_disk(config.TEST_DATA_PATH)

    print("--- [Step 1.3] COMPLETE: Data ingestion finished. ---")


if __name__ == "__main__":
    main()
