import torch

# --- Environment & Paths (Single Folder Setup) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- Source Dataset ---
SOURCE_DATASET_NAME = "artnitolog/llm-generated-texts"
SOURCE_DATASET_CONFIG = "default"

# --- Raw Columns from 'artnitolog/llm-generated-texts' ---
TEXT_COLUMNS = [
    "human",
    "GPT4 Turbo 2024-04-09",
    "GPT4 Omni",
    "Claude 3 Opus",
    "Claude 3 Sonnet",
    "Claude 3 Haiku",
    "YandexGPT 3 Pro",
    "Llama3 70B",
    "Llama3 8B",
    "Mixtral 8x7B",
    "Mixtral 8x22B",
    "GigaChat Pro",
]
HUMAN_COLUMN = "human"

# --- Processed Data File Paths ---
TRAIN_DATA_PATH = "train.arrow"
TEST_DATA_PATH = "test.arrow"

# --- Core Labels (NEW) ---
SPECIALIST_NAMES_MAP = {
    "chatgpt": ["GPT4 Turbo 2024-04-09", "GPT4 Omni"],
    "claude": ["Claude 3 Opus", "Claude 3 Sonnet", "Claude 3 Haiku"],
    "omni_unseen": ["Mixtral 8x7B", "Mixtral 8x22B"],
    "omni": ["YandexGPT 3 Pro", "Llama3 70B", "Llama3 8B", "GigaChat Pro"],
}
FINAL_SPECIALISTS = ["chatgpt", "claude", "omni"]

# --- Processing Specs ---
CHUNK_SIZE = 128
CHUNK_OVERLAP = 20
TEST_SET_SIZE = 0.1

# --- Micro-Model Config ---
MICRO_BACKBONE = "microsoft/deberta-v3-small"
MICRO_TRAIN_PARAMS = {
    "output_dir": "tmp_trainer_output",
    "max_length": CHUNK_SIZE,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 3,
    "warmup_steps": 500,
    "fp16": True,
    "logging_steps": 100,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "save_total_limit": 1,
    "report_to": "none",
}

# --- Macro-Model Config & File Paths ---
STACKING_K_FOLDS = 5
MACRO_CLASSIFIER_PATH = "macro_classifier.joblib"
MACRO_REGRESSOR_PATH = "macro_regressor.joblib"
CALIBRATION_SCALARS_PATH = "temperature_scalars.json"
CONFORMAL_RESIDUALS_PATH = "conformal_residuals.joblib"
OOF_FEATURES_PATH = "oof_features_train.csv"
TEST_FEATURES_PATH = "macro_features_test.csv"

# --- Evaluation Targets ---
PERCENT_AI_CONFIDENCE = 0.95
