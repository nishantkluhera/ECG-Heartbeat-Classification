"""Configuration for the PTB-XL diagnostic track."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PTBXL_DIR = os.path.join(DATA_DIR, "ptbxl")          # extraction target
CACHE_DIR = os.path.join(PTBXL_DIR, "cache")          # cached .npy arrays

MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Signal / label definitions
# --------------------------------------------------------------------------- #
SAMPLING_RATE = 100          # use the 100 Hz records (records100/) -> 1000 samples / 10 s
SIGNAL_LENGTH = 1000

# Standard PTB-XL lead order.
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX = {name: i for i, name in enumerate(LEADS)}

# Five diagnostic superclasses (PTB-XL aggregation via scp_statements.diagnostic_class).
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_FULL = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}
N_CLASSES = len(SUPERCLASSES)

# Recommended stratified folds (respecting patient assignment).
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD = 9
TEST_FOLD = 10

# --------------------------------------------------------------------------- #
# Model checkpoints (one per lead configuration)
# --------------------------------------------------------------------------- #
def model_path(lead_config: str) -> str:
    """lead_config is '12lead' or 'lead2'."""
    return os.path.join(MODEL_SAVE_DIR, f"ptbxl_diag_{lead_config}.pt")

# --------------------------------------------------------------------------- #
# Training hyper-parameters
# --------------------------------------------------------------------------- #
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
LR_PATIENCE = 4
LR_FACTOR = 0.3
MIN_LR = 1e-5
USE_AMP = True
NUM_WORKERS = 0
RANDOM_STATE = 42
