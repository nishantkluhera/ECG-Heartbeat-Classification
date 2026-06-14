"""Central configuration for the ECG Heartbeat Classification project.

Kept intentionally free of heavy imports (no torch) so that lightweight
utilities such as the synthetic-data generator and unit tests can import it
without pulling in the deep-learning stack.
"""
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "mitbih_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "mitbih_test.csv")

MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_NAME = "ecg_cnn_classifier.pt"          # PyTorch checkpoint
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
SCALER_NAME = "scaler.joblib"                  # fitted StandardScaler (saved during training)
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, SCALER_NAME)

VISUALIZATION_DIR = os.path.join(BASE_DIR, "visualizations")

# Create output directories up-front so downstream code can write freely.
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data / label definitions
# --------------------------------------------------------------------------- #
# The MIT-BIH Kaggle CSVs have 187 signal columns + 1 label column.
N_FEATURES = 187
N_CLASSES = 5

# AAMI standard mapping used by the Kaggle "heartbeat" dataset.
CLASS_MAP = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]

# --------------------------------------------------------------------------- #
# Training hyper-parameters
# --------------------------------------------------------------------------- #
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.2          # fraction of training data held out for validation

# Callbacks-style controls
EARLY_STOPPING_PATIENCE = 10    # stop after N epochs with no val-loss improvement
LR_PATIENCE = 5                 # ReduceLROnPlateau patience
LR_FACTOR = 0.2
MIN_LR = 1e-5

# Class-imbalance handling
APPLY_SMOTE = True              # set False to train on the raw (imbalanced) data

# Performance
USE_AMP = True                  # automatic mixed precision (only used on CUDA)
NUM_WORKERS = 0                 # DataLoader workers; 0 is safest on Windows

# Reproducibility
RANDOM_STATE = 42
