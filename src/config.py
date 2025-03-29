import os

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'mitbih_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'mitbih_test.csv')

# Model saving path
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
MODEL_NAME = 'ecg_cnn_classifier.keras' # Use .keras for TF >= 2.7
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

# Visualization path
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Data parameters
# The MIT-BIH CSV dataset often has 187 features (time steps) + 1 label column
N_FEATURES = 187
# Classes based on AAMI standard mapping used in the common Kaggle dataset:
# 0: N (Normal beat)
# 1: S (Supraventricular ectopic beat)
# 2: V (Ventricular ectopic beat)
# 3: F (Fusion beat)
# 4: Q (Unknown beat)
N_CLASSES = 5
CLASS_MAP = {
    0: 'N',
    1: 'S',
    2: 'V',
    3: 'F',
    4: 'Q'
}
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']


# Training parameters
EPOCHS = 30 # Adjust as needed
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2 # Percentage of training data to use for validation

# Resampling parameters (using SMOTE)
APPLY_SMOTE = True # Set to False to disable SMOTE

# Random state for reproducibility
RANDOM_STATE = 42