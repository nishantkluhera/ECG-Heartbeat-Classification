import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import logging

from src.config import N_FEATURES, N_CLASSES, RANDOM_STATE, APPLY_SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path):
    """Loads training and testing data from CSV files."""
    try:
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        logging.info(f"Loaded train data with shape: {df_train.shape}")
        logging.info(f"Loaded test data with shape: {df_test.shape}")
        return df_train, df_test
    except FileNotFoundError:
        logging.error(f"Error: Data files not found at {train_path} or {test_path}")
        logging.error("Please download the dataset and place it in the 'data/' directory.")
        return None, None

def preprocess_data(df_train, df_test):
    """Prepares data for model training and evaluation."""
    if df_train is None or df_test is None:
        return None, None, None, None, None

    # Separate features (X) and labels (y)
    X_train = df_train.iloc[:, :N_FEATURES].values
    y_train = df_train.iloc[:, N_FEATURES].values.astype(int)
    X_test = df_test.iloc[:, :N_FEATURES].values
    y_test = df_test.iloc[:, N_FEATURES].values.astype(int)

    logging.info(f"Initial class distribution in training data:\n{pd.Series(y_train).value_counts()}")

    # Apply SMOTE for class imbalance if enabled
    if APPLY_SMOTE:
        logging.info("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=RANDOM_STATE)
        try:
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"Class distribution after SMOTE:\n{pd.Series(y_train).value_counts()}")
        except Exception as e:
             logging.error(f"Error during SMOTE: {e}. Ensure you have enough samples for each class.")
             return None, None, None, None, None


    # Reshape data for CNN (samples, timesteps, features=1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Normalize data (StandardScaler) - Fit only on training data
    # Note: Scaling is sometimes debated for ECG raw signals, but often helpful for NNs.
    # Reshape back to 2D for scaler, then reshape back to 3D
    scaler = StandardScaler()
    n_samples_train, n_timesteps_train, n_features_train = X_train.shape
    n_samples_test, n_timesteps_test, n_features_test = X_test.shape

    X_train = scaler.fit_transform(X_train.reshape(-1, n_timesteps_train)).reshape(n_samples_train, n_timesteps_train, n_features_train)
    X_test = scaler.transform(X_test.reshape(-1, n_timesteps_test)).reshape(n_samples_test, n_timesteps_test, n_features_test)
    logging.info("Data normalized using StandardScaler.")

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=N_CLASSES)
    y_test_orig = y_test # Keep original labels for evaluation metrics
    y_test = to_categorical(y_test, num_classes=N_CLASSES)
    logging.info("Labels one-hot encoded.")

    return X_train, y_train, X_test, y_test, y_test_orig