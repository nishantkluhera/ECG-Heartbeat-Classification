import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import logging
import os

from src.config import (
    TRAIN_CSV, TEST_CSV, MODEL_PATH, EPOCHS, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_STATE, N_FEATURES, N_CLASSES
)
from src.data_loader import load_data, preprocess_data
from src.model import build_cnn_model
from src.utils import plot_training_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def train_model():
    """Loads data, preprocesses it, builds, trains, and saves the model."""
    logging.info("Starting model training process...")

    # 1. Load Data
    df_train, df_test = load_data(TRAIN_CSV, TEST_CSV)
    if df_train is None:
        return # Exit if data loading failed

    # 2. Preprocess Data
    X_train, y_train, X_test, y_test, _ = preprocess_data(df_train, df_test)
    if X_train is None:
        return # Exit if preprocessing failed

    # 3. Build Model
    model = build_cnn_model(input_shape=(N_FEATURES, 1), num_classes=N_CLASSES)

    # 4. Define Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # 5. Train Model
    logging.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, # Use a portion of training data for validation during training
        callbacks=callbacks,
        shuffle=True # Shuffle training data each epoch
    )
    logging.info("Model training finished.")

    # 6. Save Final Model (optional, checkpoint saves the best one)
    # model.save(MODEL_PATH) # Overwrites checkpoint if current model is not the best
    # logging.info(f"Best model saved to {MODEL_PATH}")

    # 7. Plot Training History
    plot_training_history(history)

    logging.info("Training process completed.")

if __name__ == "__main__":
    # Ensure the saved_models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    train_model()