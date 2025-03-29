import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics import accuracy_score

from src.config import TEST_CSV, TRAIN_CSV, MODEL_PATH, N_CLASSES, CLASS_NAMES # Need TRAIN_CSV for scaler fitting if not saved
from src.data_loader import load_data, preprocess_data # Re-use preprocessing logic
from src.utils import plot_confusion_matrix, print_classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    """Loads the test data, trained model, and evaluates performance."""
    logging.info("Starting model evaluation process...")

    # 1. Load Model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(f"Please ensure a trained model exists at {MODEL_PATH}. Run train.py first.")
        return

    # 2. Load Data
    # Need training data only to fit the scaler consistently if it wasn't saved
    # A better approach would be to save the scaler during training.
    # For simplicity here, we reload both datasets.
    df_train, df_test = load_data(TRAIN_CSV, TEST_CSV)
    if df_test is None:
         return # Exit if data loading failed

    # 3. Preprocess Test Data
    # IMPORTANT: Use the *same* preprocessing steps as in training, especially the scaler.
    # The preprocess_data function handles this if APPLY_SMOTE is False during this call,
    # or if we refactor it to return the scaler.
    # Let's slightly modify preprocess logic for evaluation (no SMOTE on test, return original y)
    _, _, X_test, y_test_cat, y_test_orig = preprocess_data(df_train, df_test) # Ignore train outputs here

    if X_test is None:
        logging.error("Failed to preprocess test data.")
        return

    # 4. Make Predictions
    logging.info("Making predictions on the test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) # Get class indices

    # 5. Evaluate Performance
    accuracy = accuracy_score(y_test_orig, y_pred)
    logging.info(f"Test Set Accuracy: {accuracy:.4f}")

    # Print classification report
    print_classification_report(y_test_orig, y_pred, class_names=CLASS_NAMES)

    # Plot confusion matrix
    plot_confusion_matrix(y_test_orig, y_pred, class_names=CLASS_NAMES)

    logging.info("Evaluation process completed.")

if __name__ == "__main__":
    evaluate_model()