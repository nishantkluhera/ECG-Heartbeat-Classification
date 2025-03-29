import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

from src.config import MODEL_PATH, N_FEATURES, CLASS_MAP, TRAIN_CSV # Need TRAIN_CSV to fit scaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for model and scaler (load once)
model = None
scaler = None

def load_prediction_resources():
    """Loads the trained model and the scaler."""
    global model, scaler
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Error loading model: {e}. Cannot make predictions.")
            return False

    if scaler is None:
        try:
            # We need to fit the scaler exactly as it was during training.
            # Ideally, the scaler object should be saved during training.
            # Here, we refit it on the training data for simplicity.
            df_train = pd.read_csv(TRAIN_CSV, header=None)
            X_train_raw = df_train.iloc[:, :N_FEATURES].values
            scaler = StandardScaler()
            scaler.fit(X_train_raw) # Fit on the raw 2D training data
            logging.info("Scaler fitted on training data.")
        except FileNotFoundError:
             logging.error(f"Training data file {TRAIN_CSV} not found. Cannot fit scaler.")
             return False
        except Exception as e:
             logging.error(f"Error fitting scaler: {e}")
             return False
    return True


def predict_heartbeat(ecg_signal_segment):
    """
    Predicts the class of a single ECG heartbeat segment.

    Args:
        ecg_signal_segment (numpy.ndarray): A 1D numpy array of shape (N_FEATURES,)
                                           representing the ECG segment.

    Returns:
        str: The predicted class name (e.g., 'Normal', 'Ventricular').
        float: The confidence score for the prediction.
        None: If prediction fails.
    """
    global model, scaler
    if model is None or scaler is None:
        if not load_prediction_resources():
            return None, None

    if ecg_signal_segment.shape != (N_FEATURES,):
        logging.error(f"Input signal must have shape ({N_FEATURES},). Received shape {ecg_signal_segment.shape}")
        return None, None

    try:
        # 1. Reshape for scaler (needs 2D) and scale
        segment_scaled = scaler.transform(ecg_signal_segment.reshape(1, -1))

        # 2. Reshape for CNN (needs 3D: samples, timesteps, features)
        segment_reshaped = segment_scaled.reshape(1, N_FEATURES, 1)

        # 3. Predict
        pred_probs = model.predict(segment_reshaped)[0] # Get probabilities for the first (only) sample
        pred_class_idx = np.argmax(pred_probs)
        pred_confidence = pred_probs[pred_class_idx]

        # 4. Map index to class name
        pred_class_name = CLASS_MAP.get(pred_class_idx, 'Unknown Error')

        logging.info(f"Prediction: {pred_class_name}, Confidence: {pred_confidence:.4f}")
        return pred_class_name, pred_confidence

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None, None

if __name__ == '__main__':
    # Example Usage:
    print("Attempting to load prediction resources...")
    if load_prediction_resources():
        print("Resources loaded.")
        # Create a dummy ECG segment (replace with actual data)
        # Ensure this dummy data has the correct number of features (e.g., 187)
        dummy_ecg = np.random.rand(N_FEATURES) * 0.5 # Example random signal

        print(f"\nPredicting class for a dummy ECG segment (shape {dummy_ecg.shape})...")
        predicted_class, confidence = predict_heartbeat(dummy_ecg)

        if predicted_class:
            print(f"==> Predicted Heartbeat Class: {predicted_class} (Confidence: {confidence:.2f})")
        else:
            print("==> Prediction failed.")

        # Example loading from the test set (if available)
        try:
            df_test = pd.read_csv('data/mitbih_test.csv', header=None)
            sample_ecg = df_test.iloc[100, :N_FEATURES].values # Take 100th sample
            actual_class_idx = int(df_test.iloc[100, N_FEATURES])
            actual_class_name = CLASS_MAP.get(actual_class_idx, 'Unknown')

            print(f"\nPredicting class for a sample ECG segment from test set (Actual: {actual_class_name})...")
            predicted_class, confidence = predict_heartbeat(sample_ecg)

            if predicted_class:
                print(f"==> Predicted Heartbeat Class: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                print("==> Prediction failed.")

        except FileNotFoundError:
            print("\nTest CSV not found, skipping prediction test on real data.")
        except Exception as e:
            print(f"\nError running prediction on test sample: {e}")

    else:
        print("Failed to load resources. Prediction unavailable.")