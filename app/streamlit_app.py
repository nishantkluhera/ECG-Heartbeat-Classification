"""Interactive Streamlit demo for the ECG Heartbeat Classifier.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import os
import sys

# Make the project root importable when launched via `streamlit run`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from src.config import TEST_CSV, MODEL_PATH, N_FEATURES, CLASS_NAMES, CLASS_MAP

st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="🫀", layout="centered")

st.title("🫀 ECG Heartbeat Classifier")
st.caption(
    "1D CNN (PyTorch) trained on the MIT-BIH Arrhythmia Database. "
    "Classifies a single heartbeat into the five AAMI categories: "
    "Normal, Supraventricular, Ventricular, Fusion, Unknown."
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    """Load model + scaler once and return the prediction callables."""
    from src.predict import predict_heartbeat, predict_proba, load_prediction_resources
    ok = load_prediction_resources()
    return ok, predict_heartbeat, predict_proba


@st.cache_data(show_spinner="Loading test data...")
def load_test_df():
    if os.path.exists(TEST_CSV):
        return pd.read_csv(TEST_CSV, header=None)
    return None


if not os.path.exists(MODEL_PATH):
    st.warning(
        "No trained model found. Train one first:\n\n"
        "```bash\npython main.py train\n```"
    )
    st.stop()

ok, predict_heartbeat, predict_proba = get_predictor()
if not ok:
    st.error("Could not load the model or scaler. Re-run `python main.py train`.")
    st.stop()

df_test = load_test_df()

# --- Input selection ------------------------------------------------------- #
st.subheader("Choose an input")
source = st.radio(
    "Signal source",
    ["Sample from test set", "Random synthetic segment"],
    horizontal=True,
    disabled=df_test is None,
)

actual_label = None
if source == "Sample from test set" and df_test is not None:
    idx = st.slider("Test sample index", 0, len(df_test) - 1, 100)
    signal = df_test.iloc[idx, :N_FEATURES].values.astype(np.float32)
    actual_label = CLASS_NAMES[int(df_test.iloc[idx, N_FEATURES])]
else:
    if st.button("🎲 New random segment") or "rand_sig" not in st.session_state:
        st.session_state.rand_sig = np.random.rand(N_FEATURES).astype(np.float32)
    signal = st.session_state.rand_sig

# --- Waveform -------------------------------------------------------------- #
st.subheader("Heartbeat waveform")
st.line_chart(pd.DataFrame({"amplitude": signal}))

# --- Prediction ------------------------------------------------------------ #
if st.button("Classify heartbeat", type="primary"):
    name, conf = predict_heartbeat(signal)
    probs = predict_proba(signal)
    if name is None:
        st.error("Prediction failed.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Predicted class", name, f"{conf:.1%} confidence")
        if actual_label is not None:
            verdict = "✅ correct" if actual_label == name else "❌ wrong"
            col2.metric("Actual class", actual_label, verdict)

        st.subheader("Class probabilities")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))

st.divider()
st.caption(
    "Model: 3× Conv1D blocks → dense head · CrossEntropyLoss · Adam · "
    "SMOTE for class imbalance. See the README for training details."
)
