"""Interactive ECG analysis demo (Streamlit).

Two modes:
  1. Clinical diagnosis from an ECG image  (upload a single-lead strip ->
     digitize -> PTB-XL diagnostic model -> calibrated findings + saliency)
  2. Heartbeat type classifier             (MIT-BIH single-beat CNN)

EDUCATIONAL / RESEARCH ONLY - not a medical device, not a diagnosis.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ECG Analyzer", page_icon="🫀", layout="centered")

DISCLAIMER = (
    "**⚠️ Educational / research use only.** This is **not** a medical device, "
    "does **not** provide a diagnosis, and can be wrong. ECG interpretation must "
    "be done by a qualified clinician. A doctor's opinion always comes first — "
    "if you have any health concern, seek professional medical care."
)


def disclaimer_banner():
    st.error(DISCLAIMER)


st.title("🫀 ECG Analyzer")
mode = st.sidebar.radio(
    "Mode",
    ["Clinical diagnosis (from ECG image)", "Heartbeat type (MIT-BIH beat)"],
)
disclaimer_banner()


# --------------------------------------------------------------------------- #
# Mode 1: image -> diagnosis
# --------------------------------------------------------------------------- #
def render_diagnosis_mode():
    from src.diagnostic.config import model_path, SIGNAL_LENGTH, SAMPLING_RATE
    st.subheader("Diagnostic screening from a single-lead ECG image")
    st.caption(
        "Upload a clean single-lead ECG strip (screenshot / scan). The image is "
        "digitized to a waveform, then a PTB-XL-trained 1D-ResNet estimates the "
        "likelihood of five diagnostic superclasses. Single-lead screening only — "
        "a full 12-lead ECG read by a clinician is far more reliable."
    )

    has_model = os.path.exists(model_path("lead2"))
    if not has_model:
        st.warning(
            "The diagnostic model isn't trained yet. Once PTB-XL is downloaded, run:\n\n"
            "```bash\npython main.py diagnose-train --leads lead2\n```\n\n"
            "Digitization still works below so you can preview the pipeline."
        )

    up = st.file_uploader("ECG strip image", type=["png", "jpg", "jpeg", "bmp"])
    if up is None:
        st.info("Upload an ECG strip image to begin.")
        return

    import cv2
    file_bytes = np.frombuffer(up.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)

    from src.digitization import digitize_ecg_image
    try:
        dig = digitize_ecg_image(img, target_len=SIGNAL_LENGTH, target_fs=SAMPLING_RATE)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not digitize the image: {e}")
        return

    st.subheader("Digitized waveform")
    st.line_chart(pd.DataFrame({"amplitude": dig.signal}))
    cal = "✅ grid-calibrated (mV)" if dig.calibrated else "⚠️ uncalibrated (shape only)"
    st.caption(f"Calibration: {cal}")

    if not has_model:
        return

    if st.button("Analyze", type="primary"):
        from src.diagnose import diagnose_image
        result = diagnose_image(img, lead_config="lead2", with_saliency=True)
        if result.get("warning"):
            st.warning(result["warning"])

        st.subheader("Diagnostic estimate")
        for f in result["findings"]:
            label = f"**{f['name']}**" + ("  🚩" if f["flagged"] else "")
            st.write(label)
            st.progress(min(1.0, f["probability"]), text=f"{f['probability']:.1%}")

        sal = result.get("saliency")
        if sal is not None:
            st.subheader("Where the model focused (Grad-CAM)")
            _plot_saliency(dig.signal, sal)
        st.info("This is a screening estimate, not a diagnosis. " + DISCLAIMER)


def _plot_saliency(signal, saliency):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(signal))
    ax.plot(x, signal, color="black", linewidth=1)
    ax.scatter(x, signal, c=saliency, cmap="Reds", s=6)
    ax.set_yticks([])
    ax.set_xlabel("time step")
    ax.set_title("Saliency (red = more influential)")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Mode 2: MIT-BIH beat classifier
# --------------------------------------------------------------------------- #
def render_beat_mode():
    from src.config import TEST_CSV, MODEL_PATH, N_FEATURES, CLASS_NAMES
    st.subheader("Heartbeat type classifier (MIT-BIH)")
    st.caption("Classifies a single pre-segmented heartbeat into 5 AAMI beat types.")

    if not os.path.exists(MODEL_PATH):
        st.warning("Beat model not trained. Run `python main.py train`.")
        return

    from src.predict import predict_heartbeat, predict_proba, load_prediction_resources
    if not load_prediction_resources():
        st.error("Could not load the beat model/scaler.")
        return

    if os.path.exists(TEST_CSV):
        df = pd.read_csv(TEST_CSV, header=None)
        idx = st.slider("Test sample index", 0, len(df) - 1, 100)
        signal = df.iloc[idx, :N_FEATURES].values.astype(np.float32)
        actual = CLASS_NAMES[int(df.iloc[idx, N_FEATURES])]
    else:
        signal = np.random.rand(N_FEATURES).astype(np.float32)
        actual = None

    st.line_chart(pd.DataFrame({"amplitude": signal}))
    if st.button("Classify beat", type="primary"):
        name, conf = predict_heartbeat(signal)
        probs = predict_proba(signal)
        c1, c2 = st.columns(2)
        c1.metric("Predicted", name, f"{conf:.1%}")
        if actual is not None:
            c2.metric("Actual", actual, "✅" if actual == name else "❌")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))


if mode.startswith("Clinical"):
    render_diagnosis_mode()
else:
    render_beat_mode()

st.divider()
st.caption("PyTorch · MIT-BIH beat CNN + PTB-XL 1D-ResNet · OpenCV digitization. " + DISCLAIMER)
