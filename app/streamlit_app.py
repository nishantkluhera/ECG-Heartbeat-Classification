"""ECG Insight - interactive ECG analysis demo (Streamlit).

A polished front-end for two models:
  1. Diagnostic screening from an ECG image  (digitize -> PTB-XL 1D-ResNet)
  2. Heartbeat-type classification           (MIT-BIH single-beat CNN)

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

st.set_page_config(page_title="ECG Insight — AI ECG Screening", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

REPO_URL = "https://github.com/nishantkluhera/ECG-Heartbeat-Classification"
SHORT_DISCLAIMER = ("Research & educational demo — not a medical device, not a diagnosis. "
                    "Always consult a qualified clinician.")


# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown("""
    <style>
      .stApp { background:
        radial-gradient(1200px 600px at 80% -10%, #16203a 0%, rgba(11,16,32,0) 55%),
        radial-gradient(900px 500px at -10% 0%, #141a36 0%, rgba(11,16,32,0) 50%), #0b1020; }
      .block-container { max-width: 940px; padding-top: 1.2rem; padding-bottom: 2rem; }
      header[data-testid="stHeader"] { background: transparent; }
      /* bordered containers -> elevated dark cards */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#121a2e; border:1px solid #25304d !important; border-radius:14px; }
      .hero { background: linear-gradient(135deg,#0ea5e9 0%,#6366f1 100%);
              color:#fff; border-radius:18px; padding:24px 28px; margin-bottom:14px;
              box-shadow:0 16px 44px -16px rgba(99,102,241,.7); }
      .hero h1 { margin:0; font-size:1.7rem; font-weight:800; letter-spacing:-.02em; }
      .hero p { margin:.4rem 0 0; opacity:.96; font-size:.96rem; max-width:660px; }
      .badges { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
      .badge { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.26);
               color:#fff; padding:4px 11px; border-radius:999px; font-size:.72rem; font-weight:600; }
      .notice { background:rgba(245,158,11,.10); border:1px solid rgba(245,158,11,.30);
                color:#fcd34d; border-radius:12px; padding:11px 14px; font-size:.83rem;
                margin-bottom:14px; }
      .notice b { color:#fde68a; }
      .sec { font-size:.74rem; text-transform:uppercase; letter-spacing:.07em;
             color:#8b9bb4; font-weight:700; margin-bottom:.4rem; }
      .topcard { background:#121a2e; border:1px solid #25304d; border-left:5px solid #818cf8;
                 border-radius:14px; padding:16px 20px; margin:4px 0 12px; }
      .topcard .lbl { color:#8b9bb4; font-size:.74rem; text-transform:uppercase;
                      letter-spacing:.07em; font-weight:700; }
      .topcard .val { font-size:1.5rem; font-weight:800; color:#f1f5f9; margin-top:3px; }
      .chip { display:inline-block; padding:3px 11px; border-radius:999px;
              font-size:.74rem; font-weight:700; vertical-align:middle; }
      .chip-ok { background:rgba(16,185,129,.14); color:#34d399; border:1px solid rgba(16,185,129,.4); }
      .chip-warn { background:rgba(245,158,11,.14); color:#fbbf24; border:1px solid rgba(245,158,11,.4); }
      .chip-info { background:rgba(129,140,248,.16); color:#a5b4fc; border:1px solid rgba(129,140,248,.45); }
      .pcard { background:#121a2e; border:1px solid #25304d; border-radius:14px;
               padding:14px 18px; margin-bottom:12px; }
      .prow { display:flex; align-items:center; gap:12px; margin:9px 0; }
      .plabel { width:200px; font-size:.86rem; color:#dbe2f0; font-weight:500; }
      .ptrack { flex:1; height:11px; background:#1e2742; border-radius:999px; overflow:hidden; }
      .pfill { height:100%; border-radius:999px; }
      .pval { width:48px; text-align:right; font-variant-numeric:tabular-nums;
              font-size:.84rem; color:#9fb0cc; font-weight:600; }
      .foot { color:#64748b; font-size:.78rem; text-align:center; margin-top:18px; }
      .foot a { color:#a5b4fc; }
      .mc { font-size:.83rem; color:#c3ccde; line-height:1.55; }
      .mc b { color:#f1f5f9; }
    </style>
    """, unsafe_allow_html=True)


def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)


def hero():
    st.markdown("""
    <div class="hero">
      <h1>🫀 ECG Insight</h1>
      <p>Deep-learning ECG screening — turn a single-lead ECG strip image into a
      calibrated estimate across five diagnostic categories, with an explanation
      of what the model looked at.</p>
      <div class="badges">
        <span class="badge">PyTorch · CNN ensemble</span>
        <span class="badge">PTB-XL · 0.935 macro-AUROC</span>
        <span class="badge">OpenCV digitization</span>
        <span class="badge">Grad-CAM explainability</span>
        <span class="badge">Calibrated probabilities</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="notice"><b>⚠️ {SHORT_DISCLAIMER}</b> '
                "A doctor's reading always comes first.</div>", unsafe_allow_html=True)


def prob_bars_html(findings):
    rows = []
    for f in findings:
        pct = f["probability"] * 100
        is_normal = f["class"] in ("NORM", "Normal")
        flagged = f["flagged"] and not is_normal
        if is_normal:
            color = "linear-gradient(90deg,#34d399,#10b981)"      # green = normal (good)
        elif flagged:
            color = "linear-gradient(90deg,#f59e0b,#ef4444)"      # warm = flagged abnormality
        else:
            color = "linear-gradient(90deg,#38bdf8,#6366f1)"      # cool = abnormal, low signal
        chip = ' <span class="chip chip-warn">review</span>' if flagged else ""
        rows.append(
            f'<div class="prow"><div class="plabel">{f["name"]}{chip}</div>'
            f'<div class="ptrack"><div class="pfill" style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'<div class="pval">{pct:.0f}%</div></div>'
        )
    return '<div class="pcard"><div class="sec">All categories</div>' + "".join(rows) + "</div>"


def model_card_sidebar():
    st.sidebar.markdown("#### Model card")
    st.sidebar.markdown("""
    <div class="mc">
      <b>Models</b>: 1D-ResNet / resnet1d_wang / inception1d<br>
      <b>Training data</b>: PTB-XL (21,799 clinical ECGs)<br>
      <b>Task</b>: multi-label, 5 superclasses<br>
      <b>Best 12-lead</b>: <b>0.935</b> macro-AUROC (ensemble + TTA)<br>
      <b>This demo</b>: single-lead II, 0.85 macro-AUROC<br>
      <b>Calibration</b>: temperature-scaled · <b>Digitization</b>: OpenCV → mV
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"[📦 Source code on GitHub]({REPO_URL})")


# --------------------------------------------------------------------------- #
# Mode 1: image -> diagnosis
# --------------------------------------------------------------------------- #
def render_diagnosis_mode():
    from src.diagnostic.config import model_path, SIGNAL_LENGTH, SAMPLING_RATE
    import cv2

    has_model = os.path.exists(model_path("lead2"))
    if not has_model:
        st.warning("Diagnostic model not trained yet. Run "
                   "`python main.py diagnose-train --leads lead2`. "
                   "Digitization preview still works below.")

    SAMPLES = {
        "Normal ECG": "assets/samples/sample_normal.png",
        "ST/T change": "assets/samples/sample_st_t_change.png",
        "Conduction disturbance": "assets/samples/sample_conduction_disturbance.png",
    }
    img = None
    with st.container(border=True):
        sec("1 · Choose an ECG")
        source = st.radio("Input", ["Try a sample", "Upload your own"],
                          horizontal=True, label_visibility="collapsed")
        if source == "Try a sample":
            choice = st.selectbox("Sample single-lead strip (real PTB-XL test records)",
                                  list(SAMPLES.keys()))
            img = cv2.imread(os.path.join(ROOT, SAMPLES[choice]), cv2.IMREAD_COLOR)
        else:
            up = st.file_uploader("Upload a single-lead ECG strip (PNG/JPG)",
                                  type=["png", "jpg", "jpeg", "bmp"])
            if up is not None:
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        st.info("Pick a sample or upload an ECG strip to begin.")
        return

    from src.digitization import digitize_ecg_image
    try:
        dig = digitize_ecg_image(img, target_len=SIGNAL_LENGTH, target_fs=SAMPLING_RATE)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not digitize the image: {e}")
        return

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            sec("Input image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        with st.container(border=True):
            sec("2 · Digitized waveform")
            st.line_chart(pd.DataFrame({"mV": dig.signal}), height=190)
            cal = ('<span class="chip chip-ok">grid-calibrated (mV)</span>' if dig.calibrated
                   else '<span class="chip chip-warn">uncalibrated — shape only</span>')
            st.markdown(cal, unsafe_allow_html=True)

    if not has_model:
        return

    if st.button("🔬 Analyze ECG", type="primary", use_container_width=True):
        with st.spinner("Running the diagnostic model…"):
            from src.diagnose import diagnose_image
            result = diagnose_image(img, lead_config="lead2", with_saliency=True)

        top = result["top"]
        conf = top["probability"]
        is_concern = top["flagged"] and top["class"] != "NORM"
        chip_cls = "chip-warn" if is_concern else "chip-info"
        chip_txt = "flagged for review" if is_concern else (
            "likely normal" if top["class"] == "NORM" else "below screening threshold")
        accent = "#10b981" if top["class"] == "NORM" else ("#ef4444" if is_concern else "#6366f1")
        st.markdown(f"""
        <div class="topcard" style="border-left-color:{accent};">
          <div class="lbl">Most likely category</div>
          <div class="val">{top['name']}
            <span class="chip {chip_cls}">{conf:.0%} · {chip_txt}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if result.get("warning"):
            st.markdown(f'<div class="notice">{result["warning"]}</div>', unsafe_allow_html=True)

        st.markdown(prob_bars_html(result["findings"]), unsafe_allow_html=True)

        sal = result.get("saliency")
        if sal is not None:
            with st.expander("🔍 Explainability — where the model focused (Grad-CAM)"):
                _plot_saliency(result["digitization"]["signal"], sal)

        st.markdown('<div class="notice">This is a <b>single-lead screening estimate</b>, '
                    'not a diagnosis. A full 12-lead ECG read by a clinician is substantially '
                    'more reliable.</div>', unsafe_allow_html=True)


def _plot_saliency(signal, saliency):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.6))
    x = np.arange(len(signal))
    ax.plot(x, signal, color="#334155", linewidth=1.1, zorder=2)
    ax.scatter(x, signal, c=saliency, cmap="plasma", s=7, zorder=3)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Brighter = more influential for the prediction", fontsize=9, color="#64748b")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Mode 2: MIT-BIH beat classifier
# --------------------------------------------------------------------------- #
def render_beat_mode():
    from src.config import TEST_CSV, MODEL_PATH, N_FEATURES, CLASS_NAMES

    st.markdown('<div class="sec">Heartbeat-type classifier · MIT-BIH</div>'
                '<p class="mc">Classifies a single pre-segmented heartbeat into 5 AAMI '
                'beat types (Normal, Supraventricular, Ventricular, Fusion, Unknown).</p>',
                unsafe_allow_html=True)

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

    with st.container(border=True):
        sec("Heartbeat waveform")
        st.line_chart(pd.DataFrame({"amplitude": signal}), height=190)

    if st.button("Classify beat", type="primary", use_container_width=True):
        name, conf = predict_heartbeat(signal)
        probs = predict_proba(signal)
        findings = [{"name": CLASS_NAMES[i], "class": CLASS_NAMES[i],
                     "probability": float(probs[i]), "flagged": False}
                    for i in range(len(CLASS_NAMES))]
        findings.sort(key=lambda d: d["probability"], reverse=True)

        verdict_html = ""
        if actual is not None:
            if actual == name:
                verdict_html = '<span class="chip chip-ok">✓ matches ground truth</span>'
            else:
                verdict_html = f'<span class="chip chip-warn">✗ actual: {actual}</span>'
        st.markdown(f"""
        <div class="topcard">
          <div class="lbl">Predicted beat type</div>
          <div class="val">{name}
            <span class="chip chip-info">{conf:.0%}</span> {verdict_html}
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(prob_bars_html(findings), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
inject_css()
hero()

st.sidebar.title("ECG Insight")
mode = st.sidebar.radio("Tool", ["🩺 Diagnose from ECG image", "💓 Heartbeat type (MIT-BIH)"])
st.sidebar.divider()
model_card_sidebar()
st.sidebar.divider()
with st.sidebar.expander("How it works"):
    st.markdown(
        "1. **Digitize** the strip image to a 1-D signal (OpenCV: de-grid → trace → mV).\n"
        "2. **Standardize** and run a **1D-ResNet** trained on PTB-XL.\n"
        "3. **Temperature-calibrated** probabilities for 5 superclasses.\n"
        "4. **Grad-CAM** highlights the influential regions.\n\n"
        "Trained both 12-lead and single-lead; the demo uses single-lead to match the image."
    )

if mode.startswith("🩺"):
    render_diagnosis_mode()
else:
    render_beat_mode()

st.markdown(
    f'<div class="foot">Built with PyTorch · OpenCV · Streamlit &nbsp;·&nbsp; '
    f'<a href="{REPO_URL}">source</a> &nbsp;·&nbsp; {SHORT_DISCLAIMER}</div>',
    unsafe_allow_html=True,
)
