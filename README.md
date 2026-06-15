# ECG Analysis Suite — Heartbeat Classification + Diagnosis from ECG Images

A PyTorch, GPU‑accelerated ECG analysis project with **two models** and an
**image‑to‑diagnosis** pipeline, wrapped in an interactive demo:

1. **Clinical diagnostic screening from an ECG image** — upload a single‑lead
   ECG strip; it is **digitized** to a waveform and a **PTB‑XL‑trained 1D‑ResNet**
   estimates five diagnostic superclasses, with calibrated confidence and a
   Grad‑CAM explanation.
2. **Heartbeat‑type classification** — a 1D‑CNN classifies a single
   pre‑segmented beat (MIT‑BIH) into 5 AAMI arrhythmia types (**98.5% test acc**).

> ## ⚠️ Medical disclaimer
> This project is for **education and research only**. It is **not a medical
> device**, does **not** provide a diagnosis, and can be wrong. ECG
> interpretation must be performed by a qualified clinician. **A doctor's
> opinion always comes first** — if you have any health concern, seek
> professional medical care.

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA-ee4c2c)
![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [Pipeline](#pipeline)
- [Results](#results)
- [Datasets](#datasets)
- [Models](#models)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Interactive Demo](#interactive-demo)
- [Limitations & Responsible Use](#limitations--responsible-use)
- [Testing](#testing)
- [License & Acknowledgements](#license--acknowledgements)

## Pipeline

```
        ECG strip image (single lead)
                  │
                  ▼
  OpenCV digitization  ── de-grid → trace extraction → mV calibration → resample
                  │
                  ▼
      1D-ResNet (PTB-XL, Lead II)
                  │
                  ▼
  temperature-calibrated probabilities + Grad-CAM saliency
                  │
                  ▼
   5 diagnostic superclasses  +  ⚠️ "not a diagnosis — see a clinician"
```

## Results

**Heartbeat classifier (MIT‑BIH, 5 beat types)** — 1D‑CNN, 30 epochs on an
RTX 3060:

| Metric | Value |
|---|---|
| Test accuracy | **98.49%** |
| Weighted F1 | 0.985 |

**Diagnostic model (PTB‑XL, 5 diagnostic superclasses)** — 1D‑ResNet, multi‑label:

| Configuration | Macro AUROC |
|---|---|
| 12‑lead (benchmark) | _pending training run_ |
| Single‑lead (Lead II, used by the image demo) | _pending training run_ |

> Single‑lead screening is inherently less reliable than a full 12‑lead read —
> the app states this explicitly and recommends clinician review.

## Datasets

| Dataset | Used for | Notes |
|---|---|---|
| [MIT‑BIH (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) | beat classifier | 109k single beats, 187 samples each |
| [PTB‑XL 1.0.3 (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.3/) | diagnostic model | 21,799 twelve‑lead 10 s ECGs, multi‑label diagnoses |

Both are large and **git‑ignored** (never committed). Get them with:

```bash
# MIT-BIH: download the two CSVs from Kaggle into data/
#   data/mitbih_train.csv, data/mitbih_test.csv

# PTB-XL: download + extract the 100 Hz subset
python scripts/get_ptbxl.py --download
```

## Models

**Beat CNN** (`src/model.py`) — 3 Conv1D blocks → dense head, input `(1, 187)`,
5 classes, `CrossEntropyLoss`, SMOTE for imbalance.

**Diagnostic 1D‑ResNet** (`src/diagnostic/model.py`) — ResNet‑18‑style 1D CNN,
input `(n_leads, 1000)` at 100 Hz, 5 multi‑label outputs (NORM, MI, STTC, CD,
HYP), `BCEWithLogitsLoss` with inverse‑frequency weighting, macro‑AUROC,
mixed‑precision training, temperature‑scaled probabilities. Lead count is
configurable: `12lead` (benchmark) or `lead2` (matches the single‑lead image input).

**Digitization** (`src/digitization.py`) — classical OpenCV: color‑aware trace
isolation (rejects the pink grid), content cropping, per‑column trace
extraction, grid‑spacing calibration to millivolts, and resampling. Targets
clean single‑lead strips; phone‑photo robustness is a known limitation.

## Project Structure

```
ECG-Heartbeat-Classification/
├── main.py                  # CLI: train / evaluate / predict / diagnose-*
├── scripts/get_ptbxl.py     # PTB-XL download + 100 Hz extraction
├── app/streamlit_app.py     # interactive demo (image diagnosis + beat classifier)
├── src/
│   ├── config.py model.py data_loader.py train.py evaluate.py predict.py
│   ├── digitization.py      # ECG image -> 1D signal
│   ├── diagnose.py          # image -> diagnosis (+ Grad-CAM, disclaimer)
│   └── diagnostic/          # PTB-XL track: config, data, model, train, evaluate
├── tests/                   # 19 pytest tests
├── data/  saved_models/  visualizations/  notebooks/
```

## Setup

Requires **Python 3.11**. For GPU training, an NVIDIA GPU + recent driver.

```bash
python -m venv .venv && .venv\Scripts\activate        # (Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU
pip install -r requirements.txt
```
CPU‑only: skip the CUDA line; `pip install -r requirements.txt` pulls CPU torch.

## Usage

```bash
# Beat classifier (MIT-BIH)
python main.py train                 # train the beat CNN
python main.py evaluate              # test metrics + confusion matrix
python main.py predict --sample-index 100

# Diagnostic model (PTB-XL)
python main.py diagnose-train --leads lead2     # single-lead (image demo)
python main.py diagnose-train --leads 12lead    # 12-lead benchmark
python main.py diagnose-eval  --leads 12lead    # per-class + macro AUROC

# Diagnose an ECG image from the CLI
python main.py diagnose-image --image path/to/ecg_strip.png
```

## Interactive Demo

```bash
streamlit run app/streamlit_app.py
```
Mode 1 — upload a single‑lead ECG strip → digitized waveform → calibrated
diagnostic estimate + Grad‑CAM. Mode 2 — the MIT‑BIH beat classifier. The
disclaimer is shown on every screen.

## Limitations & Responsible Use

- **Not a medical device / not a diagnosis.** Screening estimates only; always
  defer to a clinician.
- **Single‑lead** input is far less informative than a clinical 12‑lead ECG;
  many diagnoses genuinely require 12 leads.
- **Digitization** targets clean strips; skewed phone photos, 12‑lead sheets,
  and unusual layouts are out of scope and reduce reliability (flagged in‑app
  when grid calibration fails).
- The model reflects **PTB‑XL's population and label distribution** and will not
  generalize to all devices or patients.

## Testing

```bash
pytest -q          # 19 tests: preprocessing, models, digitization round-trip, Grad-CAM
```

## License & Acknowledgements

MIT License — see [LICENSE](LICENSE).

- **PTB‑XL**: Wagner et al., *PTB‑XL, a large publicly available ECG dataset*,
  Scientific Data (2020); PhysioNet.
- **MIT‑BIH**: Moody & Mark, *IEEE Eng Med Biol Mag* (2001); Goldberger et al.,
  *Circulation* (2000).
- Kaggle MIT‑BIH preparation by Shayan Fazeli.
