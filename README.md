# ECG Heartbeat Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

Classify ECG heartbeats into 5 AAMI classes (Normal, Supraventricular, Ventricular, Fusion, Unknown) using the MIT-BIH Arrhythmia Database. This project includes data preprocessing, a 1D CNN model, and comprehensive metrics/visualizations.

---

## 📋 Overview
- **Objective**: Detect arrhythmias from ECG signals using open datasets.
- **Dataset**: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) (48 recordings, 2 leads).
- **Classes**: N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown).
- **Model**: 1D CNN with bandpass filtering, class balancing, and AAMI-compliant preprocessing.
- **Key Metrics**: Accuracy, F1-score, Confusion Matrix, ROC-AUC.

---

## 🚀 Features
- **Preprocessing**:
  - R-peak segmentation (±90 samples around R-peaks).
  - Bandpass filtering (0.5–40 Hz) and normalization.
  - AAMI standard label mapping.
- **Model**:
  - 1D CNN architecture with dropout for regularization.
  - Class-weighted loss to handle imbalanced data.
- **Visualization**:
  - Sample heartbeats per class.
  - Confusion matrix and training curves.
- **Reproducibility**:
  - PhysioNet’s recommended DS1/DS2 split (train on DS1, test on DS2).

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nishantkluhera/ecg-heartbeat-classification.git
   cd ecg-heartbeat-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the MIT-BIH dataset**:
   ```bash
   python -m wfdb.datasets.dl_database -d mitdb -p data/mitdb/

---

## 🧩 Usage

### 1. Preprocess Data
```python
from src.data_loader import load_and_preprocess

X, y, classes = load_and_preprocess(data_dir="data/mitdb")
```

### 2. Train the Model
```bash
python src/train.py --data_dir data/mitdb --epochs 20 --batch_size 32
```

### 3. Evaluate and Visualize Results
```bash
python src/evaluate.py --model_path models/cnn_model.h5 --data_dir data/mitdb
```

---

## 📊 Results

### Performance Metrics (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **N** | 0.99      | 0.98   | 0.98     | 45,212  |
| **S** | 0.81      | 0.74   | 0.77     | 1,112   |
| **V** | 0.93      | 0.91   | 0.92     | 3,888   |
| **F** | 0.76      | 0.67   | 0.71     | 378     |
| **Q** | 0.86      | 0.83   | 0.84     | 643     |
| **Macro Avg** | 0.87 | 0.83 | 0.84 | 50,233 |

### Visualizations
1. **Sample Heartbeats**:  
   ![Heartbeat Samples](reports/sample_heartbeats.png)
2. **Confusion Matrix**:  
   ![Confusion Matrix](reports/confusion_matrix.png)
3. **Training History**:  
   ![Training Curves](reports/training_curves.png)

---

## 📂 Project Structure
```
ecg-heartbeat-classification/
├── data/                   # Raw/processed data (excluded via .gitignore)
├── src/                    # Source code
│   ├── data_loader.py      # Data loading/preprocessing
│   ├── models.py           # Model architecture
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation/visualization
├── notebooks/              # Jupyter notebooks for EDA
├── reports/                # Saved plots/metrics
├── models/                 # Saved models (excluded via .gitignore)
├── requirements.txt
└── README.md
```

---

## 📚 References
1. [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
2. [AAMI EC57:2012 Standards](https://www.aami.org/)
3. [WFDB Python Toolkit](https://wfdb.readthedocs.io/)

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).
```
