# ECG Heartbeat Classification using CNN

This project implements a 1D Convolutional Neural Network (CNN) using TensorFlow/Keras to classify ECG heartbeat signals into different arrhythmia categories based on the AAMI standard. It utilizes a publicly available ECG dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to automatically classify individual heartbeats from Electrocardiogram (ECG) signals. This is a crucial task in diagnosing cardiac arrhythmias. We use deep learning, specifically a 1D CNN, to learn features directly from the raw ECG signal segments corresponding to individual heartbeats. The model is trained to distinguish between Normal beats and several types of arrhythmic beats.

## Dataset

This project uses the **MIT-BIH Arrhythmia Database**, pre-processed and made available in CSV format. A common version is available on Kaggle:

- **Source:** [ECG Heartbeat Categorization Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (Derived from PhysioNet's MIT-BIH Arrhythmia Database)
- **Files:**
    - `mitbih_train.csv`: Training dataset.
    - `mitbih_test.csv`: Testing dataset.
- **Format:** Each row represents a single heartbeat segment. The first 187 columns are the ECG signal values (time steps), and the last column (188th) is the target class label.
- **Classes:** The labels correspond to the Association for the Advancement of Medical Instrumentation (AAMI) standard categories:
    - `0`: **N** - Normal beat
    - `1`: **S** - Supraventricular ectopic beat (SVEB)
    - `2`: **V** - Ventricular ectopic beat (VEB)
    - `3`: **F** - Fusion beat
    - `4`: **Q** - Unknown beat

**Important:** The dataset is known to be highly imbalanced (many Normal beats, fewer arrhythmic ones). This project includes an option to use SMOTE (Synthetic Minority Over-sampling TEchnique) during preprocessing to address this imbalance (see `src/config.py`).

**Download:**
1. Download the dataset files (`mitbih_train.csv` and `mitbih_test.csv`) from the Kaggle link above.
2. Place the downloaded CSV files inside the `data/` directory in the project's root folder.

## Model Architecture

The classification model is a 1D Convolutional Neural Network (CNN) built using TensorFlow/Keras. The architecture consists of:

1.  **Input Layer:** Expects segments of shape `(187, 1)`.
2.  **Convolutional Blocks:** Multiple blocks of `Conv1D`, `BatchNormalization`, `ReLU` activation, `MaxPooling1D`, and `Dropout` layers to extract hierarchical features from the time-series signal.
3.  **Flatten Layer:** To convert the 3D feature maps into a 1D vector.
4.  **Dense Blocks:** Fully connected (`Dense`) layers with `BatchNormalization`, `ReLU`, and `Dropout` for further feature processing and regularization.
5.  **Output Layer:** A `Dense` layer with `N_CLASSES` (5) units and `softmax` activation to output class probabilities.

The model is compiled using the Adam optimizer and categorical cross-entropy loss function. See `src/model.py` for the detailed implementation.

## Project Structure
ecg-heartbeat-classifier/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/                      # Data files (or instructions to download)
│   ├── .gitkeep               # Keep directory in git even if empty
│   └── mitbih_test.csv        # Example: Test dataset
│   └── mitbih_train.csv       # Example: Training dataset
│   └── (Download instructions in README)
│
├── notebooks/                 # Jupyter notebooks for exploration/visualization (Optional)
│   ├── 1_Data_Exploration.ipynb
│   └── 2_Model_Training_Experiments.ipynb
│
├── saved_models/              # Trained model files (added to .gitignore)
│   └── .gitkeep
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration variables (paths, parameters)
│   ├── data_loader.py         # Functions to load and preprocess data
│   ├── model.py               # Model definition (CNN)
│   ├── train.py               # Script to train the model
│   ├── evaluate.py            # Script to evaluate the model
│   └── predict.py             # Script/function for making predictions (Optional)
│   └── utils.py               # Utility functions (e.g., plotting)
│
└── visualizations/            # Saved plots/images (Optional)
    └── .gitkeep
    └── confusion_matrix.png
    └── training_history.png
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/ecg-heartbeat-classifier.git
    cd ecg-heartbeat-classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install TensorFlow. If you don't have a compatible GPU and CUDA installed, consider using `tensorflow-cpu` instead by modifying `requirements.txt` before installing.*

4.  **Download the dataset:**
    - Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).
    - Download `mitbih_train.csv` and `mitbih_test.csv`.
    - Place these files inside the `data/` directory.

## Usage

1.  **Configure Parameters (Optional):**
    - Edit `src/config.py` to adjust parameters like `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, or `APPLY_SMOTE`.

2.  **Train the Model:**
    - Run the training script from the project root directory:
      ```bash
      python src/train.py
      ```
    - This script will:
        - Load and preprocess the data (applying SMOTE if enabled in `config.py`).
        - Build the CNN model.
        - Train the model using the training data, validating on a split portion.
        - Save the best model found during training (based on validation accuracy) to the `saved_models/` directory.
        - Save plots of the training/validation accuracy and loss curves to the `visualizations/` directory.

3.  **Evaluate the Model:**
    - After training, evaluate the saved model on the test set:
      ```bash
      python src/evaluate.py
      ```
    - This script will:
        - Load the test data and the saved model.
        - Preprocess the test data (using the same scaler parameters as training).
        - Make predictions on the test set.
        - Print a classification report (precision, recall, F1-score per class).
        - Save a confusion matrix plot to the `visualizations/` directory.

4.  **Make Predictions (Example):**
    - Use the `src/predict.py` script to load the model and predict the class for a new ECG segment (example uses dummy data or a sample from the test set):
      ```bash
      python src/predict.py
      ```
    - You can adapt the `predict_heartbeat` function in `src/predict.py` to integrate it into other applications.

## Results

After running the evaluation script (`src/evaluate.py`), the following outputs are generated:

-   **Console Output:** Detailed classification report showing precision, recall, and F1-score for each heartbeat class, along with overall accuracy.
-   **`visualizations/confusion_matrix.png`:** A heatmap visualizing the model's predictions against the true labels for the test set. This helps identify which classes the model confuses.
-   **`visualizations/training_history.png`:** Plots showing the model's accuracy and loss on the training and validation sets over epochs during training. Helps diagnose overfitting or underfitting.

*(Optionally, add a summary of your specific results here after running the code, e.g., "The model achieved an overall accuracy of ~XX.X% on the test set. Performance was highest for Normal beats and lowest for Fusion beats, as expected due to class imbalance and morphological similarity...")*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You would need to add a LICENSE file, e.g., containing the standard MIT License text).

## Acknowledgements

-   This project uses the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
    - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag. 2001 May-Jun;20(3):45-50. PMID: 11446249.
    - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13). PMID: 10851218.
-   Dataset pre-processing inspiration from the Kaggle dataset by Shayan Fazeli.
