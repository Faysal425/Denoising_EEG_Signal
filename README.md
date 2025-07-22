# DFA-ELM: Dual-Stage EEG Framework for Cognitive Workload Assessment

This repository provides the implementation of a two-stage neural framework for real-world EEG-based cognitive workload analysis. The system consists of:

1. **EEG Signal Reconstruction**: Denoises EEG signals using a deep encoder-decoder with advanced attention mechanisms.
2. **EEG Signal Classification**: Predicts workload levels from clean/reconstructed/noisy signals using a lightweight CNN-based architecture with Extreme Learning Machine (ELM) classifier.

The framework is robust to various artifacts (e.g., EMG, EOG, ECG, PL noise) and has been validated on public and self-collected EEG datasets.

---

# 📁 Repository Structure
DFA-ELM/
├── DFAELM_reconstruction.py # EEG reconstruction model (Stage 1)
├── DFAELM_classification.py # EEG classification model (Stage 2)
├── data/ # Folder for storing datasets (MAT/self-collected)
├── models/ # Optional: pretrained models can be stored here
├── utils/ # Optional: utility scripts (e.g., metrics, preprocessing)
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

# Installation

1. Clone the repository:

```python
git clone https://github.com/yourusername/DFA-ELM.git
cd DFA-ELM
```
2. Create a virtual environment (optional but recommended):
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```python
pip install -r requirements.txt
```
Note: You must have PyTorch >= 1.12, scikit-learn, and imbalanced-learn installed.

# Usage
## Stage 1: EEG Signal Reconstruction
Run the reconstruction model to denoise EEG data (including powerline, EMG, EOG, and ECG noise):
```python
python DFAELM_reconstruction.py
```
The model outputs reconstructed EEG signals and learned latent embeddings.

## Stage 2: EEG Classification using Reconstructed Signals
Train the classification model on clean, noisy, or reconstructed EEG signals:

```python
python DFAELM_classification.py
```
The model applies:
- Channel & temporal attention
- Multi-head self-attention (MHSA)
- Multiscale fusion
- Residual ELM classifier
- Metrics such as accuracy, loss, and training plots are logged during execution.

 Dataset
MAT Dataset: Public dataset used for baseline evaluation.

Self-Collected Dataset: EEG recorded under realistic workload scenarios with ethical approval.

To use the provided data:

```python
Place your `.edf` or `.EDF` EEG files in the `data/` folder
```
For more information on data structure and preprocessing, refer to the corresponding sections inside the scripts.

# Results
- Reconstruction: Achieved SNR improvement of +12 dB over noisy signals.
- Classification: Accuracy of >95% on MAT dataset and >93% on real-world data using reconstructed signals.
- Interpretability: Topographic and spectral plots confirm spatial signal fidelity and denoising effectiveness.

# License
This project is licensed under the MIT License.

# Citation
If you use this code or dataset in your work, please cite:

```scss
(Will be updated upon paper acceptance)
```
