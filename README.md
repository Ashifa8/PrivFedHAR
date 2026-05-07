# PrivFedHAR: Privacy-Preserving Personalized Federated Learning for Human Activity Recognition

## Overview

PrivFedHAR is a privacy-preserving personalized federated learning framework for Human Activity Recognition (HAR) using wearable sensor data. The system is designed to handle non-IID data distributions, ensure user privacy using Differential Privacy (DP), and support subject-level personalization through Subject-Adaptive Layer Normalization (SALN).

The proposed approach integrates federated learning with deep temporal modeling and privacy mechanisms to enable secure and accurate human activity recognition in real-world healthcare environments.

---

## Key Components

- Hybrid GRU–LSTM encoder for temporal feature learning  
- Multi-Head Attention mechanism for discriminative feature extraction  
- Subject-Adaptive Layer Normalization (SALN) for client-specific personalization  
- Differential Privacy (DP-SGD) for formal (ε, δ)-DP guarantees  
- Federated Averaging (FedAvg) for decentralized training  

---

## Dataset

The framework is evaluated on the PAMAP2 dataset under a Leave-One-Subject-Out (LOSO) protocol.

Dataset characteristics:
- 9 subjects (8 male, 1 female)
- 18 physical activities
- 3 IMU sensors (hand, chest, ankle)
- 100 Hz sampling rate

### Dataset Access

https://drive.google.com/file/d/1P_7neoQg9JIubZ4gfRGzhjgbrusctuTJ/view?usp=drive_link

The dataset must be set to “anyone with link can view”.

---

## Preprocessing

- Removal of missing and invalid values (NaN filtering)  
- Selection of 15 activity classes  
- Sliding window segmentation:
  - Window size: 128  
  - Stride: 64  
- Leave-One-Subject-Out cross-validation setup  
- Standard normalization per subject  

---

## Experimental Setup

- Framework: Federated Learning (FedAvg)  
- Optimizer: Adam  
- Communication rounds: 30 (full model), 10 (ablation)  
- Differential Privacy: Gaussian noise injection  
  - σ = 0.8  
  - C = 1.0  
  - δ = 10⁻⁵  
- Implementation: PyTorch  

---

## Results

### LOSO Evaluation on PAMAP2

- Accuracy: 94.43% ± 4.13%  
- Macro F1-score: 92.78% ± 5.83%  
- AUC-ROC: 98.55% (excluding single-class subjects)  

The model shows consistent performance across unseen subjects under strict LOSO evaluation.

---

## Ablation Study

The contribution of each component is evaluated under controlled settings.

| Configuration              | Accuracy | F1-score | Observation |
|---------------------------|----------|----------|-------------|
| Full PrivFedHAR          | 94.43%   | 92.78%   | Best performance |
| Without SALN             | 53.15%   | 40.32%   | Significant performance drop |
| Without Differential Privacy | 97.41% | 96.94% | Slight improvement, no privacy |

SALN is the most critical component for performance, while DP introduces a small but acceptable utility trade-off.

---

## How to Run

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py

# Run ablation study
python ablation/run_ablation.py
