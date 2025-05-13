# Enhance Speech Extraction And Speaker Identification

This project explores a hybrid approach for speaker identification by integrating feature extraction methods (MFCC, Wavelet, XVector, DVector) with classical and quantum machine learning models such as HMM, RNN, and Quantum Convolutional Neural Networks (QCNN).

## 📌 Overview

In the era of virtual communication, the demand for intelligent audio analysis is rapidly growing. This work focuses on improving **speech extraction** and **speaker identification** through machine learning and quantum deep learning techniques.

Our best-performing model — a hybrid **QCNN + HMM + DVector** architecture — achieved **97% accuracy** and **95% F1-score** across 3-fold cross-validation.

## 🧠 Techniques Used

- Feature Extraction:
  - Mel-Frequency Cepstral Coefficients (MFCC)
  - Wavelet Transform (DWT)
  - X-Vector and D-Vector embeddings
- Models:
  - Hidden Markov Models (HMM)
  - Recurrent Neural Networks (RNN)
  - Quantum Convolutional Neural Networks (QCNN)
- Hybrid: QCNN combined with HMM for enhanced temporal modeling

## 📊 Dataset

| Statistic                     | Value           |
|------------------------------|-----------------|
| Total number of speakers     | 83              |
| Total number of dialogues    | 10,430          |
| Number of speaker changes    | 751             |
| Total number of audio files  | 85              |
| Total duration               | 17.93 hours     |
| Average dialogue length      | 6.19 seconds    |
| Std. deviation of length     | 11.96 seconds   |

## 🧪 Experimental Results

> **Table: Accuracy, Precision, Recall, and F1-Score Across 3 Folds**

| **Metric / Fold**    | HMM-MFCC | HMM-XVec | HMM-Wavelet | HMM-DVec | RNN-MFCC | RNN-XVec | RNN-Wavelet | RNN-DVec | QCNN-MFCC | QCNN-Wavelet | QCNN-XVec | QCNN-DVec | QCNN-HMM-MFCC | QCNN-HMM-Wavelet | QCNN-HMM-XVec | **QCNN-HMM-DVec** |
|----------------------|----------|----------|-------------|----------|-----------|-----------|--------------|-----------|-------------|----------------|------------|-------------|------------------|--------------------|----------------|----------------------|
| **Acc Fold 1**       | 0.49     | 0.66     | 0.15        | 0.36     | 0.78      | 0.69      | 0.39         | 0.86      | 0.42        | 0.16           | 0.11       | 0.44        | 0.83             | 0.32               | 0.95           | **0.97**             |
| **Acc Fold 2**       | 0.55     | 0.71     | 0.18        | 0.36     | 0.79      | 0.70      | 0.39         | 0.86      | 0.42        | 0.21           | 0.11       | 0.49        | 0.82             | 0.33               | 0.95           | **0.97**             |
| **Acc Fold 3**       | 0.48     | 0.67     | 0.17        | 0.35     | 0.78      | 0.75      | 0.39         | 0.86      | 0.56        | 0.18           | 0.11       | 0.48        | 0.83             | 0.34               | 0.95           | **0.97**             |
| **Prec Fold 1**      | 0.74     | 0.67     | 0.23        | 0.04     | 0.59      | 0.27      | 0.15         | 0.80      | 0.06        | 0.04           | 0.00       | 0.07        | 0.89             | 0.26               | 0.96           | **0.97**             |
| **Prec Fold 2**      | 0.74     | 0.74     | 0.27        | 0.04     | 0.70      | 0.34      | 0.17         | 0.80      | 0.06        | 0.07           | 0.00       | 0.07        | 0.88             | 0.26               | 0.96           | **0.97**             |
| **Prec Fold 3**      | 0.76     | 0.66     | 0.22        | 0.04     | 0.63      | 0.39      | 0.17         | 0.80      | 0.36        | 0.04           | 0.00       | 0.08        | 0.88             | 0.27               | 0.97           | **0.98**             |
| **Recall Fold 1**    | 0.69     | 0.59     | 0.12        | 0.10     | 0.53      | 0.28      | 0.14         | 0.74      | 0.09        | 0.05           | 0.01       | 0.10        | 0.87             | 0.19               | 0.93           | **0.95**             |
| **Recall Fold 2**    | 0.73     | 0.62     | 0.15        | 0.10     | 0.58      | 0.32      | 0.15         | 0.73      | 0.07        | 0.06           | 0.01       | 0.13        | 0.85             | 0.19               | 0.90           | **0.92**             |
| **Recall Fold 3**    | 0.71     | 0.56     | 0.12        | 0.10     | 0.56      | 0.39      | 0.16         | 0.74      | 0.31        | 0.04           | 0.01       | 0.13        | 0.86             | 0.21               | 0.92           | **0.94**             |
| **F1 Fold 1**        | 0.64     | 0.59     | 0.11        | 0.05     | 0.54      | 0.25      | 0.13         | 0.75      | 0.07        | 0.03           | 0.00       | 0.07        | 0.87             | 0.18               | 0.94           | **0.96**             |
| **F1 Fold 2**        | 0.66     | 0.63     | 0.14        | 0.05     | 0.59      | 0.30      | 0.14         | 0.74      | 0.05        | 0.04           | 0.00       | 0.08        | 0.85             | 0.17               | 0.92           | **0.94**             |
| **F1 Fold 3**        | 0.66     | 0.57     | 0.13        | 0.05     | 0.57      | 0.37      | 0.15         | 0.75      | 0.30        | 0.03           | 0.00       | 0.09        | 0.86             | 0.19               | 0.94           | **0.95**             |

## 🔍 Key Findings

- **DVector** features combined with **QCNN + HMM** yielded the best performance.
- **Wavelet + HMM** was the weakest performer.
- **RNN-DVec** also showed strong results but was outperformed by the hybrid quantum model.

## 🛠 Tools & Frameworks

- Python
- [SpeechBrain](https://speechbrain.readthedocs.io/)
- Pennylane (Quantum ML)
- PyTorch
- Scikit-learn

## 📈 Future Work

- Expand to noisy environments
- Add multilingual and emotion-aware capabilities
- Real-time deployment for smart assistants or security systems

## 🤝 Acknowledgments

This research was supported by the VNUHCM–University of Information Technology’s Scientific Research Support Fund.

---

## 📄 Citation

If you use this work, please cite our paper:

> Cao Hoài Sang, Thi Thành Công, Nguyễn Minh Nhựt, Nguyễn Đình Thuân. *Enhancing Speech Extraction And Speaker Recognition Through Machine Learning And Deep Learning Integration*. UIT, 2024.

---
