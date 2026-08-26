# Predictive Maintenance for Aerospace Systems — RUL Prediction

**Cheng (Tony) Ke, Felipe Villanueva, Yicun (Ethan) Wu**
Department of Electrical and Computer Engineering, University of Rochester

Predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines — i.e. how many more operating cycles an engine has before it needs maintenance — from multivariate sensor data, using the NASA CMAPSS FD001 dataset. Five modeling approaches (one statistical baseline, three CNN/RNN architectures, and one exploratory extension) are implemented and compared to find out which best captures engine degradation over time.

## Why this matters

Traditional maintenance is either reactive (fix it after it breaks) or scheduled at fixed intervals (which wastes good engine life or misses failures). Accurate RUL prediction lets maintenance be scheduled based on an engine's actual condition — reducing both unplanned failures and unnecessary inspections.

## Dataset

- **NASA CMAPSS FD001**: multivariate time-series sensor data from turbofan engines, each run to failure under a single operating condition and fault mode.
- 24 sensor + operational-setting measurements per cycle; RUL for each cycle computed as `(engine's final cycle) − (current cycle)`.
- Data is normalized and split into **sliding windows of 30 timesteps** per sample, so each model sees a short history of engine behavior rather than a single snapshot.

## Models compared

| Model | Approach |
|---|---|
| Linear Regression | Baseline — flattened sensor windows into feature vectors |
| 1D CNN | Learns local temporal patterns directly from raw sensor sequences |
| 2D CNN (STFT) | Converts sensor windows into time-frequency spectrograms, learns from those | 
| LSTM | Recurrent model with memory cells, captures long-range temporal dependencies |
| BiLSTM | Extends LSTM to process the sequence in both directions |

All neural models trained with the Adam optimizer and MSE loss for 25 epochs; evaluated on held-out validation data using RMSE, MAE, and R².

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 37.53 | 28.14 | 0.63 |
| 1D CNN | 19.96 | 14.55 | 0.90 |
| 2D CNN (STFT) | 27.17 | 20.29 | 0.81 |
| **LSTM** | **12.74** | **8.66** | **0.96** |
| BiLSTM | 112.00 | 94.00 | −2.00 |

**LSTM performed best** — its memory cells captured the long-term, gradual nature of engine degradation more effectively than convolutional or statistical approaches. The 2D CNN's spectrogram conversion lost temporal resolution and underperformed the time-domain models. BiLSTM performed far worse than plain LSTM (R² well below zero, meaning its predictions were less accurate than simply predicting the average) — likely because bidirectional context added little on this single-operating-condition dataset while making the model harder to optimize.

## Repository structure

```
ECE208_final_project.ipynb   # full pipeline: data loading, preprocessing, all 5 models, evaluation, plots
CMAPSSData/
└── train_FD001.txt          # NASA CMAPSS FD001 training data (not included — download separately)
```

## Running it

1. Download the CMAPSS FD001 dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place `train_FD001.txt` in a `CMAPSSData/` folder.
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `torch`, `matplotlib`.
3. Run `ECE208_final_project.ipynb` top to bottom.

## Team contributions

- **Cheng (Tony) Ke** — 1D CNN and LSTM: sequential data preprocessing, architecture design, training/evaluation, and analysis of temporal learning capability.
- **Felipe Villanueva** — dataset exploration and system organization: CMAPSS structure analysis, project pipeline/block diagram, codebase revisions, and the BiLSTM extension.
- **Yicun (Ethan) Wu** — 2D CNN and Linear Regression: STFT spectrogram transformation for the 2D CNN, baseline Linear Regression model, and comparative evaluation of frequency-domain vs. statistical-feature approaches.

## References

1. NASA Prognostics Center of Excellence Data Set Repository. [nasa.gov](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
2. Y. Zhao, "RNN, LSTM, and Bidirectional LSTM: Complete Guide," DagsHub Blog, 2023.
3. F. Xue, G. Jin, L. Tan, C. Zhang, and Y. Yu, "Predictive maintenance programs for aircraft engines based on remaining useful life prediction," *Scientific Reports*, vol. 15, 2025.
