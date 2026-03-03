# 🫀 Kaggle Playground Series S6E2: Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-yellow)

## 📌 Project Overview
This repository contains the code and methodology for the **Kaggle Playground Series Season 6, Episode 2 (S6E2)**. The objective of this competition is to predict the presence of heart disease based on various medical indicators.

This project was built to explore and implement advanced machine learning pipelines, from aggressive feature engineering to complex ensemble optimization strategies.

## 🏆 Competition Results
- **Private Leaderboard Score (ROC AUC):** `0.95506`
- **Public Leaderboard Score (ROC AUC):** `0.95349`
- **Standing:** Top 35% (approx. 1568 / 4371)

## 🚀 Key Strategies & Innovations

### 1. Advanced Feature Engineering
- **Domain-Specific Features:** Derived features such as `MaxHR_Age_Ratio` (Maximum Heart Rate to Age ratio) to capture physiological stress limits.
- **Target Encoding:** Applied robust target encoding with smoothing for high-cardinality categorical variables like `Thallium` and `Vessels`.
- **Interaction Terms:** Created non-linear interaction features across critical clinical indicators to help tree-based models split more effectively.

### 2. Semi-Supervised Learning (Pseudo-Labeling)
- Leveraged the test set by predicting initial probabilities with a strong baseline model.
- High-confidence predictions were extracted and appended to the training set as pseudo-labels.
- **Result:** Improved local Cross-Validation (CV) scores to `0.961+` by exposing the model to the target distribution of the test set.

### 3. Model Diversity
A robust stacking framework was built using diverse architectures:
- **CatBoost:** Excelled at handling categorical features natively (achieved the highest single-model score).
- **LightGBM & XGBoost:** Provided fast, stable, and highly tuned baseline predictions.
- **Simple Neural Network (MLP):** Added to the ensemble to capture non-linear relationships that tree-based models might miss.

### 4. Ensemble Optimization (Hill Climbing / SLSQP)
Instead of simple averaging, the final ensemble weights were dynamically optimized using **Sequential Least SQuares Programming (SLSQP)**:
- **Objective:** Maximize Out-of-Fold (OOF) ROC AUC.
- **Constraints:** Weights must sum to 1 and be bounded between [0, 1].
- The optimizer mathematically proved the dominance of the CatBoost model for this specific dataset while finding the exact fractional contributions of the other models to maximize the metric.

## 📂 Project Structure

```text
├── src/
│   ├── analyze_importance.py     # Feature importance extraction
│   ├── augment_data.py           # Integration of original UCI dataset
│   ├── train_*.py                # Model training scripts (LGBM, XGB, CatBoost, NN)
│   ├── stacking.py               # Out-of-fold generation & Stacking logic
│   ├── hill_climbing_v2.py       # SLSQP ensemble weight optimization
│   └── train_pseudo_round2.py    # Multi-round pseudo-labeling pipeline
├── docs/
│   ├── final_report.md           # Detailed experiment logs
│   ├── eda_report.md             # Exploratory Data Analysis
│   └── workflow_strategy.md      # MLOps and Kaggle submission strategy
├── notebooks/
│   └── kaggle_runner/            # Kaggle cloud execution wrappers
└── submissions/                  # Final prediction outputs
```

## ⚙️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train Base Models:**
   ```bash
   python src/train_lgbm.py
   python src/train_catboost.py
   python src/train_xgb.py
   python src/train_nn.py
   ```
3. **Optimize Ensemble:**
   ```bash
   python src/hill_climbing_v2.py
   ```

## 📝 Future Improvements
- Implement TabNet or ResNet architectures for tabular data to increase deep learning diversity.
- Apply Adversarial Validation to identify and filter out training samples that significantly differ from the test distribution.
- Implement Power Averaging as a post-processing step for extreme probability calibration.
