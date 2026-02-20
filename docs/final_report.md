# Kaggle Playground Series S6E2 - Final Report

## 1. Project Overview
- **Objective**: Predict Heart Disease (Binary Classification)
- **Status**: Completed (Model Development & Final Submission)
- **Best Strategy**: Hill Climbing Ensemble (SLSQP Optimized) + Pseudo-Labeling

## 2. Key Strategies & Experiments

### A. Data Engineering
- **Augmentation**: UCI Heart Disease 원본 데이터셋(920 rows)을 통합하여 학습 데이터 보강.
- **Feature Engineering**: 
    - `MaxHR_Age_Ratio`: 나이 대비 심박수 (Key Feature).
    - `Target Encoding`: `Thallium`, `Vessels` 등 범주형 변수의 정보량 극대화.

### B. Modeling Pipeline
1. **Diverse Base Models**:
    - **CatBoost**: Categorical Feature 처리에 탁월 (단일 모델 최고 성능).
    - **LightGBM / XGBoost**: 안정적인 성능 및 다양성 확보.
    - **Simple NN**: MLP 구조로 비선형 패턴 학습 (Ensemble 기여).
2. **Optimization**:
    - **Optuna**: 각 모델의 하이퍼파라미터 정밀 튜닝.
    - **Pseudo-Labeling**: Test 데이터 중 확신도 높은 샘플을 Train에 추가하여 재학습 (CV 0.961+ 달성).
3. **Ensemble**:
    - **Hill Climbing (v2)**: `scipy.optimize.minimize(SLSQP)`를 사용하여 OOF AUC를 최대화하는 가중치 산출.
    - **CatBoost Dominance**: 최적화 결과 CatBoost의 가중치가 압도적으로 높음(약 0.85).

## 3. Kaggle Cloud Execution (MLOps)
로컬 자원 한계를 극복하기 위해 `runner.py` 패턴을 정립함.
- **Dynamic Path Injection**: 데이터 경로를 하드코딩하지 않고 `argparse`로 주입.
- **Dynamic Search**: `/kaggle/input` 하위를 탐색하여 데이터셋 위치 자동 식별.
- **Result Handling**: 실행 후 불필요한 파일을 정리하고 결과물만 Output으로 남김.

## 4. Final Submission Files
- `submissions/hill_climbing_v2.csv`: SLSQP 최적화 가중치 적용 (Most Recommended).
- `submissions/pseudo_labeling_final.csv`: Pseudo-Labeling 적용 단일 모델.
- `submissions/ultimate_winning_ensemble_fixed.csv`: 수동 가중치 앙상블.

## 5. Future Work (If continuing)
- **Deep Learning**: TabNet 또는 ResNet 구조를 도입하여 앙상블 다양성 추가 확보.
- **Adversarial Validation**: Train/Test 분포 차이가 큰 샘플 제거.
- **Post-Processing**: 리더보드 1위권 도약을 위한 Power Averaging 실험.
