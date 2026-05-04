# Kaggle Playground Series S6E2 - Portfolio Summary

## 1. Project Overview
- **Competition**: [Playground Series - Season 6, Episode 2](https://www.kaggle.com/competitions/playground-series-s6e2)
- **Objective**: Predict Heart Disease (Binary Classification)
- **Evaluation Metric**: Area Under the ROC Curve (AUC-ROC)
- **Development Period**: Feb 13, 2026 ~ Feb 20, 2026 (Active Phase)
- **Status**: Post-Deadline Study & ML Pipeline Optimization

## 2. Dynamic Performance Review
*Scores retrieved dynamically via Kaggle CLI API.*

- **Highest Public Score**: **0.95349** (Approx. Top 37%, Rank ~1617 / 4371)
- **Highest Private Score**: **0.95506**
- **Top 1 Private Score (Reference)**: **0.95535**
- **Gap to #1**: 단 **0.00029** 차이로, 매우 경쟁력 있는 모델(Private Score 기준)을 구축했습니다.

가장 높은 성과를 낸 모델은 **Hill Climbing Optimized Weights (CatBoost dominant)** 및 **Pseudo-Labeling (CatBoost) - Semi-Supervised Learning** 이었습니다. 이를 통해 CatBoost가 해당 합성 데이터셋(Synthetic Tabular Data)에서 가장 강력한 성능을 발휘함을 실증적으로 확인했습니다.

## 3. Core Approaches & Engineering

### A. Data Engineering
- **External Data Integration**: 모델의 일반화 성능을 높이기 위해 UCI Heart Disease 원본 데이터셋을 통합.
- **Feature Engineering**: 
    - 도메인 지식 기반 파생 변수 생성: `MaxHR_Age_Ratio` (나이 대비 심박수).
    - 범주형 변수의 정보량 극대화를 위한 `Target Encoding`.

### B. Modeling Pipeline
1. **Diverse Base Models**: CatBoost, LightGBM, XGBoost, Simple NN(MLP) 등 트리기반 및 신경망 앙상블을 통한 다양성 확보.
2. **Pseudo-Labeling**: Test 데이터 중 확신도가 높은(High Confidence) 샘플을 Train 셋에 편입시켜 재학습하는 Semi-Supervised Learning 적용.
3. **Advanced Ensemble**: `scipy.optimize.minimize(SLSQP)`를 활용한 **Hill Climbing 기법**으로 OOF(Out-of-Fold) AUC를 최대화하는 최적 가중치(Optimal Weights) 탐색.

### C. MLOps & Kaggle Pipeline
- **Kaggle API Integration**: `kaggle kernels push`와 `kernel-metadata.json`을 활용하여 로컬에서 Github으로 소스를 관리하고, Kaggle 클라우드 서버에서 비동기 학습을 수행하는 CI/CD 유사 파이프라인 구축.
- **Dynamic File Handling**: 데이터셋의 동적 경로 주입(argparse 활용) 및 Kaggle 환경 최적화 스크립트 작성.

## 4. Retrospective & Future Scope
과거 기록 분석 결과 `train_pseudo_round2.py` 및 `fix_submission.py` 등 후반부 코드에서 런타임 에러(SubmissionStatus.ERROR)가 발생했던 내역이 확인되었습니다.
- **Immediate Next Steps**: 
  - 과거 발생했던 에러 원인 분석 및 디버깅.
  - 상위 1% 리더보드 솔루션 벤치마킹을 통한 앙상블 가중치(Power Averaging 등) 재조정 테스트 (Late Submission 활용).
  - Github 중앙집중형 관리로 폴더(src, ensemble, models) 리팩토링.
