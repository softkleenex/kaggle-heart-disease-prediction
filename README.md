# Kaggle Playground Series S6E2 - Heart Disease Prediction 🫀

[![Kaggle](https://img.shields.io/badge/Kaggle-Competitions-blue?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e2)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed/Portfolio-success.svg)]()

이 레포지토리는 [Kaggle Playground Series - Season 6, Episode 2](https://www.kaggle.com/competitions/playground-series-s6e2) 대회의 심장 질환 예측(Heart Disease Prediction) 솔루션을 담고 있습니다. 
단순한 모델 학습을 넘어, **MLOps 기반의 클라우드 자동화 파이프라인 구축**과 **합성 데이터(Synthetic Data)에 최적화된 앙상블 기법**을 연구하는 데 초점을 맞추었습니다.

---

## 🏆 Final Results & Performance
*   **Highest Private Score**: **0.95506** (AUC-ROC)
*   **Highest Public Score**: **0.95349** (Rank ~1617 / 4371, Top 37%)
*   **1위 솔루션(0.95535)과의 격차**: 단 **0.00029**

모델의 단일 성능 및 Hill Climbing 앙상블을 통해 프라이빗 리더보드 기준 1위권 점수에 근접하는 매우 강력한 모델을 구축하는 데 성공했습니다.

---

## 🧠 Core Engineering Strategy (과정 및 분석)

이 프로젝트는 크게 **Data Engineering, Advanced Modeling, MLOps** 세 가지 축으로 진행되었습니다.

### 1. Data Engineering & Feature Extraction
*   **External Data Integration**: 합성 데이터의 노이즈를 완화하고 견고한 결정 경계를 형성하기 위해 UCI Heart Disease 원본 데이터셋(920 rows)을 학습에 통합했습니다.
*   **Domain Knowledge Features**: `MaxHR_Age_Ratio` (나이 대비 최대 심박수)와 같은 도메인 지식 기반 파생 변수를 생성하여 모델의 예측력을 높였습니다.
*   **Target Encoding**: `Thallium`, `Vessels` 등 주요 범주형 변수의 정보량을 극대화하기 위해 Target Encoding을 적극 활용했습니다.

### 2. Modeling & Advanced Ensembling
*   **CatBoost Dominance**: 합성 탭ULAR 데이터(Synthetic Tabular Data)에서 범주형 변수 처리에 압도적인 성능을 보이는 CatBoost를 주력 모델로 채택했습니다.
*   **Semi-Supervised Learning (Pseudo-Labeling)**: Test 데이터 중 모델의 예측 확신도(Confidence)가 98% 이상인 샘플들을 추출하여 Train 셋에 편입시켜 재학습(Re-training)하는 방식으로 단일 모델의 일반화 성능을 크게 끌어올렸습니다.
*   **Hill Climbing Optimization (SLSQP)**: 단순 가중 평균(Weighted Average) 앙상블이 아닌, `scipy.optimize.minimize`의 **SLSQP 알고리즘**을 사용하여 OOF(Out-of-Fold) AUC를 극대화하는 수학적 최적 가중치를 탐색했습니다. (최적화 결과, CatBoost에 0.85라는 높은 가중치가 할당됨을 확인했습니다.)

### 3. MLOps & CI/CD Pipeline (Local to Cloud)
로컬 컴퓨팅 자원의 한계를 극복하고 모델링 생산성을 높이기 위해 Kaggle API를 활용한 원격 배포(Remote Execution) 파이프라인을 구축했습니다.
*   **Centralized Github Repo**: 모든 모델 구조와 데이터 전처리 로직은 객체 지향 및 모듈화하여 Github에서 중앙 관리합니다. (`src/` 폴더 참조)
*   **Kaggle CLI Deployment**: `kaggle kernels push` 와 `kernel-metadata.json`을 사용하여 로컬에서 코드를 작성한 뒤, Kaggle 클라우드 서버 백그라운드 환경으로 비동기(Asynchronous) 훈련을 위임하는 워크플로우를 정립했습니다.

---

## 🛠 Tech Stack
*   **Languages & Frameworks**: Python 3.10+, Pandas, Numpy, Scikit-Learn
*   **Machine Learning**: CatBoost, LightGBM, XGBoost, PyTorch (Simple NN)
*   **Optimization**: SciPy (`scipy.optimize.minimize`), Optuna
*   **MLOps & Tools**: Kaggle CLI, Git, Gemini CLI

---

## 🚀 Quick Start / Reproducibility

이 프로젝트를 로컬 환경 또는 본인의 Kaggle 계정에서 재현하는 방법입니다.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/softkleenex/kaggle-heart-disease-prediction.git
   cd kaggle-heart-disease-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure Kaggle API is configured in `~/.kaggle/kaggle.json`)*

3. **Fetch Data**:
   ```bash
   kaggle competitions download -c playground-series-s6e2 -p data/
   unzip data/playground-series-s6e2.zip -d data/
   ```

4. **Run Training Pipeline (Local or Kaggle Cloud)**:
   *   **Local**: `python src/models/train_catboost.py`
   *   **Kaggle Cloud (Recommended)**: `kaggle kernels push -p notebooks/kaggle_runner`

---

## 💡 Retrospective (사후 분석 및 피드백)

이 프로젝트를 진행하며 겪은 실패와 교훈은 다음과 같습니다. 자세한 분석은 [Retrospective Report](docs/retrospective_analysis.md)에서 확인할 수 있습니다.

*   **Defensive Programming의 부재**: 대회 막바지 `train_pseudo_round2.py` 등의 앙상블 스크립트에서 Kaggle 시스템 상 `SubmissionStatus.ERROR`가 여러 차례 발생했습니다. 원인 분석 결과, 예측 확률값이 `[0, 1]`의 범위를 미세하게 벗어났거나, Test 셋의 `id` 인덱스가 어긋났을 가능성이 컸습니다.
*   **Actionable Feedback**: 이를 통해 파이프라인의 가장 마지막 단계에는 반드시 `assert submission['Heart Disease'].between(0, 1).all()`과 같은 **Post-Processing Validation(자동 검증 로직)**을 포함해야 한다는 매우 중요한 엔지니어링 교훈을 얻었습니다.
*   **앙상블의 함정**: 무작정 많은 모델을 엮었던 `ultimate_winning_ensemble` 시도는 오히려 성능을 하락시켰습니다. 상관관계(Correlation)가 낮고 각각의 예측력이 강력한 소수의 정예 모델(CatBoost + LGBM) 구성이 훨씬 견고함을 실증했습니다.

---

## 📂 Repository Structure (Production-Ready)

과거 수십 개의 노트북과 스크립트가 혼재되어 있던 구조를 프로덕션 레벨에 맞게 모듈화 및 리팩토링했습니다.

```text
├── docs/
│   ├── archive/                # 과거 정적 분석 문서 아카이브
│   ├── portfolio_summary.md    # 포트폴리오용 성과 요약본
│   └── retrospective_analysis.md # 상세 사후 분석 및 엔지니어링 피드백
├── notebooks/
│   └── kaggle_runner/          # Kaggle Cloud 실행을 위한 Entry Point 스크립트
├── src/
│   ├── data_processing/        # 데이터 증강(Augmentation) 및 피처 엔지니어링
│   ├── ensemble/               # Stacking, Rank Averaging, Hill Climbing 로직
│   ├── models/                 # CatBoost, LGBM, NN 등 개별 학습 모듈
│   ├── pseudo_labeling/        # Semi-Supervised Learning 로직
│   └── utils_and_analysis/     # 제출 검증(Fix) 및 피처 중요도 분석 툴
└── README.md                   # You are here!
```

---
*Developed & MLOps automated via Gemini CLI*

<!-- BLOG-URL:START -->

## Blog

- Blog note: [Kaggle Playground Series S6E2 - Heart Disease Prediction 🫀](https://softkleenex.github.io/coding_training/kaggle/kaggle-heart-disease-prediction)

<!-- BLOG-URL:END -->
