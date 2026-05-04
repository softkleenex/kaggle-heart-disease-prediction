# 📊 Retrospective & Process Analysis: Heart Disease Prediction

## 1. Introduction
이 문서는 Kaggle Playground Series S6E2 (Heart Disease Prediction) 대회에 참여하며 겪은 전체적인 머신러닝 파이프라인 설계 과정, 직면했던 문제들, 그리고 이를 해결하기 위한 전략과 피드백을 상세히 기록한 사후 분석(Post-Mortem) 보고서입니다.

대회는 공식적으로 종료되었으나, 해당 프로젝트를 통해 **로컬 환경과 클라우드 환경(Kaggle Notebook)을 매끄럽게 연결하는 MLOps 워크플로우**를 완성하고, 합성 데이터(Synthetic Data)에 대한 앙상블 기법을 극대화하는 테스트베드로 활용하였습니다.

---

## 2. ML Pipeline & MLOps Architecture

단순히 주피터 노트북에서 코드를 한 줄씩 실행하는 것을 넘어, Github 기반의 코드 형상 관리와 Kaggle API를 활용한 원격 실행(Remote Execution) 파이프라인을 구축했습니다.

### 2.1. Local to Cloud Workflow (Kaggle CLI)
*   **Centralized Codebase (`src/`)**: 모든 모델, 데이터 전처리, 앙상블 코드는 `.py` 모듈로 분리하여 로컬 IDE(VSCode/Cursor)에서 객체 지향 및 함수 형태로 개발.
*   **Notebook Runner (`notebooks/kaggle_runner/runner.py`)**: Kaggle Cloud에서 백그라운드로 코드를 실행하기 위한 Entry Point.
*   **Deployment via API**: `kaggle kernels push` 명령어를 통해 로컬에서 개발한 코드를 Github에 푸시하는 동시에 Kaggle 서버로 배포하여 모델을 훈련시킴. 이 구조를 통해 무거운 컴퓨팅 리소스를 로컬에서 부담하지 않고 클라우드로 오프로딩(Off-loading) 할 수 있었습니다.

---

## 3. Detailed Process & Feedback (What Went Right & Wrong)

### ✅ What Went Right (성공 요인)

1. **Hill Climbing 앙상블의 극대화 (0.95506 Private Score 달성)**
    *   **접근법**: 일반적인 가중 평균(Weighted Average) 대신 `scipy.optimize.minimize`의 **SLSQP(Sequential Least SQuares Programming)** 알고리즘을 사용하여 OOF(Out-of-Fold) AUC를 최대화하는 가중치를 수학적으로 탐색했습니다.
    *   **성과**: 이 최적화 결과 **CatBoost** 모델에 매우 높은 가중치(약 0.85)가 부여되었습니다. 트리 기반 부스팅 모델 중 범주형 변수에 특히 강한 CatBoost가 Tabular 합성 데이터에서 핵심적인 역할을 한다는 것을 실증적으로 증명했습니다. 결과적으로 1위 점수(0.95535)와 단 **0.00029점 차이**라는 뛰어난 성과를 기록했습니다.

2. **Semi-Supervised Learning (Pseudo-Labeling)**
    *   **접근법**: 모델의 신뢰도(Confidence)가 98% 이상인 Test 데이터를 선별하여 이를 Train 데이터에 포함시킨 뒤 모델을 재학습시켰습니다.
    *   **성과**: 학습 데이터 볼륨을 늘려 모델의 일반화(Generalization) 성능을 높였으며, 단일 모델 기준 가장 높은 점수 향상 폭을 보여주었습니다.

3. **External Data 통합**
    *   원본 UCI Heart Disease 데이터셋 920개를 합쳐서 학습에 사용함으로써 합성 데이터가 가질 수 있는 노이즈를 완화하고 견고한 결정 경계(Decision Boundary)를 형성했습니다.

### ❌ What Went Wrong & Lessons Learned (실패 원인 및 피드백)

1. **후반부 제출 에러 (`SubmissionStatus.ERROR`)**
    *   **현상**: 대회가 막바지에 다다른 2월 20일, `fix_submission.py` 및 `train_pseudo_round2.py` 스크립트를 통해 생성된 결과물이 Kaggle 시스템에서 지속적으로 에러를 발생시켰습니다.
    *   **원인 분석**: 
        1. **확률 값 범위 오류(Out of Bounds)**: 앙상블 과정에서 로직 실수로 인해 예측 확률값이 `[0, 1]`의 범위를 미세하게 벗어났을 가능성이 매우 높습니다. AUC-ROC를 평가 지표로 사용하는 대회에서는 타겟 변수가 반드시 확률 값이어야 합니다.
        2. **ID 매핑 불일치**: Pseudo-Labeling을 적용하고 Test 셋을 다시 분리하는 과정에서 `id` 컬럼의 순서가 뒤섞이거나 유실되어, Sample Submission의 양식과 불일치했을 수 있습니다.
    *   **개선 피드백 (Actionable Feedback)**:
        *   제출 전 반드시 **Post-Processing Validation**을 파이프라인에 추가해야 합니다.
        *   `assert submission['Heart Disease'].between(0, 1).all()` 
        *   `assert len(submission) == len(sample_submission)`
        *   위와 같은 자동 검증(Assertion) 로직을 스크립트 마지막에 강제하여 소중한 일일 제출 횟수를 날리지 않도록 방어 로직(Defensive Programming)을 세워야 합니다.

2. **과도한 앙상블로 인한 오버피팅 가능성 (`ultimate_winning_ensemble.csv`)**
    *   **현상**: `ultimate_winning_ensemble.csv` 등 여러 모델을 강제로 엮으려 했던 시도에서 오히려 점수 하락이나 에러가 발생했습니다.
    *   **원인 분석**: 성능이 상대적으로 떨어지는 모델(Simple NN 등)을 억지로 앙상블에 포함시키려다 보니 전체적인 예측력이 하향 평준화되는 문제가 있었습니다.
    *   **개선 피드백**: 무조건 많은 모델을 섞는 것이 정답이 아님을 깨달았습니다. 상관관계(Correlation)가 낮고 각각의 성능이 뛰어난 소수의 정예 모델(예: CatBoost + LGBM)만으로 앙상블을 구성하는 것이 훨씬 강력함을 배웠습니다.

---

## 4. Conclusion & Next Steps
이 프로젝트는 단순히 점수를 높이는 것을 넘어, 머신러닝 코드를 **프로덕션 레벨(Production-Level)**에 가깝게 리팩토링하고 클라우드 연동 자동화 스크립트를 경험하는 데 큰 의의가 있었습니다. 

현재 모든 난잡했던 스크립트는 `src/models`, `src/ensemble`, `src/data_processing` 등으로 완벽히 모듈화되었으며, 차후 다른 Kaggle Tabular 대회나 실무 데이터 분석 프로젝트에 **보일러플레이트(Boilerplate) 파이프라인**으로 즉시 재사용할 수 있을 만큼 견고해졌습니다.
