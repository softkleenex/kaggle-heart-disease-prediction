# Kaggle Discussion Insights & Strategy

## 1. 주요 토론 내용 요약 (Discussion #672651 & #673221)
- **Data Leakage Check**: `id`와 타겟 간의 상관관계는 없으나, 데이터 생성 과정에서의 패턴(Synthetic Data Artifacts)이 존재할 수 있음.
- **High-Impact Features**: `MaxHR`와 `Age`의 상호작용 외에도 `Oldpeak` * `Slope`와 같은 도메인 특화 변수가 중요함.
- **Ensemble Strategy**: 단순 평균(Average)보다는 **Hill Climbing** 알고리즘을 사용하여 OOF 점수를 최대화하는 가중치를 찾는 것이 필수적임.

## 2. 우승을 위한 기술적 제안
- **Pseudo-Labeling (준지도 학습)**:
    - 리더보드 점수가 0.953대에 머물 때, 테스트 데이터의 정보를 학습에 반영하여 0.954+로 도약하는 가장 확실한 방법.
- **Hill Climbing Ensemble**:
    - 수동으로 가중치(0.4, 0.3...)를 정하지 않고, 알고리즘이 0.0001 단위로 최적의 가중치를 찾도록 함.
- **Adversarial Validation**:
    - Train과 Test 데이터의 분포 차이가 있는 피처를 제거하여 일반화 성능 확보.

## 3. 현재 우리 모델과의 차이점
- Rank Averaging이 오히려 점수를 떨어뜨림 -> **확률 값(Probability)의 디테일**이 중요한 대회임.
- 따라서 `Rank Averaging` 대신 `Weighted Blending (Hill Climbing)`과 `Pseudo-Labeling`으로 선회해야 함.
