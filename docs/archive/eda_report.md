# EDA Report - Playground Series S6E2

## 1. 타겟 변수 분석 (Heart Disease)
- **Absence**: 55.17%
- **Presence**: 44.83%
- **결론**: 클래스 불균형이 크지 않아 일반적인 정확도(Accuracy) 및 AUC-ROC를 지표로 사용 가능.

## 2. 데이터 품질 확인
- **결측치**: 없음 (모든 컬럼 630,000개 데이터 보유).
- **데이터 타입**:
    - `id`: 고유 식별자 (학습 제외).
    - 수치형: `Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`.
    - 범주형(Encoding 완료): `Sex`, `Chest pain type`, `FBS over 120`, `EKG results`, `Exercise angina`, `Slope of ST`, `Number of vessels fluro`, `Thallium`.

## 3. 피처 인사이트
- 데이터가 이미 정수 형태로 잘 인코딩되어 있어, 즉시 트리 기반 모델(LightGBM, XGBoost)에 투입 가능.
- `ST depression`만 소수점 데이터를 포함하므로 스케일링 여부는 모델에 따라 결정 (트리 모델은 불필요).

## 4. 베이스라인 전략
- **Model**: LightGBM
- **Validation**: Stratified 5-Fold
- **Preprocessing**: 
    - `Heart Disease` -> `1 (Presence)`, `0 (Absence)`
    - `id` 컬럼 제거
