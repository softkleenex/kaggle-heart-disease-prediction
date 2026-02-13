# Kaggle Playground Series S6E2 - Heart Disease Prediction

이 프로젝트는 Kaggle Playground Series S6E2 대회의 우승을 목표로 하는 워크스페이스입니다.

## 📊 현재 성적
- **Best Public LB**: 0.95334
- **Best CV (OOF)**: 0.95541 (Hill Climbing Ensemble)

## 🛠️ 주요 적용 기술
1. **Model Ensemble**: LightGBM, XGBoost, CatBoost, Simple NN 결합
2. **Advanced FE**: Target Encoding, Interaction Features, 나이 대비 심박수 비율 등
3. **Optimized Weights**: Hill Climbing 알고리즘을 통한 모델별 최적 가중치 도출
4. **Original Data Integration**: UCI Heart Disease 원본 데이터셋 통합 학습
5. **Pseudo-Labeling**: 테스트 데이터의 고확률 샘플을 학습에 활용하는 준지도 학습

## 📂 폴더 구조
- `src/`: 모델 학습 및 데이터 가공 스크립트
- `docs/`: 분석 리포트 및 토론 인사이트
- `data/`: 원본, 가공 및 외부 데이터 (Git 제외)
- `submissions/`: 제출 파일 및 로그

## 🚀 향후 계획
- Pseudo-Labeling 반복을 통한 성능 극대화
- 다양한 딥러닝 아키텍처(TabNet 등) 추가 실험
- 최종 Private LB Shake-up 대비를 위한 교차 검증 전략 강화
