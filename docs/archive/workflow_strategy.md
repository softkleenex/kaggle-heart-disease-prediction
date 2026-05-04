# Kaggle & GitHub Workflow Strategy

본 프로젝트는 로컬 개발 환경(Gemini CLI)과 클라우드 환경(Kaggle Notebook)을 효율적으로 연결하는 것을 목표로 합니다.

## 1. Kaggle Notebook - GitHub 연동
- **연결 방법**:
    1. Kaggle Notebook 우측 패널의 'Persistence' 설정을 확인합니다.
    2. 'Add Data'를 통해 본 GitHub 리포지토리를 직접 연결하거나, 
    3. Kaggle의 'GitHub' 연동 기능을 사용하여 Notebook 버전을 Git에 직접 커밋합니다.
- **추천 방식**: 
    - `src/` 폴더의 공통 모듈을 GitHub에 관리하고, Kaggle Notebook 상단에서 `!git clone`을 통해 모듈을 불러와 사용하는 방식을 추천합니다.

## 2. 코드 및 라이브러리 관리
- **로컬 (Gemini CLI)**: 
    - `src/` 내에 전처리(preprocessing), 모델 정의(models), 유틸리티(utils) 코드를 작성합니다.
    - 단위 테스트 및 소규모 실험을 수행합니다.
- **Kaggle (GPU/TPU)**:
    - 대규모 학습 및 앙상블을 수행합니다.
    - `notebooks/`에 있는 `.ipynb` 파일을 사용하여 최종 제출물을 생성합니다.

## 3. 데이터 및 결과물 관리
- **Data**: 로컬 `data/raw`에는 소량의 샘플 또는 Kaggle API로 다운로드한 원본을 두며, Git에는 포함하지 않습니다.
- **Submissions**: 제출 결과물은 `submissions/` 폴더에 기록하며, 제출 시점의 노트북 버전과 점수를 기록 파일(`submission_log.csv`)로 관리합니다.

## 4. 향후 로드맵
1. **1단계**: 데이터 탐색(EDA) 및 데이터 스펙 분석
2. **2단계**: 베이스라인 모델 구축 (Random Forest or XGBoost)
3. **3단계**: 피처 엔지니어링 및 하이퍼파라미터 튜닝
4. **4단계**: 앙상블 및 최종 제출
