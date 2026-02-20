# Kaggle Cloud Execution Troubleshooting Log

## 1. Problem: "Output file not found" / "No output files"
- **증상**: 커널은 `Complete` 상태지만, Output 탭에 파일이 없거나 제출 시 에러 발생.
- **원인**: 
    1. 학습 스크립트가 중간에 에러로 종료되었으나, `runner.py`가 이를 캐치하지 못하고 `exit code 0`으로 종료됨.
    2. `shutil.rmtree`로 작업 디렉토리를 너무 빨리 삭제하여, 생성된 파일까지 함께 지워짐.
- **해결책**:
    - **Clean Up 삭제**: 디버깅을 위해 모든 파일을 남겨둘 것.
    - **Exit Code 전파**: `os.system`의 리턴값을 확인하여 비정상 종료 시 `sys.exit(1)` 처리.

## 2. Problem: "FileNotFoundError: data/raw/test.csv"
- **증상**: GitHub에서 Clone한 코드에서 데이터를 찾지 못함.
- **원인**: 
    - GitHub에는 데이터 파일(`.csv`)이 없음 (`.gitignore` 때문).
    - Kaggle Kernel의 마운트 경로는 `/kaggle/input/`이지만, 코드는 로컬 구조인 `data/raw/`를 참조함.
- **해결책**:
    - `runner.py` 시작 부분에 **데이터 복사(Copy)** 로직 필수.
    - `/kaggle/input/` 하위를 동적으로 탐색(`os.walk`)하여 파일 위치를 찾고, `kaggle-s6e2-playground/data/raw/`로 복사해줘야 함.

## 3. Kaggle Runner Architecture (Best Practice)
1. **Runner Script (`runner.py`)**:
    - `git clone` -> `os.makedirs` -> `shutil.copy` (Data) -> `python src/train.py` -> `sys.exit`
2. **Metadata (`kernel-metadata.json`)**:
    - `enable_internet: true` (GitHub Clone용)
    - `dataset_sources`: 필요한 모든 데이터셋 포함.

## 4. Current Status (Pseudo-Labeling R2)
- 데이터 경로 문제 해결됨 (Version 6).
- 학습 중 에러가 발생했으나 로그 확인 필요.
- 다음 시도: `runner.py`에서 `rmtree` 제거 및 `traceback` 출력 강화.
