# Kaggle Cloud Execution Guide & Best Practices

## 1. 개요
로컬 자원을 아끼고 Kaggle의 GPU/TPU를 활용하기 위해, **GitHub 코드를 Kaggle Kernel에서 실행(Clone & Run)**하는 방식을 사용합니다.

## 2. 핵심 아키텍처 (Runner Pattern)
- **Local**: 코드를 작성하고 GitHub에 Push.
- **Kaggle**: `runner.py`가 GitHub 코드를 Clone하고, 데이터를 찾아 스크립트를 실행.

## 3. 시행착오 및 해결책 (Troubleshooting)

### Q1. "FileNotFoundError: data/raw/test.csv"
- **원인**: GitHub 리포지토리에는 데이터 파일이 없음 (`.gitignore`).
- **해결**: `runner.py`에서 `/kaggle/input/`을 탐색하여 데이터를 찾고, 스크립트 실행 시 인자(`--train_path`)로 전달해야 함.

### Q2. "Output file not found"
- **원인**: 실행 스크립트가 실패했거나, 결과물이 `/kaggle/working/`이 아닌 하위 폴더에 생성됨.
- **해결**:
    - 스크립트 성공 여부(`exit_code`)를 반드시 체크.
    - 결과물을 최상위 폴더(`/kaggle/working/`)로 복사(`shutil.copy`).
    - **주의**: `shutil.rmtree`로 프로젝트 폴더를 지울 때 결과물까지 지우지 않도록 주의.

### Q3. "SyntaxError" on Cloud
- **원인**: 로컬에서 수정한 코드가 GitHub에 Push되지 않았거나, 파일이 꼬임.
- **해결**: 코드 수정 후 반드시 `git push`를 하고, 커널 로그를 통해 실행된 코드를 확인할 것.

## 4. 모범 사례 (Best Practices)
1. **Path Injection**: 모든 파일 경로는 `argparse`를 통해 외부에서 주입받도록 코딩한다.
   ```python
   parser.add_argument('--train_path', default='data/train.csv')
   ```
2. **Dynamic Search**: 데이터셋 경로는 하드코딩하지 말고 `os.walk('/kaggle/input')`으로 찾는다.
3. **Keep it Clean**: `runner.py`는 오직 실행 환경 구성과 명령 전달에만 집중한다.

## 5. 현재 사용 중인 리소스
- **Kernel**: `softkleenex/runner-pseudo-r2` (GitHub 연동 Runner)
- **Dataset**: `softkleenex/s6e2-artifacts` (OOF, Augmented Data 포함)
