import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. 모든 OOF/Test 데이터 로드
lgbm_test = np.load('data/processed/lgbm_test.npy')
xgb_test = np.load('data/processed/xgb_test.npy')
cat_test = np.load('data/processed/catboost_test.npy')
nn_test = np.load('data/processed/nn_test.npy')

# 2. 가중치 결합 (Weight Blending)
# 트리 모델들에 높은 비중을 주되, NN을 섞어 일반화 성능 향상
final_test = (cat_test * 0.4) + (lgbm_test * 0.3) + (xgb_test * 0.2) + (nn_test * 0.1)

# 3. 최종 제출 파일 생성
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = np.where(final_test > 0.5, 'Presence', 'Absence')
sample_sub.to_csv('submissions/ultimate_winning_ensemble.csv', index=False)

print("Ultimate Winning Ensemble Submission saved.")
