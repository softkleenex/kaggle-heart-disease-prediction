import pandas as pd
import numpy as np

# 1. 모든 확률 데이터 로드
lgbm_test = np.load('data/processed/lgbm_test.npy')
xgb_test = np.load('data/processed/xgb_test.npy')
cat_test = np.load('data/processed/catboost_test.npy')
nn_test = np.load('data/processed/nn_test.npy')

# 2. Rank Averaging 적용
def get_rank(probs):
    return pd.Series(probs).rank(pct=True).values

lgbm_rank = get_rank(lgbm_test)
xgb_rank = get_rank(xgb_test)
cat_rank = get_rank(cat_test)
nn_rank = get_rank(nn_test)

# 가중치 결합 (Rank 기반)
final_rank_avg = (cat_rank * 0.4) + (lgbm_rank * 0.3) + (xgb_rank * 0.2) + (nn_rank * 0.1)

# 3. 제출 파일 생성
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = final_rank_avg

output_path = 'submissions/rank_averaging_ensemble.csv'
sample_sub.to_csv(output_path, index=False)
print(f"Rank Averaging submission saved to {output_path}")
