import pandas as pd
import numpy as np

# 모든 확률 데이터 로드
lgbm_test = np.load('data/processed/lgbm_test.npy')
xgb_test = np.load('data/processed/xgb_test.npy')
cat_test = np.load('data/processed/catboost_test.npy')
nn_test = np.load('data/processed/nn_test.npy')

# 가중치 결합 (확률 유지)
final_probs = (cat_test * 0.4) + (lgbm_test * 0.3) + (xgb_test * 0.2) + (nn_test * 0.1)

# 제출 파일 생성
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = final_probs # 문자열 대신 확률값 입력

output_path = 'submissions/ultimate_winning_ensemble_fixed.csv'
sample_sub.to_csv(output_path, index=False)
print(f"Fixed submission saved to {output_path}")
