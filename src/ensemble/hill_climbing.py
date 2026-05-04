import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. 모든 OOF 및 Test 데이터 로드
lgbm_oof = np.load('data/processed/lgbm_oof.npy')
xgb_oof = np.load('data/processed/xgb_oof.npy')
cat_oof = np.load('data/processed/catboost_oof.npy')
nn_oof = np.load('data/processed/nn_oof.npy')

# OOF 길이 맞추기 (Augmented 학습 시 크기가 다를 수 있음)
min_len = 630000
lgbm_oof = lgbm_oof[:min_len]
xgb_oof = xgb_oof[:min_len]
cat_oof = cat_oof[:min_len]
nn_oof = nn_oof[:min_len]

lgbm_test = np.load('data/processed/lgbm_test.npy')
xgb_test = np.load('data/processed/xgb_test.npy')
cat_test = np.load('data/processed/catboost_test.npy')
nn_test = np.load('data/processed/nn_test.npy')

# 타겟 데이터 로드
train = pd.read_csv('data/raw/train.csv')
y = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# 2. Hill Climbing 알고리즘 구현 (최적화)
def optimize_weights(oofs, y_true):
    best_weights = np.ones(len(oofs)) / len(oofs)  # 초기 가중치: 균등 (0.25, 0.25...)
    best_auc = roc_auc_score(y_true, np.average(oofs, axis=0, weights=best_weights))
    
    print(f"Initial AUC: {best_auc:.5f}")
    
    # 1000번 반복 (간단한 Random Search + Hill Climbing 변형)
    for i in range(1000):
        # 현재 가중치에서 무작위 변동
        new_weights = best_weights + np.random.normal(0, 0.01, size=len(oofs))
        new_weights = np.maximum(new_weights, 0) # 음수 방지
        new_weights /= np.sum(new_weights) # 합이 1이 되도록 정규화
        
        current_auc = roc_auc_score(y_true, np.average(oofs, axis=0, weights=new_weights))
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_weights = new_weights
            # print(f"Improved at iter {i}: {best_auc:.5f}")
            
    return best_weights, best_auc

models_oof = [lgbm_oof, xgb_oof, cat_oof, nn_oof]
model_names = ['LGBM', 'XGB', 'CatBoost', 'NN']
models_test = [lgbm_test, xgb_test, cat_test, nn_test]

print("Starting Hill Climbing Optimization...")
best_weights, best_auc = optimize_weights(models_oof, y)

print("\n--- Optimized Weights ---")
for name, w in zip(model_names, best_weights):
    print(f"{name}: {w:.4f}")

print(f"\nOptimized CV AUC: {best_auc:.5f}")

# 3. 최적 가중치로 최종 예측 생성
final_preds = np.average(models_test, axis=0, weights=best_weights)

# 결과 저장
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = final_preds
sample_sub.to_csv('submissions/hill_climbing_ensemble.csv', index=False)
print("Hill Climbing submission saved.")
