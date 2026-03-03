import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

# 1. 데이터 로드 (Kaggle Input 경로 직접 지정)
INPUT_DIR = '/kaggle/input/s6e2-artifacts'
COMP_DIR = '/kaggle/input/playground-series-s6e2'

print("Loading OOFs and Test Predictions...")

# 모델별 파일명 정의
models = {
    'CatBoost': ('catboost_oof.npy', 'catboost_test.npy'),
    'LGBM': ('lgbm_oof.npy', 'lgbm_test.npy'),
    'XGB': ('xgb_oof.npy', 'xgb_test.npy'),
    'NN': ('nn_oof.npy', 'nn_test.npy')
}

oofs = []
tests = []
model_names = []

# 정답(Target) 로드
train = pd.read_csv(os.path.join(COMP_DIR, 'train.csv'))
y_true = train['Heart Disease'].map({'Absence': 0, 'Presence': 1}).values
n_train = len(y_true)

# OOF 로드 및 정합성 체크
for name, (oof_file, test_file) in models.items():
    oof_path = os.path.join(INPUT_DIR, oof_file)
    test_path = os.path.join(INPUT_DIR, test_file)
    
    if os.path.exists(oof_path):
        oof = np.load(oof_path)
        test = np.load(test_path)
        
        # 크기 맞추기 (Augmented 학습 등으로 인해 OOF가 더 클 경우)
        if len(oof) > n_train:
            oof = oof[:n_train]
            
        if len(oof) == n_train:
            oofs.append(oof)
            tests.append(test)
            model_names.append(name)
            print(f"Loaded {name}: AUC {roc_auc_score(y_true, oof):.5f}")
        else:
            print(f"Skipping {name}: Size mismatch")
    else:
        print(f"Warning: {name} not found")

oofs = np.array(oofs).T
tests = np.array(tests).T

# 2. 최적화 (SLSQP)
def auc_func(weights):
    final_oof = np.dot(oofs, weights)
    return -roc_auc_score(y_true, final_oof)

cons = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
bounds = [(0, 1)] * len(oofs[0])
init_weights = np.ones(len(oofs[0])) / len(oofs[0])

print("
Starting Hill Climbing Optimization...")
result = minimize(auc_func, init_weights, method='SLSQP', bounds=bounds, constraints=cons)

best_weights = result.x
best_auc = -result.fun

print("
--- Optimized Weights ---")
for name, w in zip(model_names, best_weights):
    print(f"{name}: {w:.4f}")

print(f"
Final Optimized CV AUC: {best_auc:.5f}")

# 3. 제출 파일 생성
final_preds = np.dot(tests, best_weights)
sample_sub = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'))
sample_sub['Heart Disease'] = final_preds
sample_sub.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
