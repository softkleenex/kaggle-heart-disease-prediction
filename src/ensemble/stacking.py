import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# OOF 및 Test 데이터 로드
lgbm_oof = np.load('data/processed/lgbm_oof.npy')
xgb_oof = np.load('data/processed/xgb_oof.npy')
cat_oof = np.load('data/processed/catboost_oof.npy')

lgbm_test = np.load('data/processed/lgbm_test.npy')
xgb_test = np.load('data/processed/xgb_test.npy')
cat_test = np.load('data/processed/catboost_test.npy')

# 타겟 데이터 로드
train = pd.read_csv('data/raw/train.csv')
y = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})

# Meta Features 구성
X_meta = np.column_stack([lgbm_oof, xgb_oof, cat_oof])
X_test_meta = np.column_stack([lgbm_test, xgb_test, cat_test])

# Meta Model: Logistic Regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_oof = np.zeros(len(train))
final_test = np.zeros(len(X_test_meta))

for fold, (idx_t, idx_v) in enumerate(skf.split(X_meta, y)):
    xt, xv = X_meta[idx_t], X_meta[idx_v]
    yt, yv = y.iloc[idx_t], y.iloc[idx_v]
    
    meta_model = LogisticRegression()
    meta_model.fit(xt, yt)
    
    final_oof[idx_v] = meta_model.predict_proba(xv)[:, 1]
    final_test += meta_model.predict_proba(X_test_meta)[:, 1] / 5
    auc_val = roc_auc_score(yv, final_oof[idx_v])
    print(f"Stacking Fold {fold+1} AUC: {auc_val:.5f}")

overall_auc = roc_auc_score(y, final_oof)
print(f"Final Stacking OOF AUC: {overall_auc:.5f}")

# 최종 제출 파일 생성
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = np.where(final_test > 0.5, 'Presence', 'Absence')
sample_sub.to_csv('submissions/stacking_final.csv', index=False)
print("Final Stacking submission saved to submissions/stacking_final.csv")
