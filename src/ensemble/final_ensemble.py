import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample_sub = pd.read_csv('data/raw/sample_submission.csv')

target_map = {'Absence': 0, 'Presence': 1}
train['target'] = train['Heart Disease'].map(target_map)
features = [c for c in train.columns if c not in ['id', 'Heart Disease', 'target']]

X = train[features]
y = train['target']
X_test = test[features]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. LightGBM (Optimized)
lgbm_params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'random_state': 42,
    'learning_rate': 0.0951, 'num_leaves': 144, 'feature_fraction': 0.5018,
    'bagging_fraction': 0.9831, 'bagging_freq': 2, 'min_child_samples': 42
}

lgbm_test_preds = np.zeros(len(test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    lgbm_test_preds += model.predict_proba(X_test)[:, 1] / 5

# 2. XGBoost (Standard)
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42,
    'learning_rate': 0.05, 'max_depth': 6
}

xgb_test_preds = np.zeros(len(test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    xgb_test_preds += model.predict_proba(X_test)[:, 1] / 5

# 3. Blending (LGBM 60% + XGB 40%)
final_preds = (lgbm_test_preds * 0.6) + (xgb_test_preds * 0.4)

# 4. 제출
sample_sub['Heart Disease'] = np.where(final_preds > 0.5, 'Presence', 'Absence')
sample_sub.to_csv('submissions/final_ensemble.csv', index=False)
print("Final ensemble submission saved.")
