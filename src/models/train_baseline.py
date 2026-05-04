import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# 1. 데이터 로드
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample_sub = pd.read_csv('data/raw/sample_submission.csv')

# 2. 전처리
target_map = {'Absence': 0, 'Presence': 1}
train['target'] = train['Heart Disease'].map(target_map)

drop_cols = ['id', 'Heart Disease', 'target']
features = [c for c in train.columns if c not in drop_cols]

X = train[features]
y = train['target']
X_test = test[features]

# 3. 모델 설정 및 교차 검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.05,
    'num_leaves': 31
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF AUC: {overall_auc:.5f}")

# 4. 제출 파일 생성
sample_sub['Heart Disease'] = np.where(test_preds > 0.5, 'Presence', 'Absence')
output_path = 'submissions/baseline_lgbm.csv'
sample_sub.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
