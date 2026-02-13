import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def apply_fe(df):
    df['BP_Cholesterol'] = df['BP'] * df['Cholesterol']
    df['Age_MaxHR'] = df['Age'] * df['Max HR']
    df['High_BP'] = (df['BP'] > 140).astype(int)
    df['High_Cholesterol'] = (df['Cholesterol'] > 240).astype(int)
    df['ST_per_Age'] = df['ST depression'] / (df['Age'] + 1)
    return df

train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample_sub = pd.read_csv('data/raw/sample_submission.csv')

train = apply_fe(train)
test = apply_fe(test)

target_map = {'Absence': 0, 'Presence': 1}
train['target'] = train['Heart Disease'].map(target_map)

drop_cols = ['id', 'Heart Disease', 'target']
features = [c for c in train.columns if c not in drop_cols]

X = train[features]
y = train['target']
X_test = test[features]

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
    auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF AUC with FE: {overall_auc:.5f}")

sample_sub['Heart Disease'] = np.where(test_preds > 0.5, 'Presence', 'Absence')
sample_sub.to_csv('submissions/fe_lgbm.csv', index=False)
