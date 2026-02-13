import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. 데이터 로드 및 Target Encoding 준비
train = pd.read_csv('data/processed/train_v2.csv')
test = pd.read_csv('data/processed/test_v2.csv')
y = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
X = train.drop(['id', 'Heart Disease'], axis=1)
X_test = test.drop(['id'], axis=1)

# 2. Target Encoding (Smoothing 적용하여 과적합 방지)
def target_encode(train_df, test_df, y_train, col):
    temp = pd.concat([train_df[col], y_train], axis=1)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train_df))
    
    for train_idx, val_idx in kf.split(train_df, y_train):
        agg = temp.iloc[train_idx].groupby(col)[y_train.name].mean()
        train_encoded[val_idx] = train_df.iloc[val_idx][col].map(agg)
        
    global_mean = y_train.mean()
    test_encoded = test_df[col].map(temp.groupby(col)[y_train.name].mean()).fillna(global_mean)
    return train_encoded, test_encoded

cat_cols_for_te = ['Number of vessels fluro', 'Thallium', 'Chest pain type', 'Sex_CPT']
for col in cat_cols_for_te:
    train[f'{col}_TE'], test[f'{col}_TE'] = target_encode(X, X_test, y, col)

# 3. 10-Fold CV 학습
X_final = train.drop(['id', 'Heart Disease'], axis=1)
X_test_final = test.drop(['id'], axis=1)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'random_state': 42,
    'learning_rate': 0.02, 'num_leaves': 144, 'feature_fraction': 0.5, 'n_estimators': 3000
}

for fold, (t_idx, v_idx) in enumerate(skf.split(X_final, y)):
    xt, xv = X_final.iloc[t_idx], X_final.iloc[v_idx]
    yt, yv = y.iloc[t_idx], y.iloc[v_idx]
    
    m = lgb.LGBMClassifier(**params)
    m.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(100)])
    
    oof_preds[v_idx] = m.predict_proba(xv)[:, 1]
    test_preds += m.predict_proba(X_test_final)[:, 1] / 10
    auc_score = roc_auc_score(yv, oof_preds[v_idx])
    print(f"Winning Fold {fold+1} AUC: {auc_score:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Final Winning OOF AUC: {overall_auc:.5f}")

sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = np.where(test_preds > 0.5, 'Presence', 'Absence')
sample_sub.to_csv('submissions/winning_v1_te.csv', index=False)
