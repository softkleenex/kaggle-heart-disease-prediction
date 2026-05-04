import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드
train = pd.read_csv('data/processed/train_v2.csv')
test = pd.read_csv('data/processed/test_v2.csv')
y = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
X = train.drop(['id', 'Heart Disease'], axis=1)
X_test = test.drop(['id'], axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_model(model_type, params):
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    
    for fold, (idx_t, idx_v) in enumerate(skf.split(X, y)):
        xt, xv = X.iloc[idx_t], X.iloc[idx_v]
        yt, yv = y.iloc[idx_t], y.iloc[idx_v]
        
        if model_type == 'lgbm':
            m = lgb.LGBMClassifier(**params)
            m.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(100)])
        elif model_type == 'xgb':
            m = xgb.XGBClassifier(**params)
            m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
            
        oof[idx_v] = m.predict_proba(xv)[:, 1]
        preds += m.predict_proba(X_test)[:, 1] / 5
        auc = roc_auc_score(yv, oof[idx_v])
        print(f"{model_type} Fold {fold+1} AUC: {auc:.5f}")
        
    return oof, preds

# 1. LightGBM Optimized
lgbm_params = {
    'n_estimators': 2000, 'learning_rate': 0.05, 'num_leaves': 144, 
    'feature_fraction': 0.5, 'bagging_fraction': 0.9, 'random_state': 42, 'verbosity': -1
}
lgbm_oof, lgbm_test = train_model('lgbm', lgbm_params)

# 2. XGBoost Optimized
xgb_params = {
    'n_estimators': 2000, 'learning_rate': 0.05, 'max_depth': 6, 
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'tree_method': 'hist'
}
xgb_oof, xgb_test = train_model('xgb', xgb_params)

# 저장
np.save('data/processed/lgbm_oof.npy', lgbm_oof)
np.save('data/processed/lgbm_test.npy', lgbm_test)
np.save('data/processed/xgb_oof.npy', xgb_oof)
np.save('data/processed/xgb_test.npy', xgb_test)

print("All OOF and Test predictions saved for Stacking.")
