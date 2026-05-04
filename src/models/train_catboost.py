import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드 (Augmented)
train = pd.read_csv('data/processed/train_augmented.csv')
test = pd.read_csv('data/raw/test.csv')

# 피처 엔지니어링 재적용 (Advanced FE 로직을 직접 포함하거나 호출)
def apply_fe(df):
    df['MaxHR_Age_Ratio'] = df['Max HR'] / (df['Age'] + 1)
    df['Sex_CPT'] = df['Sex'].astype(str) + "_" + df['Chest pain type'].astype(str)
    le = LabelEncoder()
    df['Sex_CPT'] = le.fit_transform(df['Sex_CPT'])
    return df

from sklearn.preprocessing import LabelEncoder
train = apply_fe(train)
test = apply_fe(test)

target_map = {'Absence': 0, 'Presence': 1}
y = train['Heart Disease'].map(target_map)
X = train.drop(['id', 'Heart Disease'], axis=1)
X_test = test.drop(['id'], axis=1)

# 범주형 변수 지정
cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
                'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 
                'Thallium', 'Sex_CPT']
cat_features = [c for c in cat_features if c in X.columns]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=100
    )
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    auc_val = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {auc_val:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall CatBoost OOF AUC: {overall_auc:.5f}")

# OOF 및 Test 예측값 저장 (Stacking용)
np.save('data/processed/catboost_oof.npy', oof_preds)
np.save('data/processed/catboost_test.npy', test_preds)
