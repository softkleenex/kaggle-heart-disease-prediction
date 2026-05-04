import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. 데이터 로드
train_aug = pd.read_csv('data/processed/train_augmented.csv')
test = pd.read_csv('data/raw/test.csv')

# 2. Hill Climbing 결과(Best Prediction) 로드
best_sub = pd.read_csv('submissions/hill_climbing_ensemble.csv')
test_preds = best_sub['Heart Disease'].values

# 3. Pseudo-Labeling (확률 임계값: 0.99 / 0.01)
# 확신이 강한 데이터만 추출
high_conf_indices = np.where((test_preds > 0.99) | (test_preds < 0.01))[0]
pseudo_test = test.iloc[high_conf_indices].copy()
pseudo_labels = (test_preds[high_conf_indices] > 0.5).astype(int) # 0 or 1
pseudo_test['Heart Disease'] = np.where(pseudo_labels == 1, 'Presence', 'Absence')

# Train에 추가
train_final = pd.concat([train_aug, pseudo_test], axis=0).reset_index(drop=True)

# 피처 엔지니어링 (동일하게 적용)
def apply_fe(df):
    df['MaxHR_Age_Ratio'] = df['Max HR'] / (df['Age'] + 1)
    df['Sex_CPT'] = df['Sex'].astype(str) + "_" + df['Chest pain type'].astype(str)
    return df

train_final = apply_fe(train_final)
test_final = apply_fe(test)

# 데이터셋 준비
target_map = {'Absence': 0, 'Presence': 1}
y = train_final['Heart Disease'].map(target_map)
X = train_final.drop(['id', 'Heart Disease'], axis=1)
X_test = test_final.drop(['id'], axis=1)

cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
                'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 
                'Thallium', 'Sex_CPT']

# 4. 5-Fold 재학습
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_pseudo = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=1500, # 반복 횟수 증가
        learning_rate=0.03, # 학습률 감소 (더 정밀하게)
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=100
    )
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    test_preds_pseudo += model.predict_proba(X_test)[:, 1] / 5

# 결과 저장
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = test_preds_pseudo
sample_sub.to_csv('submissions/pseudo_labeling_final.csv', index=False)
print("Pseudo-Labeling submission saved.")
