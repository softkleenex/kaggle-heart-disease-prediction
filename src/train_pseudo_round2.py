import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. 데이터 로드
train_aug = pd.read_csv('data/processed/train_augmented.csv')
test = pd.read_csv('data/raw/test.csv')

# 2. Round 1 결과(Pseudo-Labeling Final) 로드
# 이전 단계에서 생성한 가장 강력한 예측값 사용
prev_sub = pd.read_csv('submissions/pseudo_labeling_final.csv')
test_preds = prev_sub['Heart Disease'].values

# 3. Pseudo-Labeling Round 2 (임계값 완화: 0.98 / 0.02)
# 더 많은 데이터를 학습에 포함시키기 위해 임계값을 살짝 넓힘
high_conf_indices = np.where((test_preds > 0.98) | (test_preds < 0.02))[0]
pseudo_test = test.iloc[high_conf_indices].copy()
pseudo_labels = (test_preds[high_conf_indices] > 0.5).astype(int)
pseudo_test['Heart Disease'] = np.where(pseudo_labels == 1, 'Presence', 'Absence')

print(f"Round 2 Pseudo-Labels Count: {len(pseudo_test)} (Previous Round was smaller)")

# Train에 추가
train_final = pd.concat([train_aug, pseudo_test], axis=0).reset_index(drop=True)

# 피처 엔지니어링
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

# 4. 5-Fold 재학습 (Round 2)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_pseudo_r2 = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=2000, # 반복 횟수 더 증가
        learning_rate=0.02, # 학습률 더 미세하게
        depth=7, # 깊이 약간 증가 (복잡한 패턴 학습)
        eval_metric='AUC',
        random_seed=42,
        verbose=200
    )
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    test_preds_pseudo_r2 += model.predict_proba(X_test)[:, 1] / 5

# 결과 저장
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sample_sub['Heart Disease'] = test_preds_pseudo_r2
sample_sub.to_csv('submissions/pseudo_labeling_round2.csv', index=False)
print("Pseudo-Labeling Round 2 submission saved.")
