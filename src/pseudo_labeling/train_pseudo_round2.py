import pandas as pd
import numpy as np
import argparse
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo-Labeling Round 2 Training")
    # Default paths are set for Local Environment
    parser.add_argument('--train_path', type=str, default='data/processed/train_augmented.csv', help='Path to augmented train data')
    parser.add_argument('--test_path', type=str, default='data/raw/test.csv', help='Path to test data')
    parser.add_argument('--prev_sub_path', type=str, default='submissions/pseudo_labeling_final.csv', help='Path to previous submission for pseudo-labeling')
    parser.add_argument('--sample_sub_path', type=str, default='data/raw/sample_submission.csv', help='Path to sample submission')
    parser.add_argument('--output_path', type=str, default='submissions/pseudo_labeling_round2.csv', help='Path to save output csv')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from:\n Train: {args.train_path}\n Test: {args.test_path}\n Prev Sub: {args.prev_sub_path}")

    # 1. 데이터 로드
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Train file not found at {args.train_path}")
    
    train_aug = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    
    # 2. Round 1 결과 로드
    prev_sub = pd.read_csv(args.prev_sub_path)
    test_preds = prev_sub['Heart Disease'].values

    # 3. Pseudo-Labeling Round 2 (임계값: 0.98 / 0.02)
    high_conf_indices = np.where((test_preds > 0.98) | (test_preds < 0.02))[0]
    pseudo_test = test.iloc[high_conf_indices].copy()
    pseudo_labels = (test_preds[high_conf_indices] > 0.5).astype(int)
    pseudo_test['Heart Disease'] = np.where(pseudo_labels == 1, 'Presence', 'Absence')

    print(f"Round 2 Pseudo-Labels Count: {len(pseudo_test)}")

    # Train에 추가
    train_final = pd.concat([train_aug, pseudo_test], axis=0).reset_index(drop=True)

    # 피처 엔지니어링
    def apply_fe(df):
        df['MaxHR_Age_Ratio'] = df['Max HR'] / (df['Age'] + 1)
        df['Sex_CPT'] = df['Sex'].astype(str) + "_" + df['Chest pain type'].astype(str)
        return df

    train_final = apply_fe(train_final)
    test_final = apply_fe(test)

    target_map = {'Absence': 0, 'Presence': 1}
    y = train_final['Heart Disease'].map(target_map)
    X = train_final.drop(['id', 'Heart Disease'], axis=1)
    X_test = test_final.drop(['id'], axis=1)

    cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
                    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 
                    'Thallium', 'Sex_CPT']

    # 4. 5-Fold 재학습
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_pseudo_r2 = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,
            depth=7,
            eval_metric='AUC',
            random_seed=42,
            verbose=200
        )
        
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        test_preds_pseudo_r2 += model.predict_proba(X_test)[:, 1] / 5

    # 결과 저장
    sample_sub = pd.read_csv(args.sample_sub_path)
    sample_sub['Heart Disease'] = test_preds_pseudo_r2
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sample_sub.to_csv(args.output_path, index=False)
    print(f"Submission saved to {args.output_path}")

if __name__ == "__main__":
    main()
