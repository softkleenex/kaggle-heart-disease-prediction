import pandas as pd
import numpy as np

# 1. 대회 데이터 로드
train = pd.read_csv('data/raw/train.csv')

# 2. 원본 데이터 로드 및 변환
external = pd.read_csv('data/external/heart_disease_uci.csv')

# 컬럼 매핑 및 전처리
ext_mapped = pd.DataFrame()
ext_mapped['Age'] = external['age']
ext_mapped['Sex'] = external['sex'].map({'Male': 1, 'Female': 0})
ext_mapped['Chest pain type'] = external['cp'].map({
    'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4
})
ext_mapped['BP'] = external['trestbps'].fillna(external['trestbps'].mean()).astype(int)
ext_mapped['Cholesterol'] = external['chol'].fillna(external['chol'].mean()).astype(int)
ext_mapped['FBS over 120'] = external['fbs'].map({True: 1, False: 0}).fillna(0).astype(int)
ext_mapped['EKG results'] = external['restecg'].map({
    'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2
}).fillna(0).astype(int)
ext_mapped['Max HR'] = external['thalch'].fillna(external['thalch'].mean()).astype(int)
ext_mapped['Exercise angina'] = external['exang'].map({True: 1, False: 0}).fillna(0).astype(int)
ext_mapped['ST depression'] = external['oldpeak'].fillna(0.0)
ext_mapped['Slope of ST'] = external['slope'].map({
    'upsloping': 1, 'flat': 2, 'downsloping': 3
}).fillna(2).astype(int)
ext_mapped['Number of vessels fluro'] = external['ca'].fillna(0).astype(int)
ext_mapped['Thallium'] = external['thal'].map({
    'normal': 3, 'fixed defect': 6, 'reversable defect': 7
}).fillna(3).astype(int)
ext_mapped['Heart Disease'] = external['num'].apply(lambda x: 'Presence' if x > 0 else 'Absence')

# 3. 데이터 합치기 (id는 무시)
train_aug = pd.concat([train.drop('id', axis=1), ext_mapped], axis=0).reset_index(drop=True)
train_aug['id'] = range(len(train_aug))

train_aug.to_csv('data/processed/train_augmented.csv', index=False)
print(f"Augmented data saved: {train_aug.shape}")
