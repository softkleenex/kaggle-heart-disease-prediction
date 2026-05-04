import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def advanced_fe(df):
    # 1. 수치형 변수 상호작용 (비율 및 차이)
    df['MaxHR_Age_Ratio'] = df['Max HR'] / (df['Age'] + 1)
    df['ST_MaxHR_Interaction'] = df['ST depression'] * df['Max HR']
    df['BP_Age_Product'] = df['BP'] * df['Age']
    
    # 2. 범주형 변수 조합 (강력한 신호 생성)
    # 예: 성별 + 가슴통증 타입 조합
    df['Sex_CPT'] = df['Sex'].astype(str) + "_" + df['Chest pain type'].astype(str)
    
    # 3. 통계적 변수 (평균 대비 편차)
    for col in ['Max HR', 'BP', 'Cholesterol']:
        df[f'{col}_diff_mean'] = df[col] - df[col].mean()
    
    # Label Encoding for new categorical combos
    le = LabelEncoder()
    df['Sex_CPT'] = le.fit_transform(df['Sex_CPT'])
    
    return df

if __name__ == "__main__":
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    
    train = advanced_fe(train)
    test = advanced_fe(test)
    
    train.to_csv('data/processed/train_v2.csv', index=False)
    test.to_csv('data/processed/test_v2.csv', index=False)
    print("Advanced FE completed and saved to data/processed/")
