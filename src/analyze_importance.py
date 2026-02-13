import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# 데이터 및 모델 재학습 (간단히)
train = pd.read_csv('data/processed/train_v2.csv')
y = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
X = train.drop(['id', 'Heart Disease'], axis=1)

model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 중요도 추출
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print(importance.head(15))

# 향후 FE 아이디어 도출을 위한 기록
importance.to_csv('docs/feature_importance.csv', index=False)
