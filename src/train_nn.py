import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
y = (train['Heart Disease'] == 'Presence').astype(int).values

# NN을 위한 전처리 (Scaling이 필수)
drop_cols = ['id', 'Heart Disease']
X = train.drop(drop_cols, axis=1)
X_test = test.drop(['id'], axis=1)

# 범주형 변수 One-hot encoding
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 간단한 MLP 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 5-Fold 학습 (NN은 시간이 걸리므로 5-Fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_nn = np.zeros(len(train))
test_nn = np.zeros(len(test))

for fold, (t_idx, v_idx) in enumerate(skf.split(X_scaled, y)):
    xt, xv = torch.FloatTensor(X_scaled[t_idx]), torch.FloatTensor(X_scaled[v_idx])
    yt, yv = torch.FloatTensor(y[t_idx]).reshape(-1, 1), torch.FloatTensor(y[v_idx]).reshape(-1, 1)
    
    model = SimpleNN(X_scaled.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simple Training Loop
    for epoch in range(20): # 빠른 실험을 위해 20 에폭
        model.train()
        optimizer.zero_grad()
        out = model(xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        oof_nn[v_idx] = model(xv).numpy().flatten()
        test_nn += model(torch.FloatTensor(X_test_scaled)).numpy().flatten() / 5
    print(f"NN Fold {fold+1} AUC: {roc_auc_score(y[v_idx], oof_nn[v_idx]):.5f}")

np.save('data/processed/nn_oof.npy', oof_nn)
np.save('data/processed/nn_test.npy', test_nn)
print(f"Final NN OOF AUC: {roc_auc_score(y, oof_nn):.5f}")
