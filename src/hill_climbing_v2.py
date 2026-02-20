import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Hill Climbing Ensemble")
    parser.add_argument('--oof_dir', type=str, default='data/processed', help='Directory containing OOF files')
    parser.add_argument('--test_dir', type=str, default='data/processed', help='Directory containing Test prediction files')
    parser.add_argument('--train_path', type=str, default='data/raw/train.csv', help='Path to original train csv')
    parser.add_argument('--sample_sub_path', type=str, default='data/raw/sample_submission.csv', help='Path to sample submission')
    parser.add_argument('--output_path', type=str, default='submissions/hill_climbing_v2.csv', help='Output path')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading OOFs from {args.oof_dir}...")

    # 1. Load Data
    model_names = ['lgbm', 'xgb', 'catboost', 'nn']
    oofs = []
    tests = []
    loaded_models = []

    train_df = pd.read_csv(args.train_path)
    y_true = train_df['Heart Disease'].map({'Absence': 0, 'Presence': 1}).values
    n_train = len(y_true)

    for name in model_names:
        oof_path = os.path.join(args.oof_dir, f'{name}_oof.npy')
        test_path = os.path.join(args.test_dir, f'{name}_test.npy')
        
        if os.path.exists(oof_path) and os.path.exists(test_path):
            oof = np.load(oof_path)
            test = np.load(test_path)
            
            if len(oof) > n_train:
                oof = oof[:n_train]
            
            if len(oof) == n_train:
                oofs.append(oof)
                tests.append(test)
                loaded_models.append(name)
                print(f"Loaded {name}: OOF shape {oof.shape}")
            else:
                print(f"Skipping {name}: OOF shape mismatch {oof.shape} vs {n_train}")
        else:
            print(f"Warning: {name} files not found.")

    if not oofs:
        raise ValueError("No valid OOF files found!")

    oofs = np.array(oofs).T
    tests = np.array(tests).T

    # 2. Optimization Objective
    def auc_func(weights):
        final_oof = np.dot(oofs, weights)
        return -roc_auc_score(y_true, final_oof)

    cons = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
    bounds = [(0, 1)] * len(loaded_models)
    init_weights = np.ones(len(loaded_models)) / len(loaded_models)

    # 3. Optimize (SLSQP)
    print("\nStarting Optimization (SLSQP)...")
    result = minimize(auc_func, init_weights, method='SLSQP', bounds=bounds, constraints=cons)

    best_weights = result.x
    best_auc = -result.fun

    print("\n--- Optimized Weights ---")
    for name, w in zip(loaded_models, best_weights):
        print(f"{name}: {w:.4f}")
    print(f"\nFinal Optimized AUC: {best_auc:.5f}")

    # 4. Create Submission
    final_test_preds = np.dot(tests, best_weights)
    
    sample_sub = pd.read_csv(args.sample_sub_path)
    sample_sub['Heart Disease'] = final_test_preds
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sample_sub.to_csv(args.output_path, index=False)
    print(f"Submission saved to {args.output_path}")

if __name__ == "__main__":
    main()
