# Kaggle Runner Script (Hill Climbing V2)
import os
import shutil
import sys

# 1. Clone GitHub Repo
print("\nCloning GitHub Repository...")
if os.path.exists('kaggle-s6e2-playground'):
    shutil.rmtree('kaggle-s6e2-playground')
os.system('git clone https://github.com/softkleenex/kaggle-s6e2-playground.git')

# 2. Find Data Paths
print("\nSearching for datasets...")
oof_dir = None
test_preds_dir = None
train_path = None
sample_sub_path = None

for root, dirs, files in os.walk('/kaggle/input/'):
    if 'lgbm_oof.npy' in files:
        oof_dir = root
        test_preds_dir = root # Usually same dir
    if 'train.csv' in files and 'playground-series-s6e2' in root:
        train_path = os.path.join(root, 'train.csv')
    if 'sample_submission.csv' in files and 'playground-series-s6e2' in root:
        sample_sub_path = os.path.join(root, 'sample_submission.csv')

if not oof_dir or not train_path:
    print("Error: Required files (OOF or Train) not found.")
    sys.exit(1)

print(f"OOF Dir: {oof_dir}")
print(f"Train Path: {train_path}")

# 3. Run Hill Climbing V2
print("\nRunning Hill Climbing V2...")
os.chdir('kaggle-s6e2-playground')

cmd = (
    f"python3 src/hill_climbing_v2.py "
    f"--oof_dir '{oof_dir}' "
    f"--test_dir '{test_preds_dir}' "
    f"--train_path '{train_path}' "
    f"--sample_sub_path '{sample_sub_path}' "
    f"--output_path '../hill_climbing_v2.csv'"
)

exit_code = os.system(cmd)

if exit_code == 0:
    print("\nHill Climbing Completed Successfully!")
    os.chdir('..')
    shutil.rmtree('kaggle-s6e2-playground')
else:
    print(f"\nHill Climbing Failed with exit code {exit_code}")
    sys.exit(exit_code)
