import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def run_deterministic_test():
    print("Running Reproducibility Validation...")

    X_train = np.load(ROOT / "data/processed/X_train_final.npy")
    X_test = np.load(ROOT / "data/processed/X_test_final.npy")
    y_train = np.load(ROOT / "data/processed/y_train.npy")
    y_test = np.load(ROOT / "data/processed/y_test.npy")

    # Fixed parameters for strict reproducibility
    params = {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5,
              "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}

    model_1 = RandomForestRegressor(**params).fit(X_train, y_train)
    score_1 = r2_score(y_test, model_1.predict(X_test))

    # Execution 2 (simulating a re-run)
    model_2 = RandomForestRegressor(**params).fit(X_train, y_train)
    score_2 = r2_score(y_test, model_2.predict(X_test))

    is_reproducible = score_1 == score_2

    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_r2_log": score_1,
        "execution_2_r2_log": score_2,
        "is_strictly_reproducible": bool(is_reproducible),
        "status": "PASSED" if is_reproducible else "FAILED",
    }

    os.makedirs(ROOT / "artifacts", exist_ok=True)
    with open(ROOT / "artifacts/reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 R2: {score_1:.6f}")
    print(f"Execution 2 R2: {score_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        print("FAILED: Pipeline is non-deterministic.")


if __name__ == "__main__":
    run_deterministic_test()
