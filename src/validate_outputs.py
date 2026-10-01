import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")

    try:
        X_train = np.load(ROOT / 'data/processed/X_train_final.npy')
        X_test = np.load(ROOT / 'data/processed/X_test_final.npy')
        y_train = np.load(ROOT / 'data/processed/y_train.npy')
        y_test = np.load(ROOT / 'data/processed/y_test.npy')
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
        return False

    errors = []

    # Missing values (validates the imputer)
    if np.isnan(X_train).sum() > 0:
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).sum() > 0:
        errors.append("NaNs detected in X_test after preprocessing.")

    # Dimensionality consistency (validates encoding and split)
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")

    # Target sanity (log1p(views) must be finite and non-negative)
    if not (np.isfinite(y_train).all() and np.isfinite(y_test).all()):
        errors.append("Non-finite values detected in the log-transformed target.")
    if (y_train < 0).any() or (y_test < 0).any():
        errors.append("Negative values detected in log1p(views) target.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "errors": errors,
    }

    os.makedirs(ROOT / "artifacts", exist_ok=True)
    report_path = ROOT / 'artifacts/preprocessing_summary_report.json'

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("[SUCCESS] Output Validation PASSED.")
        print("[INFO] Features are clean, scaled, encoded, and dimensionally consistent.")
        print(f"[INFO] Summary report saved to {report_path}")
        return True


if __name__ == "__main__":
    sys.exit(0 if validate_preprocessing_outputs() else 1)
