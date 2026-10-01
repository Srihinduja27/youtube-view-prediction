import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.common import (RANDOM_STATE, TEST_SIZE, build_preprocessor, engineer_features,
                        feature_lists, load_raw, split_xy)


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load raw data and engineer features
    df, _ = engineer_features(load_raw())
    X, y = split_xy(df)

    # 2. Train-Test Split (before transformations to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Column types
    num_cols, cat_cols = feature_lists(X_train)

    # 4. Modular ColumnTransformer: numeric -> median impute + scale; categorical -> one-hot
    preprocessor = build_preprocessor(num_cols, cat_cols, impute=True)

    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 5. Save artifacts
    os.makedirs(ROOT / "data/processed", exist_ok=True)
    os.makedirs(ROOT / "models", exist_ok=True)

    np.save(ROOT / "data/processed/X_train_final.npy", X_train_final)
    np.save(ROOT / "data/processed/X_test_final.npy", X_test_final)
    np.save(ROOT / "data/processed/y_train.npy", y_train.to_numpy(dtype=np.float64))
    np.save(ROOT / "data/processed/y_test.npy", y_test.to_numpy(dtype=np.float64))

    # Single preprocessor artifact
    joblib.dump(preprocessor, ROOT / "models/preprocessor.pkl")

    metadata = {
        "dataset_name": "YouTube Trending Video Statistics",
        "target": "log1p(views)",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open(ROOT / "data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
