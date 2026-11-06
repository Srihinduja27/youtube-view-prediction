import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.common import (RANDOM_STATE, TEST_SIZE, build_preprocessor, engineer_features,
                        feature_lists, split_xy)


def main():
    print("[INFO] Starting Preprocessing Pipeline...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path, low_memory=False)
    print(f"[INFO] Loaded raw data: {df_raw.shape}")

    df_featured, _ = engineer_features(df_raw)
    X, y = split_xy(df_featured)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["split"]["test_size"], random_state=config["split"]["random_state"]
    )

    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Save X_test_raw for drift monitoring in Lab 12
    X_test_raw_path = processed_dir / "X_test_raw.csv"
    X_test.to_csv(X_test_raw_path, index=False)
    print(f"[INFO] Saved test features baseline to {X_test_raw_path}")

    num_cols, cat_cols = feature_lists(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols, impute=True)

    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(processed_dir / "X_train_final.npy", X_train_final)
    np.save(processed_dir / "X_test_final.npy", X_test_final)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy(dtype=np.float64))
    np.save(processed_dir / "y_test.npy", y_test.to_numpy(dtype=np.float64))

    preprocessor_path = config["paths"]["preprocessor"]
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "dataset_name": "YouTube Trending Video Statistics",
        "target": "log1p(views)",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open(processed_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {preprocessor_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
