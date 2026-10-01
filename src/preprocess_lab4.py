import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import RANDOM_STATE, TEST_SIZE, engineer_features, feature_lists, load_raw, split_xy


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load raw data and engineer features
    df, _ = engineer_features(load_raw())
    X, y = split_xy(df)

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Separate column types
    num_cols, cat_cols = feature_lists(X_train)

    # 4. Scale & Encode (fit on train only)
    scaler = StandardScaler()
    x_train_parts = [scaler.fit_transform(X_train[num_cols])]
    x_test_parts = [scaler.transform(X_test[num_cols])]

    ohe = None
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        x_train_parts.append(ohe.fit_transform(X_train[cat_cols]))
        x_test_parts.append(ohe.transform(X_test[cat_cols]))

    X_train_final = np.hstack(x_train_parts)
    X_test_final = np.hstack(x_test_parts)

    # 5. Save artifacts
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    np.save(ROOT / "data/processed/X_train_final.npy", X_train_final)
    np.save(ROOT / "data/processed/X_test_final.npy", X_test_final)
    np.save(ROOT / "data/processed/y_train.npy", y_train.to_numpy(dtype=np.float64))
    np.save(ROOT / "data/processed/y_test.npy", y_test.to_numpy(dtype=np.float64))

    joblib.dump(scaler, ROOT / "models/scaler.pkl")
    if ohe is not None:
        joblib.dump(ohe, ROOT / "models/ohe.pkl")

    # 6. Metadata
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

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
