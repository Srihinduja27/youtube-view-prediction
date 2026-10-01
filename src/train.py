import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.common import RANDOM_STATE, TEST_SIZE, build_preprocessor, feature_lists, split_xy

DATA = ROOT / "data" / "processed" / "youtube_processed.csv"
MODEL = ROOT / "models" / "random_forest_model.pkl"
SPLIT = ROOT / "artifacts" / "split_data.json"


def train():
    df = pd.read_csv(DATA, low_memory=False)
    X, y = split_xy(df)  # y = log1p(views)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    num_cols, cat_cols = feature_lists(X_train)
    model = Pipeline(steps=[
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    SPLIT.write_text(json.dumps({
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "features": X_train.columns.tolist(),
    }, indent=2), encoding="utf-8")
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train()
    print(f"Model saved to {MODEL}")
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
