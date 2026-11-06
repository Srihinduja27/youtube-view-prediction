import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def create_evolved_datasets():
    print("[INFO] Creating evolved drift monitoring datasets...")

    baseline_x_path = ROOT / "data/processed/X_test_raw.csv"
    baseline_y_path = ROOT / "data/processed/y_test.npy"

    if not baseline_x_path.exists():
        # Build baseline X_test_raw from processed youtube if needed
        proc_csv = ROOT / "data/processed/youtube_processed.csv"
        if proc_csv.exists():
            from src.common import DROP_COLUMNS, TARGET, split_xy
            df = pd.read_csv(proc_csv, low_memory=False)
            X, y = split_xy(df)
            X = X.tail(min(1500, len(X))).copy()
            y = y.tail(min(1500, len(y))).to_numpy()
        else:
            print("[ERROR] Neither X_test_raw.csv nor youtube_processed.csv found.")
            return False
    else:
        X = pd.read_csv(baseline_x_path).head(1500).copy()
        if baseline_y_path.exists():
            y = np.load(baseline_y_path)[:len(X)]
        else:
            y = np.ones(len(X)) * 10.0

    # Simulate realistic drift
    evolved_X = X.copy()
    np.random.seed(42)

    # Shift days_to_trending (viral videos trend faster or slower)
    if "days_to_trending" in evolved_X.columns:
        evolved_X["days_to_trending"] = np.maximum(0, evolved_X["days_to_trending"] + np.random.normal(5, 2, len(evolved_X))).round()

    # Shift title_length (longer clickbait titles)
    if "title_length" in evolved_X.columns:
        evolved_X["title_length"] = np.maximum(1, evolved_X["title_length"] + np.random.normal(15, 5, len(evolved_X))).round()

    # Shift publish_hour
    if "publish_hour" in evolved_X.columns:
        evolved_X["publish_hour"] = ((evolved_X["publish_hour"] + np.random.choice([0, 3, 5], len(evolved_X))) % 24).astype(int)

    out_dir = ROOT / "data/monitoring/drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    evolved_x_file = out_dir / "evolved_data.csv"
    evolved_X.to_csv(evolved_x_file, index=False)
    print(f"[SUCCESS] Saved unlabeled evolved data to {evolved_x_file}")

    # Labeled evolved data for concept drift
    evolved_labeled = evolved_X.copy()
    # Evolve target views (shifted log-views with variance)
    evolved_log_y = y + np.random.normal(0.8, 0.4, len(y))
    evolved_labeled["views"] = np.maximum(np.expm1(evolved_log_y), 0).round(0)
    evolved_labeled["log_views"] = evolved_log_y

    evolved_labeled_file = out_dir / "evolved_labeled_data.csv"
    evolved_labeled.to_csv(evolved_labeled_file, index=False)
    print(f"[SUCCESS] Saved labeled evolved data to {evolved_labeled_file}")

    # Save matching baseline slice for 1-to-1 comparison
    baseline_slice = X.copy()
    baseline_slice.to_csv(baseline_x_path, index=False)
    np.save(baseline_y_path, y)

    return True


if __name__ == "__main__":
    ok = create_evolved_datasets()
    sys.exit(0 if ok else 1)
