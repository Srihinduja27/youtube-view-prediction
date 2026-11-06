from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from src.common import regression_metrics

BASELINE_X_PATH = ROOT / "data/processed/X_test_raw.csv"
BASELINE_Y_PATH = ROOT / "data/processed/y_test.npy"
EVOLVED_LABELED_PATH = ROOT / "data/monitoring/drift/evolved_labeled_data.csv"
REPORT_PATH = ROOT / "outputs/drift/concept_drift_report.csv"


def main():
    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS")
    print("=" * 70)

    loader = ModelLoader()
    loader.load()

    baseline_X = pd.read_csv(BASELINE_X_PATH)
    baseline_y = np.load(BASELINE_Y_PATH)

    evolved = pd.read_csv(EVOLVED_LABELED_PATH)
    evolved_X = evolved.drop(columns=[c for c in ["views", "log_views"] if c in evolved.columns])
    evolved_y = evolved["log_views"].to_numpy()

    # Evaluate baseline
    b_trans = loader.preprocessor.transform(clean_prediction_data(baseline_X))
    b_preds = loader.model.predict(b_trans)
    b_metrics = regression_metrics(baseline_y[:len(b_preds)], b_preds)

    # Evaluate evolved
    e_trans = loader.preprocessor.transform(clean_prediction_data(evolved_X))
    e_preds = loader.model.predict(e_trans)
    e_metrics = regression_metrics(evolved_y[:len(e_preds)], e_preds)

    rows = []
    for k in ["log_r2", "log_mae", "log_rmse", "r2", "mae", "rmse"]:
        b_val = b_metrics.get(k, 0.0)
        e_val = e_metrics.get(k, 0.0)
        delta = e_val - b_val
        degraded = (delta < -0.05) if "r2" in k else (delta > 0.05 * b_val)
        rows.append({
            "metric": k,
            "baseline": round(float(b_val), 4),
            "evolved": round(float(e_val), 4),
            "change": round(float(delta), 4),
            "status": "DEGRADED" if degraded else "STABLE",
        })

    report_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    print(report_df.to_string(index=False))
    print(f"\nReport saved to {REPORT_PATH}")
    return True


if __name__ == "__main__":
    main()
