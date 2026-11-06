from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_PATH = ROOT / "data/processed/X_test_raw.csv"
EVOLVED_PATH = ROOT / "data/monitoring/drift/evolved_data.csv"
REPORT_PATH = ROOT / "outputs/drift/prediction_drift_report.csv"
KS_THRESHOLD = 0.10


def main():
    print("=" * 70)
    print("PREDICTION DRIFT ANALYSIS")
    print("=" * 70)

    loader = ModelLoader()
    loader.load()

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    b_clean = clean_prediction_data(baseline)
    e_clean = clean_prediction_data(evolved)

    b_trans = loader.preprocessor.transform(b_clean)
    e_trans = loader.preprocessor.transform(e_clean)

    b_preds = loader.model.predict(b_trans)
    e_preds = loader.model.predict(e_trans)

    stat, pval = ks_2samp(b_preds, e_preds)
    status = "DRIFT" if stat >= KS_THRESHOLD else "OK"

    report_df = pd.DataFrame([{
        "metric": "KS_2samp",
        "statistic": round(float(stat), 6),
        "p_value": round(float(pval), 6),
        "threshold": KS_THRESHOLD,
        "status": status,
        "baseline_mean_log_views": round(float(np.mean(b_preds)), 4),
        "evolved_mean_log_views": round(float(np.mean(e_preds)), 4),
    }])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    print(report_df.to_string(index=False))
    print(f"\nPrediction Drift Status: {status}")
    print(f"Saved report to {REPORT_PATH}")
    return True


if __name__ == "__main__":
    main()
