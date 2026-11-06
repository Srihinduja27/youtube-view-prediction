import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "data/processed/X_test_raw.csv"
EVOLVED_PATH = ROOT / "data/monitoring/drift/evolved_data.csv"
REPORT_PATH = ROOT / "outputs/drift/data_drift_report.csv"

NUMERICAL_COLUMNS = ["days_to_trending", "title_length", "description_length", "publish_hour", "tag_count"]
KS_THRESHOLD = 0.10


def main():
    print("=" * 70)
    print("DATA DRIFT ANALYSIS")
    print("=" * 70)

    if not BASELINE_PATH.exists() or not EVOLVED_PATH.exists():
        print("[INFO] Creating evolved dataset first...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src/mlops_pipeline/12_create_evolved_dataset.py")], check=True)

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    results = []
    features_to_check = [c for c in NUMERICAL_COLUMNS if c in baseline.columns and c in evolved.columns]

    for col in features_to_check:
        b_vals = baseline[col].dropna()
        e_vals = evolved[col].dropna()
        stat, pval = ks_2samp(b_vals, e_vals)
        status = "DRIFT" if stat >= KS_THRESHOLD else "OK"
        results.append({
            "feature": col,
            "type": "numerical",
            "metric": "KS",
            "score": round(float(stat), 6),
            "p_value": round(float(pval), 6),
            "threshold": KS_THRESHOLD,
            "status": status,
        })

    report_df = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    drift_count = (report_df["status"] == "DRIFT").sum()
    print(report_df.to_string(index=False))
    print("=" * 70)
    print(f"Features analyzed : {len(report_df)}")
    print(f"Drift detected    : {drift_count}")
    print(f"Report saved      : {REPORT_PATH}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
