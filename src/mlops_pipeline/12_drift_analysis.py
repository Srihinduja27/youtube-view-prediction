from pathlib import Path
import pandas as pd
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA_DRIFT_PATH = ROOT / "outputs/drift/data_drift_report.csv"
PRED_DRIFT_PATH = ROOT / "outputs/drift/prediction_drift_report.csv"
CONCEPT_DRIFT_PATH = ROOT / "outputs/drift/concept_drift_report.csv"
COMBINED_REPORT_PATH = ROOT / "outputs/drift/drift_analysis_report.csv"


def ensure_reports():
    if not DATA_DRIFT_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "src/mlops_pipeline/12_data_drift.py")], check=True)
    if not PRED_DRIFT_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "src/mlops_pipeline/12_prediction_drift.py")], check=True)
    if not CONCEPT_DRIFT_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "src/mlops_pipeline/12_concept_drift.py")], check=True)


def main():
    print("=" * 70)
    print("COMBINED DRIFT ANALYSIS")
    print("=" * 70)

    ensure_reports()

    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    pred_drift = pd.read_csv(PRED_DRIFT_PATH)
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH)

    data_drift_detected = (data_drift["status"] == "DRIFT").any()
    pred_drift_detected = (pred_drift["status"] == "DRIFT").any()
    concept_drift_detected = (concept_drift["status"] == "DEGRADED").any()

    drift_observed = data_drift_detected or pred_drift_detected or concept_drift_detected

    print("\n1. DATA DRIFT")
    print(f"Features drifted: {(data_drift['status'] == 'DRIFT').sum()} / {len(data_drift)}")

    print("\n2. PREDICTION DRIFT")
    print(f"Status: {pred_drift.iloc[0]['status']}")

    print("\n3. CONCEPT DRIFT")
    print(f"Metrics degraded: {(concept_drift['status'] == 'DEGRADED').sum()} / {len(concept_drift)}")

    overall = "DRIFT_OBSERVED" if drift_observed else "NO_DRIFT"
    print("\n" + "=" * 70)
    print(f"OVERALL STATUS: {overall}")
    print("=" * 70)

    summary = pd.DataFrame([{
        "data_drift": data_drift_detected,
        "prediction_drift": pred_drift_detected,
        "concept_drift": concept_drift_detected,
        "overall_status": overall,
        "action_required": "RETRAIN_MODEL" if drift_observed else "MONITOR",
    }])

    COMBINED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(COMBINED_REPORT_PATH, index=False)
    print(f"\nCombined report saved to {COMBINED_REPORT_PATH}")
    return drift_observed


if __name__ == "__main__":
    drift = main()
    sys.exit(0 if drift else 1)
