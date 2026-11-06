import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "outputs/monitoring/monitor_trigger_report.json"
NEW_DATA_PATH = ROOT / "data/raw/new_youtube.csv"


def main():
    print("=" * 70)
    print("LAB 13: MONITORING + CONDITIONAL RETRAINING TRIGGER")
    print("=" * 70)

    # 1. Run drift analysis
    drift_res = subprocess.run([sys.executable, str(ROOT / "src/mlops_pipeline/12_drift_analysis.py")])
    drift_detected = drift_res.returncode == 0

    new_data_present = NEW_DATA_PATH.exists()
    status = "no_drift"
    action_taken = "monitoring_only"

    if drift_detected:
        print("[INFO] Drift detected: True")
        print(f"[INFO] New batch present: {new_data_present}")

        if new_data_present:
            print("[INFO] Triggering MLOps Retraining Pipeline...")
            pipe_res = subprocess.run([
                sys.executable,
                str(ROOT / "src/mlops_pipeline/youtube_prediction_pipeline.py")
            ])
            if pipe_res.returncode == 0:
                status = "retraining_completed"
                action_taken = "pipeline_executed_successfully"
            else:
                status = "retraining_failed"
                action_taken = "pipeline_failed"
        else:
            print("[INFO] Drift observed but new_youtube.csv not available. Waiting for data.")
            status = "drift_waiting_for_data"
            action_taken = "waiting_for_data_batch"
    else:
        print("[INFO] No significant drift detected. Model is performing within tolerance.")

    report = {
        "status": status,
        "action_taken": action_taken,
        "drift_detected": drift_detected,
        "new_data_available": new_data_present,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n[SUCCESS] Monitoring trigger report saved to {REPORT_PATH}")
    print(f"Outcome: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
