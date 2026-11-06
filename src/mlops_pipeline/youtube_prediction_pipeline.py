import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("Data Loading & Ingestion", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Schema Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing & Feature Engineering", "src/mlops_pipeline/03_preprocessing.py"),
    ("Preprocessing Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Model Training & MLflow Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle Manager", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Deployment Bundle & Serving", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference Demo", "src/mlops_pipeline/09_inference.py"),
]


def load_data_load_report():
    report_path = ROOT / "artifacts/data_load_report.json"
    if not report_path.exists():
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(force=False):
    print("=" * 70)
    print("STARTING YOUTUBE PREDICTION MLOPS PIPELINE")
    print("=" * 70)

    run_log = {
        "pipeline": "YouTube Prediction MLOps",
        "stages": [],
        "force": force,
    }

    for stage_idx, (stage_name, script_rel) in enumerate(STAGES):
        print("\n" + "=" * 70)
        print(f"STAGE {stage_idx + 1}/{len(STAGES)}: {stage_name}")
        print("=" * 70)

        script_path = str(ROOT / script_rel)
        result = subprocess.run([sys.executable, script_path], cwd=ROOT)
        status = "PASSED" if result.returncode == 0 else "FAILED"

        run_log["stages"].append({
            "stage": stage_name,
            "script": script_rel,
            "status": status,
            "exit_code": result.returncode,
        })

        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print(f"[HALT] Pipeline stopped because '{stage_name}' failed.")
            run_log["overall_status"] = "FAILED"
            break

        # Check data load status on stage 0
        if stage_idx == 0 and not force:
            report = load_data_load_report()
            data_status = report.get("status")
            if data_status in ("already_ingested", "no_new_data"):
                print(f"[INFO] Ingestion status: {data_status}")
                print("[INFO] No new batch detected. Downstream retraining skipped.")
                print("[TIP] Use '--force' to execute all ML stages unconditionally.")
                run_log["overall_status"] = "NO_NEW_DATA"
                break

    all_passed = (len(run_log["stages"]) == len(STAGES)) and all(s["status"] == "PASSED" for s in run_log["stages"])
    if all_passed:
        run_log["overall_status"] = "PASSED"

    os.makedirs(ROOT / "logs", exist_ok=True)
    log_file = ROOT / "logs/youtube_prediction_pipeline_run_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    if run_log["overall_status"] == "PASSED":
        print("[SUCCESS] YouTube Prediction MLOps Pipeline completed successfully!")
    elif run_log["overall_status"] == "NO_NEW_DATA":
        print("[INFO] Pipeline finished cleanly without retraining (no new data).")
    else:
        print("[ERROR] YouTube Prediction MLOps Pipeline encountered failures.")
    print(f"Execution Log: {log_file}")
    print("=" * 70)
    return run_log["overall_status"] in ("PASSED", "NO_NEW_DATA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Prediction MLOps Pipeline Orchestrator")
    parser.add_argument("--force", action="store_true", help="Force execution of all stages even if no new batch was ingested.")
    args = parser.parse_args()

    passed = run_pipeline(force=args.force)
    sys.exit(0 if passed else 1)
