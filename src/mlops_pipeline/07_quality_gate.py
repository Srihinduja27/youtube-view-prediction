import json
import os
import sys
import yaml


def main():
    print("[INFO] Starting Model Quality Gate...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = config["paths"]["evaluation_report"]
    if not os.path.exists(report_path):
        print(f"[ERROR] Evaluation report not found at {report_path}")
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

    gate_cfg = config.get("quality_gate", {"metric": "log_r2", "minimum_value": 0.65})
    metric_name = gate_cfg["metric"]
    threshold = float(gate_cfg["minimum_value"])

    actual_val = float(eval_metrics.get(metric_name, 0.0))
    passed = actual_val >= threshold

    qg_report = {
        "status": "PASSED" if passed else "FAILED",
        "metric": metric_name,
        "actual_value": actual_val,
        "threshold": threshold,
        "passed": passed,
    }

    os.makedirs("reports", exist_ok=True)
    out_path = config["paths"]["quality_gate_report"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qg_report, f, indent=4)

    print(f"[INFO] Quality Gate: {metric_name}={actual_val:.4f} | required>={threshold:.4f}")
    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
        sys.exit(0)
    else:
        print("[ERROR] Model Quality Gate FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
