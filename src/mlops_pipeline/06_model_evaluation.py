import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.common import regression_metrics


def main():
    print("[INFO] Starting ML Pipeline Model Evaluation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        sys.exit(1)

    model = joblib.load(model_path)
    processed_dir = config["data"]["processed_dir"]

    X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    print("\n--- ML Pipeline: Model Evaluation ---")
    print(f"Log R2    : {metrics['log_r2']:.4f}")
    print(f"Log MAE   : {metrics['log_mae']:.4f}")
    print(f"Log RMSE  : {metrics['log_rmse']:.4f}")
    print(f"R2 (views): {metrics['r2']:.4f}")
    print(f"MAE       : {metrics['mae']:,.2f}")
    print(f"RMSE      : {metrics['rmse']:,.2f}")
    print("---------------------------------------\n")

    os.makedirs("reports", exist_ok=True)
    report_path = config["paths"]["evaluation_report"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Plot sample scatter
    os.makedirs("artifacts", exist_ok=True)
    idx = np.random.RandomState(42).choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.3, s=8, color="#2563EB")
    lims = [float(np.min(y_test)), float(np.max(y_test))]
    plt.plot(lims, lims, "--", color="#EF4444")
    plt.title("Actual vs Predicted log1p(views)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig("artifacts/actual_vs_predicted.png", dpi=120)
    plt.close()

    print(f"[SUCCESS] Evaluation report saved to {report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
