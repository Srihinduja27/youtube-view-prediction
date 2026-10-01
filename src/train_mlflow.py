import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.common import regression_metrics

EXPERIMENT = "YouTube_Views_Prediction"


def train_and_track(run_name="RandomForest_Baseline", params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # 1. Load Data
    X_train = np.load(ROOT / "data/processed/X_train_final.npy")
    X_test = np.load(ROOT / "data/processed/X_test_final.npy")
    y_train = np.load(ROOT / "data/processed/y_train.npy")
    y_test = np.load(ROOT / "data/processed/y_test.npy")

    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        # 2. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestRegressor")
        mlflow.log_param("target", "log1p(views)")

        # 3. Train Model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 4. Evaluate
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        # 5. Log Metrics
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: R2 (original) = {metrics['r2']:.4f} | R2 (log) = {metrics['log_r2']:.4f}")

        # 6. Diagnostic plots
        os.makedirs(ROOT / "artifacts", exist_ok=True)

        idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
        fig_ap, ax_ap = plt.subplots(figsize=(6, 6))
        ax_ap.scatter(y_test[idx], y_pred[idx], alpha=0.3, s=8)
        lims = [float(y_test.min()), float(y_test.max())]
        ax_ap.plot(lims, lims, "--", color="red")
        ax_ap.set_title(f"Actual vs Predicted log(1+views) - {run_name}")
        ax_ap.set_xlabel("Actual")
        ax_ap.set_ylabel("Predicted")
        ap_path = ROOT / "artifacts/actual_vs_predicted.png"
        fig_ap.savefig(ap_path, bbox_inches="tight")
        plt.close(fig_ap)
        mlflow.log_artifact(str(ap_path), artifact_path="plots")

        fig_res, ax_res = plt.subplots(figsize=(6, 5))
        ax_res.hist(y_test - y_pred, bins=60)
        ax_res.set_title(f"Residuals (log scale) - {run_name}")
        ax_res.set_xlabel("Actual - Predicted")
        res_path = ROOT / "artifacts/residuals.png"
        fig_res.savefig(res_path, bbox_inches="tight")
        plt.close(fig_res)
        mlflow.log_artifact(str(res_path), artifact_path="plots")

        # 7. Lineage metadata
        meta = ROOT / "data/processed/dataset_metadata.json"
        if meta.exists():
            mlflow.log_artifact(str(meta), artifact_path="metadata")

        # 8. Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Local backup
        os.makedirs(ROOT / "models", exist_ok=True)
        joblib.dump(model, ROOT / "models/random_forest_mlflow.pkl")
        print(f"Run '{run_name}' successfully tracked!")


if __name__ == "__main__":
    # Baseline Run 1 (notebook baseline parameters)
    train_and_track(run_name="RandomForest")

    # Run 2: tuned parameters found by RandomizedSearchCV in the notebook
    # tuned_params = {
    #     "n_estimators": 200,
    #     "max_depth": 30,
    #     "min_samples_split": 5,
    #     "min_samples_leaf": 1,
    #     "max_features": "log2",
    #     "random_state": 42,
    #     "n_jobs": -1,
    # }
    # train_and_track(run_name="RandomForest_Tuned", params=tuned_params)
