import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.common import regression_metrics


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = config["data"]["processed_dir"]
    try:
        X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run preprocessing first.")
        return False

    model_cfg = config["model"]
    params = {
        "n_estimators": model_cfg.get("n_estimators", 100),
        "max_depth": model_cfg.get("max_depth", 25),
        "min_samples_split": model_cfg.get("min_samples_split", 5),
        "min_samples_leaf": model_cfg.get("min_samples_leaf", 2),
        "max_features": model_cfg.get("max_features", "log2"),
        "random_state": model_cfg.get("random_state", 42),
        "n_jobs": model_cfg.get("n_jobs", -1),
    }

    model_name = config["registry"]["model_name"]

    with mlflow.start_run(run_name="RandomForest_MLOps_Candidate") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestRegressor")
        mlflow.log_param("lab", "YouTube MLOps Pipeline")

        print("[INFO] Training RandomForestRegressor model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating candidate model on held-out test split...")
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        preprocessor_path = config["paths"]["preprocessor"]
        if os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessing_pipeline")

        print("[INFO] Logging and registering model with MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        model_save_path = config["paths"]["model"]
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_save_path)
        joblib.dump(model, "models/random_forest_model.pkl")

        print(f"[SUCCESS] Metrics: log_r2={metrics['log_r2']:.4f} | R2={metrics['r2']:.4f} | MAE={metrics['mae']:,.2f}")
        print(f"[SUCCESS] Model saved to {model_save_path}")
        print(f"[SUCCESS] Model registered under '{model_name}'")
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")

    return True


if __name__ == "__main__":
    ok = train_and_register_model()
    sys.exit(0 if ok else 1)
