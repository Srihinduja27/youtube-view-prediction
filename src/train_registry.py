import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.common import regression_metrics

MODEL_NAME = "YouTube_Views_Production_Model"


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    mlflow.set_experiment("YouTube_Views_Prediction")

    try:
        X_train = np.load(ROOT / 'data/processed/X_train_final.npy')
        X_test = np.load(ROOT / 'data/processed/X_test_final.npy')
        y_train = np.load(ROOT / 'data/processed/y_train.npy')
        y_test = np.load(ROOT / 'data/processed/y_test.npy')
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        sys.exit(1)

    # Notebook baseline (V1):
    # params = {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5,
    #           "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}
    # run_name = "RandomForest_Registry_V1"

    # Tuned parameters from the notebook's RandomizedSearchCV (V2)
    params = {
        "n_estimators": 200,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "random_state": 42,
        "n_jobs": -1,
    }
    run_name = "RandomForest_Registry_V2"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestRegressor")

        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestRegressor(**params)
        rf_model.fit(X_train, y_train)

        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # Preprocessor lineage
        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact(str(ROOT / "models/preprocessor.pkl"), artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"[SUCCESS] Metrics logged: R2 (log) = {metrics['log_r2']:.4f} | R2 (views) = {metrics['r2']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: '{MODEL_NAME}'")


if __name__ == "__main__":
    train_and_register_model()
