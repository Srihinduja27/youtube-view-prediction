import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_deployment():
    config = load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    model_version = prod_versions[0]
    version = model_version.version
    run_id = model_version.run_id

    deployment_dir = ROOT / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")

    print("[INFO] Loading and exporting Production model...")
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to local saved model
        model = joblib.load(ROOT / config["paths"]["model"])

    model_path = deployment_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Model saved to {model_path}")

    print("[INFO] Exporting matching preprocessor...")
    src_preprocessor = ROOT / config["paths"]["preprocessor"]
    dst_preprocessor = deployment_dir / "preprocessor.pkl"
    if src_preprocessor.exists():
        shutil.copy2(src_preprocessor, dst_preprocessor)
    else:
        # Attempt download from run
        client.download_artifacts(run_id, "preprocessing_pipeline/preprocessor.pkl", dst_path=str(deployment_dir))

    print(f"[SUCCESS] Preprocessor saved to {dst_preprocessor}")

    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": "Production",
        "run_id": run_id,
        "target": "views",
        "target_transform": "log1p(views)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "MLflow",
    }

    metadata_path = deployment_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Metadata saved to {metadata_path}")
    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("|-- model.pkl")
    print("|-- preprocessor.pkl")
    print("\\-- model_metadata.json")
    print("\n[SUCCESS] Deployment preparation completed.")
    return True


if __name__ == "__main__":
    ok = prepare_deployment()
    sys.exit(0 if ok else 1)
