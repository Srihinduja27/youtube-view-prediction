import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from src.common import engineer_features, split_xy


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    model = None
    try:
        print(f"[INFO] Loading Production model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        print(f"[INFO] MLflow load failed ({exc}). Falling back to local deployment/model.pkl")
        fallback_paths = ["deployment/model.pkl", config["paths"]["model"]]
        for p in fallback_paths:
            if os.path.exists(p):
                model = joblib.load(p)
                break
        if model is None:
            print("[ERROR] No trained model artifact found.")
            return False

    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found at {preprocessor_path}")
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path, low_memory=False)

    sample = df.sample(n=10, random_state=42).copy()
    sample_feat, _ = engineer_features(sample)
    X_sample, _ = split_xy(sample_feat)

    X_trans = preprocessor.transform(X_sample)
    pred_log = model.predict(X_trans)
    pred_views = np.maximum(np.expm1(pred_log), 0).round(0)

    result = sample[["video_id", "title", "views"]].copy()
    result["Predicted_Views"] = pred_views.astype(np.int64)
    result["Predicted_Log1p"] = np.round(pred_log, 4)
    result["Absolute_Error"] = np.abs(result["views"] - result["Predicted_Views"])

    print("\n--- ML Pipeline Production Inference Results ---")
    print(result[["video_id", "views", "Predicted_Views", "Absolute_Error"]].head(10).to_string(index=False))
    print("------------------------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    out_path = config["paths"]["inference_output"]
    result.to_csv(out_path, index=False)
    print(f"[SUCCESS] Inferences saved to {out_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
