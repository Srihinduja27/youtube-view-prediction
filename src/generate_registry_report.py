import json
import os

from mlflow.tracking import MlflowClient


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    client = MlflowClient()
    model_name = "YouTube_Views_Production_Model"

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in versions if mv.current_stage == "Production"), None)

        if not champion:
            print("[ERROR] No model found in Production stage!")
            return

        run = client.get_run(champion.run_id)

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source,
            },
            "performance_metrics": {
                "log_r2": run.data.metrics.get("log_r2"),
                "log_mae": run.data.metrics.get("log_mae"),
                "log_rmse": run.data.metrics.get("log_rmse"),
                "r2": run.data.metrics.get("r2"),
                "mae": run.data.metrics.get("mae"),
                "rmse": run.data.metrics.get("rmse"),
            },
            "target_transform": "log1p(views); apply expm1 to predictions",
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl",
        }

        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")


if __name__ == "__main__":
    generate_registry_report()
