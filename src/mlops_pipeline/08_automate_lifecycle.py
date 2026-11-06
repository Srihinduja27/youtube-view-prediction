import json
import os
import sys
import yaml
from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)


def main():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    metric_name = config["registry"].get("decision_metric", "log_r2")

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"[ERROR] No versions found for model '{model_name}'.")
        sys.exit(1)

    # 1. Promote untracked models (None) to Staging
    new_versions = [v for v in versions if v.current_stage in ("None", "")]
    for v in new_versions:
        print(f"[INFO] Moving Version {v.version} from None to Staging...")
        client.transition_model_version_stage(
            name=model_name, version=v.version, stage="Staging", archive_existing_versions=False
        )

    # Refresh
    versions = client.search_model_versions(f"name='{model_name}'")
    staging_versions = [v for v in versions if v.current_stage == "Staging"]
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not staging_versions:
        print("[INFO] No candidate models in Staging. Keeping current Production.")
        report = {
            "promoted": False,
            "candidate_version": None,
            "production_version_before": int(prod_versions[0].version) if prod_versions else None,
            "production_version_after": int(prod_versions[0].version) if prod_versions else None,
            "reason": "No staging candidate present.",
        }
        os.makedirs("reports", exist_ok=True)
        with open("reports/lifecycle_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        sys.exit(0)

    candidate = staging_versions[-1]
    candidate_run = client.get_run(candidate.run_id)
    candidate_score = float(candidate_run.data.metrics.get(metric_name, 0.0))

    prod_version_before = int(prod_versions[0].version) if prod_versions else None
    prod_score = None
    promoted = False

    if not prod_versions:
        print("[INFO] No existing Production model. Staging model becomes Production.")
        client.transition_model_version_stage(
            name=model_name, version=candidate.version, stage="Production", archive_existing_versions=False
        )
        promoted = True
        prod_version_after = int(candidate.version)
    else:
        current_prod = prod_versions[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_score = float(prod_run.data.metrics.get(metric_name, 0.0))

        print(f"[INFO] Staging Candidate Version {candidate.version} | {metric_name}: {candidate_score:.4f}")
        print(f"[INFO] Production Champion Version {current_prod.version} | {metric_name}: {prod_score:.4f}")

        if candidate_score > prod_score:
            print(f"[SUCCESS] Candidate beat Champion ({candidate_score:.4f} > {prod_score:.4f})! Promoting Version {candidate.version} to Production.")
            client.transition_model_version_stage(
                name=model_name, version=current_prod.version, stage="Archived"
            )
            client.transition_model_version_stage(
                name=model_name, version=candidate.version, stage="Production"
            )
            promoted = True
            prod_version_after = int(candidate.version)
        else:
            print(f"[INFO] Challenger failed to beat Champion ({candidate_score:.4f} <= {prod_score:.4f}). Archiving Version {candidate.version}.")
            client.transition_model_version_stage(
                name=model_name, version=candidate.version, stage="Archived"
            )
            promoted = False
            prod_version_after = int(current_prod.version)

    report = {
        "promoted": promoted,
        "candidate_version": int(candidate.version),
        "production_version_before": prod_version_before,
        "production_version_after": prod_version_after,
        "metric": metric_name,
        "candidate_metric": candidate_score,
        "production_metric": prod_score,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/lifecycle_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Automated Model Lifecycle execution complete! Saved report to reports/lifecycle_report.json")
    sys.exit(0)


if __name__ == "__main__":
    main()
