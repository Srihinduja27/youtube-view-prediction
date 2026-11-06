from mlflow.tracking import MlflowClient


def check_registry():
    client = MlflowClient()
    name = "YouTube_Views_Production_Model"
    print(f"Checking MLflow Registry for '{name}'...")
    versions = client.search_model_versions(f"name='{name}'")
    for v in versions:
        print(f"Version: {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id}")


if __name__ == "__main__":
    check_registry()
