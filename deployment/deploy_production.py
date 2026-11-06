import json
import subprocess
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
K8S_DEPLOYMENT_PATH = PROJECT_ROOT / "k8s" / "deployment.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"

IMAGE_NAME = "youtube-prediction-api"


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Deployment metadata not found: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def deploy_production():
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT")
    print("=" * 70)

    metadata = load_metadata()
    model_name = metadata["model_name"]
    model_version = metadata["model_version"]
    model_stage = metadata["model_stage"]
    run_id = metadata["run_id"]

    if model_stage != "Production":
        raise RuntimeError(f"Deployment blocked. Model stage is '{model_stage}', not 'Production'.")

    model_path = DEPLOYMENT_DIR / "model.pkl"
    preprocessor_path = DEPLOYMENT_DIR / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    image_tag = f"{IMAGE_NAME}:model-v{model_version}"
    print(f"[INFO] Model        : {model_name}")
    print(f"[INFO] Model version: {model_version}")
    print(f"[INFO] Run ID       : {run_id}")
    print(f"[INFO] Target image : {image_tag}")

    # Check Docker daemon availability
    docker_ready = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not docker_ready:
        print("[INFO] Docker daemon is not running. Docker/Kubernetes rollout skipped.")
        deploy_status = "prepared_without_container"
    else:
        print("[INFO] Docker daemon is active. Building container image...")
        build_res = subprocess.run(["docker", "build", "-t", image_tag, "."], cwd=PROJECT_ROOT)
        if build_res.returncode == 0:
            print(f"[SUCCESS] Docker image '{image_tag}' built successfully.")
            deploy_status = "docker_image_built"
        else:
            deploy_status = "docker_build_failed"

    report = {
        "status": deploy_status,
        "model_name": model_name,
        "model_version": str(model_version),
        "run_id": run_id,
        "docker_image": image_tag,
        "deployment_bundle": {
            "model": str(model_path.relative_to(PROJECT_ROOT)),
            "preprocessor": str(preprocessor_path.relative_to(PROJECT_ROOT)),
            "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Deployment report saved to {REPORT_PATH}")
    return True


if __name__ == "__main__":
    try:
        passed = deploy_production()
        sys.exit(0 if passed else 1)
    except Exception as exc:
        print(f"[ERROR] Production deployment failed: {exc}")
        sys.exit(1)
