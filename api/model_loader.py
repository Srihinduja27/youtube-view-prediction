from pathlib import Path
from typing import Any
import json
import joblib
import yaml


class ModelLoader:
    """Loads the ML model and preprocessing pipeline from verified deployment artifacts."""

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path
        self.config: dict[str, Any] = {}
        self.model = None
        self.preprocessor = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.metadata = {}

    def load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file) or {}

    def load(self) -> None:
        self.load_config()
        deployment_dir = self.project_root / "deployment"
        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        metadata_path = deployment_dir / "model_metadata.json"

        # Fallback to models/ if deployment/ not ready yet
        if not model_path.exists():
            fallback_model = self.project_root / "models" / "random_forest_mlflow.pkl"
            if not fallback_model.exists():
                fallback_model = self.project_root / "models" / "random_forest_model.pkl"
            model_path = fallback_model

        if not preprocessor_path.exists():
            preprocessor_path = self.project_root / "models" / "preprocessor.pkl"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.model_name = self.metadata.get("model_name")
            self.model_version = self.metadata.get("model_version")
            self.run_id = self.metadata.get("run_id")
        else:
            self.model_name = "YouTube_Views_Production_Model"
            self.model_version = "1"
            self.run_id = "initial"

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None


if __name__ == "__main__":
    loader = ModelLoader()
    loader.load()
    print("Model ready:", loader.is_ready())
