import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "inference" / "inference_log.jsonl"


def log_inference(request_id: str, input_data: dict, predicted_views: int, predicted_log: float, latency_ms: float):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "predicted_views": predicted_views,
        "predicted_log_views": predicted_log,
        "latency_ms": round(latency_ms, 2),
        "inputs": input_data,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
