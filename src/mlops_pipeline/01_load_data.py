from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import sys

import pandas as pd
import yaml

# Allow local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import validate_schema
from src.common import load_raw

CONFIG_PATH = Path("config.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_lineage(lineage_path):
    if not lineage_path.exists():
        return []
    try:
        with lineage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_lineage(lineage_path, lineage):
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)


def get_next_archive_version(archive_dir):
    versions = []
    for file_path in archive_dir.glob("youtube_v*.csv"):
        try:
            version = int(file_path.stem.replace("youtube_v", ""))
            versions.append(version)
        except ValueError:
            continue
    return max(versions, default=0) + 1


def ensure_current_data(raw_path):
    if not raw_path.exists():
        print(f"[INFO] Building base combined dataset at {raw_path}...")
        df = load_raw()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        return df
    return pd.read_csv(raw_path, low_memory=False)


def main():
    print("=" * 70)
    print("DATA LOADING & LINEAGE TRACKING")
    print("=" * 70)

    config = load_config()
    raw_path = Path(config["data"]["raw_path"])
    new_data_path = Path(config["data"].get("new_data_path", "data/raw/new_youtube.csv"))
    archive_dir = raw_path.parent / "archive"
    lineage_path = archive_dir / "dataset_lineage.json"
    report_path = Path("artifacts/data_load_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    current_df = ensure_current_data(raw_path)
    print(f"[INFO] Current dataset records: {len(current_df)}")

    if not new_data_path.exists():
        print("[INFO] No new_youtube.csv found. Using current baseline.")
        report = {
            "status": "no_new_data",
            "ingestion_performed": False,
            "current_rows": len(current_df),
            "current_columns": len(current_df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"[SUCCESS] Data load report saved to {report_path}")
        return True

    print("[INFO] New dataset batch detected at", new_data_path)
    new_df = pd.read_csv(new_data_path, low_memory=False)
    print(f"[INFO] Validating schema of new batch ({len(new_df)} records)...")
    if not validate_schema(new_df, output_report_name="new_data_validation.csv"):
        print("[ERROR] New batch failed schema validation.")
        return False

    new_hash = calculate_file_hash(new_data_path)
    lineage = load_lineage(lineage_path)
    for record in lineage:
        if record.get("file_hash") == new_hash:
            print("[INFO] Batch SHA256 already recorded in lineage. Skipping duplicate ingestion.")
            report = {
                "status": "already_ingested",
                "ingestion_performed": False,
                "new_data_hash": new_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_rows": len(current_df),
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            return True

    # Ingest new batch
    next_v = get_next_archive_version(archive_dir)
    archive_file = archive_dir / f"youtube_v{next_v:02d}.csv"
    shutil.copy2(raw_path, archive_file)
    print(f"[INFO] Archived prior dataset to {archive_file}")

    updated_df = pd.concat([current_df, new_df], ignore_index=True)
    updated_df.to_csv(raw_path, index=False)
    print(f"[SUCCESS] Appended new batch. Total rows now: {len(updated_df)}")

    lineage_record = {
        "dataset_version": f"v{next_v:02d}",
        "source_file": str(new_data_path.name),
        "archive_file": str(archive_file.name),
        "file_hash": new_hash,
        "rows_added": len(new_df),
        "total_rows": len(updated_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    lineage.append(lineage_record)
    save_lineage(lineage_path, lineage)

    report = {
        "status": "ingested",
        "ingestion_performed": True,
        "dataset_version": lineage_record["dataset_version"],
        "archive_file": lineage_record["archive_file"],
        "source_file": lineage_record["source_file"],
        "new_data_hash": new_hash,
        "current_rows": len(updated_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[SUCCESS] Data ingestion complete. Report saved to {report_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
