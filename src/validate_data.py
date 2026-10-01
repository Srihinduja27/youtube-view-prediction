import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

from src.common import list_raw_files, load_raw


def get_youtube_schema():
    """Schema specification for the raw Kaggle YouTube trending files (all countries combined)."""
    return pa.DataFrameSchema({
        "video_id": Column(pa.String),
        "trending_date": Column(pa.String, Check.str_matches(r"^\d{2}\.\d{2}\.\d{2}$")),
        "title": Column(pa.String, nullable=True),
        "channel_title": Column(pa.String, nullable=True),
        "category_id": Column(pa.Int, Check.ge(1)),
        "publish_time": Column(pa.String),
        "tags": Column(pa.String, nullable=True),
        "views": Column(pa.Int, Check.ge(0)),
        "likes": Column(pa.Int, Check.ge(0)),
        "dislikes": Column(pa.Int, Check.ge(0)),
        "comment_count": Column(pa.Int, Check.ge(0)),
        "thumbnail_link": Column(pa.String, nullable=True),
        "comments_disabled": Column(pa.Bool),
        "ratings_disabled": Column(pa.Bool),
        "video_error_or_removed": Column(pa.Bool),
        "description": Column(pa.String, nullable=True),
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_youtube_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        print(failures.head(50).to_string())

        os.makedirs(ROOT / "artifacts", exist_ok=True)
        report_path = ROOT / "artifacts" / output_report_name
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    csv_files, json_files = list_raw_files()
    print(f"[INFO] Found {len(csv_files)} CSV files and {len(json_files)} category JSON files.")
    if csv_files:
        ok = validate_schema(load_raw(), output_report_name="baseline_validation.csv")
        sys.exit(0 if ok else 1)
    else:
        print(f"[ERROR] No *videos.csv files found in data/raw/. See README for the dataset link.")
        sys.exit(1)
