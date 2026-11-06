import os
import sys
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_youtube_schema():
    """Schema specification for raw YouTube trending records."""
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
    }, strict=False)


def validate_schema(df, output_report_name="baseline_validation.csv"):
    """Validates DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_youtube_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[['schema_context', 'column', 'check', 'failure_case', 'index']]
        os.makedirs(PROJECT_ROOT / "artifacts", exist_ok=True)
        report_path = PROJECT_ROOT / "artifacts" / output_report_name
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False
