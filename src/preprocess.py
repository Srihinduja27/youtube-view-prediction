import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ENGINEERED_NUMERIC, DROP_COLUMNS, TARGET, engineer_features, list_raw_files, load_raw

PROCESSED = ROOT / "data" / "processed" / "youtube_processed.csv"
METADATA = ROOT / "data" / "processed" / "dataset_metadata.json"
ARTIFACT = ROOT / "artifacts" / "preprocessing_metadata.json"


def preprocess():
    raw = load_raw()
    original_shape = list(raw.shape)
    csv_files, _ = list_raw_files()

    # Same preprocessing steps used in the notebook.
    df, info = engineer_features(raw)

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)

    metadata = {
        "dataset": "YouTube Trending Video Statistics (kaggle: datasnaek/youtube-new)",
        "source_files": [Path(f).name for f in csv_files],
        "original_shape": original_shape,
        "processed_shape": list(df.shape),
        "date_format_trending_date": "%y.%d.%m",
        "engineered_features": ENGINEERED_NUMERIC,
        "missing_before_median_fill": info["missing_before_fill"],
        "fill_strategy": "median",
        "boolean_columns_cast_to_int": ["comments_disabled", "ratings_disabled", "video_error_or_removed"],
        "columns_excluded_from_model": DROP_COLUMNS,
        "target": TARGET,
        "target_transform": "log1p",
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ARTIFACT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    df = preprocess()
    print(f"Processed dataset saved to {PROCESSED}")
    print(f"Shape: {df.shape}")
