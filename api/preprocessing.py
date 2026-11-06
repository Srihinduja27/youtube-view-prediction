import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import DROP_COLUMNS, TARGET, engineer_features


EXPECTED_FEATURES = [
    "category_id", "comments_disabled", "ratings_disabled", "video_error_or_removed",
    "days_to_trending", "publish_year", "publish_month", "publish_day", "publish_dayofweek", "publish_hour",
    "trending_year", "trending_month", "trending_day", "trending_dayofweek",
    "title_length", "title_word_count", "description_length", "description_word_count", "tag_count"
]


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares DataFrame for inference."""
    df_copy = df.copy()

    # If raw video data with dates/text is provided, engineer features
    if "publish_time" in df_copy.columns or "title" in df_copy.columns:
        df_copy, _ = engineer_features(df_copy)
        drop_cols = [c for c in DROP_COLUMNS + [TARGET] if c in df_copy.columns]
        df_copy = df_copy.drop(columns=drop_cols)

    # Ensure all expected 19 features exist
    for col in EXPECTED_FEATURES:
        if col not in df_copy.columns:
            df_copy[col] = 0

    return df_copy[EXPECTED_FEATURES]
