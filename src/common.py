
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "views"
TEST_SIZE = 0.2
RANDOM_STATE = 42

BOOLEAN_COLUMNS = ["comments_disabled", "ratings_disabled", "video_error_or_removed"]

# Identifier / raw-text / leakage columns removed before modelling (same list as the notebook).
DROP_COLUMNS = [
    "video_id", "title", "channel_title", "tags", "description",
    "thumbnail_link", "publish_time", "trending_date",
    "likes", "dislikes", "comment_count",
]

ENGINEERED_NUMERIC = [
    "publish_year", "publish_month", "publish_day", "publish_dayofweek", "publish_hour",
    "trending_year", "trending_month", "trending_day", "trending_dayofweek",
    "days_to_trending", "title_length", "title_word_count",
    "description_length", "description_word_count", "tag_count",
]


def list_raw_files(raw_dir=RAW_DIR):
    csv_files = sorted(glob.glob(os.path.join(str(raw_dir), "*videos.csv")))
    json_files = sorted(glob.glob(os.path.join(str(raw_dir), "*_category_id.json")))
    return csv_files, json_files


def load_csv_safely(file_path):
    for encoding in ("utf-8", "ISO-8859-1", "latin1"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode the file: {file_path}")


def load_raw(raw_dir=RAW_DIR):
    """Concatenate all <CC>videos.csv files found in data/raw."""
    csv_files, _ = list_raw_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No *videos.csv files found in {raw_dir}. Download the dataset from "
            "https://www.kaggle.com/datasets/datasnaek/youtube-new and extract it into data/raw/."
        )
    return pd.concat([load_csv_safely(f) for f in csv_files], ignore_index=True)


def engineer_features(raw_df):
    """Notebook feature engineering. Returns (dataframe, info_dict)."""
    df = raw_df.copy()

    # trending_date is stored as YY.DD.MM in the Kaggle files.
    df["trending_date"] = pd.to_datetime(
        df["trending_date"].astype(str), format="%y.%d.%m", errors="coerce"
    )
    df["publish_time"] = pd.to_datetime(
        df["publish_time"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    df["days_to_trending"] = (
        df["trending_date"].dt.normalize() - df["publish_time"].dt.normalize()
    ).dt.days

    for part in ("year", "month", "day", "dayofweek", "hour"):
        df[f"publish_{part}"] = getattr(df["publish_time"].dt, part)
    for part in ("year", "month", "day", "dayofweek"):
        df[f"trending_{part}"] = getattr(df["trending_date"].dt, part)

    df["title_length"] = df["title"].fillna("").str.len()
    df["title_word_count"] = df["title"].fillna("").str.split().str.len()
    df["description_length"] = df["description"].fillna("").str.len()
    df["description_word_count"] = df["description"].fillna("").str.split().str.len()
    df["tag_count"] = df["tags"].fillna("").apply(
        lambda x: 0 if x == "[none]" else len(x.split("|"))
    )

    for column in BOOLEAN_COLUMNS:
        df[column] = df[column].astype(int)

    missing_before_fill = {
        c: int(df[c].isna().sum()) for c in ENGINEERED_NUMERIC if df[c].isna().sum() > 0
    }
    for column in ENGINEERED_NUMERIC:
        if df[column].isna().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    return df, {"missing_before_fill": missing_before_fill}


def split_xy(df):
    """Model matrix X and the log1p(views) target."""
    X = df.drop(columns=[c for c in DROP_COLUMNS + [TARGET] if c in df.columns])
    y = np.log1p(df[TARGET])
    return X, y


def feature_lists(X):
    """Numeric = every numeric dtype (including the int32 date parts); categorical = the rest."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, impute=False):
    num_steps = []
    if impute:
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), num_cols),
            ("cat", Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ]
    )


def regression_metrics(y_true_log, y_pred_log):
    """Metrics on the original views scale and on the log1p scale."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    actual = np.expm1(y_true_log)
    pred = np.maximum(np.expm1(y_pred_log), 0)
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
        "log_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "log_rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "log_r2": float(r2_score(y_true_log, y_pred_log)),
    }
