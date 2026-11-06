from typing import List, Optional
from pydantic import BaseModel, Field


class VideoInput(BaseModel):
    category_id: int = Field(default=24, description="YouTube Category ID (e.g., 10=Music, 24=Entertainment)")
    comments_disabled: int = Field(default=0, ge=0, le=1)
    ratings_disabled: int = Field(default=0, ge=0, le=1)
    video_error_or_removed: int = Field(default=0, ge=0, le=1)
    days_to_trending: float = Field(default=2.0, ge=0.0)
    publish_year: int = Field(default=2018)
    publish_month: int = Field(default=5, ge=1, le=12)
    publish_day: int = Field(default=15, ge=1, le=31)
    publish_dayofweek: int = Field(default=1, ge=0, le=6)
    publish_hour: int = Field(default=16, ge=0, le=23)
    trending_year: int = Field(default=2018)
    trending_month: int = Field(default=5, ge=1, le=12)
    trending_day: int = Field(default=17, ge=1, le=31)
    trending_dayofweek: int = Field(default=3, ge=0, le=6)
    title_length: int = Field(default=40, ge=1)
    title_word_count: int = Field(default=7, ge=1)
    description_length: int = Field(default=300, ge=0)
    description_word_count: int = Field(default=35, ge=0)
    tag_count: int = Field(default=12, ge=0)


class PredictionResponse(BaseModel):
    predicted_views: int
    predicted_log_views: float
    model_version: str
    request_id: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    videos: List[VideoInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_records: int
    processing_time_ms: float
