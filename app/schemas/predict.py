from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bmi: float = Field(..., gt=0, le=80)
    age: float = Field(..., gt=0, le=120)
    fgb: float = Field(..., gt=0, le=500)
    avg_systolic: float = Field(..., gt=50, le=300)
    avg_diastolic: float = Field(..., gt=30, le=200)
    insulin: float = Field(..., ge=0, le=500)


class MealPlan(BaseModel):
    breakfast: str
    lunch: str
    dinner: str


class NutritionInfo(BaseModel):
    energy_kcal: float
    carbs: float
    protein_g: float
    fat_g: float
    freesugar_g: float
    fibre_g: float
    cholestrol_mg: float
    calcium_mg: float


class MealPlanItem(BaseModel):
    meal_type: Literal["breakfast", "lunch", "dinner"]
    food_name: str
    nutrition: NutritionInfo | None = None


class TopPrediction(BaseModel):
    label: str
    probability: float = Field(..., ge=0, le=1)


class PredictionMetadata(BaseModel):
    model_version: str
    inference_timestamp: datetime


class PredictData(BaseModel):
    prediction: str
    normalized_prediction: str
    confidence: float = Field(..., ge=0, le=1)
    top_predictions: list[TopPrediction] = Field(default_factory=list)
    meal_plan: list[MealPlanItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: PredictionMetadata


class PredictResponse(BaseModel):
    success: Literal[True] = True
    data: PredictData
    request_id: str | None = None
