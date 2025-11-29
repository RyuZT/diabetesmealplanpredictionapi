from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Import fungsi dari meal_logic (hanya yang ADA)
from meal_logic import (
    normalize_status,
    recommend_meal
)

# === LOAD MODEL & SCALER ===
model = joblib.load("best_xgb.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === LOAD FOOD DATA ===
df_food = pd.read_csv("foods_prepared.csv")

app = FastAPI()

# ==== REQUEST BODY ====
class UserInput(BaseModel):
    bmi: float
    age: float
    fgb: float
    avg_systolyc: float
    avg_dystolyc: float
    insulin: float


@app.get("/")
def home():
    return {"message": "API is running successfully!"}


@app.post("/predict")
def predict_diabetes(data: UserInput):

    insulin_log = __import__("numpy").log1p(data.insulin)

    X = pd.DataFrame([{
        "bmi": data.bmi,
        "age": data.age,
        "fgb": data.fgb,
        "avg_systolyc": data.avg_systolyc,
        "avg_dystolyc": data.avg_dystolyc,
        "insulin_log": insulin_log
    }])

    X_scaled = scaler.transform(X)
    pred_class = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Normalize untuk meal plan
    status_norm = normalize_status(pred_label)

    # Generate meals
    meal_plan = recommend_meal(df_food, status_norm)

    return {
        "status_prediction": pred_label,
        "meal_status_normalized": status_norm,
        "meal_plan": meal_plan
    }
