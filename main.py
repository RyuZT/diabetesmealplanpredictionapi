# =============================================================
# main.py — FastAPI Backend for Diabetes + Meal Plan Prediction
# =============================================================

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import pickle
from meal_logic import (
    normalize_status,
    recommend_meal,
    show_nutrition_data
)

# =============================================================
# Initialize FastAPI App
# =============================================================
app = FastAPI(
    title="Diabetes Prediction & Meal Plan API",
    description="Predict diabetes status and generate meal plan",
    version="1.0"
)

# =============================================================
# Load Model & Artifacts
# =============================================================
print("🔄 Loading model and assets...")

# Load model
with open("best_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load food dataset
df_food = pd.read_csv("foods_prepared.csv")


print("✅ All assets loaded successfully!")


# =============================================================
# Request Body Model
# =============================================================
class UserInput(BaseModel):
    bmi: float
    age: float
    fgb: float
    avg_systolyc: float
    avg_dystolyc: float
    insulin: float


# =============================================================
# Root Endpoint
# =============================================================
@app.get("/")
def home():
    return {"message": "API is running successfully 🎉"}


# =============================================================
# Prediction Endpoint
# =============================================================
@app.post("/predict")
def predict_diabetes(data: UserInput):
    # Convert to dict
    user_data = data.dict()

    # Transform insulin → insulin_log
    insulin_log = np.log1p(user_data["insulin"])

    # Arrange final input
    X_input = np.array([[ 
        user_data["bmi"],
        user_data["age"],
        user_data["fgb"],
        user_data["avg_systolyc"],
        user_data["avg_dystolyc"],
        insulin_log
    ]])

    # Scaling
    X_scaled = scaler.transform(X_input)

    # Predict class
    pred_class = model.predict(X_scaled)[0]

    # Convert class index → label
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    return {
        "status": pred_label
    }


# =============================================================
# Meal Plan Endpoint
# =============================================================
@app.post("/meal-plan")
def get_meal_plan(data: UserInput):

    # Predict diabetes status first
    user_data = data.dict()

    insulin_log = np.log1p(user_data["insulin"])

    X_input = np.array([[ 
        user_data["bmi"],
        user_data["age"],
        user_data["fgb"],
        user_data["avg_systolyc"],
        user_data["avg_dystolyc"],
        insulin_log
    ]])

    X_scaled = scaler.transform(X_input)
    pred_class = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Convert to meal logic format
    normalized_status = normalize_status(pred_label)

    # Get meal plan
    meal_plan = recommend_meal(df_food, normalized_status)

    # With nutrition breakdown
    breakfast_info = show_nutrition_data(df_food, meal_plan["breakfast"])
    lunch_info = show_nutrition_data(df_food, meal_plan["lunch"])
    dinner_info = show_nutrition_data(df_food, meal_plan["dinner"])

    return {
        "status_predicted": normalized_status,
        "meal_plan": meal_plan,
        "nutrition": {
            "breakfast": breakfast_info,
            "lunch": lunch_info,
            "dinner": dinner_info
        }
    }


# =============================================================
# Run in Railway (or local)
# =============================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
