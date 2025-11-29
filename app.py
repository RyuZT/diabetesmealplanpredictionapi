from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

from meal_logic import (
    normalize_status,
    recommend_meal,
    nutrition_info
)

app = FastAPI(title="Diabetes Meal Plan API")

# =======================================================
# LOAD MODEL + SCALER + LABEL ENCODER + FOOD DATA
# =======================================================
model = joblib.load("best_xgb.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

df_food = pd.read_csv("foods_prepared.csv", delimiter=";")

numeric_cols = ["energy_kcal","carbs","protein_g","fat_g","freesugar_g",
                "fibre_g","cholestrol_mg","calcium_mg"]

for col in numeric_cols:
    df_food[col] = pd.to_numeric(df_food[col], errors="coerce").fillna(0)

# =======================================================
# INPUT FEATURES
# =======================================================
model_features = ["bmi","age","fgb","avg_systolyc","avg_dystolyc","insulin_log"]

@app.post("/predict")
def predict(data: dict):
    try:
        # Ambil nilai
        bmi = float(data["bmi"])
        age = float(data["age"])
        fgb = float(data["fgb"])
        sys = float(data["avg_systolyc"])
        dys = float(data["avg_dystolyc"])
        insulin = float(data["insulin"])

        # Transform insulin
        insulin_log = np.log1p(insulin)

        # Buat dataframe
        df = pd.DataFrame([{
            "bmi": bmi,
            "age": age,
            "fgb": fgb,
            "avg_systolyc": sys,
            "avg_dystolyc": dys,
            "insulin_log": insulin_log
        }], columns=model_features)

        df_scaled = scaler.transform(df)

        # Prediksi
        prob = model.predict_proba(df_scaled)[0]
        pred_class = np.argmax(prob)
        label = le.inverse_transform([pred_class])[0]
        confidence = float(prob[pred_class])

        # Normalisasi label
        norm_label = normalize_status(label)

        # Buat meal plan
        meal = recommend_meal(df_food, norm_label)

        if meal is None:
            return {"error": "Dataset makanan tidak cukup untuk status ini"}

        # Nutrisi
        breakfast_info = nutrition_info(df_food, meal["breakfast"])
        lunch_info = nutrition_info(df_food, meal["lunch"])
        dinner_info = nutrition_info(df_food, meal["dinner"])

        return {
            "prediction": label,
            "normalized_prediction": norm_label,
            "confidence": confidence,
            "meal_plan": {
                "breakfast": meal["breakfast"],
                "lunch": meal["lunch"],
                "dinner": meal["dinner"]
            },
            "nutrition": {
                "breakfast": breakfast_info,
                "lunch": lunch_info,
                "dinner": dinner_info
            }
        }

    except Exception as e:
        return {"error": str(e)}
