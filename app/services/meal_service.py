import pandas as pd

from app.schemas.predict import MealPlan, NutritionInfo, NutritionSummary

STATUS_RULES = {
    "Non-Diabetic": {
        "max_sugar": 20,
        "max_fat": 20,
        "min_protein": 5,
        "max_carbs": 60,
        "max_chol": 300,
        "min_fibre": 2,
        "cal_min": 150,
        "cal_max": 600,
    },
    "Prediabetic": {
        "max_sugar": 10,
        "max_fat": 15,
        "min_protein": 8,
        "max_carbs": 45,
        "max_chol": 250,
        "min_fibre": 3,
        "cal_min": 150,
        "cal_max": 500,
    },
    "Diabetic": {
        "max_sugar": 5,
        "max_fat": 10,
        "min_protein": 10,
        "max_carbs": 35,
        "max_chol": 200,
        "min_fibre": 4,
        "cal_min": 120,
        "cal_max": 450,
    },
}


def normalize_status(label: str) -> str:
    text = label.lower().replace("-", "").replace(" ", "")
    if text in {"nondiabetes", "nondiabetic", "normal"}:
        return "Non-Diabetic"
    if text in {"prediabetes", "prediabetic"}:
        return "Prediabetic"
    if text in {"diabetes", "diabetic"}:
        return "Diabetic"
    return "Non-Diabetic"


def _filter_food(df: pd.DataFrame, rules: dict[str, float]) -> pd.DataFrame:
    return df[
        (df["freesugar_g"] <= rules["max_sugar"])
        & (df["fat_g"] <= rules["max_fat"])
        & (df["protein_g"] >= rules["min_protein"])
        & (df["carbs"] <= rules["max_carbs"])
        & (df["cholestrol_mg"] <= rules["max_chol"])
        & (df["fibre_g"] >= rules["min_fibre"])
        & (df["energy_kcal"] >= rules["cal_min"])
        & (df["energy_kcal"] <= rules["cal_max"])
    ]


def recommend_meal_plan(df_food: pd.DataFrame, status: str) -> MealPlan | None:
    rules = STATUS_RULES.get(status, STATUS_RULES["Non-Diabetic"])
    filtered = _filter_food(df_food, rules)

    if len(filtered) < 3:
        return None

    picks = filtered.sample(n=3, replace=False)
    items = picks["fooditems"].tolist()

    return MealPlan(breakfast=items[0], lunch=items[1], dinner=items[2])


def nutrition_info(df_food: pd.DataFrame, food_name: str) -> NutritionInfo | None:
    row = df_food[df_food["fooditems"].str.lower() == food_name.lower()]
    if row.empty:
        return None

    record = row.iloc[0]
    return NutritionInfo(
        energy_kcal=float(record["energy_kcal"]),
        carbs=float(record["carbs"]),
        protein_g=float(record["protein_g"]),
        fat_g=float(record["fat_g"]),
        freesugar_g=float(record["freesugar_g"]),
        fibre_g=float(record["fibre_g"]),
        cholestrol_mg=float(record["cholestrol_mg"]),
        calcium_mg=float(record["calcium_mg"]),
    )


def build_nutrition_summary(
    df_food: pd.DataFrame, meal_plan: MealPlan
) -> NutritionSummary | None:
    breakfast = nutrition_info(df_food, meal_plan.breakfast)
    lunch = nutrition_info(df_food, meal_plan.lunch)
    dinner = nutrition_info(df_food, meal_plan.dinner)

    if not breakfast or not lunch or not dinner:
        return None

    return NutritionSummary(breakfast=breakfast, lunch=lunch, dinner=dinner)
