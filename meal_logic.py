import pandas as pd

# ============================
# NORMALISASI STATUS
# ============================
def normalize_status(label):
    t = label.lower().replace("-", "").replace(" ", "")
    if t in ["nondiabetes", "nondiabetic", "normal"]:
        return "Non-Diabetic"
    if t in ["prediabetes", "prediabetic"]:
        return "Prediabetic"
    if t in ["diabetes", "diabetic"]:
        return "Diabetic"
    return "Non-Diabetic"

# ============================
# RULE BERDASARKAN STATUS
# ============================
def get_rules(status):
    if status == "Non-Diabetic":
        return {"max_sugar":20,"max_fat":20,"min_protein":5,
                "max_carbs":60,"max_chol":300,"min_fibre":2,
                "cal_min":150,"cal_max":600}

    if status == "Prediabetic":
        return {"max_sugar":10,"max_fat":15,"min_protein":8,
                "max_carbs":45,"max_chol":250,"min_fibre":3,
                "cal_min":150,"cal_max":500}

    return {"max_sugar":5,"max_fat":10,"min_protein":10,
            "max_carbs":35,"max_chol":200,"min_fibre":4,
            "cal_min":120,"cal_max":450}

# ============================
# FILTER MAKANAN
# ============================
def filter_food(df, rules):
    return df[
        (df["freesugar_g"] <= rules["max_sugar"]) &
        (df["fat_g"] <= rules["max_fat"]) &
        (df["protein_g"] >= rules["min_protein"]) &
        (df["carbs"] <= rules["max_carbs"]) &
        (df["cholestrol_mg"] <= rules["max_chol"]) &
        (df["fibre_g"] >= rules["min_fibre"]) &
        (df["energy_kcal"] >= rules["cal_min"]) &
        (df["energy_kcal"] <= rules["cal_max"])
    ]

# ============================
# BUAT REKOMENDASI
# ============================
def recommend_meal(df, status):
    rules = get_rules(status)
    filtered = filter_food(df, rules)

    if len(filtered) < 3:
        return None

    return {
        "breakfast": filtered.sample(1).iloc[0]["fooditems"],
        "lunch": filtered.sample(1).iloc[0]["fooditems"],
        "dinner": filtered.sample(1).iloc[0]["fooditems"]
    }

# ============================
# DETAIL NUTRISI
# ============================
def nutrition_info(df, food):
    row = df[df["fooditems"].str.lower() == food.lower()]
    if row.empty:
        return None
    
    r = row.iloc[0]
    return {
        "energy_kcal": r["energy_kcal"],
        "carbs": r["carbs"],
        "protein_g": r["protein_g"],
        "fat_g": r["fat_g"],
        "freesugar_g": r["freesugar_g"],
        "fibre_g": r["fibre_g"],
        "cholestrol_mg": r["cholestrol_mg"],
        "calcium_mg": r["calcium_mg"],
    }
