# Diabetes Meal Plan Inference API

FastAPI backend untuk ML inference diabetes status dan rekomendasi meal plan, dengan kontrak JSON yang stabil dan client-agnostic untuk Android, web, Flutter, Python, atau service backend lain.

## Architecture Summary

Arsitektur dipisah menjadi layer yang jelas:

- `api`: HTTP routes (tipis)
- `schemas`: contract request/response Pydantic
- `services`: orchestration inference flow
- `ml`: model loader, preprocessing, postprocessing
- `core`: settings, constants, error codes, exceptions, handlers, response builder, logging

## Project Structure

```text
app/
  main.py
  api/
    router.py
    routes/
      health.py
      predict.py
  schemas/
    common.py
    predict.py
  services/
    inference_service.py
    meal_service.py
  ml/
    model_loader.py
    preprocessing.py
    postprocessing.py
  core/
    settings.py
    constants.py
    error_codes.py
    exceptions.py
    exception_handlers.py
    responses.py
    logging.py
  models/
    best_xgb.pkl
    scaler.pkl
    label_encoder.pkl
  data/
    foods_prepared.csv
tests/
  test_health.py
  test_predict.py
```

## Setup

- Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
```

## Run Local

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /api/v1/healthz`
- `GET /api/v1/readyz`
- `POST /api/v1/predict`

## Health vs Ready

- `healthz`: memastikan service process hidup.
- `readyz`: memastikan resource penting siap (`model`, `scaler`, `label_encoder`, `food_data`).

## Request Example

```json
{
  "bmi": 24.7,
  "age": 43,
  "fgb": 98,
  "avg_systolic": 122,
  "avg_diastolic": 80,
  "insulin": 12
}
```

## Success Response Example

```json
{
  "success": true,
  "data": {
    "prediction": "Non-Diabetic",
    "normalized_prediction": "Non-Diabetic",
    "confidence": 0.91,
    "top_predictions": [
      {
        "label": "Non-Diabetic",
        "probability": 0.91
      },
      {
        "label": "Prediabetic",
        "probability": 0.07
      }
    ],
    "meal_plan": [
      {
        "meal_type": "breakfast",
        "food_name": "Dhokla",
        "nutrition": {
          "energy_kcal": 216.49,
          "carbs": 30.68,
          "protein_g": 13.45,
          "fat_g": 5.28,
          "freesugar_g": 4.78,
          "fibre_g": 4.95,
          "cholestrol_mg": 5.16,
          "calcium_mg": 123.21
        }
      },
      {
        "meal_type": "lunch",
        "food_name": "Paneerstuffedcheela/chilla",
        "nutrition": null
      },
      {
        "meal_type": "dinner",
        "food_name": "Edamame Boiled",
        "nutrition": null
      }
    ],
    "warnings": [],
    "metadata": {
      "model_version": "best_xgb.pkl",
      "inference_timestamp": "2026-03-28T08:10:31Z"
    }
  },
  "request_id": "e4aa9a4e-ef21-4ea6-b97f-8e37691d8ed8"
}
```

`meal_plan` selalu berupa array. Jika rekomendasi tidak tersedia, nilainya `[]` dan warning code akan diisi.

## Warning Codes (Predict Response)

- `meal_plan_unavailable`
- `nutrition_unavailable`
- `low_confidence`

## Error Response Example

```json
{
  "success": false,
  "error": {
    "code": "validation_error",
    "message": "input is invalid",
    "details": [
      {
        "field": "bmi",
        "message": "Input should be greater than 0",
        "type": "greater_than"
      }
    ]
  },
  "request_id": "a1c3125f-11ff-4cdc-8f7f-f7e9189a5c11"
}
```

## Main Error Codes

- `validation_error`
- `bad_request`
- `model_not_loaded`
- `prediction_failed`
- `resource_not_ready`
- `internal_server_error`
- `http_error`

## Request ID / Correlation

- Setiap request diproses dengan `request_id`.
- Jika client mengirim header `X-Request-ID`, nilai itu dipakai.
- Jika tidak, server generate UUID otomatis.
- `request_id` dimasukkan ke response body error/success dan response header `X-Request-ID`.

## Update Model Artifacts

Simpan artifact baru di:

- `app/models/best_xgb.pkl`
- `app/models/scaler.pkl`
- `app/models/label_encoder.pkl`

Jika format input model berubah, update:

- `app/ml/preprocessing.py` (mapping dan urutan feature)
- `app/core/constants.py` (`MODEL_FEATURES`)

## Run Tests

```bash
pytest -q
```

## Docker

Build:

```bash
docker build -t diabetes-ml-api .
```

Run:

```bash
docker run --rm -p 8000:8000 diabetes-ml-api
```
