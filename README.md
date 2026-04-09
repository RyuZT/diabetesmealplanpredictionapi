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
Dockerfile
.dockerignore
docker-compose.yml
Caddyfile
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

## Local App Run (Without Compose)

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /api/v1/healthz`
- `GET /api/v1/readyz`
- `POST /api/v1/predict`

## Docker Compose Deployment (Recommended for VPS)

Deploy architecture:

- `app` service: FastAPI inference API (internal network only)
- `proxy` service: Caddy reverse proxy (public port 80/443)
- internal Docker network `app_net`
- Caddy forwards requests to `app:8000`
- app tidak dipublish langsung ke host

### Environment

Gunakan `.env` untuk konfigurasi app. Tambahan penting untuk proxy:

- `CADDY_SITE_ADDRESS=:80` untuk local/VPS tanpa domain
- set `CADDY_SITE_ADDRESS=api.your-domain.com` untuk domain + auto TLS Caddy

### Compose Commands

Build and start:

```bash
docker compose up -d --build
```

Stop and remove containers:

```bash
docker compose down
```

Show logs:

```bash
docker compose logs -f
```

Show logs per service:

```bash
docker compose logs -f app
docker compose logs -f proxy
```

Restart service:

```bash
docker compose restart app
docker compose restart proxy
```

Check status and health:

```bash
docker compose ps
docker inspect --format='{{json .State.Health}}' $(docker compose ps -q app)
```

## Local Runtime Test Checklist

1. Health endpoint (through proxy):

```bash
curl http://localhost/api/v1/healthz
```

Expected: `200` + `success=true`.

2. Readiness endpoint (through proxy):

```bash
curl http://localhost/api/v1/readyz
```

Expected: `200` jika artifact model + data siap.

3. Prediction endpoint (through proxy):

```bash
curl -X POST http://localhost/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"bmi":24.7,"age":43,"fgb":98,"avg_systolic":122,"avg_diastolic":80,"insulin":12}'
```

Expected: `200` + payload prediksi lengkap.

Troubleshooting cepat:

- `proxy` up, `app` unhealthy -> cek `docker compose logs -f app`
- `/healthz` ok tapi `/readyz` fail -> biasanya artifact/data path
- request dapat 502 dari proxy -> biasanya app belum ready / app crash

## Monitoring Dasar

- Gunakan `/api/v1/healthz` untuk uptime monitor eksternal (Uptime Kuma, Better Stack, dll).
- Gunakan `/api/v1/readyz` untuk memonitor kesiapan inference resource.
- Operasional harian cukup dengan:
  - `docker compose ps`
  - `docker compose logs -f app`
  - `docker compose logs -f proxy`

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

## Run Tests

```bash
pytest -q
```
