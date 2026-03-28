import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent


class Settings(BaseSettings):
    app_name: str = "Diabetes Meal Plan Inference API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Stored as raw text to support CSV in .env (e.g. a,b,c) and JSON list string.
    allowed_origins: str = "http://localhost,http://localhost:3000"
    cors_allow_credentials: bool = False

    model_dir: Path = APP_DIR / "models"
    data_dir: Path = APP_DIR / "data"

    model_filename: str = "best_xgb.pkl"
    scaler_filename: str = "scaler.pkl"
    label_encoder_filename: str = "label_encoder.pkl"
    food_data_filename: str = "foods_prepared.csv"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "debug"}:
            return True
        if text in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False

        return False

    @staticmethod
    def _parse_origins(raw_value: str) -> list[str]:
        text = raw_value.strip()
        if not text:
            return []

        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass

        return [item.strip() for item in text.split(",") if item.strip()]

    @property
    def allowed_origins_list(self) -> list[str]:
        return self._parse_origins(self.allowed_origins)

    @model_validator(mode="after")
    def validate_cors_config(self) -> "Settings":
        if self.cors_allow_credentials and "*" in self.allowed_origins_list:
            raise ValueError(
                "allow_credentials=true is not valid when allowed_origins contains '*'."
            )
        return self

    @model_validator(mode="after")
    def resolve_dirs(self) -> "Settings":
        if not self.model_dir.is_absolute():
            self.model_dir = (PROJECT_ROOT / self.model_dir).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
