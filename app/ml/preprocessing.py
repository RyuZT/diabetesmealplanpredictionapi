import numpy as np
import pandas as pd

from app.core.constants import MODEL_FEATURES
from app.schemas.predict import PredictRequest


def build_feature_dataframe(payload: PredictRequest) -> pd.DataFrame:
    # Public API fields stay typo-free, while internal model columns follow trained artifacts.
    feature_row = {
        "bmi": payload.bmi,
        "age": payload.age,
        "fgb": payload.fgb,
        "avg_systolyc": payload.avg_systolic,
        "avg_dystolyc": payload.avg_diastolic,
        "insulin_log": float(np.log1p(payload.insulin)),
    }
    return pd.DataFrame([feature_row], columns=list(MODEL_FEATURES))
