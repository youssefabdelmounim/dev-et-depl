"""FastAPI service exposing the EduAccess model.

Run from the project root (any working directory works):

    uvicorn api.main:app --reload

Endpoints
---------
* ``GET  /``                          service description
* ``GET  /health``                    liveness + model status
* ``GET  /model/info``                registered model metadata
* ``GET  /metrics``                   evaluation metrics of the compared models
* ``GET  /indicators``                dataset indicators for the dashboard
* ``GET  /figures/{name}```           evaluation figures (whitelisted)
* ``GET  /students/sample``           candidate students for a test prediction
* ``POST /predict``                   predict from a feature dictionary
* ``POST /predict/student/{id}```     predict by looking up a real student
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    INDICATORS_FILE,
    KEY_COLUMNS,
    METRICS_FILE,
    MODEL_BUNDLE,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.features import build_and_cache
from src.indicators import build_indicators


class PredictRequest(BaseModel):
    """Feature dictionary; missing fields are imputed like during training."""

    features: Dict[str, Any] = Field(
        default_factory=dict,
        examples=[
            {
                "code_module": "BBB",
                "code_presentation": "2014J",
                "gender": "M",
                "region": "Scotland",
                "highest_education": "A Level or Equivalent",
                "imd_band": "20-30%",
                "age_band": "35-55",
                "disability": "N",
                "vle_clicks_30d": 42,
                "vle_active_days_30d": 8,
            }
        ],
    )


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 = at risk, 0 = success")
    risk_label: str = Field(..., description="'at_risk' or 'success'")
    probability_at_risk: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="'high', 'medium' or 'low'")
    model_name: str
    provided_features: int
    total_features: int
    imputed_features: List[str]


class _State:
    bundle: Optional[Dict[str, Any]] = None
    bundle_error: Optional[str] = None
    features: Optional[pd.DataFrame] = None
    features_error: Optional[str] = None


state = _State()


def load_artifacts() -> None:
    """Load the model bundle and the cached feature table (best effort)."""
    if MODEL_BUNDLE.exists():
        try:
            state.bundle = joblib.load(MODEL_BUNDLE)
            state.bundle_error = None
        except Exception as exc:  # pragma: no cover - corrupt artifact
            state.bundle_error = f"cannot load model bundle: {exc}"
    else:
        state.bundle_error = (
            f"{MODEL_BUNDLE.name} not found - run `python -m src.train` first"
        )

    if state.features is None:
        if ARTIFACTS_DIR.joinpath("features.csv").exists():
            try:
                state.features = build_and_cache(force=False)
                state.features_error = None
            except Exception as exc:  # pragma: no cover
                state.features_error = f"cannot load features cache: {exc}"
        else:
            state.features_error = (
                "features.csv not found - run `python -m src.features` first"
            )


def _risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "high"
    if probability >= 0.33:
        return "medium"
    return "low"


def _to_dataframe(features: Dict[str, Any]) -> tuple[pd.DataFrame, List[str]]:
    """Build an ordered single-row frame; absent/NaN columns are imputed later."""
    provided: Dict[str, Any] = {}
    for key, value in features.items():
        if key not in FEATURE_COLUMNS or value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        provided[key] = value
    row: Dict[str, Any] = {c: provided.get(c, np.nan) for c in FEATURE_COLUMNS}
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in frame.columns:
        if col not in NUMERIC_FEATURES and not pd.isna(frame[col]).all():
            frame[col] = frame[col].astype(str)
    imputed = [c for c in FEATURE_COLUMNS if c not in provided]
    return frame, imputed


def run_prediction(frame: pd.DataFrame) -> Dict[str, Any]:
    bundle = state.bundle
    assert bundle is not None
    pipe = bundle["pipeline"]
    proba = float(pipe.predict_proba(frame)[0, 1])
    prediction = int(proba >= 0.5)
    return {
        "prediction": prediction,
        "risk_label": "at_risk" if prediction == 1 else "success",
        "probability_at_risk": round(proba, 4),
        "risk_level": _risk_level(proba),
        "model_name": bundle["model_name"],
        "provided_features": int(len(FEATURE_COLUMNS)),
        "total_features": len(FEATURE_COLUMNS),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="EduAccess API",
    description=(
        "Early detection of learning difficulties (OULAD dataset). "
        "Decision-support tool: it estimates the probability that a student "
        "profile shows signs of academic difficulty during the first weeks "
        "of a presentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "project": "EduAccess - détection précoce des difficultés d'apprentissage",
        "docs": "/docs",
        "endpoints": [
            "/health", "/model/info", "/metrics", "/indicators",
            "/figures/{name}", "/students/sample", "/predict",
            "/predict/student/{id}",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": state.bundle is not None,
        "model_error": state.bundle_error,
        "features_loaded": state.features is not None,
        "features_error": state.features_error,
    }


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    if state.bundle is None:
        raise HTTPException(status_code=503, detail=state.bundle_error)
    bundle = state.bundle
    return {
        "model_name": bundle["model_name"],
        "trained_at": bundle["trained_at"],
        "n_features": len(bundle["feature_columns"]),
        "feature_columns": bundle["feature_columns"],
        "metrics": bundle["metrics"],
        "all_metrics": bundle["all_metrics"],
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="metrics.json not found - run `python -m src.train` first",
        )
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


@app.get("/indicators")
def indicators() -> Dict[str, Any]:
    if state.features is None:
        raise HTTPException(status_code=503, detail=state.features_error)
    if INDICATORS_FILE.exists() and state.bundle is not None:
        cached = json.loads(INDICATORS_FILE.read_text(encoding="utf-8"))
        if cached.get("model"):
            return cached
    fresh = build_indicators(state.features)
    INDICATORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDICATORS_FILE.write_text(
        json.dumps(fresh, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return fresh


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    allowed = {p.name for p in FIGURES_DIR.glob("*.png")} if FIGURES_DIR.exists() else set()
    if name not in allowed:
        raise HTTPException(status_code=404, detail=f"unknown figure '{name}'")
    return FileResponse(FIGURES_DIR / name, media_type="image/png")


@app.get("/students/sample")
def students_sample(
    n: int = Query(20, ge=1, le=200),
    module: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Random real students (with their actual outcome) for UI testing."""
    if state.features is None:
        raise HTTPException(status_code=503, detail=state.features_error)
    frame = state.features
    if module:
        frame = frame[frame["code_module"] == module]
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"unknown module '{module}'")
    sample = frame.sample(n=min(n, len(frame)), random_state=seed)
    columns = KEY_COLUMNS + ["gender", "region", "highest_education",
                             "final_result", TARGET_COLUMN]
    records = json.loads(sample[columns].to_json(orient="records"))
    return {"count": len(records), "students": records}


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    if state.bundle is None:
        raise HTTPException(status_code=503, detail=state.bundle_error)
    if not request.features:
        raise HTTPException(status_code=422, detail="'features' must not be empty")
    frame, imputed = _to_dataframe(request.features)
    result = run_prediction(frame)
    result["provided_features"] = len(FEATURE_COLUMNS) - len(imputed)
    result["imputed_features"] = imputed
    return Prediction(**result)


@app.post("/predict/student/{id_student}", response_model=Prediction)
def predict_student(
    id_student: int,
    code_module: str = Query(..., description="e.g. BBB"),
    code_presentation: str = Query(..., description="e.g. 2014J"),
) -> Prediction:
    if state.bundle is None:
        raise HTTPException(status_code=503, detail=state.bundle_error)
    if state.features is None:
        raise HTTPException(status_code=503, detail=state.features_error)
    match = state.features[
        (state.features["id_student"] == id_student)
        & (state.features["code_module"] == code_module)
        & (state.features["code_presentation"] == code_presentation)
    ]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail="student not found for this module/presentation",
        )
    row = match.iloc[0]
    frame, imputed = _to_dataframe({c: row[c] for c in FEATURE_COLUMNS})
    result = run_prediction(frame)
    result["provided_features"] = len(FEATURE_COLUMNS) - len(imputed)
    result["imputed_features"] = imputed
    return Prediction(**result)
