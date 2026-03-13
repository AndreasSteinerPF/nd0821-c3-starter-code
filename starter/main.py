from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import (
    CAT_FEATURES,
    ENCODER_PATH,
    LB_PATH,
    MODEL_PATH,
    ensure_artifacts,
)

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent


def pull_dvc_data_on_dyno_startup() -> None:
    if "DYNO" in os.environ and (REPO_DIR / ".dvc").is_dir():
        subprocess.run(
            ["dvc", "config", "core.no_scm", "true"],
            cwd=REPO_DIR,
            check=True,
        )
        if subprocess.run(["dvc", "pull"], cwd=REPO_DIR, check=False).returncode != 0:
            raise RuntimeError("dvc pull failed")
        shutil.rmtree(REPO_DIR / ".dvc", ignore_errors=True)
        shutil.rmtree(REPO_DIR / ".apt" / "usr" / "lib" / "dvc", ignore_errors=True)


pull_dvc_data_on_dyno_startup()
app = FastAPI(title="Census Income Prediction API")


class CensusRecord(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 31,
                "workclass": "Private",
                "fnlgt": 45781,
                "education": "Masters",
                "education-num": 14,
                "marital-status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital-gain": 14084,
                "capital-loss": 0,
                "hours-per-week": 50,
                "native-country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class PredictionResponse(BaseModel):
    prediction: str


@lru_cache(maxsize=1)
def load_artifacts():
    ensure_artifacts()
    model = load(MODEL_PATH)
    encoder = load(ENCODER_PATH)
    lb = load(LB_PATH)
    return model, encoder, lb


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the census income prediction API."}


@app.post("/predict", response_model=PredictionResponse)
def predict_salary(record: CensusRecord) -> PredictionResponse:
    model, encoder, lb = load_artifacts()
    record_frame = pd.DataFrame([record.model_dump(by_alias=True)])
    processed_data, _, _, _ = process_data(
        record_frame,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    prediction = inference(model, processed_data)
    label = lb.inverse_transform(prediction)[0]
    return PredictionResponse(prediction=str(label))
