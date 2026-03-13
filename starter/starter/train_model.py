from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"

RAW_DATA_PATH = DATA_DIR / "census.csv"
CLEAN_DATA_PATH = DATA_DIR / "census_clean.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "encoder.joblib"
LB_PATH = MODEL_DIR / "label_binarizer.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SLICE_OUTPUT_PATH = MODEL_DIR / "slice_output.txt"
SAMPLE_REQUESTS_PATH = MODEL_DIR / "sample_requests.json"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def clean_census_data(data: pd.DataFrame) -> pd.DataFrame:
    """Strip the inconsistent whitespace from column names and string values."""
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.strip()
    object_columns = cleaned.select_dtypes(include="object").columns
    for column in object_columns:
        cleaned[column] = cleaned[column].str.strip()
    return cleaned


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_clean_data(
    raw_path: Path = RAW_DATA_PATH,
    cleaned_path: Path = CLEAN_DATA_PATH,
) -> pd.DataFrame:
    data = clean_census_data(load_raw_data(raw_path))
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cleaned_path, index=False)
    return data


def train_test_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[LABEL],
    )


def choose_sample_requests(
    model,
    data: pd.DataFrame,
    encoder,
    lb,
) -> dict[str, dict[str, int | str]]:
    features = data.drop(columns=[LABEL])
    X_data, _, _, _ = process_data(
        features,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    predictions = inference(model, X_data)
    prediction_labels = lb.inverse_transform(predictions)

    sample_requests: dict[str, dict[str, int | str]] = {}
    for target in ["<=50K", ">50K"]:
        sample_index = features.index[prediction_labels == target][0]
        sample_requests[target] = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in features.loc[sample_index].to_dict().items()
        }
    return sample_requests


def train_and_save_model() -> dict[str, float]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_clean_data()
    train, test = train_test_data(data)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    dump(model, MODEL_PATH)
    dump(encoder, ENCODER_PATH)
    dump(lb, LB_PATH)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "fbeta": float(fbeta),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    slice_metrics = compute_slice_metrics(
        model,
        test,
        CAT_FEATURES,
        encoder,
        lb,
        LABEL,
    )
    slice_lines = [
        "feature,value,count,precision,recall,fbeta",
        *[
            (
                f"{row['feature']},{row['value']},{row['count']},"
                f"{row['precision']:.4f},{row['recall']:.4f},{row['fbeta']:.4f}"
            )
            for row in slice_metrics
        ],
    ]
    SLICE_OUTPUT_PATH.write_text("\n".join(slice_lines) + "\n", encoding="utf-8")

    sample_requests = choose_sample_requests(model, test, encoder, lb)
    SAMPLE_REQUESTS_PATH.write_text(
        json.dumps(sample_requests, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def artifacts_exist() -> bool:
    required_paths = [
        CLEAN_DATA_PATH,
        MODEL_PATH,
        ENCODER_PATH,
        LB_PATH,
        METRICS_PATH,
        SLICE_OUTPUT_PATH,
        SAMPLE_REQUESTS_PATH,
    ]
    return all(path.exists() for path in required_paths)


def ensure_artifacts() -> dict[str, float]:
    if artifacts_exist():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train_and_save_model()


if __name__ == "__main__":
    metrics = train_and_save_model()
    print(json.dumps(metrics, indent=2))
