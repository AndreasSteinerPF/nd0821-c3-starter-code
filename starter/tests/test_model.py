from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model


def test_process_data_training_encodes_features_and_labels() -> None:
    data = pd.DataFrame(
        {
            "age": [39, 50, 28],
            "workclass": ["State-gov", "Private", "Private"],
            "education": ["Bachelors", "Masters", "HS-grad"],
            "salary": ["<=50K", ">50K", "<=50K"],
        }
    )

    X, y, encoder, lb = process_data(
        data,
        categorical_features=["workclass", "education"],
        label="salary",
        training=True,
    )

    assert X.shape[0] == 3
    assert y.tolist() == [0, 1, 0]
    assert encoder is not None
    assert lb is not None


def test_compute_model_metrics_returns_expected_scores() -> None:
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_train_model_and_inference_fit_a_classifier() -> None:
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 2],
        ]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert preds.shape == y_train.shape
    assert set(np.unique(preds)).issubset({0, 1})
    assert (preds == y_train).mean() >= 0.83
