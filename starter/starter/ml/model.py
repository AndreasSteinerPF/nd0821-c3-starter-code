from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=4,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    model,
    data: pd.DataFrame,
    categorical_features: list[str],
    encoder,
    lb,
    label: str,
) -> list[dict[str, float | str | int]]:
    """Compute model metrics over each categorical feature slice."""
    results: list[dict[str, float | str | int]] = []
    for feature in categorical_features:
        for value in sorted(data[feature].unique()):
            sliced_data = data[data[feature] == value]
            X_slice, y_slice, _, _ = process_data(
                sliced_data,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            results.append(
                {
                    "feature": feature,
                    "value": value,
                    "count": int(len(sliced_data)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "fbeta": float(fbeta),
                }
            )
    return results
