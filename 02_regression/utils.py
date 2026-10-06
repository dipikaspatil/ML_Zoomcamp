"""Helper functions for ML Zoomcamp, chapter 2 (regression)."""

import numpy as np


def fit_fill_values(df_train, features, strategy="zero"):
    """Learn one fill value per feature column, from the TRAINING data only.

    strategy="zero": every column is filled with 0
    strategy="mean": every column is filled with its training mean

    Columns without missing values are harmless: their fill value is never used.
    Returns a dict {column_name: fill_value}.
    """
    missing = [c for c in features if c not in df_train.columns]
    if missing:
        raise KeyError(f"Columns not found in the data: {missing}")

    if strategy == "zero":
        return {c: 0.0 for c in features}

    if strategy == "mean":
        fill_values = df_train[features].mean().to_dict()
        bad = [c for c, v in fill_values.items() if np.isnan(v)]
        if bad:
            raise ValueError(f"Cannot compute a mean for all-missing columns: {bad}")
        return fill_values

    raise ValueError("strategy must be 'zero' or 'mean'")


def prepare_X(df, features, fill_values):
    """Select the feature columns, fill missing values per column, return a NumPy array.

    fill_values is a dict from fit_fill_values (one value per column).
    Raises an error if any missing value is left, so a silent NaN can never reach training.
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in the data: {missing}")

    not_covered = [c for c in features if c not in fill_values]
    if not_covered:
        raise KeyError(f"No fill value defined for columns: {not_covered}")

    X = df[features].fillna(value=fill_values).values.astype(float)

    if not np.isfinite(X).all():
        raise ValueError("X still contains NaN or infinite values after filling")

    return X


def train_linear_regression(X, y):
    """Linear regression without regularization (normal equation).

    Adds the column of ones for the bias internally.
    Returns (w0, w): the bias and the feature weights.
    """
    if np.isnan(y).any():
        raise ValueError("y contains missing values")

    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def predict(X, w0, w):
    """Predictions of a trained linear regression model."""
    return w0 + X.dot(w)


def rmse(y, y_pred):
    """Root mean squared error."""
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


def train_linear_regression_reg(X, y, r=0.0):
    """Linear regression with L2 regularization (ridge), normal equation.

    r is added to the diagonal of X^T X (the bias term included, as in the lesson).
    r=0 gives the same weights as train_linear_regression.
    Returns (w0, w).
    """
    if np.isnan(y).any():
        raise ValueError("y contains missing values")

    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]