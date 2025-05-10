"""
Utility functions (lag creation, metric name cleaning, etc.)
"""

"""
Utility functions
"""

import re
import pandas as pd

def clean_metric_name(name: str) -> str:
    """
    Sanitize station names into MLflow‑safe metric names.
    """
    # allow alphanumerics, underscores, hyphens, periods, colons, slashes, spaces
    return re.sub(r"[^\w\-\.:/ ]", "_", name)

def clean_model_name(name: str) -> str:
    """
    Sanitize station names into MLflow-safe model names.
    Removes non-alphanumeric characters and replaces them with underscores.
    """
    return re.sub(r"[^\w]+", "_", name).strip("_")

def make_lags(df, n_lags=28):
    """
    Generate lag features for each column in a wide-format dataframe.
    Returns (X, y) where X is the lagged input and y is the target (t).
    """
    X, y = [], []
    for i in range(n_lags, len(df)):
        X.append(df.iloc[i - n_lags:i].values.flatten())
        y.append(df.iloc[i].values)
    X = pd.DataFrame(X, columns=[f"{col}_lag{lag+1}" for lag in range(n_lags) for col in df.columns])
    y = pd.DataFrame(y, columns=df.columns)
    return X, y

