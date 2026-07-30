"""
salary_utils.py
===============
Custom transformers and helpers used by the salary-prediction pipeline.

This module is imported by the training notebook AND must be importable wherever
``salary_predictor.pkl`` is loaded, because pickle stores classes and functions by
reference.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# YearsCode parsing
# ---------------------------------------------------------------------------
def parse_years_value(value):
    """Convert a single raw ``YearsCode`` entry into a float.

    Rules
    -----
    * ``"Less than 1 year"``  -> 0.5
    * ``"More than 50 years"`` -> 50.0
    * numeric strings          -> float
    * anything else / missing  -> NaN (imputed later in the pipeline)
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.nan if pd.isna(value) else float(value)

    text = str(value).strip().lower()
    if text in ("", "nan", "none", "na"):
        return np.nan
    if text.startswith("less than"):
        return 0.5
    if text.startswith("more than"):
        return 50.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_years_code(X):
    """Vectorised version of :func:`parse_years_value` for a DataFrame/array column."""
    frame = pd.DataFrame(X).copy()
    for column in frame.columns:
        frame[column] = frame[column].map(parse_years_value)
    return frame.astype(float)


# ---------------------------------------------------------------------------
# Multi-hot encoder for ";"-separated multi-label columns
# ---------------------------------------------------------------------------
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """One binary column per individual label of a ``;``-separated text column.

    This is the multi-label generalisation of one-hot encoding. Labels seen at
    prediction time but not during ``fit`` are ignored, and missing values are
    treated as an empty label set (all output columns equal 0).

    Parameters
    ----------
    separator : str
        Character separating the labels inside one cell.
    min_frequency : float or int
        Labels rarer than this are dropped. A value < 1 is read as a fraction of
        the training rows, a value >= 1 as an absolute row count.
    prefix : str
        Prefix used when building output feature names.
    """

    def __init__(self, separator=";", min_frequency=0.002, prefix="label"):
        self.separator = separator
        self.min_frequency = min_frequency
        self.prefix = prefix

    # -- helpers ------------------------------------------------------------
    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        array = np.asarray(X)
        if array.ndim == 2:
            array = array[:, 0]
        return pd.Series(array)

    def _split(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return set()
        text = str(value)
        if text.strip().lower() in ("", "nan", "none"):
            return set()
        return {part.strip() for part in text.split(self.separator) if part.strip()}

    # -- scikit-learn API ---------------------------------------------------
    def fit(self, X, y=None):
        series = self._to_series(X)
        counts = Counter()
        for value in series:
            counts.update(self._split(value))

        n_rows = max(len(series), 1)
        if self.min_frequency < 1:
            threshold = max(1, int(self.min_frequency * n_rows))
        else:
            threshold = int(self.min_frequency)

        self.classes_ = sorted(label for label, count in counts.items() if count >= threshold)
        self.class_to_index_ = {label: i for i, label in enumerate(self.classes_)}
        self.n_features_out_ = len(self.classes_)
        return self

    def transform(self, X):
        series = self._to_series(X)
        out = np.zeros((len(series), self.n_features_out_), dtype=np.float64)
        for row, value in enumerate(series):
            for label in self._split(value):
                index = self.class_to_index_.get(label)
                if index is not None:
                    out[row, index] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.prefix}__{label}" for label in self.classes_], dtype=object)


# ---------------------------------------------------------------------------
# Version-tolerant OneHotEncoder factory
# ---------------------------------------------------------------------------
def make_one_hot_encoder(min_frequency=50):
    """Return a dense OneHotEncoder that tolerates unseen categories.

    Falls back gracefully on scikit-learn versions that do not support the
    ``sparse_output`` / ``min_frequency`` arguments.
    """
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
        )
    except TypeError:
        pass
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
