from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone


class SeparateTargetRegressor(BaseEstimator):
    """Fit independent estimators for RV change and IV change."""

    def __init__(self, rv_model, iv_model):
        self.rv_model = rv_model
        self.iv_model = iv_model

    def fit(self, x, y):
        values = np.asarray(y, float)
        self.rv_model_ = clone(self.rv_model).fit(x, values[:, 0])
        self.iv_model_ = clone(self.iv_model).fit(x, values[:, 1])
        return self

    def predict(self, x):
        return np.column_stack([self.rv_model_.predict(x), self.iv_model_.predict(x)])
