from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class ModelSpec:
    estimator: object
    parameter_space: dict
    feature_scope: str = "all"


def har_features(all_features: list[str]) -> list[str]:
    fixed = {
        "days_to_expiry", "entry_iv_call", "entry_iv_put", "entry_log_moneyness",
        "entry_straddle_price", "entry_spot_price", "entry_future_price",
        "surface_atm_iv_call", "surface_atm_iv_put", "surface_skew_call", "surface_skew_put",
        "surface_smile_call", "surface_smile_put", "entry_time_sin", "entry_time_cos", "entry_day_of_week",
    }
    return [column for column in all_features if column in fixed or column.startswith(("lag_", "change_"))]


def model_specs(seed: int, n_jobs: int) -> dict[str, ModelSpec]:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return {
        "har_ridge": ModelSpec(
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "har",
        ),
        "elastic_net": ModelSpec(
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
                      ("model", ElasticNet(max_iter=4000, tol=1e-3, random_state=seed))]),
            {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2], "model__l1_ratio": [0.1, 0.5, 0.9]}, "har",
        ),
        "hist_gradient_boosting": ModelSpec(
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", HistGradientBoostingRegressor(loss="absolute_error", early_stopping=True, random_state=seed))]),
            {"model__learning_rate": [0.025, 0.05, 0.1], "model__max_iter": [150, 300],
             "model__max_leaf_nodes": [15, 31, 63], "model__l2_regularization": [0.1, 1.0, 10.0]},
        ),
        "random_forest": ModelSpec(
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", RandomForestRegressor(n_jobs=n_jobs, random_state=seed))]),
            {"model__n_estimators": [100, 200], "model__max_depth": [10, 16, None],
             "model__min_samples_leaf": [10, 30, 60], "model__max_features": [0.4, 0.7]},
        ),
        "extra_trees": ModelSpec(
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", ExtraTreesRegressor(n_jobs=n_jobs, random_state=seed))]),
            {"model__n_estimators": [100, 200], "model__max_depth": [10, 16, None],
             "model__min_samples_leaf": [5, 15, 40], "model__max_features": [0.4, 0.7]},
        ),
        "xgboost": ModelSpec(
            XGBRegressor(objective="reg:pseudohubererror", tree_method="hist", n_jobs=n_jobs, random_state=seed),
            {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.025, 0.05, 0.1],
             "subsample": [0.7, 0.9], "colsample_bytree": [0.5, 0.8], "reg_lambda": [1.0, 10.0]},
        ),
        "lightgbm": ModelSpec(
            LGBMRegressor(objective="huber", n_jobs=n_jobs, random_state=seed, verbosity=-1),
            {"n_estimators": [200, 400], "num_leaves": [15, 31, 63], "learning_rate": [0.025, 0.05, 0.1],
             "min_child_samples": [20, 60, 120], "subsample": [0.7, 0.9], "colsample_bytree": [0.5, 0.8],
             "reg_lambda": [1.0, 10.0]},
        ),
        "catboost": ModelSpec(
            CatBoostRegressor(loss_function="MAE", verbose=False, allow_writing_files=False,
                              thread_count=n_jobs, random_seed=seed),
            {"iterations": [200, 400], "depth": [4, 6, 8], "learning_rate": [0.025, 0.05, 0.1],
             "l2_leaf_reg": [3.0, 10.0, 30.0], "random_strength": [0.5, 1.0]},
        ),
    }


class ResidualHybrid:
    """HAR Ridge level plus a LightGBM nonlinear residual correction."""

    def __init__(self, base_estimator, residual_estimator, har_columns: list[str], all_columns: list[str]):
        self.base_estimator = base_estimator
        self.residual_estimator = residual_estimator
        self.har_columns = har_columns
        self.all_columns = all_columns

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.base_estimator.fit(X[self.har_columns], y)
        residual = y.to_numpy() - self.base_estimator.predict(X[self.har_columns])
        self.residual_estimator.fit(X[self.all_columns], residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_estimator.predict(X[self.har_columns]) + self.residual_estimator.predict(X[self.all_columns])

