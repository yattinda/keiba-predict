"""Stage1 top3: LightGBM / CatBoost の二値2本のみ。"""

import pandas as pd
from catboost import CatBoostClassifier, Pool

from train_utils import (
    CATBOOST_PARAMS,
    LGB_PARAMS,
    _prepare_frame,
    binary_target,
    feature_matrix,
    fit_catboost_binary,
    fit_lightgbm,
    inner_time_split,
    predict_catboost,
    predict_proba_positive,
)

TOP3_BASE_COLS = ["lgb_binary", "cat_binary"]


def _best_n(model, default=200):
    for attr in ("best_iteration_", "best_iteration"):
        n = getattr(model, attr, None)
        if n is not None and int(n) > 0:
            return int(n)
    return default


class Top3FundamentalModels:
    def __init__(self, include_odds=False):
        self.include_odds = include_odds
        self.models = {}

    def fit(self, df):
        X = feature_matrix(df, include_odds=self.include_odds)
        y = binary_target(df["着順"])
        inner_train, inner_val = inner_time_split(df, df.index)
        X_tr = X.loc[inner_train]
        X_va = X.loc[inner_val] if inner_val is not None else None
        y_va = y.loc[inner_val] if inner_val is not None else None

        probe_bin = fit_lightgbm(X_tr, y.loc[inner_train], X_va, y_va)
        probe_cat = fit_catboost_binary(X_tr, y.loc[inner_train], X_va, y_va)

        bin_params = dict(LGB_PARAMS)
        bin_params["n_estimators"] = _best_n(probe_bin, LGB_PARAMS["n_estimators"])
        self.models["lgb_binary"] = fit_lightgbm(X, y, params=bin_params)

        from column_select import catboost_categorical_for_columns

        n_cat = _best_n(probe_cat, CATBOOST_PARAMS["iterations"])
        Xp, _ = _prepare_frame(X)
        cats = catboost_categorical_for_columns(Xp.columns)
        for c in cats:
            Xp[c] = Xp[c].astype(str)
        params = dict(CATBOOST_PARAMS)
        params["iterations"] = max(n_cat, 50)
        params.pop("early_stopping_rounds", None)
        params.pop("eval_metric", None)
        cat = CatBoostClassifier(**params)
        cat.fit(Pool(Xp, y, cat_features=cats))
        self.models["cat_binary"] = cat
        return self

    def predict_frame(self, df):
        X = feature_matrix(df, include_odds=self.include_odds)
        out = pd.DataFrame(index=df.index)
        out["lgb_binary"] = predict_proba_positive(self.models["lgb_binary"], X)
        out["cat_binary"] = predict_catboost(self.models["cat_binary"], X)
        return out