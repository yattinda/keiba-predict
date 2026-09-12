"""Stage1 ファンダメンタルモデル群（市場情報なし・全特徴）。"""

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier, Pool

from train_utils import (
    CATBOOST_PARAMS,
    LGB_PARAMS,
    LGB_RANK_PARAMS,
    _prepare_frame,
    binary_target,
    feature_matrix,
    fit_catboost_binary,
    fit_lightgbm,
    fit_lightgbm_ranker,
    fit_lightgbm_regressor,
    inner_time_split,
    predict_catboost,
    predict_proba_positive,
    rank_target,
)


BASE_MODEL_NAMES = ["lgb_binary", "lgb_rank", "lgb_speed", "cat_binary"]


def _best_n(model, default=200):
    for attr in ("best_iteration_", "best_iteration"):
        n = getattr(model, attr, None)
        if n is not None and int(n) > 0:
            return int(n)
    return default


class FundamentalModels:
    def __init__(self, include_odds=False):
        self.include_odds = include_odds
        self.models = {}

    def _xy(self, df):
        X = feature_matrix(df, include_odds=self.include_odds)
        y_bin = pd.Series(binary_target(df["着順"]), index=df.index)
        y_rank = pd.Series(rank_target(df["着順"]), index=df.index)
        speed_col = "スピード指数" if "スピード指数" in df.columns else "走破時間"
        y_speed = pd.to_numeric(df[speed_col], errors="coerce")
        return X, y_bin, y_rank, y_speed

    def fit(self, df):
        X, y_bin, y_rank, y_speed = self._xy(df)
        inner_train, inner_val = inner_time_split(df, df.index)
        X_tr = X.loc[inner_train]
        X_va = X.loc[inner_val] if inner_val is not None else None
        y_va_bin = y_bin.loc[inner_val] if inner_val is not None else None
        y_va_rank = y_rank.loc[inner_val] if inner_val is not None else None

        probe_bin = fit_lightgbm(X_tr, y_bin.loc[inner_train], X_va, y_va_bin)
        probe_rank = fit_lightgbm_ranker(
            df, X_tr, y_rank.loc[inner_train], X_va, y_va_rank
        )
        speed_ok = y_speed.loc[inner_train].notna()
        va_speed = None
        y_va_speed = None
        if inner_val is not None:
            va_ok = y_speed.loc[inner_val].notna()
            va_speed = X_va.loc[va_ok]
            y_va_speed = y_speed.loc[inner_val].loc[va_ok]
        probe_speed = fit_lightgbm_regressor(
            X_tr.loc[speed_ok],
            y_speed.loc[inner_train].loc[speed_ok],
            va_speed,
            y_va_speed,
        )
        probe_cat = fit_catboost_binary(
            X_tr, y_bin.loc[inner_train], X_va, y_va_bin
        )

        bin_params = dict(LGB_PARAMS)
        bin_params["n_estimators"] = _best_n(probe_bin, LGB_PARAMS["n_estimators"])
        self.models["lgb_binary"] = fit_lightgbm(X, y_bin, params=bin_params)

        rank_params = dict(LGB_RANK_PARAMS)
        rank_params["n_estimators"] = _best_n(probe_rank, LGB_RANK_PARAMS["n_estimators"])
        self.models["lgb_rank"] = fit_lightgbm_ranker(df, X, y_rank, params=rank_params)

        n_reg = _best_n(probe_speed, LGB_PARAMS["n_estimators"])
        ok = y_speed.notna()
        X_ok, cats = _prepare_frame(X.loc[ok])
        self.models["lgb_speed"] = lgb.LGBMRegressor(
            **{**LGB_PARAMS, "n_estimators": n_reg}
        )
        fit_kwargs = {}
        if cats:
            fit_kwargs["categorical_feature"] = cats
        self.models["lgb_speed"].fit(X_ok, y_speed.loc[ok], **fit_kwargs)

        n_cat = _best_n(probe_cat, CATBOOST_PARAMS["iterations"])
        Xp, _ = _prepare_frame(X)
        from column_select import catboost_categorical_for_columns

        cats = catboost_categorical_for_columns(Xp.columns)
        for c in cats:
            Xp[c] = Xp[c].astype(str)
        params = dict(CATBOOST_PARAMS)
        params["iterations"] = max(n_cat, 50)
        params.pop("early_stopping_rounds", None)
        params.pop("eval_metric", None)
        cat = CatBoostClassifier(**params)
        cat.fit(Pool(Xp, y_bin, cat_features=cats))
        self.models["cat_binary"] = cat
        return self

    def predict_frame(self, df):
        X = feature_matrix(df, include_odds=self.include_odds)
        out = pd.DataFrame(index=df.index)
        out["lgb_binary"] = predict_proba_positive(self.models["lgb_binary"], X)
        out["lgb_rank"] = predict_proba_positive(self.models["lgb_rank"], X)
        out["lgb_speed"] = predict_proba_positive(self.models["lgb_speed"], X)
        out["cat_binary"] = predict_catboost(self.models["cat_binary"], X)
        return out
