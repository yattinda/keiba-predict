"""学習共通: ターゲット変換、レース単位の時系列分割、LightGBM / CatBoost 学習。"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from column_select import categorical_for_columns, catboost_categorical_for_columns, full_feature_columns


LGB_PARAMS = {
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "random_state": 42,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "verbose": -1,
    "n_jobs": 1,
}

LGB_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "random_state": 42,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "verbose": -1,
    "label_gain": list(range(5)),
    "n_jobs": 1,
}


CATBOOST_PARAMS = {
    "iterations": 400,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
    "thread_count": 1,
}


def binary_target(finish):
    return (pd.to_numeric(finish, errors="coerce") < 4).astype(int)


def rank_target(finish):
    finish = pd.to_numeric(finish, errors="coerce")
    return np.select(
        [finish == 1, finish == 2, finish == 3, finish <= 5],
        [4, 3, 2, 1],
        default=0,
    ).astype(int)


def feature_matrix(df, include_odds=False):
    cols = full_feature_columns(df, include_odds=include_odds)
    return df[cols]


def race_time_series_splits(df, n_splits=5, date_col="日付", race_col="race_id"):
    races = (
        df[[race_col, date_col]]
        .drop_duplicates()
        .sort_values([date_col, race_col])
        .reset_index(drop=True)
    )
    race_ids = races[race_col].to_numpy()
    n_splits = min(n_splits, max(len(race_ids) - 1, 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(race_ids):
        train_races = set(race_ids[train_idx])
        val_races = set(race_ids[val_idx])
        yield (
            df[df[race_col].isin(train_races)].index,
            df[df[race_col].isin(val_races)].index,
        )


def inner_time_split(df, train_idx, val_frac=0.15, date_col="日付", race_col="race_id"):
    train_df = df.loc[train_idx]
    races = (
        train_df[[race_col, date_col]]
        .drop_duplicates()
        .sort_values([date_col, race_col])
    )
    n_val = max(1, int(len(races) * val_frac))
    if len(races) < 4:
        return train_idx, None
    val_races = set(races.tail(n_val)[race_col])
    inner_val = train_df.index[train_df[race_col].isin(val_races)]
    inner_train = train_df.index[~train_df[race_col].isin(val_races)]
    if len(inner_train) == 0 or len(inner_val) == 0:
        return train_idx, None
    return inner_train, inner_val


def _prepare_frame(X):
    X = X.copy()
    cats = categorical_for_columns(X.columns)
    for c in cats:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int).clip(lower=0)
    for c in X.columns:
        if c not in cats:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cats


def fit_lightgbm(X_train, y_train, X_val=None, y_val=None, params=None):
    X_train, cats = _prepare_frame(X_train)
    model_params = dict(LGB_PARAMS if params is None else params)
    model = lgb.LGBMClassifier(**model_params)
    fit_kwargs = {}
    if cats:
        fit_kwargs["categorical_feature"] = cats
    if X_val is not None and y_val is not None and len(X_val):
        X_val, _ = _prepare_frame(X_val)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def fit_lightgbm_regressor(X_train, y_train, X_val=None, y_val=None):
    X_train, cats = _prepare_frame(X_train)
    params = dict(LGB_PARAMS)
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {}
    if cats:
        fit_kwargs["categorical_feature"] = cats
    if X_val is not None and y_val is not None and len(X_val):
        X_val, _ = _prepare_frame(X_val)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _sort_by_race(df, X, y):
    order = df.loc[X.index].sort_values(["日付", "race_id"]).index
    return df.loc[order], X.loc[order], pd.Series(y, index=X.index).loc[order]


def fit_lightgbm_ranker(df, X_train, y_train, X_val=None, y_val=None, params=None):
    train_df, X_train, y_train = _sort_by_race(df, X_train, y_train)
    group_train = train_df.groupby("race_id", sort=False).size().to_numpy()
    X_train, cats = _prepare_frame(X_train)
    model_params = dict(LGB_RANK_PARAMS if params is None else params)
    model_params.pop("objective", None)
    model = lgb.LGBMRanker(**model_params)
    fit_kwargs = {"group": group_train}
    if cats:
        fit_kwargs["categorical_feature"] = cats
    if X_val is not None and y_val is not None and len(X_val):
        val_df, X_val, y_val = _sort_by_race(df, X_val, y_val)
        group_val = val_df.groupby("race_id", sort=False).size().to_numpy()
        X_val, _ = _prepare_frame(X_val)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [group_val]
        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def fit_catboost_binary(X_train, y_train, X_val=None, y_val=None):
    from catboost import CatBoostClassifier, Pool

    X_train, _ = _prepare_frame(X_train)
    cats = catboost_categorical_for_columns(X_train.columns)
    for c in cats:
        X_train[c] = X_train[c].astype(str)
    train_pool = Pool(X_train, y_train, cat_features=cats)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    if X_val is not None and y_val is not None and len(X_val):
        X_val, _ = _prepare_frame(X_val)
        for c in cats:
            X_val[c] = X_val[c].astype(str)
        val_pool = Pool(X_val, y_val, cat_features=cats)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    else:
        model.fit(train_pool)
    return model


def predict_proba_positive(model, X):
    X, _ = _prepare_frame(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2:
            return proba[:, 1]
        return np.asarray(proba, dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def predict_catboost(model, X):
    from catboost import Pool

    X, _ = _prepare_frame(X)
    cats = catboost_categorical_for_columns(X.columns)
    for c in cats:
        X[c] = X[c].astype(str)
    return model.predict_proba(Pool(X, cat_features=cats))[:, 1]


def oof_predict(df, feature_df, y, n_splits=5, kind="binary"):
    oof = pd.Series(np.nan, index=df.index, dtype=float)
    X = feature_df.loc[df.index]
    y_aligned = pd.Series(y, index=df.index)
    for train_idx, val_idx in race_time_series_splits(df, n_splits=n_splits):
        inner_train, inner_val = inner_time_split(df, train_idx)
        X_tr, y_tr = X.loc[inner_train], y_aligned.loc[inner_train]
        X_inner = X.loc[inner_val] if inner_val is not None else None
        y_inner = y_aligned.loc[inner_val] if inner_val is not None else None
        if kind == "binary":
            model = fit_lightgbm(X_tr, y_tr, X_inner, y_inner)
            oof.loc[val_idx] = predict_proba_positive(model, X.loc[val_idx])
        elif kind == "reg":
            model = fit_lightgbm_regressor(X_tr, y_tr, X_inner, y_inner)
            oof.loc[val_idx] = predict_proba_positive(model, X.loc[val_idx])
        elif kind == "rank":
            model = fit_lightgbm_ranker(df, X_tr, y_tr, X_inner, y_inner)
            oof.loc[val_idx] = predict_proba_positive(model, X.loc[val_idx])
        elif kind == "cat":
            model = fit_catboost_binary(X_tr, y_tr, X_inner, y_inner)
            oof.loc[val_idx] = predict_catboost(model, X.loc[val_idx])
        else:
            raise ValueError(kind)
    return oof.to_numpy()


def softmax_by_race(values, race_id):
    values = np.asarray(values, dtype=float)
    race_id = np.asarray(race_id)
    s = pd.Series(values)
    shifted = s - s.groupby(race_id, sort=False).transform("max")
    exp = np.exp(shifted.fillna(0))
    denom = pd.Series(exp).groupby(race_id, sort=False).transform("sum").replace(0, np.nan)
    return (exp / denom).to_numpy()


def zscore_by_race(values, race_id):
    s = pd.Series(np.asarray(values, dtype=float))
    g = s.groupby(np.asarray(race_id), sort=False)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).to_numpy()


def rank_by_race(values, race_id, ascending=False):
    s = pd.Series(np.asarray(values, dtype=float))
    return (
        s.groupby(np.asarray(race_id), sort=False)
        .rank(method="average", ascending=ascending)
        .to_numpy()
    )
