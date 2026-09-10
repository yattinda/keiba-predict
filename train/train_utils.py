"""学習共通: ターゲット変換、レース単位の時系列分割、LightGBM 学習。"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from column_select import categorical_for_columns


LGB_PARAMS = {
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "random_state": 42,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "verbose": -1,
}


def binary_target(finish):
    return (pd.to_numeric(finish, errors="coerce") < 4).astype(int)


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


def _prepare_frame(X):
    X = X.copy()
    cats = categorical_for_columns(X.columns)
    for c in cats:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int).clip(lower=0)
    return X, cats


def fit_lightgbm(X_train, y_train, X_val=None, y_val=None):
    X_train, cats = _prepare_frame(X_train)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    fit_kwargs = {}
    if cats:
        fit_kwargs["categorical_feature"] = cats
    if X_val is not None and y_val is not None:
        X_val, _ = _prepare_frame(X_val)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_proba_positive(model, X):
    X, _ = _prepare_frame(X)
    return model.predict_proba(X)[:, 1]


def oof_predict(df, feature_df, y, n_splits=5):
    oof = pd.Series(np.nan, index=df.index, dtype=float)
    X = feature_df.loc[df.index]
    y_aligned = y.loc[df.index]
    for train_idx, val_idx in race_time_series_splits(df, n_splits=n_splits):
        model = fit_lightgbm(
            X.loc[train_idx],
            y_aligned.loc[train_idx],
            X.loc[val_idx],
            y_aligned.loc[val_idx],
        )
        oof.loc[val_idx] = predict_proba_positive(model, X.loc[val_idx])
    return oof.to_numpy()
