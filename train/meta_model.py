"""Stage2 メタモデル。"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_utils import (
    binary_target,
    inner_time_split,
    rank_by_race,
    softmax_by_race,
    zscore_by_race,
)


BASE_PRED_COLS = ["lgb_binary", "lgb_rank", "lgb_speed", "cat_binary", "race_nn"]
CONTEXT_COLS = ["頭数", "クラス", "距離帯", "馬場", "距離"]


def add_race_normalized(meta, race_id, cols):
    race_id = np.asarray(race_id)
    out = meta.copy()
    for col in cols:
        if col not in out.columns:
            continue
        values = out[col].to_numpy()
        out[f"{col}_softmax"] = softmax_by_race(values, race_id)
        out[f"{col}_z"] = zscore_by_race(values, race_id)
        out[f"{col}_rank"] = rank_by_race(values, race_id)
        grouped_max = (
            pd.Series(values).groupby(race_id, sort=False).transform("max").to_numpy()
        )
        out[f"{col}_gap"] = values - grouped_max
    return out


def build_meta_frame(df, pred_df):
    meta = pred_df.copy()
    for col in CONTEXT_COLS:
        if col in df.columns:
            meta[col] = pd.to_numeric(df[col], errors="coerce").to_numpy()
    pred_cols = [c for c in BASE_PRED_COLS if c in meta.columns]
    meta = add_race_normalized(meta, df["race_id"], pred_cols)
    return meta


def _lgb_meta():
    return lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.02,
        n_estimators=400,
        min_data_in_leaf=50,
        random_state=42,
        verbose=-1,
    )


def _lr_meta():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


class MetaModel:
    def __init__(self):
        self.lgb = None
        self.lr = None
        self.chosen = "lgb"
        self.feature_cols_ = None

    def fit(self, df, pred_df):
        meta = build_meta_frame(df, pred_df)
        y = pd.Series(binary_target(df["着順"]), index=df.index)
        inner_train, inner_val = inner_time_split(df, df.index)
        X = meta.fillna(0)
        self.feature_cols_ = list(X.columns)
        self.lgb = _lgb_meta()
        self.lr = _lr_meta()
        self.lgb.fit(X.loc[inner_train], y.loc[inner_train])
        self.lr.fit(X.loc[inner_train], y.loc[inner_train])
        if inner_val is not None and len(inner_val):
            from sklearn.metrics import log_loss

            p_lgb = self.lgb.predict_proba(X.loc[inner_val])[:, 1]
            p_lr = self.lr.predict_proba(X.loc[inner_val])[:, 1]
            yv = y.loc[inner_val]
            ll_lgb = log_loss(yv, np.clip(p_lgb, 1e-6, 1 - 1e-6), labels=[0, 1])
            ll_lr = log_loss(yv, np.clip(p_lr, 1e-6, 1 - 1e-6), labels=[0, 1])
            self.chosen = "lgb" if ll_lgb <= ll_lr else "lr"
        else:
            self.chosen = "lgb"
        model = self.lgb if self.chosen == "lgb" else self.lr
        model.fit(X, y)
        return self

    def _raw(self, df, pred_df):
        meta = build_meta_frame(df, pred_df).fillna(0)
        for col in self.feature_cols_:
            if col not in meta.columns:
                meta[col] = 0
        X = meta[self.feature_cols_]
        model = self.lgb if self.chosen == "lgb" else self.lr
        return model.predict_proba(X)[:, 1]

    def predict(self, df, pred_df):
        raw = self._raw(df, pred_df)
        race_id = df["race_id"].to_numpy()
        win = softmax_by_race(raw, race_id)
        if "頭数" in df.columns:
            field = pd.to_numeric(df["頭数"], errors="coerce").to_numpy()
        else:
            field = df.groupby("race_id")["race_id"].transform("size").to_numpy()
        field = np.where(np.isnan(field), 3, field)
        top3 = win * np.clip(field, 3, None)
        share = pd.Series(top3).groupby(race_id).transform("sum").replace(0, np.nan)
        top3 = np.clip(top3 * (3.0 / share), 0, 1)
        return pd.DataFrame({"p_win": win, "p_top3": top3, "p_raw": raw}, index=df.index)
