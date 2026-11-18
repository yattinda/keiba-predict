"""Stage2 top3 メタ: ベース確率のロジットを非負重みで束ねる（レース softmax なし）。"""

import numpy as np
import pandas as pd

from horse_logit import apply_horse_bernoulli, clip_proba, fit_horse_bernoulli, proba_logit
from train_utils import binary_target, inner_time_split

TOP3_BASE_PRED_COLS = ["lgb_binary", "cat_binary"]


def build_logit_frame(pred_df):
    out = pd.DataFrame(index=pred_df.index)
    for col in TOP3_BASE_PRED_COLS:
        if col not in pred_df.columns:
            continue
        values = pd.to_numeric(pred_df[col], errors="coerce").to_numpy()
        out[col] = proba_logit(values)
    return out


def _bernoulli_nll(p, y):
    p = clip_proba(p)
    y = np.asarray(y, dtype=int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Top3MetaModel:
    def __init__(self):
        self.model = None
        self.chosen = "stack"
        self.feature_cols_ = None
        self.base_only_col_ = None

    def _features(self, pred_df):
        return build_logit_frame(pred_df), None

    def fit(self, df, pred_df):
        finish = df["着順"]
        y = binary_target(finish).to_numpy()
        features, _ = self._features(pred_df)
        cols = [c for c in TOP3_BASE_PRED_COLS if c in features.columns]
        if not cols:
            raise ValueError("top3 メタ: ベース予測列がありません")
        X = features[cols].fillna(0.0).to_numpy()
        n_feat = X.shape[1]
        bounds = [(0.0, None)] * n_feat

        inner_train, inner_val = inner_time_split(df, df.index)
        if inner_val is not None and len(inner_val):
            val_mask = df.index.isin(inner_val)
            val_X = X[val_mask]
            val_y = y[val_mask]
            stack_model, stack_p = fit_horse_bernoulli(
                val_X,
                val_y,
                x0=np.ones(n_feat),
                bounds=bounds,
                l2_target=np.ones(n_feat),
                l2_coef=1e-3,
            )
            stack_nll = _bernoulli_nll(stack_p, val_y)
            best_nll = stack_nll
            best_choice = "stack"
            for name in cols:
                raw = pd.to_numeric(pred_df.loc[inner_val, name], errors="coerce").to_numpy()
                solo_nll = _bernoulli_nll(raw, val_y)
                if solo_nll < best_nll:
                    best_nll = solo_nll
                    best_choice = f"base:{name}"

            self.chosen = best_choice
            if best_choice.startswith("base:"):
                self.base_only_col_ = best_choice.split(":", 1)[1]
                w = np.zeros(n_feat)
                w[cols.index(self.base_only_col_)] = 1.0
                self.model = fit_horse_bernoulli(
                    X,
                    y,
                    x0=w,
                    bounds=bounds,
                    l2_target=w,
                    l2_coef=0.0,
                )[0]
            else:
                self.model, _ = fit_horse_bernoulli(
                    X,
                    y,
                    x0=np.ones(n_feat),
                    bounds=bounds,
                    l2_target=np.ones(n_feat),
                    l2_coef=1e-3,
                )
        else:
            self.chosen = "stack"
            self.model, _ = fit_horse_bernoulli(
                X,
                y,
                x0=np.ones(n_feat),
                bounds=bounds,
                l2_target=np.ones(n_feat),
                l2_coef=1e-3,
            )
        self.feature_cols_ = cols
        return self

    def predict(self, df, pred_df):
        features, _ = self._features(pred_df)
        for col in self.feature_cols_:
            if col not in features.columns:
                features[col] = 0.0
        X = features[self.feature_cols_].fillna(0.0).to_numpy()
        if self.base_only_col_ and self.chosen.startswith("base:"):
            raw = pd.to_numeric(pred_df[self.base_only_col_], errors="coerce").to_numpy()
            p = clip_proba(raw)
            return pd.DataFrame({"p_top3": p, "p_raw": p}, index=df.index)
        p, _ = apply_horse_bernoulli(self.model, X)
        return pd.DataFrame({"p_top3": p, "p_raw": p}, index=df.index)