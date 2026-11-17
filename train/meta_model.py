"""Stage2 メタモデル: ベース予測シェアの条件付きロジット。"""

import numpy as np
import pandas as pd

from race_logit import apply_race_softmax, fit_race_softmax
from train_utils import inner_time_split, softmax_by_race


BASE_PRED_COLS = ["lgb_binary", "lgb_rank", "lgb_speed", "cat_binary", "race_nn"]


def build_log_share_frame(pred_df, race_id):
    race_id = np.asarray(race_id)
    out = pd.DataFrame(index=pred_df.index)
    for col in BASE_PRED_COLS:
        if col not in pred_df.columns:
            continue
        values = pd.to_numeric(pred_df[col], errors="coerce").to_numpy()
        share = softmax_by_race(values, race_id)
        out[col] = np.log(np.clip(share, 1e-6, 1.0))
    return out


def _winner_nll(p, race_id, finish):
    finish = pd.to_numeric(finish, errors="coerce")
    y_win = (finish == 1).astype(int).to_numpy()
    win_idx = np.where(y_win == 1)[0]
    if len(win_idx) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-np.mean(np.log(p[win_idx])))


class MetaModel:
    def __init__(self):
        self.model = None
        self.chosen = "stack"
        self.feature_cols_ = None
        self.base_only_col_ = None

    def _features(self, pred_df, race_id):
        frame = build_log_share_frame(pred_df, race_id)
        return frame, list(frame.columns)

    def fit(self, df, pred_df):
        race_id = df["race_id"].to_numpy()
        finish = df["着順"]
        features, cols = self._features(pred_df, race_id)
        if not cols:
            raise ValueError("メタモデル: ベース予測列がありません")
        X = features[cols].fillna(np.log(1e-6)).to_numpy()
        y_win = (pd.to_numeric(finish, errors="coerce") == 1).astype(int).to_numpy()

        n_feat = X.shape[1]
        bounds = [(0.0, None)] * n_feat
        inner_train, inner_val = inner_time_split(df, df.index)
        if inner_val is not None and len(inner_val):
            val_mask = df.index.isin(inner_val)
            val_race = race_id[val_mask]
            val_finish = finish.loc[inner_val]
            val_X = X[val_mask]

            stack_model, stack_p = fit_race_softmax(
                val_X,
                y_win[val_mask],
                val_race,
                x0=np.ones(n_feat),
                bounds=bounds,
                l2_target=np.ones(n_feat),
                l2_coef=1e-3,
            )
            stack_nll = _winner_nll(stack_p, val_race, val_finish)

            best_nll = stack_nll
            best_choice = "stack"
            for j, name in enumerate(cols):
                raw = pd.to_numeric(pred_df.loc[inner_val, name], errors="coerce").to_numpy()
                share = softmax_by_race(raw, val_race)
                solo_nll = _winner_nll(share, val_race, val_finish)
                if solo_nll < best_nll:
                    best_nll = solo_nll
                    best_choice = f"base:{name}"

            self.chosen = best_choice
            if best_choice.startswith("base:"):
                self.base_only_col_ = best_choice.split(":", 1)[1]
                w = np.zeros(n_feat)
                w[cols.index(self.base_only_col_)] = 1.0
                self.model = fit_race_softmax(
                    X,
                    y_win,
                    race_id,
                    x0=w,
                    bounds=bounds,
                    l2_target=w,
                    l2_coef=0.0,
                )[0]
            else:
                self.model, _ = fit_race_softmax(
                    X,
                    y_win,
                    race_id,
                    x0=np.ones(n_feat),
                    bounds=bounds,
                    l2_target=np.ones(n_feat),
                    l2_coef=1e-3,
                )
        else:
            self.chosen = "stack"
            self.model, _ = fit_race_softmax(
                X,
                y_win,
                race_id,
                x0=np.ones(n_feat),
                bounds=bounds,
                l2_target=np.ones(n_feat),
                l2_coef=1e-3,
            )
        self.feature_cols_ = cols
        return self

    def predict(self, df, pred_df):
        race_id = df["race_id"].to_numpy()
        features, cols = self._features(pred_df, race_id)
        for col in self.feature_cols_:
            if col not in features.columns:
                features[col] = np.log(1e-6)
        X = features[self.feature_cols_].fillna(np.log(1e-6)).to_numpy()
        if self.base_only_col_ and self.chosen.startswith("base:"):
            j = self.feature_cols_.index(self.base_only_col_)
            share = softmax_by_race(X[:, j], race_id)
            return pd.DataFrame({"p_win": share, "p_raw": share}, index=df.index)
        p_win, _ = apply_race_softmax(self.model, X, race_id)
        return pd.DataFrame({"p_win": p_win, "p_raw": p_win}, index=df.index)