"""Stage3 top3: 複勝オッズがあれば混合。診断用の均等少額精算。"""

import itertools

import numpy as np
import pandas as pd

from horse_logit import apply_horse_bernoulli, clip_proba, fit_horse_bernoulli, proba_logit
from market_roi import roi_summary, split_calibration_races

PLACE_ODDS_COL = "複勝オッズ"
MIN_FIELD_FOR_PLACE = 8
DIAGNOSTIC_UNIT_STAKE = 0.001


def place_market_probs(odds):
    raw = pd.to_numeric(odds, errors="coerce")
    return (1.0 / np.clip(np.asarray(raw, dtype=float), 1.01, None)).astype(float)


def _field_size(df):
    if "頭数" in df.columns:
        return pd.to_numeric(df["頭数"], errors="coerce").to_numpy()
    return df.groupby("race_id")["馬番"].transform("size").to_numpy()


def fit_place_blend(p_model, p_market, y_top3):
    eps = 1e-6
    pm = clip_proba(p_model)
    pk = clip_proba(p_market)
    x = np.column_stack([proba_logit(pm), proba_logit(pk)])
    bounds = [(0.0, None), (0.0, None)]
    model, _ = fit_horse_bernoulli(
        x,
        y_top3,
        x0=np.array([1.0, 1.0]),
        bounds=bounds,
        l2_target=np.array([1.0, 1.0]),
        l2_coef=1e-3,
    )
    return model


def apply_place_blend(model, p_model, p_market):
    eps = 1e-6
    pm = clip_proba(p_model)
    pk = clip_proba(p_market)
    x = np.column_stack([proba_logit(pm), proba_logit(pk)])
    return apply_horse_bernoulli(model, x)[0], model.coef_.ravel()


def _place_payout_map(payouts):
    if payouts is None or len(payouts) == 0:
        return {}
    df = payouts.copy()
    df["券種"] = df["券種"].astype(str)
    place = df[df["券種"].str.contains("複勝", na=False)]
    mapping = {}
    for _, row in place.iterrows():
        combo = str(row.get("組番", "")).split()[0]
        try:
            umaban = int(float(combo))
        except ValueError:
            continue
        payout = pd.to_numeric(
            str(row.get("払戻金", "")).replace(",", "").split()[0],
            errors="coerce",
        )
        if pd.isna(payout):
            continue
        mapping[(row["race_id"], umaban)] = float(payout) / 100.0
    return mapping


def select_place_diagnostic_bets(p, race_id, df, prob_threshold=0.25, unit_stake=DIAGNOSTIC_UNIT_STAKE):
    p = np.asarray(p, dtype=float)
    race_id = np.asarray(race_id)
    field = _field_size(df)
    stake = np.zeros(len(p), dtype=float)
    candidates = (p >= prob_threshold) & (field >= MIN_FIELD_FOR_PLACE)
    stake[candidates] = unit_stake
    return p, stake


def settle_place_bets(df, stake, payouts=None):
    finish = pd.to_numeric(df["着順"], errors="coerce").to_numpy()
    umaban = pd.to_numeric(df["馬番"], errors="coerce").to_numpy()
    race_id = df["race_id"].to_numpy()
    payout_map = _place_payout_map(payouts)
    profit = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        if stake[i] <= 0:
            continue
        if finish[i] > 3 or np.isnan(finish[i]):
            profit[i] = -stake[i]
            continue
        key = (race_id[i], int(umaban[i]) if not np.isnan(umaban[i]) else -1)
        if key not in payout_map:
            profit[i] = -stake[i]
            continue
        profit[i] = stake[i] * (payout_map[key] - 1.0)
    return pd.DataFrame(
        {
            "race_id": race_id,
            "馬番": umaban,
            "stake": stake,
            "profit": profit,
            "hit": (finish <= 3) & (stake > 0),
        }
    )


def pick_place_threshold(market, calib_tail, p_model, min_bets=20, fallback=0.25):
    p_final, _ = market.predict(calib_tail, p_model)
    grid = [0.15, 0.20, 0.25, 0.30, 0.35]
    best = fallback
    best_roi = float("-inf")
    for thresh in grid:
        market.prob_threshold = thresh
        _, stake = select_place_diagnostic_bets(
            p_final, calib_tail["race_id"], calib_tail, prob_threshold=thresh
        )
        bets = settle_place_bets(calib_tail, stake, payouts=None)
        live = bets[bets["stake"] > 0]
        if len(live) < min_bets:
            continue
        roi = float(live["profit"].sum() / live["stake"].sum()) if live["stake"].sum() else float("nan")
        if roi > best_roi:
            best_roi = roi
            best = thresh
    fallback_flag = int(best == fallback and best_roi == float("-inf"))
    return best, fallback_flag


def top3_triple_heuristics(df, p, top_n=5):
    p = np.asarray(p, dtype=float)
    rows = []
    for rid, g in df.groupby("race_id", sort=False):
        idx = g.index.to_numpy()
        order = idx[np.argsort(-p[idx])][:top_n]
        if len(order) < 3:
            continue
        for combo in itertools.combinations(order, 3):
            umaban = [int(pd.to_numeric(df.loc[i, "馬番"], errors="coerce")) for i in combo]
            score = float(np.prod([p[i] for i in combo]))
            rows.append(
                {
                    "race_id": rid,
                    "馬番1": umaban[0],
                    "馬番2": umaban[1],
                    "馬番3": umaban[2],
                    "heuristic_score": score,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["race_id", "馬番1", "馬番2", "馬番3", "heuristic_score"])
    out = pd.DataFrame(rows)
    out = out.sort_values(["race_id", "heuristic_score"], ascending=[True, False])
    return out.groupby("race_id", sort=False).head(1).reset_index(drop=True)


class Top3MarketIntegrator:
    def __init__(self, prob_threshold=0.25):
        self.prob_threshold = prob_threshold
        self.blend_model = None
        self.weights_ = None
        self.use_market = False

    def _has_place_odds(self, df):
        return PLACE_ODDS_COL in df.columns and pd.to_numeric(df[PLACE_ODDS_COL], errors="coerce").notna().any()

    def fit(self, df, p_model, y_top3):
        if not self._has_place_odds(df):
            self.use_market = False
            self.blend_model = None
            self.weights_ = None
            return self
        p_mkt = place_market_probs(df[PLACE_ODDS_COL])
        self.blend_model = fit_place_blend(p_model, p_mkt, y_top3)
        self.weights_ = self.blend_model.coef_.ravel()
        self.use_market = True
        return self

    def predict(self, df, p_model):
        if not self.use_market or not self._has_place_odds(df):
            p = clip_proba(p_model)
            return p, None
        p_mkt = place_market_probs(df[PLACE_ODDS_COL])
        p_final, weights = apply_place_blend(self.blend_model, p_model, p_mkt)
        self.weights_ = weights
        return p_final, p_mkt

    def diagnostic_bets(self, df, p_top3, payouts=None):
        _, stake = select_place_diagnostic_bets(
            p_top3,
            df["race_id"],
            df,
            prob_threshold=self.prob_threshold,
        )
        bets = settle_place_bets(df, stake, payouts=payouts)
        return bets