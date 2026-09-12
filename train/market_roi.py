"""Stage3 市場統合と ROI バックテスト。"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from train_utils import softmax_by_race


def market_probs(odds, race_id):
    inv = 1.0 / np.clip(pd.to_numeric(odds, errors="coerce").to_numpy(), 1.01, None)
    grouped = pd.Series(inv).groupby(np.asarray(race_id), sort=False)
    return (pd.Series(inv) / grouped.transform("sum")).to_numpy()


class _LinearSoftmax:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)

    def decision_function(self, x):
        w = self.coef_.ravel()
        return np.asarray(x, dtype=float) @ w


def _fit_race_softmax(features, y_win, race_id, x0=None):
    features = np.asarray(features, dtype=float)
    y_win = np.asarray(y_win).astype(int)
    race_id = np.asarray(race_id)
    win_idx = np.where(y_win == 1)[0]
    if len(win_idx) == 0:
        weights = np.ones(features.shape[1], dtype=float) if x0 is None else np.asarray(x0)
        model = _LinearSoftmax(weights)
        return model, softmax_by_race(features @ model.coef_.ravel(), race_id)
    if x0 is None:
        x0 = np.ones(features.shape[1], dtype=float)

    def loss(w):
        raw = features @ w
        p = softmax_by_race(raw, race_id)
        return -np.mean(np.log(np.clip(p[win_idx], 1e-12, 1)))

    result = minimize(loss, x0=x0, method="L-BFGS-B")
    weights = result.x
    raw = features @ weights
    return _LinearSoftmax(weights), softmax_by_race(raw, race_id)


def fit_longshot_bias(p_market, y_win, race_id):
    x = np.log(np.clip(p_market, 1e-6, 1)).reshape(-1, 1)
    return _fit_race_softmax(x, y_win, race_id, x0=np.array([1.0]))


def apply_longshot_bias(model, p_market, race_id):
    x = np.log(np.clip(p_market, 1e-6, 1)).reshape(-1, 1)
    raw = model.decision_function(x)
    return softmax_by_race(raw, race_id)


def fit_conditional_logit(p_model, p_market, y_win, race_id):
    eps = 1e-6
    x = np.column_stack(
        [
            np.log(np.clip(p_model, eps, 1)),
            np.log(np.clip(p_market, eps, 1)),
        ]
    )
    return _fit_race_softmax(x, y_win, race_id, x0=np.array([1.0, 1.0]))


def apply_conditional_logit(model, p_model, p_market, race_id):
    eps = 1e-6
    x = np.column_stack(
        [
            np.log(np.clip(p_model, eps, 1)),
            np.log(np.clip(p_market, eps, 1)),
        ]
    )
    raw = model.decision_function(x)
    return softmax_by_race(raw, race_id), model.coef_.ravel()


def harville_ordered(p_win, order):
    remaining = np.asarray(p_win, dtype=float).copy()
    remaining = remaining / remaining.sum()
    prob = 1.0
    for idx in order:
        total = remaining.sum()
        if total <= 0:
            return 0.0
        prob *= remaining[idx] / total
        remaining[idx] = 0.0
    return float(prob)


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def expected_value(p, odds):
    return p * pd.to_numeric(odds, errors="coerce").to_numpy() - 1.0


def fractional_kelly(p, odds, fraction=0.25):
    b = np.clip(pd.to_numeric(odds, errors="coerce").to_numpy() - 1.0, 1e-6, None)
    q = 1.0 - p
    f = (p * b - q) / b
    return np.clip(fraction * f, 0.0, 0.05)


def select_bets(p, odds, ev_threshold=0.05, kelly_fraction=0.25):
    ev = expected_value(p, odds)
    stake = fractional_kelly(p, odds, fraction=kelly_fraction)
    take = ev > ev_threshold
    stake = np.where(take, stake, 0.0)
    return ev, stake


def _payout_map(payouts):
    if payouts is None or len(payouts) == 0:
        return {}
    df = payouts.copy()
    df["券種"] = df["券種"].astype(str)
    win = df[df["券種"].str.contains("単勝", na=False)]
    mapping = {}
    for _, row in win.iterrows():
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


def settle_win_bets(df, p, stake, payouts=None):
    odds = pd.to_numeric(df["オッズ"], errors="coerce").to_numpy()
    finish = pd.to_numeric(df["着順"], errors="coerce").to_numpy()
    umaban = pd.to_numeric(df["馬番"], errors="coerce").to_numpy()
    race_id = df["race_id"].to_numpy()
    payout_map = _payout_map(payouts)
    profit = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        if stake[i] <= 0:
            continue
        if finish[i] != 1:
            profit[i] = -stake[i]
            continue
        key = (race_id[i], int(umaban[i]) if not np.isnan(umaban[i]) else -1)
        if key in payout_map:
            profit[i] = stake[i] * (payout_map[key] - 1.0)
        else:
            profit[i] = stake[i] * (odds[i] - 1.0)
    bets = pd.DataFrame(
        {
            "race_id": race_id,
            "馬番": umaban,
            "p": p,
            "odds": odds,
            "stake": stake,
            "profit": profit,
            "win": finish == 1,
        }
    )
    return bets


def roi_summary(bets):
    live = bets[bets["stake"] > 0]
    staked = float(live["stake"].sum())
    profit = float(live["profit"].sum())
    n = int(len(live))
    return {
        "n_bets": n,
        "n_races": int(live["race_id"].nunique()) if n else 0,
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else float("nan"),
        "hit_rate": float(live["win"].mean()) if n else float("nan"),
    }


def bootstrap_roi(bets, n_boot=1000, random_state=42):
    live = bets[bets["stake"] > 0].copy()
    if live.empty:
        return {"roi_mean": float("nan"), "roi_p05": float("nan"), "roi_p95": float("nan")}
    rng = np.random.default_rng(random_state)
    race_ids = live["race_id"].unique()
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(race_ids, size=len(race_ids), replace=True)
        part = live.set_index("race_id").loc[chosen]
        staked = part["stake"].sum()
        profit = part["profit"].sum()
        samples.append(profit / staked if staked else np.nan)
    samples = np.asarray(samples, dtype=float)
    return {
        "roi_mean": float(np.nanmean(samples)),
        "roi_p05": float(np.nanpercentile(samples, 5)),
        "roi_p95": float(np.nanpercentile(samples, 95)),
    }


class MarketIntegrator:
    def __init__(self, ev_threshold=0.05, kelly_fraction=0.25):
        self.ev_threshold = ev_threshold
        self.kelly_fraction = kelly_fraction
        self.bias_model = None
        self.logit_model = None
        self.isotonic = None
        self.weights_ = None

    def fit(self, df, p_model, y_win):
        p_mkt = market_probs(df["オッズ"], df["race_id"])
        self.bias_model, p_mkt_adj = fit_longshot_bias(p_mkt, y_win, df["race_id"])
        self.logit_model, p_comb = fit_conditional_logit(
            p_model, p_mkt_adj, y_win, df["race_id"]
        )
        self.weights_ = self.logit_model.coef_.ravel()
        self.isotonic = fit_isotonic(p_comb, y_win)
        return self

    def predict(self, df, p_model):
        p_mkt = market_probs(df["オッズ"], df["race_id"])
        p_mkt_adj = apply_longshot_bias(self.bias_model, p_mkt, df["race_id"])
        p_comb, weights = apply_conditional_logit(
            self.logit_model, p_model, p_mkt_adj, df["race_id"]
        )
        self.weights_ = weights
        p_cal = np.clip(self.isotonic.predict(p_comb), 1e-6, 1 - 1e-6)
        p_win = softmax_by_race(np.log(p_cal), df["race_id"])
        return p_win, p_mkt, p_mkt_adj

    def bets(self, df, p_win, payouts=None):
        ev, stake = select_bets(
            p_win,
            df["オッズ"],
            ev_threshold=self.ev_threshold,
            kelly_fraction=self.kelly_fraction,
        )
        bets = settle_win_bets(df, p_win, stake, payouts=payouts)
        bets["ev"] = ev
        return bets
