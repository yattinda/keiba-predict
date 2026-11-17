"""Stage3 市場統合と ROI バックテスト。"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from race_logit import apply_race_softmax, fit_race_softmax
from train_utils import softmax_by_race


def market_probs(odds, race_id):
    raw = pd.to_numeric(odds, errors="coerce")
    inv = 1.0 / np.clip(np.asarray(raw, dtype=float), 1.01, None)
    grouped = pd.Series(inv).groupby(np.asarray(race_id), sort=False)
    return (pd.Series(inv) / grouped.transform("sum")).to_numpy()


def fit_conditional_logit(p_model, p_market, y_win, race_id):
    eps = 1e-6
    x = np.column_stack(
        [
            np.log(np.clip(p_model, eps, 1)),
            np.log(np.clip(p_market, eps, 1)),
        ]
    )
    bounds = [(0.0, None), (0.0, None)]
    return fit_race_softmax(
        x,
        y_win,
        race_id,
        x0=np.array([0.5, 1.0]),
        bounds=bounds,
        l2_target=np.array([0.0, 1.0]),
        l2_coef=1e-2,
    )


def apply_conditional_logit(model, p_model, p_market, race_id):
    eps = 1e-6
    x = np.column_stack(
        [
            np.log(np.clip(p_model, eps, 1)),
            np.log(np.clip(p_market, eps, 1)),
        ]
    )
    return apply_race_softmax(model, x, race_id)


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


def expected_value(p, odds):
    o = pd.to_numeric(odds, errors="coerce")
    if isinstance(o, pd.Series):
        o = o.to_numpy()
    else:
        o = np.asarray(o, dtype=float)
    return np.asarray(p, dtype=float) * o - 1.0


def _solve_exclusive_kelly(p, odds, candidates, kelly_fraction=0.25, max_race_stake=0.05):
    p = np.asarray(p, dtype=float)
    odds = np.asarray(odds, dtype=float)
    n = len(p)
    x0 = np.zeros(n)

    def objective(f):
        total = f.sum()
        wealth = 1.0 - total + f * odds
        if np.any(wealth <= 1e-9):
            return 1e6
        return -np.sum(p * np.log(wealth))

    bounds = []
    for i in range(n):
        if candidates[i]:
            bounds.append((0.0, 0.2))
        else:
            bounds.append((0.0, 0.0))
    constraints = [{"type": "ineq", "fun": lambda f: 0.2 - f.sum()}]
    try:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success:
            return np.zeros(n)
        f = np.clip(res.x, 0.0, None) * kelly_fraction
        total = f.sum()
        if total > max_race_stake and total > 0:
            f = f * (max_race_stake / total)
        wealth = 1.0 - f.sum() + f * odds
        if np.any(wealth <= 1e-9):
            return np.zeros(n)
        return f
    except Exception:
        return np.zeros(n)


def select_bets(p, odds, race_id, ev_threshold=0.05, kelly_fraction=0.25):
    p = np.asarray(p, dtype=float)
    odds = np.asarray(odds, dtype=float)
    race_id = np.asarray(race_id)
    ev = expected_value(p, odds)
    stake = np.zeros(len(p), dtype=float)
    for rid in np.unique(race_id):
        mask = race_id == rid
        idx = np.where(mask)[0]
        ev_r = ev[mask]
        candidates = ev_r > ev_threshold
        if not candidates.any():
            continue
        stake[mask] = _solve_exclusive_kelly(
            p[mask],
            odds[mask],
            candidates,
            kelly_fraction=kelly_fraction,
            max_race_stake=0.05,
        )
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


def split_calibration_races(calib_df, train_frac=0.75):
    races = (
        calib_df[["race_id", "日付"]]
        .drop_duplicates()
        .sort_values(["日付", "race_id"])
        .reset_index(drop=True)
    )
    n_train = max(1, int(len(races) * train_frac))
    train_races = set(races.iloc[:n_train]["race_id"])
    val_races = set(races.iloc[n_train:]["race_id"])
    train_mask = calib_df["race_id"].isin(train_races)
    val_mask = calib_df["race_id"].isin(val_races)
    return calib_df.loc[train_mask], calib_df.loc[val_mask]


def pick_ev_threshold(market, calib_tail, p_model, min_bets=30, fallback=0.05):
    y_win = (pd.to_numeric(calib_tail["着順"], errors="coerce") == 1).astype(int)
    p_final, _ = market.predict(calib_tail, p_model)
    grid = [0.0, 0.02, 0.05, 0.10, 0.15]
    best = fallback
    best_roi = float("-inf")
    for thresh in grid:
        market.ev_threshold = thresh
        bets = market.bets(calib_tail, p_final, payouts=None)
        live = bets[bets["stake"] > 0]
        if len(live) < min_bets:
            continue
        roi = float(live["profit"].sum() / live["stake"].sum())
        if roi > best_roi:
            best_roi = roi
            best = thresh
    fallback_flag = int(best == fallback and best_roi == float("-inf"))
    return best, fallback_flag


class MarketIntegrator:
    def __init__(self, ev_threshold=0.05, kelly_fraction=0.25):
        self.ev_threshold = ev_threshold
        self.kelly_fraction = kelly_fraction
        self.logit_model = None
        self.weights_ = None

    def fit(self, df, p_model, y_win):
        p_mkt = market_probs(df["オッズ"], df["race_id"])
        self.logit_model, _ = fit_conditional_logit(
            p_model, p_mkt, y_win, df["race_id"].to_numpy()
        )
        self.weights_ = self.logit_model.coef_.ravel()
        return self

    def predict(self, df, p_model):
        p_mkt = market_probs(df["オッズ"], df["race_id"])
        p_final, weights = apply_conditional_logit(
            self.logit_model, p_model, p_mkt, df["race_id"].to_numpy()
        )
        self.weights_ = weights
        return p_final, p_mkt

    def bets(self, df, p_win, payouts=None):
        ev, stake = select_bets(
            p_win,
            df["オッズ"],
            df["race_id"],
            ev_threshold=self.ev_threshold,
            kelly_fraction=self.kelly_fraction,
        )
        bets = settle_win_bets(df, p_win, stake, payouts=payouts)
        bets["ev"] = ev
        return bets