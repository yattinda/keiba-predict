"""契約: Stage3 の確率はレース内で1。賭金・精算の符号と集計が破綻しない。"""

import numpy as np
import pandas as pd
import pytest

from market_roi import (
    MarketIntegrator,
    _solve_exclusive_kelly,
    apply_conditional_logit,
    bootstrap_roi,
    fit_conditional_logit,
    market_probs,
    roi_summary,
    select_bets,
    settle_win_bets,
    split_calibration_races,
)
from train_utils import softmax_by_race


def test_conditional_logit_probs_sum_to_one(two_horse_race_frame):
    df = two_horse_race_frame(8)
    p_model = np.full(len(df), 0.25)
    y_win = (df["着順"] == 1).astype(int)
    race_id = df["race_id"].to_numpy()
    model, _ = fit_conditional_logit(p_model, np.full(len(df), 0.25), y_win, race_id)
    p_final, _ = apply_conditional_logit(model, p_model, np.full(len(df), 0.25), race_id)
    sums = pd.Series(p_final).groupby(race_id).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-5)


def test_market_integrator_has_no_bias_or_isotonic():
    m = MarketIntegrator()
    assert not hasattr(m, "bias_model") or getattr(m, "bias_model", None) is None
    assert not hasattr(m, "isotonic") or getattr(m, "isotonic", None) is None


def test_exclusive_kelly_caps_race_stake():
    p = np.array([0.45, 0.40, 0.44, 0.38])
    odds = np.array([3.0, 2.8, 3.1, 2.9])
    race_id = np.array(["A", "A", "B", "B"])
    _, stake = select_bets(p, odds, race_id, ev_threshold=-0.5, kelly_fraction=0.25)
    for rid in np.unique(race_id):
        assert stake[race_id == rid].sum() <= 0.05 + 1e-9


def test_exclusive_kelly_zero_stake_when_optimization_fails():
    """非有限オッズなどで最適化が破綻したレースは見送り（stake 0）。"""
    stake = _solve_exclusive_kelly(
        np.array([0.6, 0.4]),
        np.array([np.inf, 2.0]),
        np.array([True, True]),
    )
    assert stake.sum() == 0.0


def test_market_probs_normalize_even_when_raw_implied_over_one():
    """生の 1/odds 合計が1超でも、レース内正規化で合計1。"""
    odds = np.array([1.5, 2.0, 3.0, 4.0])
    race_id = np.array(["R", "R", "R", "R"])
    p = market_probs(odds, race_id)
    assert p.sum() == pytest.approx(1.0)


def test_market_probs_clips_tiny_odds():
    p = market_probs(np.array([1.0, 5.0]), np.array(["R", "R"]))
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)


def test_settle_win_bets_profit_signs():
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "馬番": [1, 2],
            "オッズ": [3.0, 5.0],
            "着順": [1, 2],
        }
    )
    stake = np.array([0.01, 0.02])
    bets = settle_win_bets(df, np.array([0.6, 0.4]), stake)
    winner = bets.loc[bets["馬番"] == 1].iloc[0]
    loser = bets.loc[bets["馬番"] == 2].iloc[0]
    assert winner["profit"] == pytest.approx(0.01 * (3.0 - 1))
    assert loser["profit"] == pytest.approx(-0.02)


def test_settle_uses_payout_table_over_odds():
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "馬番": [1],
            "オッズ": [3.0],
            "着順": [1],
        }
    )
    payouts = pd.DataFrame(
        {"race_id": ["R1"], "券種": ["単勝"], "組番": ["1"], "払戻金": ["450"]}
    )
    bets = settle_win_bets(df, np.array([0.5]), np.array([0.1]), payouts=payouts)
    assert bets["profit"].iloc[0] == pytest.approx(0.1 * (4.5 - 1))


def test_roi_summary_empty_bets():
    bets = pd.DataFrame(
        {
            "race_id": [],
            "stake": [],
            "profit": [],
            "win": [],
        }
    )
    s = roi_summary(bets)
    assert s["n_bets"] == 0
    assert np.isnan(s["roi"])


def test_bootstrap_roi_reproducible():
    bets = pd.DataFrame(
        {
            "race_id": ["A", "A", "B"],
            "stake": [0.01, 0.01, 0.02],
            "profit": [0.01, -0.01, 0.02],
        }
    )
    a = bootstrap_roi(bets, n_boot=200, random_state=7)
    b = bootstrap_roi(bets, n_boot=200, random_state=7)
    assert a["roi_p05"] == b["roi_p05"]
    assert a["roi_p95"] == b["roi_p95"]


def test_split_calibration_races_disjoint():
    rows = []
    for i in range(8):
        rows.append(
            {
                "race_id": f"R{i}",
                "日付": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "馬番": 1,
            }
        )
    calib = pd.DataFrame(rows)
    train, val = split_calibration_races(calib, train_frac=0.75)
    assert set(train["race_id"]).isdisjoint(set(val["race_id"]))
    assert train["日付"].max() <= val["日付"].min()


def test_softmax_by_race_contract():
    p = softmax_by_race(np.array([1.0, 2.0, 0.0, 1.0]), np.array(["A", "A", "B", "B"]))
    assert p[[0, 1]].sum() == pytest.approx(1.0)
    assert p[[2, 3]].sum() == pytest.approx(1.0)