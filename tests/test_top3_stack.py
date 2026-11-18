"""契約: top3 は馬ごとの周辺確率。レース内合計1ではない。補助・診断用。"""

import numpy as np
import pandas as pd
import pytest

from column_select import full_feature_columns
from top3_market import (
    Top3MarketIntegrator,
    place_market_probs,
    settle_place_bets,
    select_place_diagnostic_bets,
    top3_triple_heuristics,
)
from top3_meta_model import Top3MetaModel, build_logit_frame
from horse_logit import fit_horse_bernoulli


def test_horse_bernoulli_probs_in_unit_interval():
    X = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]])
    y = np.array([1, 0, 1])
    model, p = fit_horse_bernoulli(X, y, x0=np.ones(2), bounds=[(0.0, None)] * 2)
    assert np.all(p >= 1e-6)
    assert np.all(p <= 1 - 1e-6)
    assert not np.isclose(p.sum(), 1.0)


def test_horse_bernoulli_weights_nonnegative():
    X = np.log(
        np.clip(
            np.column_stack([np.full(6, 0.3), np.full(6, 0.7)]),
            1e-6,
            1 - 1e-6,
        )
        / (1 - np.clip(np.column_stack([np.full(6, 0.3), np.full(6, 0.7)]), 1e-6, 1 - 1e-6))
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    model, _ = fit_horse_bernoulli(X, y, bounds=[(0.0, None)] * 2)
    assert np.all(model.coef_.ravel() >= -1e-9)


def test_logit_frame_no_race_softmax():
    pred = pd.DataFrame({"lgb_binary": [0.2, 0.8], "cat_binary": [0.6, 0.4]}, index=[0, 1])
    frame = build_logit_frame(pred)
    assert "lgb_binary" in frame.columns
    assert frame["lgb_binary"].iloc[0] < frame["lgb_binary"].iloc[1]


def test_top3_meta_predict_not_sum_to_one(two_horse_race_frame):
    df = two_horse_race_frame(n_races=10)
    pred = pd.DataFrame(
        {
            "lgb_binary": np.full(len(df), 0.55),
            "cat_binary": np.full(len(df), 0.60),
        },
        index=df.index,
    )
    meta = Top3MetaModel().fit(df, pred)
    out = meta.predict(df, pred)
    sums = out.groupby(df["race_id"])["p_top3"].sum()
    assert not np.allclose(sums.to_numpy(), 1.0, atol=0.05)


def test_stage1_excludes_win_odds_from_features():
    """top3 Stage1 も今走の単勝オッズを特徴に入れない。"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "着順": [1, 4],
            "オッズ": [2.5, 5.0],
            "人気": [1, 2],
            "着順1": [3, 2],
            "スピード指数1": [0.1, 0.2],
            "距離": [1600, 1600],
            "芝・ダート": [0, 0],
            "場id": [1, 1],
            "馬場": [0, 0],
            "騎手": ["J1", "J2"],
            "馬番": [1, 2],
        }
    )
    cols = full_feature_columns(df, include_odds=False)
    assert "オッズ" not in cols
    assert "人気" not in cols


def test_top3_market_identity_without_place_odds(two_horse_race_frame):
    df = two_horse_race_frame(6)
    p_model = np.linspace(0.2, 0.8, len(df))
    y = (pd.to_numeric(df["着順"], errors="coerce") <= 3).astype(int)
    m = Top3MarketIntegrator()
    m.fit(df, p_model, y)
    p_final, p_mkt = m.predict(df, p_model)
    np.testing.assert_allclose(p_final, p_model, rtol=1e-6)
    assert p_mkt is None or np.all(np.isnan(p_mkt))


def test_place_market_not_normalized_per_race():
    odds = np.array([2.0, 4.0, 8.0, 10.0])
    p = place_market_probs(odds)
    assert p.sum() != pytest.approx(1.0)


def test_select_place_bets_zero_stake_under_eight_horses():
    df = pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "馬番": list(range(1, 7)),
            "着順": [1, 2, 3, 4, 5, 6],
            "頭数": [6] * 6,
        }
    )
    p = np.linspace(0.1, 0.9, 6)
    _, stake = select_place_diagnostic_bets(p, df["race_id"], df, prob_threshold=0.0)
    assert stake.sum() == 0.0


def test_settle_place_two_winners_same_race():
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "馬番": [1, 2],
            "着順": [1, 2],
            "頭数": [10, 10],
        }
    )
    payouts = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "券種": ["複勝", "複勝"],
            "組番": ["1", "2"],
            "払戻金": ["180", "250"],
        }
    )
    stake = np.array([0.01, 0.01])
    bets = settle_place_bets(df, stake, payouts=payouts)
    assert (bets["profit"] > 0).sum() == 2


def test_triple_heuristic_score_is_product():
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "馬番": [1, 2, 3],
            "着順": [1, 2, 3],
        }
    )
    p = np.array([0.5, 0.4, 0.3])
    tri = top3_triple_heuristics(df, p, top_n=3)
    assert len(tri) == 1
    expected = 0.5 * 0.4 * 0.3
    assert tri["heuristic_score"].iloc[0] == pytest.approx(expected)
    assert tri["heuristic_score"].sum() != pytest.approx(1.0)