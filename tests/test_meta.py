"""契約: メタはレース内確率分布。指標は1着予測の解釈が一意。"""

import numpy as np
import pandas as pd
import pytest

from meta_model import BASE_PRED_COLS, MetaModel, build_log_share_frame
from metrics import race_winner_nll, win_pick_hit_rate
from race_nn import RaceDataset
from train_utils import softmax_by_race


def test_log_share_is_log_softmax_per_race():
    pred = pd.DataFrame({"lgb_binary": [0.2, 0.8, 0.5, 0.5]}, index=[0, 1, 2, 3])
    race_id = np.array(["A", "A", "B", "B"])
    frame = build_log_share_frame(pred, race_id)
    shares = np.exp(frame["lgb_binary"].to_numpy())
    np.testing.assert_allclose(shares.sum(), 2.0, rtol=1e-5)
    np.testing.assert_allclose(shares[:2].sum(), 1.0, rtol=1e-5)


def test_meta_model_predict_sums_to_one_per_race(two_horse_race_frame):
    df = two_horse_race_frame(n_races=12)
    pred = pd.DataFrame(
        {c: np.linspace(0.2, 0.8, len(df)) for c in BASE_PRED_COLS},
        index=df.index,
    )
    meta = MetaModel().fit(df, pred)
    out = meta.predict(df, pred)
    sums = out.groupby(df["race_id"])["p_win"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-4)


def test_meta_model_requires_base_columns():
    df = pd.DataFrame({"race_id": ["R1"], "着順": [1], "日付": pd.Timestamp("2020-01-01")})
    pred = pd.DataFrame(index=df.index)
    with pytest.raises(ValueError, match="ベース予測"):
        MetaModel().fit(df, pred)


def test_race_winner_nll_ignores_dead_heat_races():
    race_id = np.array(["R1", "R1", "R2", "R2"])
    p = np.array([0.6, 0.4, 0.5, 0.5])
    finish = pd.Series([1, 1, 1, 2])
    nll = race_winner_nll(race_id, p, finish)
    assert nll == pytest.approx(-np.log(0.5))


def test_win_pick_hit_rate():
    race_id = np.array(["R1", "R1"])
    p = np.array([0.7, 0.3])
    finish = pd.Series([1, 2])
    assert win_pick_hit_rate(race_id, p, finish) == pytest.approx(1.0)


def test_race_dataset_skips_single_horse_races():
    df = pd.DataFrame(
        {
            "race_id": ["solo", "duo", "duo"],
            "着順": [1, 1, 2],
        }
    )
    feats = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    ds = RaceDataset(df, feats, np.zeros(1), np.ones(1), train=True)
    race_ids = {s[3] for s in ds.samples}
    assert race_ids == {"duo"}


def test_softmax_by_race_sums_to_one_with_nan():
    values = np.array([1.0, 2.0, np.nan, 3.0])
    race_id = np.array(["A", "A", "B", "B"])
    p = softmax_by_race(values, race_id)
    assert np.nansum(p[[0, 1]]) == pytest.approx(1.0)
    assert p[2:].sum() == pytest.approx(1.0)