"""契約: Stage1 に入ってはいけない列、年次 walk-forward、レース単位の時系列 split。"""

import numpy as np
import pandas as pd
import pytest

from column_select import full_feature_columns
from train_utils import (
    binary_target,
    feature_matrix,
    inner_time_split,
    race_time_series_splits,
    rank_target,
)
from walk_forward import walk_forward_windows


FORBIDDEN_FEATURES = {
    "着順",
    "オッズ",
    "人気",
    "走破時間",
    "走破時間_秒",
    "スピード指数",
    "上がり",
    "通過順",
    "着差",
    "賞金",
    "単勝払戻",
}


def test_stage1_excludes_leak_and_market_columns():
    """Stage1（オッズなし）に今走リーク列と市場列が混ざらない。"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "着順": [1, 2],
            "オッズ": [2.5, 5.0],
            "人気": [1, 2],
            "走破時間": ["1:34", "1:35"],
            "走破時間_秒": [94.0, 95.0],
            "スピード指数": [0.5, -0.5],
            "上がり": [33.0, 34.0],
            "通過順": [3.0, 5.0],
            "着差": [0.1, 0.5],
            "賞金": [100, 0],
            "単勝払戻": [250, np.nan],
            "着順1": [3, 2],
            "スピード指数1": [0.1, 0.2],
            "距離": [1600, 1600],
            "斤量": [55, 56],
        }
    )
    cols = set(full_feature_columns(df, include_odds=False))
    leaked = cols & FORBIDDEN_FEATURES
    assert not leaked, f"除外すべき列が残っている: {leaked}"
    assert "着順1" in cols
    assert "スピード指数1" in cols
    assert set(feature_matrix(df, include_odds=False).columns) == cols


def test_walk_forward_windows_are_disjoint_and_named():
    """2012–2025 なら4窓。各年は1役割だけ。test 年は name に反映。"""
    years = list(range(2012, 2026))
    windows = walk_forward_windows(years)
    assert len(windows) == 4
    for w in windows:
        all_years = w["base"] + w["meta"] + w["calib"] + w["test"]
        assert len(all_years) == len(set(all_years))
        assert w["test"][0] > w["calib"][-1]
        assert w["name"] == f"test_{w['test'][0]}"
        assert w["test"][-1] > w["calib"][-1]


def test_walk_forward_empty_when_years_too_short():
    assert walk_forward_windows(list(range(2015, 2022))) == []


def test_race_time_series_split_no_race_straddles_train_and_val():
    df = pd.DataFrame(
        {
            "race_id": ["A", "A", "B", "B", "C", "C"],
            "日付": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01", "2020-03-01"]
            ),
        }
    )
    for train_idx, val_idx in race_time_series_splits(df, n_splits=2):
        train_races = set(df.loc[train_idx, "race_id"])
        val_races = set(df.loc[val_idx, "race_id"])
        assert train_races.isdisjoint(val_races)
        if len(val_idx):
            val_min = df.loc[val_idx, "日付"].min()
            train_max = df.loc[train_idx, "日付"].max()
            assert val_min >= train_max


def test_inner_time_split_uses_tail_races_or_aborts():
    df = pd.DataFrame(
        {
            "race_id": [f"R{i}" for i in range(6)],
            "日付": pd.to_datetime([f"2020-0{i+1}-01" for i in range(6)]),
        }
    )
    train_idx = df.index
    inner_train, inner_val = inner_time_split(df, train_idx, val_frac=0.2)
    if inner_val is not None:
        assert set(df.loc[inner_val, "race_id"]).issubset({f"R{i}" for i in range(4, 6)})

    tiny = df.iloc[:2]
    it, iv = inner_time_split(tiny, tiny.index)
    assert iv is None


@pytest.mark.parametrize(
    "finish,expected",
    [
        (1, 1),
        (3, 1),
        (4, 0),
    ],
)
def test_binary_target_top3(finish, expected):
    assert int(binary_target(pd.Series([finish]))[0]) == expected


def test_rank_target_grading():
    finishes = pd.Series([1, 2, 3, 5, 8])
    got = rank_target(finishes)
    assert list(got) == [4, 3, 2, 1, 0]