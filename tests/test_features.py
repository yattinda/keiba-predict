"""契約: expanding 集計・相対特徴・エンコード・芝フィルタ順序（リークと学習対象の切り分け）。"""

import numpy as np
import pandas as pd
import pytest

from data_process import (
    add_expanding_form_stats,
    add_jockey_win_rate,
    add_past_runs,
    add_relative_features,
    apply_encoders,
    fit_encoders,
    process_frame,
)


def _jockey_timeline():
    """騎手J: 1走目は別レースで3着、2走目で1着、3走目で追加の1着（未来）。"""
    return pd.DataFrame(
        {
            "日付": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "race_id": ["ra", "rb", "rc"],
            "騎手": ["J"] * 3,
            "着順": [3, 1, 1],
            "馬": ["X", "Y", "Z"],
            "馬番": [1, 1, 1],
            "距離": [1600, 1600, 1600],
            "場id": [1, 1, 1],
            "馬場": [0, 0, 0],
            "芝・ダート": [0, 0, 0],
            "調教師": ["T"] * 3,
            "斤量": [55.0, 55.0, 55.0],
            "体重": [480, 480, 480],
            "日付差": [30, 30, 30],
            "休養明け": [0, 0, 0],
        }
    )


def test_jockey_first_race_win_rate_is_nan():
    """初出走の騎手勝率は未定義（NaN）。"""
    out = add_jockey_win_rate(_jockey_timeline())
    first = out.loc[out["race_id"] == "ra", "騎手の勝率"].iloc[0]
    assert pd.isna(first)


def test_jockey_win_rate_excludes_same_race_outcome():
    """2走目の率は1走目だけ。1走目は3着なので勝率0（1/1）。"""
    out = add_jockey_win_rate(_jockey_timeline())
    second = out.loc[out["race_id"] == "rb", "騎手の勝率"].iloc[0]
    assert second == pytest.approx(0.0)


def test_jockey_win_rate_unchanged_when_future_win_added():
    """3走目を足す前後で、2走目時点の率は同じ。"""
    base = _jockey_timeline().iloc[:2]
    full = _jockey_timeline()
    rate_before = add_jockey_win_rate(base).loc[base.index[1], "騎手の勝率"]
    rate_after = add_jockey_win_rate(full).loc[full.index[1], "騎手の勝率"]
    assert rate_before == rate_after


def test_expanding_form_unchanged_when_future_race_added(horse_history_frame):
    """未来の1着を足しても、既存走の expanding 成績は変わらない。"""
    base = horse_history_frame(
        dates=["2020-01-01", "2020-02-01", "2020-03-01"],
        finishes=[2, 5, 1],
    )
    base["休養明け"] = 0
    base["日付差"] = 30
    future = horse_history_frame(dates=["2021-01-01"], finishes=[1], jockeys=["J_future"])
    future["race_id"] = "future"
    future["休養明け"] = 0
    future["日付差"] = 30
    cols = ["race_id", "馬の勝率", "馬の複勝率", "馬の距離帯複勝率", "騎手のコース複勝率", "調教師の複勝率"]
    before = add_expanding_form_stats(base)[cols].sort_values("race_id").reset_index(drop=True)
    after = add_expanding_form_stats(pd.concat([base, future], ignore_index=True))
    subset = after[after["race_id"].isin(base["race_id"])][cols].sort_values("race_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(before, subset)
    second = after.loc[after["日付"] == pd.Timestamp("2020-02-01"), "馬の勝率"].iloc[0]
    assert second == pytest.approx(0.0)


def test_distance_band_form_does_not_mix_bands(horse_history_frame):
    """距離帯ごとに集計: 短距離3走目は 1着・5着・1着 → 複勝2/3ではなく当該帯で 1/2。長距離2走目は前走1着だけ → 1/1。"""
    df = horse_history_frame(
        dates=[
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
        ],
        finishes=[1, 5, 1, 1, 2],
        distances=[1200, 1200, 1200, 2000, 2000],
    )
    df["休養明け"] = 0
    df["日付差"] = 30
    out = add_expanding_form_stats(df)
    short_third = out.loc[out["日付"] == pd.Timestamp("2020-03-01"), "馬の距離帯複勝率"].iloc[0]
    long_second = out.loc[out["日付"] == pd.Timestamp("2020-05-01"), "馬の距離帯複勝率"].iloc[0]
    assert short_third == pytest.approx(0.5)
    assert long_second == pytest.approx(1.0)


def test_relative_gap_uses_only_same_race():
    """レース内 gap はそのレースの最大との差だけ（他レースの斤量は見ない）。"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2"],
            "斤量": [54.0, 56.0, 99.0],
        }
    )
    out = add_relative_features(df)
    gap_r1_max = out.loc[out["race_id"] == "R1", "斤量_race_gap"]
    assert gap_r1_max.max() == pytest.approx(0.0)
    assert gap_r1_max.min() == pytest.approx(-2.0)
    assert out.loc[out["race_id"] == "R2", "斤量_race_gap"].iloc[0] == pytest.approx(0.0)


def test_encoder_unknown_category_gets_stable_id():
    """train に無いカテゴリは unknown 番号に落ち、mapping 長は増えない。"""
    train = pd.DataFrame({"騎手": ["A", "B", "A"]})
    test = pd.DataFrame({"騎手": ["B", "C"]})
    enc = fit_encoders(train)
    n_before = len(enc["騎手"]["mapping"])
    out = apply_encoders(test, enc)
    n_after = len(enc["騎手"]["mapping"])
    assert n_before == n_after
    assert int(out.iloc[1]["騎手"]) == n_before


def _raw_row(horse, date, surface, race_id, finish=1):
    return {
        "日付": pd.Timestamp(date).strftime("%Y年%m月%d日"),
        "race_id": race_id,
        "馬番": 1,
        "馬": horse,
        "horse_id": "h1",
        "着順": finish,
        "走破時間": "1:35.0",
        "距離": 1600,
        "場id": 1,
        "芝・ダート": surface,
        "回り": "右",
        "馬場": "良",
        "天気": "晴",
        "クラス": "未勝利",
        "性": "牡",
        "齢": 3,
        "斤量": 55,
        "騎手": "騎手A",
        "オッズ": 3.5,
        "人気": 1,
        "体重": 480,
        "体重変化": 0,
        "上がり": "34.0",
        "通過順": "3-3",
        "着差": "ハナ",
        "レース名": "テスト",
        "開催": "1回",
        "場名": "中山",
        "調教師": "調教師A",
    }


def test_turf_filter_keeps_dirt_in_past_run_column():
    """芝絞りは近走の後: 前走ダート(1)が芝レース行の芝・ダート1 に残る。"""
    raw = pd.DataFrame(
        [
            _raw_row("馬A", "2020-01-01", "ダ", "dirt1", finish=2),
            _raw_row("馬A", "2020-02-01", "芝", "turf1", finish=1),
        ]
    )
    out, _ = process_frame(raw, train_year_end=2025, min_speed_count=1)
    assert len(out) == 1
    turf = out.iloc[0]
    assert turf["race_id"] == "turf1"
    assert int(turf["芝・ダート1"]) == 1


def test_encoder_fit_ignores_categories_after_train_year():
    """train_year_end より後のカテゴリは mapping に入らず、それ以前の行の値も変わらない。"""
    early = pd.DataFrame(
        [
            _raw_row("馬A", "2020-01-01", "芝", "early1", finish=3),
            _raw_row("馬A", "2020-06-01", "芝", "early2", finish=1),
        ]
    )
    future = pd.DataFrame(
        [
            _raw_row("馬A", "2021-01-01", "芝", "future1", finish=1),
        ]
    )
    future["騎手"] = "未来騎手"
    future["レース名"] = "未来レース"
    only, enc_only = process_frame(early, train_year_end=2020, min_speed_count=1)
    both, enc_both = process_frame(
        pd.concat([early, future], ignore_index=True),
        train_year_end=2020,
        min_speed_count=1,
    )
    assert "未来騎手" not in enc_both["騎手"]["mapping"]
    assert set(enc_both["騎手"]["mapping"]) == set(enc_only["騎手"]["mapping"])
    assert int(both.loc[both["race_id"] == "future1", "騎手"].iloc[0]) == enc_both["騎手"]["unknown"]

    causal = ["着順1", "騎手の勝率", "馬の勝率", "スピード指数"]
    left = only.set_index("race_id")[causal].sort_index()
    right = both.set_index("race_id").loc[left.index, causal].sort_index()
    pd.testing.assert_frame_equal(left, right)
    assert int(only.loc[only["race_id"] == "early2", "着順1"].iloc[0]) == 3