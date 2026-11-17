"""
合成レース用の共通フィクスチャ。

実データは使わず、テスト名と docstring で「何が壊れたら困るか」を読めるようにする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def two_horse_race_frame():
    """2頭×複数レース。市場統合・メタモデル用。"""

    def build(n_races=5, alt_winner=True):
        rows = []
        for r in range(n_races):
            rid = f"R{r:03d}"
            winner1 = (r % 2 == 0) if alt_winner else True
            rows.append(
                {
                    "race_id": rid,
                    "日付": pd.Timestamp("2020-01-01") + pd.Timedelta(days=r * 7),
                    "馬番": 1,
                    "馬": f"H1_{r}",
                    "着順": 1 if winner1 else 2,
                    "オッズ": 2.5,
                    "人気": 1,
                }
            )
            rows.append(
                {
                    "race_id": rid,
                    "日付": pd.Timestamp("2020-01-01") + pd.Timedelta(days=r * 7),
                    "馬番": 2,
                    "馬": f"H2_{r}",
                    "着順": 2 if winner1 else 1,
                    "オッズ": 3.5,
                    "人気": 2,
                }
            )
        return pd.DataFrame(rows)

    return build


@pytest.fixture
def horse_history_frame():
    """1頭の時系列。近走・expanding 集計用。"""

    def build(
        horse="A",
        horse_id="h1",
        dates=None,
        finishes=None,
        distances=None,
        surfaces=None,
        jockeys=None,
        trainers=None,
    ):
        if dates is None:
            dates = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
        if finishes is None:
            finishes = [8, 2, 5, 1]
        n = len(dates)
        if distances is None:
            distances = [1600] * n
        if surfaces is None:
            surfaces = [0] * n
        if jockeys is None:
            jockeys = ["J1"] * n
        if trainers is None:
            trainers = ["T1"] * n
        return pd.DataFrame(
            {
                "馬": [horse] * n,
                "horse_id": [horse_id] * n,
                "日付": pd.to_datetime(dates),
                "着順": finishes,
                "race_id": [f"r{i+1}" for i in range(n)],
                "馬番": list(range(1, n + 1)),
                "距離": distances,
                "芝・ダート": surfaces,
                "場id": [1] * n,
                "馬場": [0] * n,
                "騎手": jockeys,
                "調教師": trainers,
                "斤量": [55.0] * n,
                "体重": [480] * n,
                "オッズ": [5.0] * n,
            }
        )

    return build


@pytest.fixture
def minimal_raw_row():
    """data_process.main と同じ列を持つ最小1行（load_raw 用ヘッダの雛形）。"""

    def build(**overrides):
        row = {
            "日付": "2020年01月01日",
            "race_id": "2020010101",
            "馬番": 1,
            "馬": "テスト馬",
            "着順": 1,
            "走破時間": "1:34.5",
            "距離": 1600,
            "場id": 1,
            "芝・ダート": "芝",
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
            "上がり": "34.5",
            "通過順": "3-3",
            "着差": "ハナ",
            "レース名": "テストS",
            "開催": "1回中山1日目",
            "場名": "中山",
        }
        row.update(overrides)
        return row

    return build