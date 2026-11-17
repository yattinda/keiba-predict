"""契約: 近走特徴は「自分より前の出走」だけ。未来を足しても過去行は不変。"""

import pandas as pd

from data_process import add_past_runs


def _horse_frame(dates, finishes, race_ids=None):
    n = len(dates)
    if race_ids is None:
        race_ids = list(range(1, n + 1))
    return pd.DataFrame(
        {
            "馬": ["A"] * n,
            "horse_id": ["h1"] * n,
            "日付": pd.to_datetime(dates),
            "着順": finishes,
            "race_id": race_ids,
        }
    )


def test_oldest_run_has_no_self_leak_in_past_finish():
    """最古走の着順1は空（今走自身を過去走に含めない）。"""
    df = _horse_frame(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        [8, 2, 5, 1],
    )
    out = add_past_runs(df)
    oldest = out.loc[out["日付"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert pd.isna(oldest["着順1"])
    assert int(oldest["着順"]) == 8


def test_past_finish_is_strictly_older_race():
    """2走目の着順1 は1走目の着順と一致。"""
    df = _horse_frame(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        [8, 2, 5, 1],
    )
    out = add_past_runs(df)
    row = out.loc[out["日付"] == pd.Timestamp("2020-02-01")].iloc[0]
    assert int(row["着順1"]) == 8


def test_adding_future_race_does_not_change_past_rows():
    """未来の出走を1件足しても、既存レースの着順1 列は同一。"""
    base = _horse_frame(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        [8, 2, 5, 1],
    )
    before = add_past_runs(base)
    key_cols = ["race_id", "着順1"]
    snapshot = before[key_cols].sort_values("race_id").reset_index(drop=True)

    extended = pd.concat(
        [
            base,
            _horse_frame(["2021-01-01"], [3], race_ids=[99]),
        ],
        ignore_index=True,
    )
    after = add_past_runs(extended)
    subset = after[after["race_id"].isin(base["race_id"])][key_cols].sort_values("race_id")
    subset = subset.reset_index(drop=True)
    pd.testing.assert_frame_equal(snapshot, subset)