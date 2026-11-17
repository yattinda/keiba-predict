"""契約: スピード指数は「過去の走破のみ」で決まり、同レースの相手や未来レースに影響されない。"""

import numpy as np
import pandas as pd

from data_process import add_causal_speed_figure


def _rows_for_race(race_id, date, seconds_list, distance=1600, venue=1, going=0):
    rows = []
    for i, sec in enumerate(seconds_list, start=1):
        rows.append(
            {
                "race_id": race_id,
                "日付": pd.Timestamp(date),
                "馬番": i,
                "馬": f"H{i}",
                "走破時間_秒": sec,
                "距離": distance,
                "場id": venue,
                "馬場": going,
            }
        )
    return rows


def test_future_races_do_not_change_earlier_indices():
    """未来レースを足しても、既出レースの指数は1ビットも変わらない。"""
    rows_a = _rows_for_race("r1", "2020-01-01", [95.0, 96.0])
    rows_b = _rows_for_race("r2", "2020-02-01", [94.0, 97.0])
    df_a = pd.DataFrame(rows_a)
    idx_a = add_causal_speed_figure(df_a, min_count=2)["スピード指数"].to_numpy()

    df_both = pd.DataFrame(rows_a + rows_b)
    idx_both = add_causal_speed_figure(df_both, min_count=2)
    idx_a2 = idx_both.loc[idx_both["race_id"] == "r1", "スピード指数"].to_numpy()
    np.testing.assert_allclose(idx_a, idx_a2, rtol=0, atol=1e-9)


def test_later_race_uses_earlier_times():
    """2走目は1走目の統計を参照する（単独2走目だけと値が違う）。"""
    rows_a = _rows_for_race("r1", "2020-01-01", [95.0, 96.0])
    rows_b = _rows_for_race("r2", "2020-02-01", [94.0, 97.0])
    full = add_causal_speed_figure(pd.DataFrame(rows_a + rows_b), min_count=2)
    only_b = add_causal_speed_figure(pd.DataFrame(rows_b), min_count=2)
    b_with_history = full.loc[full["race_id"] == "r2", "スピード指数"].to_numpy()
    b_alone = only_b["スピード指数"].to_numpy()
    assert not np.allclose(b_with_history, b_alone, equal_nan=True)


def test_same_race_opponent_times_not_in_index():
    """同レースの他馬のタイムは指数計算に使わない（初出グループは NaN）。"""
    rows = _rows_for_race("r1", "2020-01-01", [90.0, 100.0])
    out = add_causal_speed_figure(pd.DataFrame(rows), min_count=2)
    fast = out.loc[out["走破時間_秒"] == 90.0, "スピード指数"].iloc[0]
    slow = out.loc[out["走破時間_秒"] == 100.0, "スピード指数"].iloc[0]
    assert np.isnan(fast) and np.isnan(slow)