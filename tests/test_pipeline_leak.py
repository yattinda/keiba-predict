"""契約: 窓の fit と EV 閾値はテスト年の着順を見ない。壊れるとテスト年でモデルや閾値を選んだことになる。"""

import numpy as np
import pandas as pd

import train_pipeline
from market_roi import split_calibration_races


_SEEN = {}


class _RecordingBase:
    def __init__(self, include_odds=False):
        self.include_odds = include_odds

    def fit(self, data):
        _SEEN["base_index"] = set(data.index)
        return self

    def predict_frame(self, data):
        p = np.full(len(data), 0.5)
        return pd.DataFrame(
            {
                "lgb_binary": p,
                "lgb_rank": p,
                "lgb_speed": p,
                "cat_binary": p,
            },
            index=data.index,
        )


class _RecordingMeta:
    def fit(self, df, pred):
        _SEEN["meta_index"] = set(df.index)
        _SEEN["meta_pred_index"] = set(pred.index)
        return self

    def predict(self, df, pred):
        return pd.DataFrame({"p_win": np.full(len(df), 0.5)}, index=df.index)

    @property
    def chosen(self):
        return "stack"


class _RecordingMarket:
    def __init__(self, ev_threshold=0.05, kelly_fraction=0.25):
        self.ev_threshold = ev_threshold
        self.kelly_fraction = kelly_fraction
        self.weights_ = np.array([1.0, 1.0])

    def fit(self, df, p_model, y_win):
        _SEEN["market_index"] = set(df.index)
        return self

    def predict(self, df, p_model):
        p = np.full(len(df), 0.5)
        return p, p

    def bets(self, df, p_final, payouts=None):
        return pd.DataFrame(columns=["race_id", "stake", "profit", "win"])


def _record_threshold(market, calib_tail, p_model, min_bets=30, fallback=0.05):
    _SEEN["threshold_index"] = set(calib_tail.index)
    return fallback, 1


def _no_payouts(years):
    _SEEN["payout_years"] = list(years)
    return None


def _window_frame():
    """base=2018, meta=2019, calib=2020（4レース）, test=2021。各レース2頭。"""
    rows = []
    plan = [
        (2018, 1, "base"),
        (2019, 1, "meta"),
        (2020, 4, "calib"),
        (2021, 1, "test"),
    ]
    for year, n_races, prefix in plan:
        for i in range(n_races):
            rid = f"{prefix}{i}"
            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=i)
            for uma, finish, odds in ((1, 1, 2.5), (2, 2, 4.0)):
                rows.append(
                    {
                        "race_id": rid,
                        "日付": date,
                        "year": year,
                        "馬番": uma,
                        "着順": finish,
                        "オッズ": odds,
                    }
                )
    return pd.DataFrame(rows)


def test_run_window_fits_do_not_see_test_year(tmp_path, monkeypatch):
    """ベース・メタ・市場 fit と閾値選択の行集合は、それぞれの窓だけ。テスト年は評価にしか出ない。"""
    _SEEN.clear()
    monkeypatch.setattr(train_pipeline, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(train_pipeline, "FundamentalModels", _RecordingBase)
    monkeypatch.setattr(train_pipeline, "MetaModel", _RecordingMeta)
    monkeypatch.setattr(train_pipeline, "MarketIntegrator", _RecordingMarket)
    monkeypatch.setattr(train_pipeline, "pick_ev_threshold", _record_threshold)
    monkeypatch.setattr(train_pipeline, "load_payouts", _no_payouts)

    df = _window_frame()
    window = {
        "name": "test_2021",
        "base": [2018],
        "meta": [2019],
        "calib": [2020],
        "test": [2021],
    }
    result = train_pipeline.run_window(
        df,
        window,
        skip_nn=True,
        skip_cat=False,
        ev_threshold=0.05,
        kelly_fraction=0.25,
    )
    assert result is not None
    assert result["window"] == "test_2021"

    base_df = df.loc[df["year"].isin(window["base"])]
    meta_df = df.loc[df["year"].isin(window["meta"])]
    calib_df = df.loc[df["year"].isin(window["calib"])]
    test_df = df.loc[df["year"].isin(window["test"])]
    head, tail = split_calibration_races(calib_df)

    assert _SEEN["base_index"] == set(base_df.index)
    assert _SEEN["meta_index"] == set(meta_df.index)
    assert _SEEN["meta_pred_index"] == set(meta_df.index)
    assert _SEEN["market_index"] == set(head.index)
    assert _SEEN["threshold_index"] == set(tail.index)
    assert _SEEN["payout_years"] == window["test"]

    test_index = set(test_df.index)
    for key in ("base_index", "meta_index", "meta_pred_index", "market_index", "threshold_index"):
        assert _SEEN[key].isdisjoint(test_index)
    assert set(head.index).isdisjoint(set(tail.index))
    assert head["日付"].max() <= tail["日付"].min()
