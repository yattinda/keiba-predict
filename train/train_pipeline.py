"""
Benter 型パイプライン: ファンダメンタル → メタ → 市場統合 → ROI。

使用方法（リポジトリルートから）:
    python train/train_pipeline.py --train --predict
    python train/train_pipeline.py --train --predict --skip-nn
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAIN_DIR)

import metrics
from base_models import FundamentalModels
from market_roi import (
    MarketIntegrator,
    bootstrap_roi,
    pick_ev_threshold,
    roi_summary,
    split_calibration_races,
)
from meta_model import MetaModel
from race_nn import RaceSoftmaxModel
from walk_forward import mask_years, walk_forward_windows


MODEL_DIR = "./model/roi"
RAW_DIR = "./datasets/raw_data"
ENCODED_ALL = "./datasets/encoded/encoded_all.csv"
ENCODED_TRAIN = "./datasets/encoded/encoded_traindata.csv"
ENCODED_TEST = "./datasets/encoded/encoded_testdata.csv"


def load_encoded():
    if os.path.exists(ENCODED_ALL):
        return pd.read_csv(ENCODED_ALL)
    train = pd.read_csv(ENCODED_TRAIN)
    test = pd.read_csv(ENCODED_TEST)
    return pd.concat([train, test], ignore_index=True)


def load_payouts(years):
    frames = []
    for year in years:
        path = os.path.join(RAW_DIR, f"{year}_payouts.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, encoding="SHIFT-JIS"))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _predict_bases(base, race_nn, df, skip_nn):
    pred = base.predict_frame(df)
    if skip_nn or race_nn is None:
        pred["race_nn"] = np.nan
    else:
        pred["race_nn"] = race_nn.predict(df)
        if pred["race_nn"].isna().all():
            pred["race_nn"] = pred["lgb_binary"]
        else:
            pred["race_nn"] = pred["race_nn"].fillna(pred["lgb_binary"])
    return pred


def run_window(df, window, skip_nn, skip_cat, ev_threshold, kelly_fraction):
    print("=" * 60)
    print(f"window {window['name']}: base={window['base']} meta={window['meta']} "
          f"calib={window['calib']} test={window['test']}")
    base_df = df.loc[mask_years(df, window["base"])]
    meta_df = df.loc[mask_years(df, window["meta"])]
    calib_df = df.loc[mask_years(df, window["calib"])]
    test_df = df.loc[mask_years(df, window["test"])]
    if min(len(base_df), len(meta_df), len(calib_df), len(test_df)) == 0:
        print("  skip: empty split")
        return None

    print("  Stage1 ベースモデル学習...")
    base = FundamentalModels(include_odds=False)
    if skip_cat:
        from train_utils import fit_lightgbm, feature_matrix, binary_target as bint

        class _Stub:
            def fit(self, data):
                X = feature_matrix(data, include_odds=False)
                y = bint(data["着順"])
                model = fit_lightgbm(X, y)
                self.models = {
                    "lgb_binary": model,
                    "lgb_rank": model,
                    "lgb_speed": model,
                    "cat_binary": model,
                }
                return self

            def predict_frame(self, data):
                from train_utils import predict_proba_positive

                X = feature_matrix(data, include_odds=False)
                p = predict_proba_positive(self.models["lgb_binary"], X)
                return pd.DataFrame(
                    {
                        "lgb_binary": p,
                        "lgb_rank": p,
                        "lgb_speed": p,
                        "cat_binary": p,
                    },
                    index=data.index,
                )

        base = _Stub().fit(base_df)
    else:
        base.fit(base_df)

    race_nn = None
    if not skip_nn:
        print("  Stage1-E レース NN 学習...")
        race_nn = RaceSoftmaxModel(include_odds=False)
        race_nn.fit(base_df)

    print("  Stage2 メタ学習...")
    meta_pred = _predict_bases(base, race_nn, meta_df, skip_nn)
    meta = MetaModel()
    meta.fit(meta_df, meta_pred)
    print(f"    chosen meta={meta.chosen}")

    print("  Stage3 市場統合 / キャリブレーション...")
    calib_pred = _predict_bases(base, race_nn, calib_df, skip_nn)
    calib_meta = meta.predict(calib_df, calib_pred)
    calib_head, calib_tail = split_calibration_races(calib_df)
    y_win_head = (pd.to_numeric(calib_head["着順"], errors="coerce") == 1).astype(int)
    market = MarketIntegrator(ev_threshold=ev_threshold, kelly_fraction=kelly_fraction)
    market.fit(calib_head, calib_meta.loc[calib_head.index, "p_win"].to_numpy(), y_win_head.to_numpy())
    chosen_thr, threshold_fallback = pick_ev_threshold(
        market,
        calib_tail,
        calib_meta.loc[calib_tail.index, "p_win"].to_numpy(),
        min_bets=30,
        fallback=ev_threshold,
    )
    market.ev_threshold = chosen_thr
    print(f"    logit weights w_model,w_market={market.weights_}")
    print(f"    ev_threshold={chosen_thr} (fallback={threshold_fallback})")

    print("  テスト予測...")
    test_pred = _predict_bases(base, race_nn, test_df, skip_nn)
    test_meta = meta.predict(test_df, test_pred)
    p_final, p_mkt = market.predict(test_df, test_meta["p_win"].to_numpy())
    years = window["test"]
    payouts = load_payouts(years)
    bets = market.bets(test_df, p_final, payouts=payouts)
    summary = roi_summary(bets)
    boot = bootstrap_roi(bets)
    finish = pd.to_numeric(test_df["着順"], errors="coerce")
    nll_market = metrics.race_winner_nll(test_df["race_id"], p_mkt, finish)
    nll_model = metrics.race_winner_nll(test_df["race_id"], test_meta["p_win"], finish)
    nll_final = metrics.race_winner_nll(test_df["race_id"], p_final, finish)
    result = {
        "window": window["name"],
        "meta_backend": meta.chosen,
        "w_model": float(market.weights_[0]) if market.weights_ is not None else np.nan,
        "w_market": float(market.weights_[1]) if market.weights_ is not None else np.nan,
        "ev_threshold": float(chosen_thr),
        "threshold_fallback": int(threshold_fallback),
        "nll_market": nll_market,
        "nll_model": nll_model,
        "nll_final": nll_final,
        "delta_nll": float(nll_market - nll_final) if np.isfinite(nll_market) and np.isfinite(nll_final) else float("nan"),
        **{f"roi_{k}": v for k, v in summary.items()},
        **boot,
        "win_hit_rate": metrics.win_pick_hit_rate(test_df["race_id"], p_final, finish),
    }
    out_dir = os.path.join(MODEL_DIR, window["name"])
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(base, os.path.join(out_dir, "base_models.pkl"))
    joblib.dump(meta, os.path.join(out_dir, "meta_model.pkl"))
    joblib.dump(market, os.path.join(out_dir, "market.pkl"))
    if race_nn is not None:
        joblib.dump(
            {
                "state": race_nn.model.state_dict(),
                "mean": race_nn.mean_,
                "std": race_nn.std_,
                "cols": race_nn.feature_cols_,
                "include_odds": race_nn.include_odds,
            },
            os.path.join(out_dir, "race_nn.pkl"),
        )
    pred_out = test_df[["race_id", "馬番", "着順", "オッズ"]].copy()
    pred_out["p_win"] = test_meta["p_win"].to_numpy()
    pred_out["p_final"] = p_final
    pred_out["p_market"] = p_mkt
    pred_out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    bets.to_csv(os.path.join(out_dir, "bets.csv"), index=False)
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(pd.Series(result).to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description="ROI パイプライン")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--skip-nn", action="store_true")
    parser.add_argument("--skip-cat", action="store_true")
    parser.add_argument("--ev-threshold", type=float, default=0.05)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--base-years", type=int, default=6)
    parser.add_argument("--meta-years", type=int, default=3)
    parser.add_argument("--calib-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    if not args.train and not args.predict:
        parser.error("--train または --predict を指定してください")

    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_encoded()
    years = sorted(df["year"].dropna().astype(int).unique())
    windows = walk_forward_windows(
        years,
        base_years=args.base_years,
        meta_years=args.meta_years,
        calib_years=args.calib_years,
        test_years=args.test_years,
    )
    if args.max_windows:
        windows = windows[-args.max_windows :]
    print(f"walk-forward windows: {len(windows)}")
    rows = []
    if args.train:
        for window in windows:
            row = run_window(
                df,
                window,
                skip_nn=args.skip_nn,
                skip_cat=args.skip_cat,
                ev_threshold=args.ev_threshold,
                kelly_fraction=args.kelly_fraction,
            )
            if row:
                rows.append(row)
    elif args.predict:
        latest = os.path.join(MODEL_DIR, "walk_forward_metrics.csv")
        if not os.path.exists(latest):
            raise SystemExit("学習済み metrics がありません。先に --train を実行してください")
        print(pd.read_csv(latest).to_string(index=False))
    if rows:
        table = pd.DataFrame(rows)
        path = os.path.join(MODEL_DIR, "walk_forward_metrics.csv")
        table.to_csv(path, index=False)
        print(f"\n比較表: {path}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
