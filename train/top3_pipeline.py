"""
3着以内の補助パイプライン: ファンダメンタル → メタ → 必要なら市場混合 → 診断出力。

使用方法（リポジトリルートから）:
    python train/top3_pipeline.py --train --predict
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
from market_roi import bootstrap_roi, roi_summary, split_calibration_races
from top3_base_models import Top3FundamentalModels
from top3_market import Top3MarketIntegrator, pick_place_threshold, top3_triple_heuristics
from top3_meta_model import Top3MetaModel
from train_utils import binary_target
from walk_forward import mask_years, walk_forward_windows


MODEL_DIR = "./model/top3"
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


def run_window(df, window, prob_threshold):
    print("=" * 60)
    print(
        f"top3 window {window['name']}: base={window['base']} meta={window['meta']} "
        f"calib={window['calib']} test={window['test']}"
    )
    base_df = df.loc[mask_years(df, window["base"])]
    meta_df = df.loc[mask_years(df, window["meta"])]
    calib_df = df.loc[mask_years(df, window["calib"])]
    test_df = df.loc[mask_years(df, window["test"])]
    if min(len(base_df), len(meta_df), len(calib_df), len(test_df)) == 0:
        print("  skip: empty split")
        return None

    print("  Stage1 top3 ベース...")
    base = Top3FundamentalModels(include_odds=False)
    base.fit(base_df)

    print("  Stage2 メタ学習...")
    meta_pred = base.predict_frame(meta_df)
    meta = Top3MetaModel()
    meta.fit(meta_df, meta_pred)
    print(f"    chosen meta={meta.chosen}")

    print("  Stage3 市場混合 / 閾値...")
    calib_pred = base.predict_frame(calib_df)
    calib_meta = meta.predict(calib_df, calib_pred)
    calib_head, calib_tail = split_calibration_races(calib_df)
    y_top3 = binary_target(calib_head["着順"]).to_numpy()
    market = Top3MarketIntegrator(prob_threshold=prob_threshold)
    market.fit(calib_head, calib_meta.loc[calib_head.index, "p_top3"].to_numpy(), y_top3)
    chosen_thr, threshold_fallback = pick_place_threshold(
        market,
        calib_tail,
        calib_meta.loc[calib_tail.index, "p_top3"].to_numpy(),
        fallback=prob_threshold,
    )
    market.prob_threshold = chosen_thr
    if market.weights_ is not None:
        print(f"    blend weights={market.weights_}")
    else:
        print("    no place odds: identity blend")
    print(f"    prob_threshold={chosen_thr} (fallback={threshold_fallback})")

    print("  テスト予測...")
    test_pred = base.predict_frame(test_df)
    test_meta = meta.predict(test_df, test_pred)
    p_final, p_mkt = market.predict(test_df, test_meta["p_top3"].to_numpy())
    payouts = load_payouts(window["test"])
    bets = market.diagnostic_bets(test_df, p_final, payouts=payouts)
    summary = roi_summary(bets.rename(columns={"hit": "win"}))
    boot = bootstrap_roi(bets.rename(columns={"hit": "win"}))
    y_test = binary_target(test_df["着順"]).to_numpy()
    ev_model = metrics.evaluate_predictions(y_test, test_meta["p_top3"], test_df["race_id"], name="model")
    ev_final = metrics.evaluate_predictions(y_test, p_final, test_df["race_id"], name="final")
    race_sum = pd.Series(p_final, index=test_df.index).groupby(test_df["race_id"]).sum()
    result = {
        "window": window["name"],
        "meta_backend": meta.chosen,
        "w_model_blend": float(market.weights_[0]) if market.weights_ is not None else np.nan,
        "w_market_blend": float(market.weights_[1]) if market.weights_ is not None and len(market.weights_) > 1 else np.nan,
        "prob_threshold": float(chosen_thr),
        "threshold_fallback": int(threshold_fallback),
        "logloss_model": ev_model["logloss"],
        "logloss_final": ev_final["logloss"],
        "brier_final": ev_final["brier"],
        "mean_p_top3_sum_per_race": float(race_sum.mean()),
        **{f"roi_{k}": v for k, v in summary.items()},
        **boot,
    }
    out_dir = os.path.join(MODEL_DIR, window["name"])
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(base, os.path.join(out_dir, "top3_base.pkl"))
    joblib.dump(meta, os.path.join(out_dir, "top3_meta.pkl"))
    joblib.dump(market, os.path.join(out_dir, "top3_market.pkl"))

    pred_out = test_df[["race_id", "馬番", "着順", "オッズ"]].copy()
    if "頭数" in test_df.columns:
        pred_out["頭数"] = test_df["頭数"]
    pred_out["p_top3"] = test_meta["p_top3"].to_numpy()
    pred_out["p_final"] = p_final
    if p_mkt is not None:
        pred_out["p_market_place"] = p_mkt
    pred_out["race_p_top3_sum"] = pred_out.groupby("race_id")["p_final"].transform("sum")
    pred_out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    triples = top3_triple_heuristics(test_df, p_final, top_n=5)
    triples.to_csv(os.path.join(out_dir, "triple_heuristics.csv"), index=False)
    bets.to_csv(os.path.join(out_dir, "diagnostic_bets.csv"), index=False)
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(pd.Series(result).to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description="3着以内 補助パイプライン")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--prob-threshold", type=float, default=0.25)
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
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["日付"], errors="coerce").dt.year
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
            row = run_window(df, window, prob_threshold=args.prob_threshold)
            if row:
                rows.append(row)
    elif args.predict:
        path = os.path.join(MODEL_DIR, "walk_forward_metrics.csv")
        if not os.path.exists(path):
            raise SystemExit("学習済み metrics がありません。先に --train を実行してください")
        print(pd.read_csv(path).to_string(index=False))
    if rows:
        table = pd.DataFrame(rows)
        path = os.path.join(MODEL_DIR, "walk_forward_metrics.csv")
        table.to_csv(path, index=False)
        print(f"\n比較表: {path}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()