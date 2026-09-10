"""
OOF 時系列スタッキング。

4観点 LightGBM の OOF 確率をロジスティック回帰で結合する。
全特徴 LightGBM をベースラインとして同じテストで比較する。

使用方法（リポジトリルートから）:
    python train/stacking_model.py --train --predict
    python train/stacking_model.py --train --predict --ablation
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAIN_DIR)

import column_select
import metrics
from train_utils import (
    binary_target,
    fit_lightgbm,
    oof_predict,
    predict_proba_positive,
)

VIEWS = ["horse_ability", "horse_condition", "human_bias", "race_condition"]
MODEL_DIR = "./model"
META_FEATURES = [f"lgb_{view}" for view in VIEWS]


def _ensure_dirs():
    os.makedirs(f"{MODEL_DIR}/lightgbm", exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/stacking", exist_ok=True)


def _artifact_suffix(include_odds):
    return "" if include_odds else "_no_odds"


def load_train_test():
    train = pd.read_csv("./datasets/encoded/encoded_traindata.csv")
    test = pd.read_csv("./datasets/encoded/encoded_testdata.csv")
    return train, test


def make_meta_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            ),
        ]
    )


class StackingModel:
    def __init__(self, include_odds=True, n_splits=5):
        self.include_odds = include_odds
        self.n_splits = n_splits
        self.sub_models = {}
        self.full_model = None
        self.meta_model = None
        self.meta_features = META_FEATURES

    def _view_X(self, df, view):
        return column_select.select_features(df, view, include_odds=self.include_odds)

    def generate_oof_features(self, df, y):
        meta = pd.DataFrame(index=df.index)
        for view in VIEWS:
            print(f"  OOF: {view}")
            X = self._view_X(df, view)
            meta[f"lgb_{view}"] = oof_predict(df, X, y, n_splits=self.n_splits)
        return meta

    def generate_features(self, df, mode="predict"):
        del mode
        features = {}
        for view in VIEWS:
            name = f"lightgbm_{view}"
            if name not in self.sub_models:
                continue
            X = self._view_X(df, view)
            features[f"lgb_{view}"] = predict_proba_positive(self.sub_models[name], X)
        return pd.DataFrame(features, index=df.index)

    def load_sub_models(self):
        suffix = _artifact_suffix(self.include_odds)
        print("サブモデルを読み込み中...")
        for view in VIEWS:
            path = f"{MODEL_DIR}/lightgbm/lightgbm_{view}{suffix}.pkl"
            if os.path.exists(path):
                self.sub_models[f"lightgbm_{view}"] = joblib.load(path)
                print(f"  ✓ lightgbm_{view}: {path}")
            else:
                print(f"  ✗ lightgbm_{view}: {path} (存在しない)")
        full_path = f"{MODEL_DIR}/lightgbm/lightgbm_full{suffix}.pkl"
        if os.path.exists(full_path):
            self.full_model = joblib.load(full_path)
            print(f"  ✓ lightgbm_full: {full_path}")
        meta_path = f"{MODEL_DIR}/stacking/meta_model{suffix}.pkl"
        if os.path.exists(meta_path):
            payload = joblib.load(meta_path)
            if isinstance(payload, dict):
                self.meta_model = payload.get("meta_model")
                self.include_odds = payload.get("include_odds", self.include_odds)
            else:
                self.meta_model = payload
        print(f"読み込んだモデル数: {len(self.sub_models)}")

    def train(self, data):
        _ensure_dirs()
        suffix = _artifact_suffix(self.include_odds)
        y = pd.Series(binary_target(data["着順"]), index=data.index, name="y")
        odds_label = "あり" if self.include_odds else "なし"
        print(f"OOF メタ特徴量を生成中（オッズ・人気: {odds_label}）...")
        meta_oof = self.generate_oof_features(data, y)
        valid = meta_oof.notna().all(axis=1)
        print(f"  OOF 有効行: {int(valid.sum())} / {len(data)}")

        print("全特徴 LightGBM の OOF...")
        X_full = self._view_X(data, "full")
        full_oof = oof_predict(data, X_full, y, n_splits=self.n_splits)

        print("メタモデル（LogisticRegression）を学習...")
        self.meta_model = make_meta_model()
        self.meta_model.fit(meta_oof.loc[valid], y.loc[valid])
        meta_oof_pred = pd.Series(np.nan, index=data.index)
        meta_oof_pred.loc[valid] = self.meta_model.predict_proba(meta_oof.loc[valid])[:, 1]

        print("全 train でベースモデルを再学習...")
        self.sub_models = {}
        for view in VIEWS:
            model = fit_lightgbm(self._view_X(data, view), y)
            name = f"lightgbm_{view}"
            self.sub_models[name] = model
            joblib.dump(model, f"{MODEL_DIR}/lightgbm/lightgbm_{view}{suffix}.pkl")
            print(f"  saved {name}")
        self.full_model = fit_lightgbm(X_full, y)
        joblib.dump(self.full_model, f"{MODEL_DIR}/lightgbm/lightgbm_full{suffix}.pkl")

        payload = {
            "meta_model": self.meta_model,
            "include_odds": self.include_odds,
            "views": VIEWS,
        }
        joblib.dump(payload, f"{MODEL_DIR}/stacking/meta_model{suffix}.pkl")
        joblib.dump(meta_oof, f"{MODEL_DIR}/stacking/meta_features{suffix}.pkl")

        oof_out = meta_oof.copy()
        oof_out["lgb_full"] = full_oof
        oof_out["lgb_stack"] = meta_oof_pred
        oof_out["actual"] = y.to_numpy()
        oof_out["race_id"] = data["race_id"].to_numpy()
        oof_out.to_csv(f"{MODEL_DIR}/stacking/oof_predictions{suffix}.csv", index=False)
        print(f"メタモデルを保存しました: {MODEL_DIR}/stacking/meta_model{suffix}.pkl")
        return oof_out

    def predict(self, data):
        if not self.sub_models:
            self.load_sub_models()
        if self.meta_model is None:
            suffix = _artifact_suffix(self.include_odds)
            payload = joblib.load(f"{MODEL_DIR}/stacking/meta_model{suffix}.pkl")
            if isinstance(payload, dict):
                self.meta_model = payload["meta_model"]
            else:
                self.meta_model = payload
        meta_features = self.generate_features(data)
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def predict_all(self, data):
        if not self.sub_models:
            self.load_sub_models()
        out = self.generate_features(data)
        out["lgb_stack"] = self.meta_model.predict_proba(out[META_FEATURES])[:, 1]
        if self.full_model is None:
            suffix = _artifact_suffix(self.include_odds)
            path = f"{MODEL_DIR}/lightgbm/lightgbm_full{suffix}.pkl"
            if os.path.exists(path):
                self.full_model = joblib.load(path)
        if self.full_model is not None:
            out["lgb_full"] = predict_proba_positive(
                self.full_model, self._view_X(data, "full")
            )
        return out


def _eval_frame(data, pred_df, include_odds):
    y = binary_target(data["着順"])
    finish = pd.to_numeric(data["着順"], errors="coerce")
    rows = []
    for col in pred_df.columns:
        rows.append(
            {
                **metrics.evaluate_predictions(
                    y,
                    pred_df[col],
                    race_id=data["race_id"],
                    finish=finish,
                    name=col,
                ),
                "include_odds": include_odds,
            }
        )
    return rows


def train_and_eval(include_odds, n_splits, do_train, do_predict):
    stacking = StackingModel(include_odds=include_odds, n_splits=n_splits)
    train_df, test_df = load_train_test()
    rows = []
    suffix = _artifact_suffix(include_odds)

    if do_train:
        print("=" * 50)
        print(f"スタッキング学習 odds={include_odds}")
        print("=" * 50)
        oof = stacking.train(train_df)
        pred_cols = [c for c in oof.columns if c.startswith("lgb_")]
        mask = oof[pred_cols].notna().all(axis=1)
        oof_rows = _eval_frame(
            train_df.loc[mask],
            oof.loc[mask, pred_cols],
            include_odds,
        )
        for row in oof_rows:
            row["split"] = "oof_train"
        rows.extend(oof_rows)

    if do_predict:
        print("=" * 50)
        print(f"テスト予測 odds={include_odds}")
        print("=" * 50)
        if not stacking.sub_models:
            stacking.load_sub_models()
        pred_df = stacking.predict_all(test_df)
        y = binary_target(test_df["着順"])
        result = pd.DataFrame(
            {
                "race_id": test_df["race_id"],
                "horse_id": test_df["馬"],
                "prediction": pred_df["lgb_stack"].to_numpy(),
                "actual": y.to_numpy(),
                "finish": pd.to_numeric(test_df["着順"], errors="coerce"),
            }
        )
        for col in pred_df.columns:
            result[col] = pred_df[col].to_numpy()
        out_path = f"{MODEL_DIR}/stacking/predictions{suffix}.csv"
        result.to_csv(out_path, index=False)
        print(f"予測結果を保存しました: {out_path}")
        test_rows = _eval_frame(test_df, pred_df, include_odds)
        for row in test_rows:
            row["split"] = "test_2024"
        rows.extend(test_rows)
        table = metrics.format_metrics_table(test_rows)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return rows


def main(args):
    _ensure_dirs()
    odds_settings = [True, False] if args.ablation else [not args.no_odds]
    all_rows = []
    for include_odds in odds_settings:
        all_rows.extend(
            train_and_eval(
                include_odds=include_odds,
                n_splits=args.n_splits,
                do_train=args.train,
                do_predict=args.predict,
            )
        )
    if all_rows:
        table = metrics.format_metrics_table(all_rows)
        path = f"{MODEL_DIR}/stacking/comparison.csv"
        table.to_csv(path, index=False)
        print(f"\n比較表: {path}")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOF スタッキング予測モデル")
    parser.add_argument("--train", action="store_true", help="OOF メタモデルを学習")
    parser.add_argument("--predict", action="store_true", help="テストデータで評価")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="オッズ・人気あり/なしの両方を学習・評価",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="現レースのオッズ・人気を使わない",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    if not args.train and not args.predict:
        parser.error("--train または --predict を指定してください")
    main(args)
