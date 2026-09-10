"""
スタッキングモデルを使用した予測

使用方法:
    python train/predict_with_stacking.py \
      --input ./datasets/encoded/encoded_testdata.csv \
      --output ./model/stacking/predictions.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stacking_model import StackingModel


def horse_id_column(data):
    if "馬" in data.columns:
        return data["馬"]
    if "horse_id" in data.columns:
        return data["horse_id"]
    raise KeyError("入力CSVに 馬 / horse_id 列がありません")


def main():
    parser = argparse.ArgumentParser(description="スタッキングモデルを使用した予測")
    parser.add_argument(
        "--input",
        help="入力CSVファイルのパス",
        default="./datasets/encoded/encoded_testdata.csv",
    )
    parser.add_argument(
        "--output",
        help="出力ファイルパス",
        default="./model/stacking/predictions.csv",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="オッズなしで学習したメタモデルを使う",
    )
    args = parser.parse_args()

    stacking = StackingModel(include_odds=not args.no_odds)
    stacking.load_sub_models()

    if len(stacking.sub_models) == 0:
        print("エラー: サブモデルが読み込まれませんでした")
        return

    data = pd.read_csv(args.input)
    print(f"入力データ読み込み完了: {len(data)} 行")

    predictions = stacking.predict(data)
    print("予測完了")

    result = pd.DataFrame(
        {
            "race_id": data["race_id"],
            "horse_id": horse_id_column(data),
            "prediction": predictions,
        }
    )
    if "着順" in data.columns:
        finish = pd.to_numeric(data["着順"], errors="coerce")
        # 生の着順（1,2,3,...）を想定。既に 0/1 の場合はそのまま残す。
        if finish.max() > 1:
            result["actual"] = (finish < 4).astype(int)
        else:
            result["actual"] = finish.astype(int)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\n予測結果を保存しました: {args.output}")
    print(f"  平均予測値: {predictions.mean():.4f}")
    print(f"  最大予測値: {predictions.max():.4f}")
    print(f"  最小予測値: {predictions.min():.4f}")


if __name__ == "__main__":
    main()
