"""
生レース結果を学習・推論用の encoded CSV に変換する。

- 2011〜2024 をまとめて処理し、近走特徴が年をまたげるようにする
- 走破時間の標準化パラメータは train（既定: year <= 2023）だけで推定
- 日付は固定原点 2010-01-01 からの日数
- 騎手勝率は当該レースより前の成績のみ（expanding）
- カテゴリ encoders は train で fit し、未知カテゴリは専用コード
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd


DATE_ORIGIN = pd.Timestamp("2010-01-01")
CATEGORICAL_FEATURES = [
    "馬",
    "騎手",
    "レース名",
    "開催",
    "場名",
    "騎手1",
    "騎手2",
    "騎手3",
    "騎手4",
    "騎手5",
]
DATE_COLUMNS = ["日付", "日付1", "日付2", "日付3", "日付4", "日付5"]
PAST_FEATURES = [
    "馬番",
    "騎手",
    "斤量",
    "オッズ",
    "体重",
    "体重変化",
    "上がり",
    "通過順",
    "着順",
    "距離",
    "クラス",
    "走破時間",
    "芝・ダート",
    "天気",
    "馬場",
]


def class_mapping(row):
    mappings = {
        "障害": 0,
        "G1": 10,
        "G2": 9,
        "G3": 8,
        "(L)": 7,
        "オープン": 7,
        "OP": 7,
        "3勝": 6,
        "1600": 6,
        "2勝": 5,
        "1000": 5,
        "1勝": 4,
        "500": 4,
        "新馬": 3,
        "未勝利": 1,
    }
    for key, value in mappings.items():
        if key in str(row):
            return value
    return 0


def parse_race_time_seconds(series):
    time_parts = series.astype(str).str.split(":", expand=True)
    minutes = pd.to_numeric(time_parts[0], errors="coerce")
    if time_parts.shape[1] < 2:
        return minutes
    rest = time_parts[1].str.split(".", expand=True)
    sec = pd.to_numeric(rest[0], errors="coerce")
    tenth = pd.to_numeric(rest[1], errors="coerce") if rest.shape[1] > 1 else 0
    return minutes * 60 + sec + tenth.fillna(0) / 10


def load_raw(dirname, year_start, year_end):
    frames = []
    print("ファイル取得：開始")
    for year in range(year_start, year_end + 1):
        path = os.path.join(dirname, "raw_data", f"{year}.csv")
        data = pd.read_csv(path, encoding="SHIFT-JIS", header=0)
        data["日付"] = pd.to_datetime(data["日付"], format="%Y年%m月%d日")
        data["着順"] = pd.to_numeric(data["着順"], errors="coerce")
        data = data.dropna(subset=["着順"])
        data["着順"] = data["着順"].astype(int)
        data = data[data["芝・ダート"] == "芝"]
        frames.append(data)
    print("ファイル取得：完了")
    return pd.concat(frames, ignore_index=True)


def fit_time_scaler(seconds_train):
    mean1 = seconds_train.mean()
    std1 = seconds_train.std()
    if std1 == 0 or np.isnan(std1):
        std1 = 1.0
    z = -((seconds_train - mean1) / std1)
    z = z.clip(lower=-3, upper=2)
    z = z.mask(z > 2.5, 2)
    mean2 = z.mean()
    std2 = z.std()
    if std2 == 0 or np.isnan(std2):
        std2 = 1.0
    return {
        "mean1": float(mean1),
        "std1": float(std1),
        "mean2": float(mean2),
        "std2": float(std2),
    }


def apply_time_scaler(seconds, params):
    z = -((seconds - params["mean1"]) / params["std1"])
    z = z.clip(lower=-3, upper=2)
    z = z.mask(z > 2.5, 2)
    return (z - params["mean2"]) / params["std2"]


def add_past_runs(df):
    df = df.sort_values(by=["馬", "日付"], ascending=[True, False]).copy()
    for i in range(1, 6):
        df[f"日付{i}"] = df.groupby("馬")["日付"].shift(-i)
        for feature in PAST_FEATURES:
            df[f"{feature}{i}"] = df.groupby("馬")[feature].shift(-i)
    for feature in PAST_FEATURES:
        for i in range(1, 6):
            df[f"{feature}{i}"] = df.groupby("馬")[f"{feature}{i}"].ffill()
    df = df.groupby(["race_id", "馬"], as_index=False).last()
    return df.sort_values(by=["日付", "race_id"], ascending=True)


def add_diffs_and_weight(df):
    df = df.assign(
        距離差=df["距離"] - df["距離1"],
        日付差=(df["日付"] - df["日付1"]).dt.days,
        距離差1=df["距離1"] - df["距離2"],
        日付差1=(df["日付1"] - df["日付2"]).dt.days,
        距離差2=df["距離2"] - df["距離3"],
        日付差2=(df["日付2"] - df["日付3"]).dt.days,
        距離差3=df["距離3"] - df["距離4"],
        日付差3=(df["日付3"] - df["日付4"]).dt.days,
        距離差4=df["距離4"] - df["距離5"],
        日付差4=(df["日付4"] - df["日付5"]).dt.days,
    )
    kinryo_columns = ["斤量", "斤量1", "斤量2", "斤量3", "斤量4", "斤量5"]
    for col in kinryo_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["平均斤量"] = df[kinryo_columns].mean(axis=1)
    return df


def add_jockey_win_rate(df):
    """当該行より前の出走だけで勝率を計算する。同一レースの結果は含めない。"""
    df = df.sort_values(["日付", "race_id"]).copy()
    df["_win"] = (df["着順"] == 1).astype(int)
    race_jockey = (
        df.groupby(["race_id", "騎手"], as_index=False)
        .agg(race_date=("日付", "first"), wins=("_win", "sum"), starts=("_win", "size"))
        .sort_values(["race_date", "race_id"])
    )
    race_jockey["prev_wins"] = race_jockey.groupby("騎手")["wins"].cumsum() - race_jockey["wins"]
    race_jockey["prev_starts"] = race_jockey.groupby("騎手")["starts"].cumsum() - race_jockey["starts"]
    race_jockey["騎手の勝率"] = race_jockey["prev_wins"] / race_jockey["prev_starts"].replace(0, np.nan)
    df = df.drop(columns=["_win"])
    df = df.merge(
        race_jockey[["race_id", "騎手", "騎手の勝率"]],
        on=["race_id", "騎手"],
        how="left",
    )
    return df


def dates_to_ordinal(df):
    year = df["日付"].dt.year
    for col in DATE_COLUMNS:
        df[col] = (df[col] - DATE_ORIGIN).dt.days
    df["year"] = year
    return df


def fit_encoders(train_df):
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        values = train_df[col].where(train_df[col].notna(), "__NA__").astype(str)
        classes = pd.Index(values.unique())
        mapping = {c: i for i, c in enumerate(classes)}
        encoders[col] = {"mapping": mapping, "unknown": len(mapping)}
    return encoders


def apply_encoders(df, encoders):
    df = df.copy()
    for col, spec in encoders.items():
        values = df[col].where(df[col].notna(), "__NA__").astype(str)
        df[col] = values.map(spec["mapping"]).fillna(spec["unknown"]).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="レースデータの前処理")
    parser.add_argument("--year-start", type=int, default=2011)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--train-year-end", type=int, default=2023)
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    df = load_raw(dirname, args.year_start, args.year_end)

    print("データ変換：開始")
    seconds = parse_race_time_seconds(df["走破時間"]).ffill()
    train_mask_time = df["日付"].dt.year <= args.train_year_end
    time_params = fit_time_scaler(seconds[train_mask_time])
    df["走破時間"] = apply_time_scaler(seconds, time_params)

    pas = df["通過順"].astype(str).str.split("-", expand=True)
    df["通過順"] = pas.apply(lambda col: pd.to_numeric(col, errors="coerce")).mean(axis=1)

    df["クラス"] = df["クラス"].apply(class_mapping)
    df["性"] = df["性"].map({"牡": 0, "牝": 1, "セ": 2})
    df["芝・ダート"] = df["芝・ダート"].map({"芝": 0, "ダ": 1, "障": 2})
    df["回り"] = df["回り"].map({"右": 0, "左": 1, "芝": 2, "直": 2})
    df["馬場"] = df["馬場"].map({"良": 0, "稍": 1, "重": 2, "不": 3})
    df["天気"] = df["天気"].map({"晴": 0, "曇": 1, "小": 2, "雨": 3, "雪": 4})
    print("データ変換：完了")

    print("近5走取得：開始")
    df = add_past_runs(df)
    print("近5走取得：終了")

    df = df.replace("---", np.nan)
    print("日付変換：開始")
    df = add_diffs_and_weight(df)
    df = add_jockey_win_rate(df)
    df = dates_to_ordinal(df)
    print("日付変換：終了")

    train_mask = df["year"] <= args.train_year_end
    print("エンコーディング：開始")
    encoders = fit_encoders(df.loc[train_mask])
    df = apply_encoders(df, encoders)
    print("エンコーディング：終了")

    encoded_dir = os.path.join(dirname, "encoded")
    os.makedirs(encoded_dir, exist_ok=True)
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()
    train_path = os.path.join(encoded_dir, "encoded_traindata.csv")
    test_path = os.path.join(encoded_dir, "encoded_testdata.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "time_params": time_params,
        "encoders": encoders,
        "date_origin": str(DATE_ORIGIN.date()),
        "train_year_end": args.train_year_end,
        "year_start": args.year_start,
        "year_end": args.year_end,
    }
    meta_path = os.path.join(encoded_dir, "preprocess_meta.pkl")
    joblib.dump(meta, meta_path)

    print("ファイル出力：終了")
    print(f"  train: {train_path} ({len(train_df)} 行)")
    print(f"  test:  {test_path} ({len(test_df)} 行)")
    print(f"  meta:  {meta_path}")


if __name__ == "__main__":
    main()
