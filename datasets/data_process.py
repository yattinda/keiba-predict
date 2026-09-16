"""
生レース結果を学習・推論用の encoded CSV に変換する。

- 全馬場を読み、近走特徴を作ったあとに芝へ絞る（近走のダート・障害を残す）
- スピード指数は距離 × 場id × 馬場の z-score（train 期間だけで fit）
- 頭数・レース内相対特徴・リークなし expanding 集計を付与
- 日付は固定原点 2010-01-01 からの日数
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
    "調教師",
    "馬主",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "父",
    "母父",
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
    "スピード指数",
    "枠番",
    "着差",
    "調教師",
]
RELATIVE_BASE_COLS = [
    "スピード指数1",
    "スピード指数2",
    "スピード指数3",
    "着順1",
    "着順2",
    "着順3",
    "騎手の勝率",
    "斤量",
    "体重",
    "日付差",
    "馬の複勝率",
    "騎手のコース複勝率",
    "調教師の複勝率",
]
OPTIONAL_NUMERIC = ["枠番", "着差", "賞金"]
OPTIONAL_CATEGORICAL = [
    "調教師",
    "馬主",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "父",
    "母父",
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


def parse_margin(series):
    mapping = {
        "同着": 0.0,
        "ハナ": 0.1,
        "アタマ": 0.2,
        "クビ": 0.3,
        "1/2": 0.5,
        "1/4": 0.25,
        "3/4": 0.75,
    }

    def _one(val):
        if pd.isna(val):
            return np.nan
        text = str(val).strip()
        if text in ("", "---"):
            return np.nan
        if text in mapping:
            return mapping[text]
        for key, num in mapping.items():
            if key in text:
                prefix = text.replace(key, "").replace(".", "").strip()
                base = pd.to_numeric(prefix, errors="coerce")
                return (0.0 if pd.isna(base) else float(base)) + num
        return pd.to_numeric(text, errors="coerce")

    return series.map(_one)


def distance_band(distance):
    d = pd.to_numeric(distance, errors="coerce")
    return np.select(
        [d <= 1400, d <= 1800, d <= 2200],
        [0, 1, 2],
        default=3,
    )


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
        frames.append(data)
    print("ファイル取得：完了")
    return pd.concat(frames, ignore_index=True)


def ensure_optional_columns(df):
    for col in OPTIONAL_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
    for col in OPTIONAL_CATEGORICAL:
        if col not in df.columns:
            df[col] = np.nan
    return df


def fit_speed_figure_params(seconds, distance, venue, going, train_mask):
    tmp = pd.DataFrame(
        {
            "seconds": seconds,
            "距離": pd.to_numeric(distance, errors="coerce"),
            "場id": venue,
            "馬場": going,
        }
    )
    train = tmp.loc[train_mask]
    stats = (
        train.groupby(["距離", "場id", "馬場"], dropna=False)["seconds"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(1.0).replace(0, 1.0)
    stats.loc[stats["n"] < 20, ["mean", "std"]] = pd.NA
    global_mean = float(train["seconds"].mean())
    global_std = float(train["seconds"].std() or 1.0)
    return {
        "stats": stats,
        "global_mean": global_mean,
        "global_std": global_std if global_std else 1.0,
    }


def apply_speed_figure(seconds, distance, venue, going, params):
    tmp = pd.DataFrame(
        {
            "seconds": seconds.to_numpy(),
            "距離": pd.to_numeric(distance, errors="coerce").to_numpy(),
            "場id": venue.to_numpy() if hasattr(venue, "to_numpy") else venue,
            "馬場": going.to_numpy() if hasattr(going, "to_numpy") else going,
        }
    )
    merged = tmp.merge(params["stats"], on=["距離", "場id", "馬場"], how="left")
    mean = merged["mean"].fillna(params["global_mean"])
    std = merged["std"].fillna(params["global_std"]).replace(0, 1.0)
    z = -(merged["seconds"] - mean) / std
    return z.clip(lower=-4, upper=4).to_numpy()


def add_past_runs(df):
    past_features = [c for c in PAST_FEATURES if c in df.columns]
    df = df.sort_values(by=["馬", "日付"], ascending=[True, False]).copy()
    entity = "horse_id" if df["horse_id"].notna().any() else "馬"
    for i in range(1, 6):
        df[f"日付{i}"] = df.groupby(entity)["日付"].shift(-i)
        for feature in past_features:
            df[f"{feature}{i}"] = df.groupby(entity)[feature].shift(-i)
    for feature in past_features:
        for i in range(1, 6):
            df[f"{feature}{i}"] = df.groupby(entity)[f"{feature}{i}"].ffill()
    df = df.groupby(["race_id", "馬"], as_index=False).last()
    return df.sort_values(by=["日付", "race_id"], ascending=True).copy()


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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    present = [c for c in kinryo_columns if c in df.columns]
    df["平均斤量"] = df[present].mean(axis=1)
    df["休養明け"] = (pd.to_numeric(df["日付差"], errors="coerce") >= 90).astype(float)
    return df


def _race_entity_expanding(df, entity_cols, value_cols):
    keys = ["race_id"] + list(entity_cols)
    race_ent = (
        df.groupby(keys, as_index=False)
        .agg(
            race_date=("日付", "first"),
            **{name: (src, "sum") for name, src in value_cols.items()},
        )
        .sort_values(["race_date", "race_id"])
    )
    grouped = race_ent.groupby(list(entity_cols), dropna=False)
    out = race_ent[keys].copy()
    for name in value_cols:
        out[f"prev_{name}"] = grouped[name].cumsum() - race_ent[name]
    return out


def add_jockey_win_rate(df):
    df = df.sort_values(["日付", "race_id"]).copy()
    df["_win"] = (df["着順"] == 1).astype(int)
    df["_start"] = 1
    stats = _race_entity_expanding(df, ["騎手"], {"wins": "_win", "starts": "_start"})
    stats["騎手の勝率"] = stats["prev_wins"] / stats["prev_starts"].replace(0, np.nan)
    df = df.drop(columns=["_win", "_start"])
    return df.merge(stats[["race_id", "騎手", "騎手の勝率"]], on=["race_id", "騎手"], how="left")


def add_expanding_form_stats(df):
    df = df.sort_values(["日付", "race_id"]).copy()
    df["_win"] = (df["着順"] == 1).astype(int)
    df["_top3"] = (df["着順"] <= 3).astype(int)
    df["_start"] = 1
    horse_key = "horse_id" if df["horse_id"].notna().any() else "馬"

    horse = _race_entity_expanding(
        df, [horse_key], {"wins": "_win", "top3": "_top3", "starts": "_start"}
    )
    horse["馬の勝率"] = horse["prev_wins"] / horse["prev_starts"].replace(0, np.nan)
    horse["馬の複勝率"] = horse["prev_top3"] / horse["prev_starts"].replace(0, np.nan)
    horse["馬の出走数"] = horse["prev_starts"]
    df = df.merge(
        horse[["race_id", horse_key, "馬の勝率", "馬の複勝率", "馬の出走数"]],
        on=["race_id", horse_key],
        how="left",
    )

    df["_距離帯"] = distance_band(df["距離"])
    band = _race_entity_expanding(
        df,
        [horse_key, "_距離帯"],
        {"top3": "_top3", "starts": "_start"},
    )
    band["馬の距離帯複勝率"] = band["prev_top3"] / band["prev_starts"].replace(0, np.nan)
    df = df.merge(
        band[["race_id", horse_key, "_距離帯", "馬の距離帯複勝率"]],
        on=["race_id", horse_key, "_距離帯"],
        how="left",
    )

    layoff = df.loc[df["休養明け"] == 1].copy()
    if len(layoff):
        rest = _race_entity_expanding(
            layoff, [horse_key], {"top3": "_top3", "starts": "_start"}
        )
        rest["馬の休養明け複勝率"] = rest["prev_top3"] / rest["prev_starts"].replace(0, np.nan)
        df = df.merge(
            rest[["race_id", horse_key, "馬の休養明け複勝率"]],
            on=["race_id", horse_key],
            how="left",
        )
    else:
        df["馬の休養明け複勝率"] = np.nan

    course = _race_entity_expanding(
        df, ["騎手", "場id"], {"top3": "_top3", "starts": "_start"}
    )
    course["騎手のコース複勝率"] = course["prev_top3"] / course["prev_starts"].replace(0, np.nan)
    df = df.merge(
        course[["race_id", "騎手", "場id", "騎手のコース複勝率"]],
        on=["race_id", "騎手", "場id"],
        how="left",
    )

    if df["調教師"].notna().any():
        trainer = _race_entity_expanding(
            df, ["調教師"], {"top3": "_top3", "starts": "_start"}
        )
        trainer["調教師の複勝率"] = trainer["prev_top3"] / trainer["prev_starts"].replace(
            0, np.nan
        )
        df = df.merge(
            trainer[["race_id", "調教師", "調教師の複勝率"]],
            on=["race_id", "調教師"],
            how="left",
        )
    else:
        df["調教師の複勝率"] = np.nan

    df = df.drop(columns=["_win", "_top3", "_start", "_距離帯"])
    return df


def add_relative_features(df):
    df = df.copy()
    cols = [c for c in RELATIVE_BASE_COLS if c in df.columns]
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        g = values.groupby(df["race_id"])
        df[f"{col}_race_rank"] = g.rank(method="average", ascending=False, na_option="keep")
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        df[f"{col}_race_z"] = (values - mean) / std
        df[f"{col}_race_gap"] = values - g.transform("max")
    return df


def dates_to_ordinal(df):
    year = df["日付"].dt.year
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = (df[col] - DATE_ORIGIN).dt.days
    df["year"] = year
    df["距離帯"] = distance_band(df["距離"])
    return df


def fit_encoders(train_df):
    encoders = {}
    cols = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]
    for col in cols:
        values = train_df[col].where(train_df[col].notna(), "__NA__").astype(str)
        classes = pd.Index(values.unique())
        mapping = {c: i for i, c in enumerate(classes)}
        encoders[col] = {"mapping": mapping, "unknown": len(mapping)}
    return encoders


def apply_encoders(df, encoders):
    df = df.copy()
    for col, spec in encoders.items():
        if col not in df.columns:
            continue
        values = df[col].where(df[col].notna(), "__NA__").astype(str)
        df[col] = values.map(spec["mapping"]).fillna(spec["unknown"]).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="レースデータの前処理")
    parser.add_argument("--year-start", type=int, default=2015)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--train-year-end", type=int, default=2024)
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    df = load_raw(dirname, args.year_start, args.year_end)
    df = ensure_optional_columns(df)

    print("データ変換：開始")
    seconds = parse_race_time_seconds(df["走破時間"])
    df["走破時間_秒"] = seconds
    df["着差"] = parse_margin(df["着差"]) if "着差" in df.columns else np.nan

    pas = df["通過順"].astype(str).str.split("-", expand=True)
    df["通過順"] = pas.apply(lambda col: pd.to_numeric(col, errors="coerce")).mean(axis=1)

    df["クラス"] = df["クラス"].apply(class_mapping)
    df["性"] = df["性"].map({"牡": 0, "牝": 1, "セ": 2})
    df["芝・ダート"] = df["芝・ダート"].map({"芝": 0, "ダ": 1, "障": 2})
    df["回り"] = df["回り"].map({"右": 0, "左": 1, "芝": 2, "直": 2})
    df["馬場"] = df["馬場"].map({"良": 0, "稍": 1, "重": 2, "不": 3})
    df["天気"] = df["天気"].map({"晴": 0, "曇": 1, "小": 2, "雨": 3, "雪": 4})
    df["頭数"] = df.groupby("race_id")["馬番"].transform("size")
    print("データ変換：完了")

    print("スピード指数：開始")
    train_mask_time = df["日付"].dt.year <= args.train_year_end
    speed_params = fit_speed_figure_params(
        seconds, df["距離"], df["場id"], df["馬場"], train_mask_time
    )
    df["スピード指数"] = apply_speed_figure(
        seconds, df["距離"], df["場id"], df["馬場"], speed_params
    )
    df["走破時間"] = df["スピード指数"]
    print("スピード指数：完了")

    print("近5走取得：開始")
    df = add_past_runs(df)
    print("近5走取得：終了")

    df = df[df["芝・ダート"] == 0].copy()

    df = df.replace("---", np.nan)
    print("日付変換：開始")
    df = add_diffs_and_weight(df)
    df = add_jockey_win_rate(df)
    df = add_expanding_form_stats(df)
    df = add_relative_features(df)
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
    all_path = os.path.join(encoded_dir, "encoded_all.csv")
    train_path = os.path.join(encoded_dir, "encoded_traindata.csv")
    test_path = os.path.join(encoded_dir, "encoded_testdata.csv")
    df.to_csv(all_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "speed_params": {
            "stats": speed_params["stats"].to_dict(orient="list"),
            "global_mean": speed_params["global_mean"],
            "global_std": speed_params["global_std"],
        },
        "encoders": encoders,
        "date_origin": str(DATE_ORIGIN.date()),
        "train_year_end": args.train_year_end,
        "year_start": args.year_start,
        "year_end": args.year_end,
    }
    meta_path = os.path.join(encoded_dir, "preprocess_meta.pkl")
    joblib.dump(meta, meta_path)

    print("ファイル出力：終了")
    print(f"  all:   {all_path} ({len(df)} 行)")
    print(f"  train: {train_path} ({len(train_df)} 行)")
    print(f"  test:  {test_path} ({len(test_df)} 行)")
    print(f"  meta:  {meta_path}")


if __name__ == "__main__":
    main()
