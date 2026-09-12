group_horse_ability = [
    "性",
    "齢",
    "平均斤量",
    "着順1",
    "着順2",
    "着順3",
    "クラス1",
    "クラス2",
    "クラス3",
    "上がり1",
    "上がり2",
    "上がり3",
    "日付差",
]

category_horse_ability = [
    "性",
    "クラス1",
    "クラス2",
    "クラス3",
]

group_race_condition = [
    "芝・ダート",
    "距離",
    "回り",
    "馬場",
    "天気",
    "場id",
    "クラス",
    "斤量",
    "距離差",
    "距離差1",
    "距離差2",
    "芝・ダート1",
    "芝・ダート2",
    "芝・ダート3",
    "馬場1",
    "馬場2",
    "馬場3",
]

category_race_condition = [
    "芝・ダート",
    "回り",
    "馬場",
    "天気",
    "場id",
    "芝・ダート1",
    "芝・ダート2",
    "芝・ダート3",
    "馬場1",
    "馬場2",
    "馬場3",
]

group_human_bias = [
    "騎手",
    "オッズ",
    "人気",
    "馬番",
    "騎手の勝率",
    "騎手1",
    "騎手2",
    "騎手3",
    "オッズ1",
    "オッズ2",
    "オッズ3",
    "馬番1",
    "馬番2",
    "馬番3",
]

category_human_bias = [
    "騎手",
    "馬番",
    "騎手1",
    "騎手2",
    "騎手3",
    "馬番1",
    "馬番2",
    "馬番3",
]

group_horse_condition = [
    "体重",
    "体重変化",
    "体重1",
    "体重2",
    "体重3",
    "体重変化1",
    "体重変化2",
    "体重変化3",
    "通過順1",
    "通過順2",
    "通過順3",
]

MARKET_COLS = ["オッズ", "人気"]
POST_RACE_COLS = [
    "上がり",
    "走破時間",
    "走破時間_秒",
    "通過順",
    "スピード指数",
    "着差",
    "賞金",
]
ID_COLS = ["着順", "year", "race_id"]
NON_FEATURE_COLS = ID_COLS + ["券種", "組番", "払戻金"]
PAYOUT_COLS = ["単勝払戻", "複勝払戻"]

group_human_bias_no_odds = [c for c in group_human_bias if c not in MARKET_COLS]

HIGH_CARD_CATEGORICAL = [
    "馬",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "父",
    "母父",
    "調教師",
    "馬主",
    "レース名",
]

ALL_CATEGORICAL = sorted(
    (
        set(category_horse_ability + category_race_condition + category_human_bias)
        | {
            "開催",
            "場名",
            "騎手4",
            "騎手5",
            "距離帯",
            "枠番",
        }
    )
    - set(HIGH_CARD_CATEGORICAL)
)


def view_columns(view, include_odds=True):
    if view == "horse_ability":
        return list(group_horse_ability)
    if view == "race_condition":
        return list(group_race_condition)
    if view == "human_bias":
        return list(group_human_bias if include_odds else group_human_bias_no_odds)
    if view == "horse_condition":
        return list(group_horse_condition)
    raise ValueError(f"unknown view: {view}")


def full_feature_columns(df, include_odds=True):
    drop = set(NON_FEATURE_COLS) | set(POST_RACE_COLS) | set(PAYOUT_COLS)
    if not include_odds:
        drop.update(MARKET_COLS)
    return [c for c in df.columns if c not in drop]


def select_features(df, view, include_odds=True):
    if view == "full":
        cols = full_feature_columns(df, include_odds=include_odds)
    else:
        cols = view_columns(view, include_odds=include_odds)
        cols = [c for c in cols if c in df.columns]
    return df[cols]


def categorical_for_columns(columns):
    return [c for c in ALL_CATEGORICAL if c in set(columns)]


def catboost_categorical_for_columns(columns):
    cols = set(columns)
    return [c for c in ALL_CATEGORICAL + HIGH_CARD_CATEGORICAL if c in cols]
