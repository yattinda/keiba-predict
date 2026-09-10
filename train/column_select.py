group_horse_ability = [
    "性", #category
    "齢",
    "平均斤量",
    "着順1",
    "着順2",
    "着順3",
    "クラス1", #category
    "クラス2", #category
    "クラス3", #category
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

# 2. 【適性・シチュエーション重視グループ】（条件マッチモデル）
group_race_condition = [
    "芝・ダート", #category
    "距離",
    "回り", #category
    "馬場", #category
    "天気", #category
    "場id", #category
    "クラス",
    "斤量",
    "距離差",
    "距離差1",
    "距離差2",
    "芝・ダート1", #category
    "芝・ダート2", #category
    "芝・ダート3", #category
    "馬場1", #category
    "馬場2", #category
    "馬場3", #category
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

# 3. 【人間・バイアス重視グループ】（人間＆オッズモデル）
group_human_bias = [
    "騎手", #category
    "オッズ",
    "人気",
    "馬番", #category
    "騎手の勝率",
    "騎手1", #category
    "騎手2", #category
    "騎手3", #category
    "オッズ1",
    "オッズ2",
    "オッズ3",
    "馬番1", #category
    "馬番2", #category
    "馬番3", #category
]

category_human_bias = [
    "騎手",
    "馬番",
    "騎手1",
    "騎手2",
    "騎手3",
    "馬番1",
    "馬番2",
    "馬番3"
]

# 4. 【状態変化・パドック代替グループ】（馬体・調子モデル）
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
POST_RACE_COLS = ["上がり", "走破時間", "通過順"]
NON_FEATURE_COLS = ["着順", "year", "race_id", "馬"]

group_human_bias_no_odds = [c for c in group_human_bias if c not in MARKET_COLS]

ALL_CATEGORICAL = sorted(
    set(category_horse_ability + category_race_condition + category_human_bias)
    | {"レース名", "開催", "場名", "騎手4", "騎手5"}
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
    drop = set(NON_FEATURE_COLS) | set(POST_RACE_COLS)
    if not include_odds:
        drop.update(MARKET_COLS)
    return [c for c in df.columns if c not in drop]


def select_features(df, view, include_odds=True):
    if view == "full":
        cols = full_feature_columns(df, include_odds=include_odds)
    else:
        cols = view_columns(view, include_odds=include_odds)
    return df[cols]


def categorical_for_columns(columns):
    return [c for c in ALL_CATEGORICAL if c in set(columns)]


