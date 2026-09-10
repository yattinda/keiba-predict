"""馬単位・レース単位の評価指標。閾値 0.5 の混同行列には依存しない。"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _clip_proba(pred):
    return np.clip(np.asarray(pred, dtype=float), 1e-7, 1 - 1e-7)


def race_ranking_metrics(race_id, pred, y_binary, finish=None, k=3):
    df = pd.DataFrame(
        {
            "race_id": np.asarray(race_id),
            "pred": np.asarray(pred, dtype=float),
            "y": np.asarray(y_binary, dtype=int),
        }
    )
    if finish is not None:
        df["finish"] = np.asarray(finish)

    precisions = []
    recalls = []
    win_hits = []
    for _, g in df.groupby("race_id", sort=False):
        if len(g) == 0:
            continue
        n_pos = int(g["y"].sum())
        top = g.nlargest(min(k, len(g)), "pred")
        hit = int(top["y"].sum())
        precisions.append(hit / len(top))
        recalls.append(hit / n_pos if n_pos > 0 else np.nan)
        if "finish" in g.columns:
            pred_first = g.loc[g["pred"].idxmax()]
            win_hits.append(float(pred_first["finish"] == 1))

    return {
        "race_precision_at_3": float(np.nanmean(precisions)) if precisions else float("nan"),
        "race_recall_at_3": float(np.nanmean(recalls)) if recalls else float("nan"),
        "win_hit_rate": float(np.mean(win_hits)) if win_hits else float("nan"),
        "n_races": int(df["race_id"].nunique()),
    }


def evaluate_predictions(y_binary, pred, race_id=None, finish=None, name="model"):
    y = np.asarray(y_binary, dtype=int)
    p = _clip_proba(pred)
    metrics = {
        "model": name,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "pred_mean": float(np.mean(p)),
        "n_rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
    if race_id is not None:
        metrics.update(race_ranking_metrics(race_id, p, y, finish=finish))
    return metrics


def format_metrics_table(rows):
    df = pd.DataFrame(rows)
    preferred = [
        "model",
        "include_odds",
        "auc",
        "logloss",
        "brier",
        "race_precision_at_3",
        "race_recall_at_3",
        "win_hit_rate",
        "n_races",
        "n_rows",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[cols]
