"""レース単位 softmax 上の線形条件付きロジット（Stage2 / Stage3 共通）。"""

import numpy as np
from scipy.optimize import minimize

from train_utils import softmax_by_race


class LinearRaceSoftmax:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)

    def decision_function(self, x):
        w = self.coef_.ravel()
        return np.asarray(x, dtype=float) @ w


def fit_race_softmax(
    features,
    y_win,
    race_id,
    x0=None,
    bounds=None,
    l2_target=None,
    l2_coef=0.0,
):
    features = np.asarray(features, dtype=float)
    y_win = np.asarray(y_win).astype(int)
    race_id = np.asarray(race_id)
    n_feat = features.shape[1]
    win_idx = np.where(y_win == 1)[0]
    if len(win_idx) == 0:
        weights = np.ones(n_feat, dtype=float) if x0 is None else np.asarray(x0, dtype=float)
        model = LinearRaceSoftmax(weights)
        return model, softmax_by_race(features @ model.coef_.ravel(), race_id)
    if x0 is None:
        x0 = np.ones(n_feat, dtype=float)
    if l2_target is None:
        l2_target = np.ones(n_feat, dtype=float)

    def loss(w):
        raw = features @ w
        p = softmax_by_race(raw, race_id)
        nll = -np.mean(np.log(np.clip(p[win_idx], 1e-12, 1)))
        if l2_coef > 0:
            nll += l2_coef * np.sum((w - l2_target) ** 2)
        return nll

    opt_bounds = bounds
    if opt_bounds is None:
        opt_bounds = [(None, None)] * n_feat
    result = minimize(loss, x0=np.asarray(x0, dtype=float), method="L-BFGS-B", bounds=opt_bounds)
    weights = result.x
    raw = features @ weights
    return LinearRaceSoftmax(weights), softmax_by_race(raw, race_id)


def apply_race_softmax(model, features, race_id):
    raw = model.decision_function(features)
    return softmax_by_race(raw, race_id), model.coef_.ravel()