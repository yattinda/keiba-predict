"""馬単位ベルヌーイ（シグモイド）上の線形ロジット。top3 用。"""

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-6


def clip_proba(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def proba_logit(p):
    p = clip_proba(p)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LinearHorseBernoulli:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)

    def decision_function(self, x):
        w = self.coef_.ravel()
        return np.asarray(x, dtype=float) @ w


def fit_horse_bernoulli(
    features,
    y,
    x0=None,
    bounds=None,
    l2_target=None,
    l2_coef=0.0,
):
    features = np.asarray(features, dtype=float)
    y = np.asarray(y).astype(int)
    n_feat = features.shape[1]
    if x0 is None:
        x0 = np.ones(n_feat, dtype=float)
    if l2_target is None:
        l2_target = np.ones(n_feat, dtype=float)

    def loss(w):
        z = features @ w
        p = sigmoid(z)
        p = clip_proba(p)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if l2_coef > 0:
            nll += l2_coef * np.sum((w - l2_target) ** 2)
        return nll

    opt_bounds = bounds
    if opt_bounds is None:
        opt_bounds = [(None, None)] * n_feat
    result = minimize(loss, x0=np.asarray(x0, dtype=float), method="L-BFGS-B", bounds=opt_bounds)
    weights = result.x
    model = LinearHorseBernoulli(weights)
    z = features @ weights
    return model, clip_proba(sigmoid(z))


def apply_horse_bernoulli(model, features):
    z = model.decision_function(features)
    p = sigmoid(z)
    return clip_proba(p), model.coef_.ravel()