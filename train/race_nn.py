"""レース単位 softmax NN（条件付きロジット / Transformer encoder）。"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_utils import _prepare_frame, feature_matrix

MAX_HORSES = 18


class RaceDataset(Dataset):
    def __init__(self, df, features, scaler_mean, scaler_std, train=True):
        self.samples = []
        feat = features.astype(np.float32)
        feat = (feat - scaler_mean) / np.where(scaler_std == 0, 1.0, scaler_std)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        finish = pd.to_numeric(df["着順"], errors="coerce").to_numpy()
        race_ids = df["race_id"].to_numpy()
        grouped = pd.DataFrame({"race_id": race_ids}).groupby("race_id", sort=False)
        for race_id, g in grouped:
            idx = g.index.to_numpy()
            n = len(idx)
            if n < 2:
                continue
            x = np.zeros((MAX_HORSES, feat.shape[1]), dtype=np.float32)
            mask = np.zeros(MAX_HORSES, dtype=np.bool_)
            x[:n] = feat[idx]
            mask[:n] = True
            winner = np.zeros(MAX_HORSES, dtype=np.float32)
            win_pos = np.where(finish[idx] == 1)[0]
            if len(win_pos):
                winner[win_pos[0]] = 1.0
            elif not train:
                pass
            else:
                continue
            self.samples.append((x, mask, winner, race_id, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, mask, winner, race_id, idx = self.samples[i]
        return (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(winner),
            race_id,
            idx,
        )


def _collate(batch):
    x = torch.stack([b[0] for b in batch])
    mask = torch.stack([b[1] for b in batch])
    winner = torch.stack([b[2] for b in batch])
    race_ids = [b[3] for b in batch]
    indices = [b[4] for b in batch]
    return x, mask, winner, race_ids, indices


class RaceTransformer(nn.Module):
    """Deep Sets + race context. Transformer は環境によって segfault するため使わない。"""

    def __init__(self, n_features, d_model=64, nhead=4, nlayers=2, dropout=0.1):
        del nhead, nlayers
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        h = self.phi(x)
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        h = h * mask_f
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        race = h.sum(dim=1, keepdim=True) / denom
        race = race.expand_as(h)
        h = self.rho(torch.cat([h, race], dim=-1))
        logits = self.head(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits


class RaceSoftmaxModel:
    def __init__(self, include_odds=False, epochs=8, batch_size=32, lr=1e-3):
        self.include_odds = include_odds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.mean_ = None
        self.std_ = None
        self.feature_cols_ = None
        self.keep_idx_ = None
        self.model = None

    def _features(self, df):
        X = feature_matrix(df, include_odds=self.include_odds)
        if self.feature_cols_ is None:
            self.feature_cols_ = list(X.columns)
        else:
            for col in self.feature_cols_:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_cols_]
        X, _ = _prepare_frame(X)
        arr = X.to_numpy(dtype=np.float32)
        return arr

    def fit(self, df):
        work = df.reset_index(drop=True)
        feat = self._features(work)
        if feat.size == 0:
            raise ValueError("Race NN: 特徴量が空です")
        self.keep_idx_ = ~np.all(np.isnan(feat), axis=0)
        if not self.keep_idx_.any():
            self.keep_idx_ = np.ones(feat.shape[1], dtype=bool)
        feat = np.nan_to_num(feat[:, self.keep_idx_], nan=0.0)
        self.mean_ = feat.mean(axis=0)
        std = feat.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        dataset = RaceDataset(work, feat, self.mean_, self.std_)
        if len(dataset) == 0:
            raise ValueError("Race NN: 学習可能なレースがありません")
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=_collate
        )
        self.model = RaceTransformer(feat.shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for x, mask, winner, _, _ in loader:
                x = x.to(self.device)
                mask = mask.to(self.device)
                winner = winner.to(self.device)
                logits = self.model(x, mask)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(winner * log_probs).sum(dim=1)
                valid = winner.sum(dim=1) > 0
                if valid.any():
                    loss = loss[valid].mean()
                else:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict(self, df):
        original_index = df.index.to_numpy()
        work = df.reset_index(drop=True)
        feat = self._features(work)
        feat = np.nan_to_num(feat[:, self.keep_idx_], nan=0.0)
        dataset = RaceDataset(work, feat, self.mean_, self.std_, train=False)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=_collate
        )
        preds = np.full(len(df), np.nan, dtype=float)
        self.model.eval()
        with torch.no_grad():
            for x, mask, _, _, indices in loader:
                x = x.to(self.device)
                mask = mask.to(self.device)
                logits = self.model(x, mask)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                for row_prob, idx, row_mask in zip(prob, indices, mask.cpu().numpy()):
                    n = int(row_mask.sum())
                    preds[idx[:n]] = row_prob[:n]
        out = pd.Series(preds, index=original_index)
        return out.to_numpy()
