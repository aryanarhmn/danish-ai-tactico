"""Sequence model: a bidirectional GRU over the per-frame features of tactico/ml.py.

The gradient-boosted classifier sees a few seconds around each frame. The GRU reads a whole minute
in both directions, so it can carry what happened 30 seconds ago (a stoppage, the last team in
control) through a long stretch without a tracked ball. Its probabilities are averaged with the
classifier's and decoded with the same Viterbi step. Trained only on external data
(tools/train_seq.py).

The GRU runs at 5 frames per second (every 5th frame) and its output is repeated back to 25 fps.
"""

import numpy as np

STEP = 5            # 25 fps -> 5 fps
CHUNK = 300         # 60 s per training sequence
HIDDEN = 64


def prepare_inputs(X, stats=None):
    """NaN -> 0 after standardising, plus a missing flag per feature. Returns (inputs, stats)."""
    if stats is None:
        from tactico import ml
        k = len(ml.TEAM_FEATURES)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0) + 1e-6
        # same scaling for both team blocks and a zero mean for gap_d1, so swapping teams is exact
        mu[:k] = mu[k:2 * k] = (mu[:k] + mu[k:2 * k]) / 2
        sd[:k] = sd[k:2 * k] = (sd[:k] + sd[k:2 * k]) / 2
        mu[-1] = 0.0
        stats = (mu.astype(np.float32), sd.astype(np.float32))
    mu, sd = stats
    Z = (X - mu) / sd
    miss = np.isnan(Z)
    Z = np.where(miss, 0.0, np.clip(Z, -8, 8))
    return np.concatenate([Z, miss.astype(np.float32)], axis=1).astype(np.float32), stats


def build_net(n_in):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, 96), nn.ReLU())
            self.gru = nn.GRU(96, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)

    return Net()


def swap_inputs(Z, n_team):
    """Swap the home and away blocks in prepared inputs (values and missing flags)."""
    n_feat = Z.shape[1] // 2
    Y = Z.copy()
    for off in (0, n_feat):
        a, b = off, off + n_team
        Y[:, a:a + n_team], Y[:, b:b + n_team] = Z[:, b:b + n_team], Z[:, a:a + n_team]
    Y[:, n_feat - 1] = -Z[:, n_feat - 1]            # gap_d1 changes sign
    return Y


def predict_half(net, Z):
    """(n, 3) probabilities for one half: GRU on every 5th frame, repeated to 25 fps."""
    import torch
    sub = Z[::STEP]
    with torch.no_grad():
        logits = net(torch.from_numpy(sub[None]))[0]
        p = torch.softmax(logits, dim=-1).numpy()
    return np.repeat(p, STEP, axis=0)[:len(Z)]


def load(path):
    """Bundle saved by tools/train_seq.py --save."""
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(saved["n_in"])
    net.load_state_dict(saved["state"])
    net.eval()
    return {"net": net, "stats": saved["stats"]}


def predict_proba(bundle, X, halves):
    """Team-symmetric probabilities for the full array (uniform outside the halves)."""
    import torch
    from tactico import ml
    torch.set_num_threads(1)            # one thread: the same sums in the same order on every run
    net = bundle["net"]
    net.eval()
    n_team = len(ml.TEAM_FEATURES)
    Z, _ = prepare_inputs(X, bundle["stats"])
    out = np.full((len(X), 3), 1.0 / 3, dtype=np.float32)
    for _, first, last in halves:
        z = Z[first:last + 1]
        p1 = predict_half(net, z)
        p2 = predict_half(net, swap_inputs(z, n_team))[:, [1, 0, 2]]
        out[first:last + 1] = (p1 + p2) / 2
    return out
