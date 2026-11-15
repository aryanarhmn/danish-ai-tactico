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


def build_net(n_in, n_out=3):
    """n_out = 3 (H, A, D) or 4: the fourth output is a logit for "the label changes here"."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, 96), nn.ReLU())
            self.gru = nn.GRU(96, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * HIDDEN, n_out)

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


def predict_half(net, Z, boundary=False):
    """(n, 3) probabilities for one half: GRU on every 5th frame, repeated to 25 fps.
    With boundary=True also (n,) probability that the label changes there (None without that head)."""
    import torch
    sub = Z[::STEP]
    with torch.no_grad():
        logits = net(torch.from_numpy(sub[None]))[0]
        p = torch.softmax(logits[:, :3], dim=-1).numpy()
        b = torch.sigmoid(logits[:, 3]).numpy() if logits.shape[1] > 3 else None
    p = np.repeat(p, STEP, axis=0)[:len(Z)]
    if not boundary:
        return p
    return p, (None if b is None else np.repeat(b, STEP)[:len(Z)])


def load(path):
    """Bundle saved by tools/train_seq.py --save."""
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    nets = []
    for state in saved.get("states") or [saved["state"]]:
        net = build_net(saved["n_in"], saved.get("n_out", 3))
        net.load_state_dict(state)
        net.eval()
        nets.append(net)
    return {"nets": nets, "stats": saved["stats"]}


def predict_proba(bundle, X, halves):
    """Team-symmetric probabilities for the full array (uniform outside the halves)."""
    import torch
    from tactico import ml
    torch.set_num_threads(1)            # one thread: the same sums in the same order on every run
    nets = bundle.get("nets") or [bundle["net"]]
    n_team = len(ml.TEAM_FEATURES)
    Z, _ = prepare_inputs(X, bundle["stats"])
    out = np.full((len(X), 3), 1.0 / 3, dtype=np.float32)
    for _, first, last in halves:
        z = Z[first:last + 1]
        zs = swap_inputs(z, n_team)
        acc = np.zeros((len(z), 3), dtype=np.float32)
        for net in nets:              # several seeds: plain average
            net.eval()
            acc += (predict_half(net, z) + predict_half(net, zs)[:, [1, 0, 2]]) / 2
        out[first:last + 1] = acc / len(nets)
    return out


def predict_boundary(bundle, X, halves):
    """(n,) probability that the label changes at each frame, averaged over seeds and both team
    orders (a change is a change whichever team is home). None if the nets have no such head."""
    import torch
    from tactico import ml
    torch.set_num_threads(1)
    nets = bundle.get("nets") or [bundle["net"]]
    n_team = len(ml.TEAM_FEATURES)
    Z, _ = prepare_inputs(X, bundle["stats"])
    out = np.zeros(len(X), dtype=np.float32)
    for _, first, last in halves:
        z = Z[first:last + 1]
        zs = swap_inputs(z, n_team)
        acc = np.zeros(len(z), dtype=np.float32)
        for net in nets:
            net.eval()
            b1 = predict_half(net, z, boundary=True)[1]
            if b1 is None:
                return None
            acc += (b1 + predict_half(net, zs, boundary=True)[1]) / 2
        out[first:last + 1] = acc / len(nets)
    return out
