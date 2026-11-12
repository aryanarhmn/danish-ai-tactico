"""Decoding experiments on the held-out probabilities saved by `tools/train_seq.py --cv --dump DIR`.
EXTERNAL DATA ONLY.

    python tools/decode_cv.py C:\\mtah-ext\\dump3

Two ideas, both scored with the same fixed criterion as tools/blend_cv.py (mean of the DFL mean and
the SkillCorner mean):

  min-duration  Viterbi over (label, frames since the label started): a label can only be left
                after K frames. Separate K for live labels (H, A) and for dead ball.
  abstain       write X where the decoded label's probability is below a threshold. The scorer
                forward-fills X inside a possession, so this delays changes until the model is sure.
"""

import glob
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from tactico import ml  # noqa: E402
import train_ml as T  # noqa: E402

import numba  # noqa: E402

STATES = np.array(ml.STATES)


@numba.njit(cache=True)
def viterbi_min(lp, sw, dc, k_live, k_dead):
    """Viterbi with a minimum duration per label. State (l, d), d = frames since entry, capped."""
    n = lp.shape[0]
    ks = np.array([max(k_live, 1), max(k_live, 1), max(k_dead, 1)])
    off = np.array([0, ks[0], ks[0] + ks[1]])
    m = ks[0] + ks[1] + ks[2]
    trans = np.array([[0.0, sw, dc], [sw, 0.0, dc], [dc, dc, 0.0]])
    neg = -1e18
    score = np.full(m, neg)
    for l in range(3):
        score[off[l] + ks[l] - 1] = lp[0, l]          # start as if already settled
    back = np.zeros((n, m), dtype=np.int32)
    new = np.empty(m)
    for t in range(1, n):
        # best exit from each label (only from the settled sub-state)
        for l in range(3):
            k = ks[l]
            base = off[l]
            # sub-state 0: entered now from a settled other label
            best = neg
            arg = 0
            for j in range(3):
                if j == l:
                    continue
                s = score[off[j] + ks[j] - 1] - trans[j, l]
                if s > best:
                    best = s
                    arg = off[j] + ks[j] - 1
            if k == 1:
                stay = score[base]
                if stay >= best:
                    new[base] = stay + lp[t, l]
                    back[t, base] = base
                else:
                    new[base] = best + lp[t, l]
                    back[t, base] = arg
                continue
            new[base] = best + lp[t, l]
            back[t, base] = arg
            for d in range(1, k - 1):
                new[base + d] = score[base + d - 1] + lp[t, l]
                back[t, base + d] = base + d - 1
            a = score[base + k - 2]
            b = score[base + k - 1]
            if b >= a:
                new[base + k - 1] = b + lp[t, l]
                back[t, base + k - 1] = base + k - 1
            else:
                new[base + k - 1] = a + lp[t, l]
                back[t, base + k - 1] = base + k - 2
        for i in range(m):
            score[i] = new[i]
    path = np.zeros(n, dtype=np.int8)
    s = int(np.argmax(score))
    for t in range(n - 1, -1, -1):
        lab = 0 if s < off[1] else (1 if s < off[2] else 2)
        path[t] = lab
        s = back[t, s]
    return path


def decode(prob, halves, sw, dc, k_live=1, k_dead=1, abstain=0.0):
    out = np.full(len(prob), "X", dtype="<U1")
    for _, first, last in halves:
        p = prob[first:last + 1].astype(np.float64)
        lp = np.log(np.clip(p, 1e-4, 1.0))
        path = viterbi_min(lp, sw, dc, k_live, k_dead)
        lab = STATES[path]
        if abstain > 0:
            conf = p[np.arange(len(p)), path]
            lab = np.where(conf < abstain, "X", lab)
        out[first:last + 1] = lab
    return out


def load(folder):
    matches = []
    for path in sorted(glob.glob(os.path.join(folder, "fold*.pkl"))):
        with open(path, "rb") as handle:
            matches += pickle.load(handle)
    return matches


def evaluate(matches, w, **kw):
    d, s = [], []
    for m in matches:
        p = (1 - w) * m["p_hgb"] + w * m["p_gru"].mean(axis=0)
        sp, _ = T.score(m["labels"], decode(p, m["halves"], **kw))
        (d if m["tag"] == "D" else s).append(sp)
    return (np.mean(d) + np.mean(s)) / 2, np.mean(d), np.mean(s), d + s


def main():
    matches = load(sys.argv[1])
    ref = evaluate(matches, 0.8, sw=4.0, dc=12.0)
    print("v12 reference (w 0.8, 4/12): {:.4f}  DFL {:.4f}  SC {:.4f}".format(*ref[:3]), flush=True)
    rows = []
    for k_live in (1, 13, 25, 50):
        for k_dead in (1, 25, 50, 100):
            for sw, dc in ((4.0, 12.0), (2.0, 8.0), (3.0, 10.0)):
                r = evaluate(matches, 0.8, sw=sw, dc=dc, k_live=k_live, k_dead=k_dead)
                rows.append((r, "min-dur live {} dead {}  costs {:g}/{:g}".format(k_live, k_dead, sw, dc)))
        print("  k_live {} done".format(k_live), flush=True)
    for a in (0.4, 0.45, 0.5, 0.55, 0.6):
        r = evaluate(matches, 0.8, sw=4.0, dc=12.0, abstain=a)
        rows.append((r, "abstain below {:.2f}  costs 4/12".format(a)))
    rows.sort(key=lambda x: -x[0][0])
    print("\n  crit    DFL    SC     wins/27  setting")
    for r, name in rows[:15]:
        wins = sum(a > b for a, b in zip(r[3], ref[3]))
        print("  {:.4f} {:.4f} {:.4f}  {:2d}      {}".format(r[0], r[1], r[2], wins, name))
    print("\n  abstention:")
    for r, name in rows:
        if name.startswith("abstain"):
            print("  {:.4f} {:.4f} {:.4f}  {}".format(r[0], r[1], r[2], name))


if __name__ == "__main__":
    main()
