"""Choose the blend of classifier and GRU, the number of GRU seeds and the decoding costs from the
held-out probabilities that `tools/train_seq.py --cv --dump DIR` saved. EXTERNAL DATA ONLY.

    python tools/blend_cv.py C:\\mtah-ext\\dump3

Every combination is decoded and scored on the held-out DFL and SkillCorner matches of all five
folds. The ranking criterion, fixed before looking at the numbers, is the mean of the DFL mean and
the SkillCorner mean, so the twenty SkillCorner matches do not drown out the seven DFL matches.
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

WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)          # weight of the GRU in the average
COSTS = [(s, d) for s in (4.0, 6.0, 8.0, 12.0) for d in (12.0, 16.0, 24.0)]

try:                                              # same algorithm as ml.viterbi, compiled
    import numba

    @numba.njit(cache=True)
    def _viterbi(lp, sw, dc):
        n = lp.shape[0]
        trans = np.array([[0.0, sw, dc], [sw, 0.0, dc], [dc, dc, 0.0]])
        score = lp[0].copy()
        back = np.zeros((n, 3), dtype=np.int8)
        cand = np.empty((3, 3))
        new = np.empty(3)
        for t in range(1, n):
            for i in range(3):
                for j in range(3):
                    cand[i, j] = score[i] - trans[i, j]
            for j in range(3):
                b = 0
                for i in range(1, 3):
                    if cand[i, j] > cand[b, j]:
                        b = i
                back[t, j] = b
                new[j] = cand[b, j] + lp[t, j]
            score[:] = new
        path = np.zeros(n, dtype=np.int8)
        path[n - 1] = np.argmax(score)
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def label(prob, halves, sw, dc):
        out = np.full(len(prob), "X", dtype="<U1")
        for _, first, last in halves:
            lp = np.log(np.clip(prob[first:last + 1].astype(np.float64), 1e-4, 1.0))
            out[first:last + 1] = np.array(ml.STATES)[_viterbi(lp, sw, dc)]
        return out
except ImportError:
    label = ml.label


def main():
    folder = sys.argv[1]
    matches = []
    for path in sorted(glob.glob(os.path.join(folder, "fold*.pkl"))):
        with open(path, "rb") as handle:
            matches += pickle.load(handle)
    n_seeds = matches[0]["p_gru"].shape[0]
    print("{} held-out matches from {} folds, {} GRU seeds".format(
        len(matches), len(glob.glob(os.path.join(folder, "fold*.pkl"))), n_seeds))
    rows = []
    for k in sorted({1, n_seeds}):
        for w in (0.0,) + WEIGHTS:
            for c in COSTS:
                d, s = [], []
                for m in matches:
                    p = (1 - w) * m["p_hgb"] + w * m["p_gru"][:k].mean(axis=0)
                    sp, _ = T.score(m["labels"], label(p, m["halves"], *c))
                    (d if m["tag"] == "D" else s).append(sp)
                rows.append(((np.mean(d) + np.mean(s)) / 2, np.mean(d), np.mean(s), k, w, c, s, d))
        print("  scored {} seed(s)".format(k), flush=True)
    rows.sort(key=lambda r: -r[0])
    print("\n  crit    DFL    SC    seeds  w_gru  switch dead")
    for r in rows[:15]:
        print("  {:.4f} {:.4f} {:.4f}   {}    {:.1f}   {:5.1f} {:5.1f}".format(r[0], r[1], r[2], r[3], r[4], *r[5]))
    ref = [r for r in rows if r[3] == 1 and r[4] == 0.5 and r[5] == (8.0, 16.0)][0]
    best = rows[0]
    wins = sum(a > b for a, b in zip(best[6] + best[7], ref[6] + ref[7]))
    print("\n  current v11 (1 seed, w 0.5, 8/16): {:.4f}  DFL {:.4f}  SC {:.4f}".format(*ref[:3]))
    print("  best beats it on {} of {} matches".format(wins, len(ref[6]) + len(ref[7])))


if __name__ == "__main__":
    main()
