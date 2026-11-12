"""Does adding the SkillCorner matches help? EXTERNAL DATA ONLY.

    python tools/cv_multi.py --dfl-dir C:\\mtah-ext\\dfl --sc-dir C:\\mtah-ext\\sc --sample-dir C:\\mtah-ext\\sample-data

Five folds over the DFL and SkillCorner matches together. For each fold two models are trained on
the other folds: one on DFL only (the current model) and one on DFL + SkillCorner. Both are scored
on the held-out DFL matches (degraded as in train_ml.TEST_MIX) and the held-out SkillCorner matches
(their own automatic tracking, not degraded), and at the end on the Metrica games.
SkillCorner matches are used as they are: their ball and player gaps are real.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "external"))
from tactico import ml  # noqa: E402
import dfl_open  # noqa: E402
import train_ml as T  # noqa: E402

COSTS = [(4.0, 16.0), (8.0, 8.0), (8.0, 16.0), (12.0, 16.0), (8.0, 24.0), (16.0, 24.0)]


def sc_games(folder, f32=False):
    """One entry per .npz match. f32=True stores the features as float32, half the memory."""
    out = []
    for p in dfl_open.list_matches(folder):
        d = dfl_open.load_match(p)
        X, _ = ml.features(d)
        if f32:
            X = X.astype(np.float32)
        y = np.array([T.CODE.get(v, -1) for v in d["labels"]], dtype=np.int8)
        out.append({"name": "S" + os.path.basename(p)[3:-4], "X": X, "Xs": [X], "y": y,
                    "labels": d["labels"], "halves": d["halves"]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir")
    parser.add_argument("--sc-dir")
    parser.add_argument("--sample-dir")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    t0 = time.time()
    dfl_train, _ = T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TRAIN_MIX))
    dfl_test, _ = T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TEST_MIX), seed_base=500)
    sc = sc_games(args.sc_dir)
    print("loaded {} DFL + {} SkillCorner ({:.0f} s)".format(len(dfl_train), len(sc), time.time() - t0), flush=True)
    fold_d = [i % args.folds for i in range(len(dfl_train))]
    fold_s = [(i + 2) % args.folds for i in range(len(sc))]
    res = {key: {c: [] for c in COSTS} for key in ("dfl_only|D", "dfl_only|S", "both|D", "both|S")}
    for f in range(args.folds):
        tr_d = [g for g, k in zip(dfl_train, fold_d) if k != f]
        tr_s = [g for g, k in zip(sc, fold_s) if k != f]
        te_d = [g for g, k in zip(dfl_test, fold_d) if k == f]
        te_s = [g for g, k in zip(sc, fold_s) if k == f]
        models = {"dfl_only": T.fit(tr_d), "both": T.fit(tr_d + tr_s)}
        for name, m in models.items():
            for tag, games in (("D", te_d), ("S", te_s)):
                part = T.evaluate(m, games, COSTS)
                for c in COSTS:
                    res[name + "|" + tag][c] += part[c]
        print("  fold {} done ({:.0f} s)".format(f, time.time() - t0), flush=True)
    for key, r in res.items():
        T.report(r, key)
    if args.sample_dir:
        met, _ = T.prepare("metrica", args.sample_dir, None, T.parse_mix(T.TEST_MIX), seed_base=500)
        for name, games in (("dfl_only", dfl_train), ("both", dfl_train + sc)):
            T.report(T.evaluate(T.fit(games), met, COSTS), name + " | Metrica")


if __name__ == "__main__":
    main()
