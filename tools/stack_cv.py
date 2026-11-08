"""Does a second stage help? Out-of-fold first-stage probabilities on the DFL matches, then a
second-stage model evaluated leave-one-match-out on them. EXTERNAL DATA ONLY.

    python tools/stack_cv.py --dfl-dir C:\\mtah-ext\\dfl

The first-stage probabilities of each match come from a model trained on the other six. The
second stage for a held-out match is trained on the other six matches' out-of-fold probabilities;
those were produced by first-stage models that had seen the held-out match, a small optimistic
leak that the cross-dataset test on the Metrica games (--sample-dir) does not have.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from tactico import ml  # noqa: E402
import train_ml as T  # noqa: E402

COSTS = [(s, d) for s in (2.0, 4.0, 8.0, 12.0) for d in (4.0, 8.0, 16.0)]


def fit2(rows, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    S = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    m.fit(S, y)
    return m


def rows2(prob, X, y):
    keep = np.flatnonzero(y >= 0)[::T.STRIDE]
    S = ml.stack_features(prob, X)[keep]
    yy = y[keep]
    return [(S, yy), (ml.swap_stack(S), np.where(yy == 2, 2, 1 - yy))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir")
    parser.add_argument("--sample-dir")
    args = parser.parse_args()
    t0 = time.time()
    train, _ = T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TRAIN_MIX))
    test, _ = T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TEST_MIX), seed_base=500)
    # out-of-fold first stage: on the training degradation (for stage-2 training) and on the test one
    oof_train, oof_test = [], []
    for k in range(len(train)):
        m1 = T.fit([g for j, g in enumerate(train) if j != k])
        oof_train.append(ml.predict_proba(m1, train[k]["X"]))
        oof_test.append(ml.predict_proba(m1, test[k]["X"]))
        print("  stage 1 fold {} ({:.0f} s)".format(train[k]["name"], time.time() - t0), flush=True)
    res1 = {c: [] for c in COSTS}
    res2 = {c: [] for c in COSTS}
    for k in range(len(train)):
        rows = []
        for j in range(len(train)):
            if j != k:
                rows += rows2(oof_train[j], train[j]["X"], train[j]["y"])
        m2 = fit2(rows)
        p2 = ml.predict_stack(m2, oof_test[k], test[k]["X"])
        for c in COSTS:
            res1[c].append(T.score(test[k]["labels"], ml.label(oof_test[k], test[k]["halves"], *c)))
            res2[c].append(T.score(test[k]["labels"], ml.label(p2, test[k]["halves"], *c)))
        print("  stage 2 fold {} ({:.0f} s)".format(test[k]["name"], time.time() - t0), flush=True)
    T.report(res1, "one stage, DFL leave-one-match-out")
    T.report(res2, "two stages, DFL leave-one-match-out")

    if args.sample_dir:
        met, _ = T.prepare("metrica", args.sample_dir, None, T.parse_mix(T.TEST_MIX), seed_base=500)
        m1 = T.fit(train)
        rows = []
        for j in range(len(train)):
            rows += rows2(oof_train[j], train[j]["X"], train[j]["y"])
        m2 = fit2(rows)
        r1 = {c: [] for c in COSTS}
        r2 = {c: [] for c in COSTS}
        for g in met:
            p1 = ml.predict_proba(m1, g["X"])
            p2 = ml.predict_stack(m2, p1, g["X"])
            for c in COSTS:
                r1[c].append(T.score(g["labels"], ml.label(p1, g["halves"], *c)))
                r2[c].append(T.score(g["labels"], ml.label(p2, g["halves"], *c)))
        T.report(r1, "one stage, trained on DFL, tested on Metrica")
        T.report(r2, "two stages, trained on DFL, tested on Metrica")


if __name__ == "__main__":
    main()
