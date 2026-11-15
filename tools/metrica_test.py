"""Final check on the three Metrica sample games, which no setting was chosen on.

Trains the classifier and the GRUs on all the external training data (DFL degraded, SkillCorner,
the older SkillCorner matches), exactly as for the submission, then scores the Metrica games
degraded with the held-out test degradation. Run it once in each code version to compare them.

    python tools/metrica_test.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc \\
        --extra-dir C:/mtah-ext/sc-old --sample-dir C:/mtah-ext/sample-data --spell-balance 1.0
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from tactico import ml, seq  # noqa: E402
import train_ml as T  # noqa: E402
import train_seq as TS  # noqa: E402
from cv_multi import sc_games  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dfl-dir", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--extra-dir")
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--spell-balance", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    T.SPELL_BALANCE = args.spell_balance
    TS.EPOCHS = args.epochs
    t0 = time.time()
    log = lambda m: print(m + " ({:.0f} s)".format(time.time() - t0), flush=True)  # noqa: E731
    train = TS.to32(T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TRAIN_MIX))[0])
    train += sc_games(args.sc_dir, f32=True)
    if args.extra_dir:
        train += sc_games(args.extra_dir, f32=True)
    test = T.prepare("metrica", args.sample_dir, None, T.parse_mix(T.TEST_MIX), seed_base=900)[0]
    log("{} features, {} training matches, {} test games".format(len(ml.FEATURE_NAMES), len(train), len(test)))
    m1 = T.fit(train)
    nets = [TS.fit_seq(train, seed=k, log=log) for k in range(args.seeds)]
    rows = []
    for g in test:
        p1 = ml.predict_proba(m1, g["X"])
        p2 = np.mean([seq.predict_proba(n, g["X"].astype(np.float32), g["halves"]) for n in nets], axis=0)
        lab = ml.label(0.2 * p1 + 0.8 * p2, g["halves"], 4.0, 12.0)
        spell, macro = T.score(g["labels"], lab)
        rows.append((g["name"], spell, macro))
        print("  {}  spellF1 {:.4f}  macroF1 {:.4f}".format(g["name"], spell, macro), flush=True)
    print("Metrica mean  spellF1 {:.4f}  macroF1 {:.4f}".format(
        np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows])))


if __name__ == "__main__":
    main()
