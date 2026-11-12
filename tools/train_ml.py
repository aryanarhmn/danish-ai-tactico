"""Train and cross-validate the learned possession model on EXTERNAL data only.

    python tools/train_ml.py --dfl-dir C:\\mtah-ext\\dfl --cv          leave-one-match-out on DFL
    python tools/train_ml.py --dfl-dir C:\\mtah-ext\\dfl --sample-dir C:\\mtah-ext\\sample-data --cross
                                                                    train on DFL, test on Metrica
    python tools/train_ml.py --dfl-dir C:\\mtah-ext\\dfl --save models/possession.pkl

The training matches are degraded to look like the competition tracking: the ball removed on
--ball-missing of the frames (41 % per CHALLENGE.md) and --noise metres of position error. The
same degradation is applied to the held-out match, so the CV score estimates performance on
tracking of that quality. Nothing here reads the competition data.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "external"))
import possession_metrics as pm  # noqa: E402
from tactico import ml  # noqa: E402
from tune_external import degrade, raw_games  # noqa: E402

CODE = {"H": 0, "A": 1, "D": 2}
STRIDE = 5
# Measured on the competition tracking with tools/noise_match.py (summary numbers only): ball
# missing on 41 % of frames in gaps of mean 17 s (median 11 s), very smooth positions (ball jitter
# about 1 cm, i.e. about 0.015 m of noise), about 10.2 of 11 players per team in view.
# tools/gap_profile.py: the ball is missing on 80 % of frames with the players slow (< 1 m/s), 42 % at
# 1-2 m/s and 11 % above 2 m/s, so gaps are placed by player speed (fifth field = 1).
TRAIN_MIX = "0.015:0.41:200:0.07:1,0.03:0.41:150:0.12:1"
TEST_MIX = "0.02:0.41:250:0.09:1"
SPELL_BALANCE = 0.0  # >0: give short spells more weight, since spellF1 counts every spell once
BALANCE_FRAMES = 250 # 10 s: a spell of n < 250 frames gets sqrt(250 / n), capped, scaled by SPELL_BALANCE


def spell_weights(y, cap=3.0):
    """Per-frame loss weights: 1 everywhere, up to `cap` inside H/A spells shorter than BALANCE_FRAMES."""
    w = np.ones(len(y), dtype=np.float64)
    if SPELL_BALANCE <= 0:
        return w
    edges = np.flatnonzero(np.r_[True, y[1:] != y[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if y[a] in (0, 1):
            w[a:b] = 1.0 + SPELL_BALANCE * (np.clip(np.sqrt(BALANCE_FRAMES / (b - a)), 1.0, cap) - 1.0)
    return w


COSTS = [(s, d) for s in (4.0, 8.0, 12.0, 16.0, 24.0) for d in (4.0, 8.0, 12.0, 16.0)]


def parse_mix(text):
    """'noise:missing:gap:player_drop[:by_speed],...' -> list of tuples. by_speed=1 places ball
    gaps where the players are slow, as measured on the competition tracking (gap = correlation
    length in frames, missing is then set by the measured rates)."""
    return [tuple(float(v) for v in part.split(":")) for part in text.split(",")]


def prepare(data, sample_dir, dfl_dir, mix, seed_base=0):
    """One entry per game; X/y stacked over the degradations in `mix` (labels from the first)."""
    out, names = [], None
    for i, (name, d) in enumerate(raw_games(data, sample_dir, dfl_dir)):
        Xs = []
        for j, spec in enumerate(mix):
            noise, miss, gap, pdrop = spec[:4]
            by_speed = spec[4] if len(spec) > 4 else 0
            dd = degrade(d, noise, miss, int(gap), pdrop, seed=seed_base + 10 * (i + 1) + j,
                         by_speed=by_speed)
            X, names = ml.features(dd)
            Xs.append(X)
        y = np.array([CODE.get(v, -1) for v in d["labels"]], dtype=np.int8)
        out.append({"name": name, "Xs": Xs, "X": Xs[0], "y": y, "labels": d["labels"],
                    "halves": d["halves"]})
    return out, names


def training_rows(games, with_weights=False):
    Xs, ys, ws = [], [], []
    for g in games:
        wg = spell_weights(g["y"]) if with_weights else None
        for j, Xg in enumerate(g.get("Xs", [g["X"]])):
            keep = np.flatnonzero(g["y"] >= 0)[j % STRIDE::STRIDE * len(g.get("Xs", [0]))]
            X, y = Xg[keep], g["y"][keep]
            Xs += [X, ml.swap_teams(X)]
            ys += [y, np.where(y == 2, 2, 1 - y)]
            if with_weights:
                ws += [wg[keep], wg[keep]]
    if with_weights:
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    return np.concatenate(Xs), np.concatenate(ys)


def fit(games, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                           min_samples_leaf=80, l2_regularization=1.0,
                                           random_state=seed)
    if SPELL_BALANCE > 0:
        X, y, w = training_rows(games, with_weights=True)
        model.fit(X, y, sample_weight=w)
    else:
        X, y = training_rows(games)
        model.fit(X, y)
    return model


def score(labels_true, labels_pred):
    gt = pm.label_codes(labels_true)
    pred = pm.label_codes(labels_pred)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return spell, macro


def evaluate(model, games, costs):
    """{cost: [(spell, macro) per game]} and the frame accuracy of the raw argmax."""
    res = {c: [] for c in costs}
    for g in games:
        prob = ml.predict_proba(model, g["X"])
        for c in costs:
            res[c].append(score(g["labels"], ml.label(prob, g["halves"], *c)))
    return res


def report(res, title):
    rows = sorted(((np.mean([s for s, _ in v]), c, np.mean([m for _, m in v]), v)
                   for c, v in res.items()), reverse=True)
    print(title)
    print("  switch  dead   spellF1  macroF1   per match")
    for spell, c, macro, v in rows[:8]:
        print("  {:6.1f} {:5.1f}   {:.3f}    {:.3f}     {}".format(
            c[0], c[1], spell, macro, " ".join("{:.3f}".format(s) for s, _ in v)))
    return rows[0][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir")
    parser.add_argument("--sc-dir", help="SkillCorner .npz folder (tools/external/skillcorner_open.py); "
                        "added to the training matches as they are, without degradation")
    parser.add_argument("--extra-dir", help="more .npz matches for training only (tools/external/skillcorner_old.py)")
    parser.add_argument("--sample-dir")
    parser.add_argument("--train-mix", default=TRAIN_MIX,
                        help="degradations to train on, noise:ball_missing:gap:player_drop,...")
    parser.add_argument("--test-mix", default=TEST_MIX, help="degradation of the held-out games")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--save")
    parser.add_argument("--switch-cost", type=float, default=None)
    parser.add_argument("--dead-cost", type=float, default=None)
    parser.add_argument("--spell-balance", type=float, default=0.0,
                        help="extra loss weight on short possessions (0 = off, as in v13; v14 uses 1)")
    args = parser.parse_args()
    global SPELL_BALANCE
    SPELL_BALANCE = args.spell_balance

    t0 = time.time()
    train_mix, test_mix = parse_mix(args.train_mix), parse_mix(args.test_mix)
    dfl, names = prepare("dfl", None, args.dfl_dir, train_mix)
    print("DFL matches:", " ".join(g["name"] for g in dfl), "({:.0f} s)".format(time.time() - t0))
    print("train on", train_mix, " test on", test_mix)

    if args.cv:
        res = {c: [] for c in COSTS}
        dfl_test, _ = prepare("dfl", None, args.dfl_dir, test_mix, seed_base=500)
        for k, held in enumerate(dfl_test):
            model = fit([g for j, g in enumerate(dfl) if j != k])
            part = evaluate(model, [held], COSTS)
            for c in COSTS:
                res[c] += part[c]
            print("  fold {} done ({:.0f} s)".format(held["name"], time.time() - t0), flush=True)
        report(res, "leave-one-match-out on DFL (degraded)")

    if args.cross:
        met, _ = prepare("metrica", args.sample_dir, None, test_mix, seed_base=500)
        model = fit(dfl)
        report(evaluate(model, met, COSTS), "trained on DFL, tested on Metrica sample games")

    if args.save:
        train = dfl
        if args.sc_dir:
            from cv_multi import sc_games
            train = dfl + sc_games(args.sc_dir)
            print("training on {} DFL + {} SkillCorner matches".format(len(dfl), len(train) - len(dfl)))
        if args.extra_dir:
            from cv_multi import sc_games
            extra = sc_games(args.extra_dir)
            train = train + extra
            print("plus {} extra matches for training only".format(len(extra)))
        model = fit(train)
        cost = (args.switch_cost, args.dead_cost)
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "wb") as handle:
            pickle.dump({"model": model, "features": names, "switch_cost": cost[0],
                         "dead_cost": cost[1], "train_mix": train_mix}, handle)
        print("saved", args.save)


if __name__ == "__main__":
    main()
