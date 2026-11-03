"""Tune tactico/config.py on EXTERNAL data only: Metrica's public sample games.

    python tools/tune_external.py --sample-dir C:\\mtah-ext\\sample-data
    python tools/tune_external.py --grid margin          (one of the grids below)

Scores every variant on the three sample games against event-derived possession labels
(tools/external/metrica_sample.py) with the competition scorer, and prints the mean over games.
  full   whole pipeline: our dead-ball mask + possession
  poss   possession only: dead ball taken from the event labels, so only the possession
         parameters (CONTROL_*, SWITCH_FRAMES, RESTART_*) matter

This never touches the competition game, its ground truth or its video.
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tools", "external"))
import possession_metrics as pm  # noqa: E402
import metrica_sample as ms  # noqa: E402
from tactico import config as C, deadball, features, possession  # noqa: E402

GRIDS = {
    "now": {},
    "margin": {"CONTROL_MARGIN_M": [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]},
    "radius": {"CONTROL_RADIUS_M": [1.0, 1.5, 2.0, 2.5, 3.0]},
    "switch": {"SWITCH_FRAMES": [5, 10, 15, 20, 25, 35]},
    "restart": {"RESTART_FRAMES": [3, 5, 10], "USE_RESTART_KICKER": [True, False]},
    "poss": {"CONTROL_RADIUS_M": [1.5, 2.0, 2.5], "CONTROL_MARGIN_M": [0.75, 1.0, 1.25],
             "SWITCH_FRAMES": [8, 10, 12]},
    "kick": {"RESTART_KICK_MS": [2.0, 3.0, 4.0, 6.0], "RESTART_QUIET_FRAMES": [5, 12, 25]},
    "whistle": {"STOP_SPEED_MS": [1.0, 1.2, 1.5], "STOP_FRAMES": [25, 50],
                "STOP_BACKDATE_FRAMES": [0, 25, 50]},
    "play": {"PLAY_SPEED_MS": [1.5, 2.0], "PLAY_BALL_MS": [3.0, 5.0], "PLAY_FRAMES": [25, 50]},
    "rules": {"USE_OUT_LATCH": [True, False], "USE_WHISTLE": [True, False]},
}


def add_noise(d, sigma, seed):
    """Smooth random walk position error of about `sigma` metres on the ball and every player,
    so the clean sample tracking behaves more like automatic tracking."""
    if sigma <= 0:
        return d
    rng = np.random.default_rng(seed)
    k = np.ones(13) / 13.0

    def wobble(shape):
        w = rng.normal(0, 1, shape)
        w = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, w)
        return (w / w.std() * sigma).astype(np.float32)

    d = dict(d)
    d["ball"] = d["ball"] + wobble(d["ball"].shape)
    d["players"] = d["players"] + wobble(d["players"].shape)
    d["speed"] = ms._speed(d["players"])
    return d


def load_games(path, sigma=0.0):
    games = []
    for g in ms.GAMES:
        d = add_noise(ms.load_game(g, path), sigma, seed=g)
        feat = features.build(d, ms.PITCH)
        games.append((g, d, feat, pm.label_codes(d["labels"])))
    return games


def predict(d, feat, oracle_dead):
    dead = (d["labels"] == "D") if oracle_dead else deadball.mask(feat, forced=())
    return possession.label(feat, dead, d["halves"])


def score(games, oracle_dead):
    spell, macro = [], []
    for g, d, feat, gt in games:
        pred = pm.label_codes(predict(d, feat, oracle_dead))
        runs = pm.valid_runs(gt)
        spell.append(pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"])
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))
    return spell, macro


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir")
    parser.add_argument("--grid", default="now", choices=sorted(GRIDS))
    parser.add_argument("--noise", type=float, default=0.0,
                        help="metres of simulated tracking error (0 = clean sample data)")
    args = parser.parse_args()

    games = load_games(args.sample_dir, args.noise)
    grid = GRIDS[args.grid]
    names = list(grid)
    base = {k: getattr(C, k) for k in names}
    print("{:<48} {:>7} {:>7}   {:>7} {:>7}   per game (full spellF1)".format(
        "variant", "full", "macro", "poss", "macro"))
    rows = []
    for values in itertools.product(*grid.values()) if grid else [()]:
        for k, v in zip(names, values):
            setattr(C, k, v)
        fs, fm = score(games, False)
        ps, pmac = score(games, True)
        label = " ".join("{}={}".format(k, v) for k, v in zip(names, values)) or "current config"
        rows.append((np.mean(fs), label, np.mean(fm), np.mean(ps), np.mean(pmac), fs))
    for k, v in base.items():
        setattr(C, k, v)
    for fs_mean, label, fm, ps, pmac, fs in sorted(rows, reverse=True):
        print("{:<48} {:7.3f} {:7.3f}   {:7.3f} {:7.3f}   {}".format(
            label, fs_mean, fm, ps, pmac, " ".join("{:.3f}".format(x) for x in fs)))


if __name__ == "__main__":
    main()
