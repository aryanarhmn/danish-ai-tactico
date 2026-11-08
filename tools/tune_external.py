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
import dfl_open  # noqa: E402
import metrica_sample as ms  # noqa: E402
from tactico import config as C, deadball, features, possession  # noqa: E402

GRIDS = {
    "now": {},
    "rel": {"CONTROL_REL_SPEED_MS": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 99.0]},
    "relposs": {"CONTROL_REL_SPEED_MS": [5.0, 6.0, 8.0, 99.0], "CONTROL_MARGIN_M": [0.75, 1.0, 1.25],
                "SWITCH_FRAMES": [8, 10, 12]},
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


def add_noise(d, sigma, seed, sigma_ball=None):
    """Smooth random position error of about `sigma` metres on every player and `sigma_ball`
    (default: sigma) on the ball, so the clean sample tracking behaves more like automatic
    tracking. tools/noise_match.py measures which level matches the competition tracking."""
    sigma_ball = sigma if sigma_ball is None else sigma_ball
    if sigma <= 0 and sigma_ball <= 0:
        return d
    rng = np.random.default_rng(seed)
    k = np.ones(13) / 13.0

    def wobble(shape, size):
        if size <= 0:
            return np.zeros(shape, np.float32)
        w = rng.normal(0, 1, shape)
        w = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, w)
        return (w / w.std() * size).astype(np.float32)

    d = dict(d)
    d["ball"] = d["ball"] + wobble(d["ball"].shape, sigma_ball)
    d["players"] = d["players"] + wobble(d["players"].shape, sigma)
    d["speed"] = ms._speed(d["players"])
    return d


def drop_ball(d, frac, mean_gap, seed):
    """Remove the ball on about `frac` of frames, in gaps of mean `mean_gap` frames.

    The competition tracking misses the ball on 41 % of graded frames (CHALLENGE.md). The external
    games have it everywhere, so without this a tuner relies on a ball that will not be there.
    Alternating visible / missing stretches with exponential lengths."""
    if frac <= 0:
        return d
    rng = np.random.default_rng(1000 + seed)
    n = len(d["ball"])
    mean_vis = mean_gap * (1 - frac) / frac
    miss = np.zeros(n, dtype=bool)
    t, missing = 0, rng.random() < frac
    while t < n:
        length = max(1, int(rng.exponential(mean_gap if missing else mean_vis)))
        if missing:
            miss[t:t + length] = True
        t += length
        missing = not missing
    d = dict(d)
    d["ball"] = d["ball"].copy()
    d["ball"][miss] = np.nan
    return d


def drop_players(d, frac, mean_gap, seed):
    """Remove each player track on about `frac` of frames, in gaps of mean `mean_gap` frames:
    automatic tracking loses players behind others and at the edge of the picture."""
    if frac <= 0:
        return d
    rng = np.random.default_rng(2000 + seed)
    n, s = d["players"].shape[:2]
    players = d["players"].copy()
    speed = d["speed"].copy() if d.get("speed") is not None else None
    mean_vis = mean_gap * (1 - frac) / frac
    for k in range(s):
        t, missing = 0, rng.random() < frac
        while t < n:
            length = max(1, int(rng.exponential(mean_gap if missing else mean_vis)))
            if missing:
                players[t:t + length, k] = np.nan
                if speed is not None:
                    speed[t:t + length, k] = np.nan
            t += length
            missing = not missing
    d = dict(d)
    d["players"], d["speed"] = players, speed
    return d


def degrade(d, noise, ball_missing, gap, player_drop, seed):
    d = add_noise(d, noise, seed=seed)
    d = drop_ball(d, ball_missing, gap, seed=seed)
    return drop_players(d, player_drop, 125, seed=seed)


def raw_games(data, sample_dir, dfl_dir):
    out = []
    if data in ("metrica", "all"):
        out += [("M{}".format(g), ms.load_game(g, sample_dir)) for g in ms.GAMES]
    if data in ("dfl", "all"):
        out += [("D" + os.path.basename(p)[-10:-4], dfl_open.load_match(p))
                for p in dfl_open.list_matches(dfl_dir)]
    return out


def load_games(data="metrica", sample_dir=None, dfl_dir=None, sigma=0.0, sigma_ball=None,
               ball_missing=0.0, gap=50):
    games = []
    for i, (name, d) in enumerate(raw_games(data, sample_dir, dfl_dir)):
        d = add_noise(d, sigma, seed=i + 1, sigma_ball=sigma_ball)
        d = drop_ball(d, ball_missing, gap, seed=i + 1)
        feat = features.build(d, ms.PITCH)
        games.append((name, d, feat, pm.label_codes(d["labels"])))
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
    parser.add_argument("--dfl-dir", help="folder of .npz from tools/external/dfl_open.py")
    parser.add_argument("--data", default="metrica", choices=("metrica", "dfl", "all"))
    parser.add_argument("--ball-missing", type=float, default=0.0,
                        help="fraction of frames to drop the ball on (competition: 0.41)")
    parser.add_argument("--gap", type=int, default=50, help="mean length of a ball gap, frames")
    parser.add_argument("--grid", default="now", choices=sorted(GRIDS))
    parser.add_argument("--noise", type=float, default=0.0,
                        help="metres of simulated player tracking error (0 = clean sample data)")
    parser.add_argument("--noise-ball", type=float, default=None,
                        help="metres of simulated ball error (default: same as --noise)")
    args = parser.parse_args()

    games = load_games(args.data, args.sample_dir, args.dfl_dir, args.noise, args.noise_ball,
                       args.ball_missing, args.gap)
    print("games:", " ".join(g[0] for g in games))
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
