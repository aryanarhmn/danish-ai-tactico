"""How noisy is the competition tracking compared with the clean sample games?

    python tools/noise_match.py --sample-dir C:\\mtah-ext\\sample-data

An automatic check on the provided tracking only (no ground truth, no video): it measures the
frame-to-frame jitter of player and ball positions in the competition ATD and in the sample games
with increasing simulated noise (tools/tune_external.py --noise), and prints the noise level whose
jitter matches. Use that level for --noise when tuning. Only summary numbers are printed.

Jitter = median over moving objects of |x[t+1] - 2 x[t] + x[t-1]| in centimetres, i.e. the
second difference: zero for steady motion, so it measures tracking wobble, not running.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "external"))
import metrica_sample as ms  # noqa: E402
from tune_external import add_noise  # noqa: E402

LEVELS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45)


def jitter(pos):
    """Median second-difference magnitude in cm, ignoring NaN."""
    d2 = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    m = np.sqrt(np.sum(d2 * d2, axis=-1)).ravel()
    m = m[~np.isnan(m)]
    return float(np.median(m) * 100) if len(m) else float("nan")


def stats(d):
    return jitter(d["players"]), jitter(d["ball"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir")
    parser.add_argument("--no-competition", action="store_true", help="sample games only")
    args = parser.parse_args()

    if not args.no_competition:
        sys.path.insert(0, os.path.join(ROOT, "starter"))
        from games import GAME, half_video_blocks
        from tactico import load
        atd = load.load()
        p, b = stats(atd)
        print("competition ATD       players {:5.2f} cm   ball {:5.2f} cm".format(p, b))
        # ball gaps and players in view, inside the two halves only
        lens, miss, tot, seen = [], 0, 0, {0: [], 1: []}
        for _, first, last in half_video_blocks(GAME):
            m = np.isnan(atd["ball"][first:last + 1, 0])
            miss += int(m.sum())
            tot += len(m)
            e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
            lens += list(np.flatnonzero(e == -1) - np.flatnonzero(e == 1))
            vis = ~np.isnan(atd["players"][first:last + 1, :, 0])
            for t in (0, 1):
                seen[t].append(vis[:, atd["team"] == t].sum(axis=1))
        lens = np.array(lens)
        print("ball missing {:.0%} of frames, {} gaps, mean {:.1f} s, median {:.1f} s, "
              "90th pct {:.1f} s".format(miss / tot, len(lens), lens.mean() / 25,
                                       np.median(lens) / 25, np.percentile(lens, 90) / 25))
        for t, name in ((0, "home"), (1, "away")):
            v = np.concatenate(seen[t])
            print("{} players in view per frame: mean {:.1f}, 10th pct {:.0f}, 90th pct {:.0f}".format(
                name, v.mean(), np.percentile(v, 10), np.percentile(v, 90)))
        print("player speed: median {:.2f}, 90th pct {:.2f} (training data is m/s: median about 1.4)".format(
            float(np.nanmedian(atd["speed"])), float(np.nanpercentile(atd["speed"], 90))))
        print("track slots: {} home, {} away, {} other".format(
            int(np.sum(atd["team"] == 0)), int(np.sum(atd["team"] == 1)),
            int(np.sum((atd["team"] != 0) & (atd["team"] != 1)))))

    games = [ms.load_game(g, args.sample_dir) for g in ms.GAMES]
    rows = []
    for sigma in LEVELS:
        ps, bs = zip(*[stats(add_noise(d, sigma, seed=g)) for g, d in zip(ms.GAMES, games)])
        rows.append((sigma, np.mean(ps), np.mean(bs)))
        print("sample, noise {:4.2f} m  players {:5.2f} cm   ball {:5.2f} cm".format(
            sigma, np.mean(ps), np.mean(bs)))
    if not args.no_competition:
        best_p = min(rows, key=lambda r: abs(r[1] - p))[0]
        best_b = min(rows, key=lambda r: abs(r[2] - b))[0]
        print("closest noise level:  players {:.2f} m   ball {:.2f} m".format(best_p, best_b))


if __name__ == "__main__":
    main()
