"""When does the competition tracking lose the ball? Summary numbers only.

    python tools/gap_profile.py

An automatic check on the provided tracking (no ground truth, no video). For the frames inside the
two halves it prints how often the ball is missing when the players are slow, medium or fast,
and, for each gap, how fast the ball was going and how close to a line it was just before it
disappeared. tools/train_ml.py can then place simulated gaps the same way on the external matches.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from tactico import load  # noqa: E402


def main():
    atd = load.load()
    length, width = GAMES[GAME]["pitch"]
    ball, speed = atd["ball"], atd["speed"]
    with np.errstate(invalid="ignore"):
        mean_speed = np.nanmean(speed, axis=1)
    rows = []
    starts = []
    for _, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        miss = np.isnan(ball[sl, 0])
        rows.append((miss, mean_speed[sl]))
        e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if a < 3:
                continue
            p0, p1 = ball[first + a - 3], ball[first + a - 1]
            v = np.linalg.norm(p1 - p0) / 2 * 25
            edge = min(p1[0], length - p1[0], p1[1], width - p1[1])
            starts.append((b - a, v, edge, np.nanmean(mean_speed[first + max(0, a - 25):first + a])))
    miss = np.concatenate([r[0] for r in rows])
    sp = np.concatenate([r[1] for r in rows])
    print("ball missing by mean player speed:")
    for lo, hi, name in ((0, 1.0, "slow  < 1.0 m/s"), (1.0, 2.0, "medium 1-2 m/s"), (2.0, 99, "fast  > 2.0 m/s")):
        m = (sp >= lo) & (sp < hi)
        print("  {}   {:4.0%} of frames are like this, ball missing on {:4.0%} of them".format(
            name, m.mean(), miss[m].mean() if m.any() else float("nan")))
    s = np.array(starts)
    print("{} gaps. Just before the ball disappears:".format(len(s)))
    for q in (25, 50, 75):
        print("  {}th pct: ball speed {:5.1f} m/s, distance to nearest line {:5.1f} m, "
              "player speed {:4.2f} m/s".format(q, np.percentile(s[:, 1], q), np.percentile(s[:, 2], q),
                                                np.nanpercentile(s[:, 3], q)))
    long_gaps = s[s[:, 0] > 250]
    print("gaps over 10 s: {}, of which {:.0%} start within 3 m of a line, {:.0%} with the ball "
          "faster than 10 m/s".format(len(long_gaps), np.mean(long_gaps[:, 2] < 3),
                                       np.mean(long_gaps[:, 1] > 10)))


if __name__ == "__main__":
    main()
