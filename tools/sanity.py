"""Compare submission files without any ground truth.

    python tools/sanity.py evaluation/results/Danish_AI-Tactico.csv evaluation/results/rules.csv

For each file: share of H/A/D on the two halves, number of possessions, mean possession length,
possessions per 5 minutes. For two files: how often they agree. No ground truth is read, so this
is an automatic check on the provided data only, and it cannot be used to tune anything.
For reference, the external DFL matches have about 45 % dead ball and 26 possessions per 5 minutes.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402


def read(path):
    with open(path) as handle:
        next(handle)
        return np.array([line.strip().split(",")[1] for line in handle])


def mask(n):
    m = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        m[first:last + 1] = True
    return m


def stats(lab, m):
    shares = {k: float(np.mean(lab[m] == k)) for k in "HAD"}
    spells, length = 0, []
    for _, first, last in half_video_blocks(GAME):
        seg = lab[first:last + 1]
        run, cur = 0, None
        for v in seg:
            if v in "HA" and v == cur:
                run += 1
            else:
                if cur in ("H", "A"):
                    spells += 1
                    length.append(run)
                cur, run = v, 1
        if cur in ("H", "A"):
            spells += 1
            length.append(run)
    minutes = m.sum() / 25 / 60
    return shares, spells, np.mean(length) / 25 if length else 0, spells / minutes * 5


def main():
    files = sys.argv[1:]
    labs = [read(f) for f in files]
    m = mask(len(labs[0]))
    for f, lab in zip(files, labs):
        sh, n, mean, per5 = stats(lab, m)
        print("{:<32} H {:4.0%}  A {:4.0%}  D {:4.0%}   {} possessions, mean {:.1f} s, "
              "{:.1f} per 5 min".format(os.path.basename(f), sh["H"], sh["A"], sh["D"], n, mean, per5))
    if len(labs) == 2:
        a, b = labs[0][m], labs[1][m]
        print("agreement {:.0%}  (both live: same team {:.0%})".format(
            np.mean(a == b), np.mean((a == b)[np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"])])))


if __name__ == "__main__":
    main()
