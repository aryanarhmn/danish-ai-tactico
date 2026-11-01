"""Danish AI-Tactico -- frame-level possession for More than a Hack 2027.

    python run.py                                   # writes results/Danish_AI-Tactico.csv
    python run.py --out results/test.csv
    python run.py --no-tagging                      # ignore Smart Tagging
    python run.py --refresh                         # re-parse the ATD instead of using the cache

Deterministic: same data in, byte-identical file out. All tunable numbers are in tactico/config.py.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from tactico import config, deadball, features, keeper, load, possession  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "Danish_AI-Tactico.csv")


def write_csv(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def summary(labels, halves):
    graded = np.zeros(len(labels), dtype=bool)
    for _, a, b in halves:
        graded[a:b + 1] = True
    counts = {k: int(np.sum((labels == k) & graded)) for k in "HADX"}
    spells = []
    for _, a, b in halves:
        block = labels[a:b + 1]
        block = block[block != "X"]  # the scorer forward-fills X, so ignore it here too
        start = 0
        for i in range(1, len(block) + 1):
            if i == len(block) or block[i] != block[start]:
                if block[start] in "HA":
                    spells.append(i - start)
                start = i
    mean_s = np.mean(spells) / config.FPS if spells else 0.0
    print("graded frames  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("possessions    {}  mean {:.1f} s  (public GT: 17 in 5 min, mean 14.2 s)".format(
        len(spells), mean_s))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-tagging", action="store_true")
    parser.add_argument("--no-keeper", action="store_true")
    parser.add_argument("--no-out-latch", action="store_true")
    parser.add_argument("--no-whistle", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.no_tagging:
        config.USE_SMART_TAGGING = False
    if args.no_keeper:
        config.USE_KEEPER = False
    if args.no_out_latch:
        config.USE_OUT_LATCH = False
    if args.no_whistle:
        config.USE_WHISTLE = False

    atd = load.load(refresh=args.refresh)
    feat = features.build(atd, GAMES[GAME]["pitch"])

    events = None
    if config.USE_SMART_TAGGING:
        from load_smart_tagging import load as load_tagging
        events = load_tagging()

    dead = deadball.mask(feat, events)
    halves = half_video_blocks(GAME)
    hold = keeper.hold_team(atd, feat, halves, GAMES[GAME]["pitch"]) if config.USE_KEEPER else None
    labels = possession.label(feat, dead, halves, hold)

    write_csv(args.out, labels)
    summary(labels, halves)
    print("wrote", os.path.relpath(args.out, ROOT))
    print("next:  python evaluation/validate_submission.py", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
