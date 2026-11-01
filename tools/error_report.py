"""Where do we disagree with a ground truth? Biggest mistakes first.

    python tools/error_report.py val/mit_facit.csv
"""
import argparse
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS = os.path.join(ROOT, "evaluation", "results", "Danish_AI-Tactico.csv")
FPS = 25
NAMES = {"H": "Barca", "A": "Athletic", "D": "dead", "X": "none"}


def read(path):
    with open(path) as handle:
        next(handle)
        return [line.rstrip().split(",")[1] for line in handle]


def clock(frame):
    s = frame / FPS
    return "{}:{:02d}:{:04.1f}".format(int(s // 3600), int(s // 60 % 60), s % 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt")
    parser.add_argument("--ours", default=OURS)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    gt, ours = read(args.gt), read(args.ours)
    stretches, kinds = [], Counter()
    start, cur = None, None
    for f in range(len(gt) + 1):
        wrong = f < len(gt) and gt[f] != "X" and ours[f] != gt[f]
        key = (gt[f], ours[f]) if wrong else None
        if start is not None and key != cur:
            stretches.append((f - start, start, f, cur))
            kinds[cur] += f - start
            start = None
        if wrong and start is None:
            start, cur = f, key
    graded = sum(1 for g in gt if g != "X")
    print("graded frames {}  wrong {} ({:.0f}%)\n".format(
        graded, sum(kinds.values()), 100.0 * sum(kinds.values()) / max(graded, 1)))
    print("mistakes by kind (truth -> ours), seconds:")
    for (g, o), n in kinds.most_common():
        print("  {:>8} -> {:<8} {:6.1f} s".format(NAMES[g], NAMES[o], n / FPS))
    print("\nlongest stretches (watch these in the video):")
    print("  from        to          secs  truth     ours")
    for n, a, b, (g, o) in sorted(stretches, reverse=True)[:args.top]:
        print("  {}  {}  {:5.1f}  {:<8}  {}".format(clock(a), clock(b), n / FPS, NAMES[g], NAMES[o]))


if __name__ == "__main__":
    main()
