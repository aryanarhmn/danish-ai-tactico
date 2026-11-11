"""Danish AI-Tactico -- frame-level possession for More than a Hack 2027.

    python run.py                                   # writes results/Danish_AI-Tactico.csv
    python run.py --out results/test.csv
    python run.py --no-tagging                      # ignore Smart Tagging
    python run.py --refresh                         # re-parse the ATD instead of using the cache
    python run.py --model models/possession.pkl     # learned model instead of the rules
    python run.py --model models/possession.pkl --seq models/sequence.pt   # classifier + GRU average
    python run.py --model models/possession.pkl --seq models/sequence.pt --seq-weight 0.8 --switch-cost 4 --dead-cost 12

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
    parser.add_argument("--no-kicker", action="store_true")
    parser.add_argument("--no-goal", action="store_true")
    parser.add_argument("--no-lost", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--model", help="learned model from tools/train_ml.py (external data only)")
    parser.add_argument("--seq", help="GRU from tools/train_seq.py; its probabilities are averaged "
                        "with the classifier's before decoding (needs --model and torch)")
    parser.add_argument("--seq-weight", type=float, default=0.5,
                        help="weight of the GRU in the average (tools/blend_cv.py chose 0.8)")
    parser.add_argument("--switch-cost", type=float, help="override the decoding cost stored in --model")
    parser.add_argument("--dead-cost", type=float, help="override the decoding cost stored in --model")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a value in tactico/config.py for this run, e.g. --set CONTROL_MARGIN_M=1.0")
    args = parser.parse_args()

    for item in args.set:
        name, _, value = item.partition("=")
        if not hasattr(config, name):
            parser.error("no setting called {} in tactico/config.py".format(name))
        old = getattr(config, name)
        setattr(config, name, (value.lower() == "true") if isinstance(old, bool) else type(old)(value))

    if args.no_tagging:
        config.USE_SMART_TAGGING = False
    if args.no_keeper:
        config.USE_KEEPER = False
    if args.no_out_latch:
        config.USE_OUT_LATCH = False
    if args.no_whistle:
        config.USE_WHISTLE = False
    if args.no_kicker:
        config.USE_RESTART_KICKER = False
    if args.no_goal:
        config.USE_GOAL = False
    if args.no_lost:
        config.USE_LOST_OUT = False

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
    if args.model:
        import pickle
        from tactico import ml
        with open(args.model, "rb") as handle:
            bundle = pickle.load(handle)
        d = {"ball": atd["ball"], "players": atd["players"], "speed": atd["speed"],
             "team": atd["team"], "halves": halves}
        X, names = ml.features(d, GAMES[GAME]["pitch"])
        if names != bundle["features"]:
            raise SystemExit("model was trained on other features; retrain it")
        prob = ml.predict_proba(bundle["model"], X)
        if args.seq:
            from tactico import seq
            w = args.seq_weight
            prob = (1 - w) * prob + w * seq.predict_proba(seq.load(args.seq), X.astype(np.float32), halves)
            print("sequence model {} (weight {} in the average)".format(os.path.basename(args.seq), w))
        if args.switch_cost is not None:
            bundle["switch_cost"] = args.switch_cost
        if args.dead_cost is not None:
            bundle["dead_cost"] = args.dead_cost
        labels = ml.label(prob, halves, bundle["switch_cost"], bundle["dead_cost"])
        gap = np.zeros(len(labels), dtype=bool)
        gap[deadball.SECOND_HALF_GRADED_FROM:deadball.SECOND_HALF_KICKOFF] = True
        labels[gap & (labels != "X")] = "D"
        print("model          {} (switch cost {}, dead cost {})".format(
            os.path.basename(args.model), bundle["switch_cost"], bundle["dead_cost"]))

    if config.USE_GOAL:
        found = deadball.goals(feat)
        print("goal candidates {} (ball behind the line or lost at the goal mouth)".format(
            len(deadball.runs(deadball.goal_candidates(feat)))))
        print("goals found    {}  at video {}".format(len(found), ", ".join(
            "{}:{:02d}:{:02d}".format(a // 90000, a // 1500 % 60, a // 25 % 60) for a, _ in found) or "-"))
    write_csv(args.out, labels)
    summary(labels, halves)
    print("wrote", os.path.relpath(args.out, ROOT))
    print("next:  python evaluation/validate_submission.py", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
