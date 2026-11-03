"""Look at what the model saw around a moment, second by second (or frame by frame).

    python tools/inspect_frames.py 12800 13050            # one line per 5 frames
    python tools/inspect_frames.py 12800 13050 --step 1   # every frame

Columns:
    frame  video time  ball  d_H  d_A  ballspd  plyspd  out  dead  cand
    ball   1 if the tracker sees the ball
    d_H/d_A nearest Barca / Athletic player to the ball, metres
    cand   which team the possession rule sees as in control on that frame (H, A or -)
    keeper whose goalkeeper is holding the ball (H, A or -), see tactico/keeper.py

This prints match data. Keep the output on your own machine: do not paste it into a cloud AI.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
from tactico import config, deadball, features, keeper, load, possession  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OURS_PATH = os.path.join(ROOT, "evaluation", "results", "Danish_AI-Tactico.csv")


def read_labels(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip().split(",")[1] for line in handle])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    atd = load.load()
    feat = features.build(atd, GAMES[GAME]["pitch"])
    events = None
    if config.USE_SMART_TAGGING:
        from load_smart_tagging import load as load_tagging
        events = load_tagging()
    dead = deadball.mask(feat, events)
    cand = possession.candidates(feat)
    from games import half_video_blocks
    hold = keeper.hold_team(atd, feat, half_video_blocks(GAME), GAMES[GAME]["pitch"])

    print("frame   time      ball   d_H    d_A  ballspd plyspd out dead cand keeper")
    for f in range(args.start, args.end + 1, args.step):
        t = f / config.FPS
        c = {0: "H", 1: "A"}.get(int(cand[f]), "-")
        k = {0: "H", 1: "A"}.get(int(hold[f]), "-")
        print("{:6d}  {:02d}:{:05.2f}  {}   {:5.1f}  {:5.1f}  {:6.1f}  {:5.1f}   {}   {}    {}    {}".format(
            f, int(t // 60), t % 60, int(feat["ball_visible"][f]),
            feat["d_home"][f], feat["d_away"][f], feat["ball_speed"][f],
            feat["player_speed"][f], int(feat["ball_out"][f]), int(dead[f]), c, k))


if __name__ == "__main__":
    main()
