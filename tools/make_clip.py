"""Cut a short clip of the tactical video with the tracking and the labels drawn on it.

    pip install opencv-python                 (once)
    python tools/make_clip.py 12600 13100     # video frames, start and end

Writes C:\\mtah-data\\clips\\clip_12600_13100.mp4 (MTAH_DATA_DIR\\clips), next to the data and
outside the repository: the clip is match video and must not be shared or committed.

On every frame it draws:
    - the tracked players, blue = Barcelona (H), red = Athletic (A), and the ball in yellow,
      projected through the homography. If the dots sit on the players, video and tracking line up.
    - a banner: frame number, video time, ground truth (public window only), our label, and what
      the rules saw (dead ball, keeper hold, candidate team).

This removes any doubt about timestamps: the frame number on screen IS the row in the submission.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402
from tactico import config, deadball, features, keeper, load, possession  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OURS_PATH = os.path.join(ROOT, "evaluation", "results", "Danish_AI-Tactico.csv")
COLOURS = {0: (230, 120, 30), 1: (40, 40, 220)}   # BGR: blue-ish home, red away
BALL = (0, 230, 255)


def read_labels(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip().split(",")[1] for line in handle])


def main():
    import cv2
    from load_homography import load as load_h, project

    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    args = parser.parse_args()

    video_path = require(["video"])[0]
    length, width = GAMES[GAME]["pitch"]

    atd = load.load()
    feat = features.build(atd, (length, width))
    events = None
    if config.USE_SMART_TAGGING:
        from load_smart_tagging import load as load_tagging
        events = load_tagging()
    dead = deadball.mask(feat, events)
    cand = possession.candidates(feat)
    hold = keeper.hold_team(atd, feat, half_video_blocks(GAME), (length, width))
    gt = read_labels(GT_PATH)
    ours = read_labels(OURS_PATH)
    H, _, valid = load_h()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.join(os.environ.get("MTAH_DATA_DIR") or DATA_DIR, "clips")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clip_{}_{}.mp4".format(args.start, args.end))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for f in range(args.start, args.end + 1):
        ok, img = cap.read()
        if not ok:
            break
        if valid[f]:
            m = H[f]
            for s, side in enumerate(atd["team"]):
                x, y = atd["players"][f, s]
                if np.isnan(x):
                    continue
                px, py = project(m, x / length, y / width)
                cv2.circle(img, (int(px), int(py)), 9, COLOURS[int(side)], 2)
            bx, by = atd["ball"][f]
            if not np.isnan(bx):
                px, py = project(m, bx / length, by / width)
                cv2.circle(img, (int(px), int(py)), 6, BALL, -1)

        t = f / config.FPS
        g = gt[f] if gt is not None else "?"
        o = ours[f] if ours is not None else "?"
        c = {0: "H", 1: "A"}.get(int(cand[f]), "-")
        k = {0: "H", 1: "A"}.get(int(hold[f]), "-")
        text = "frame {}  {:02d}:{:05.2f}   GT {}   ours {}   dead {}  keeper {}  cand {}".format(
            f, int(t // 60), t % 60, g if g != "X" else "-", o, int(dead[f]), k, c)
        cv2.rectangle(img, (0, 0), (w, 48), (0, 0, 0), -1)
        cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)

    writer.release()
    cap.release()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
