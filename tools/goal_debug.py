"""Why does the goal rule not confirm a goal? One line per goal candidate.

    python tools/goal_debug.py

For each candidate (ball behind the goal line, or lost at the goal mouth) it prints the video time,
then over the next GOAL_MAX_FRAMES: when the ball first comes within 2 m and 5 m of the centre spot,
the longest run of frames it is tracked within 2 m, the longest run within 2 m AND slower than
RESTART_QUIET_MS, and the most open play in a row. Only times and counts: safe to share.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
from tactico import config as C, deadball, features, load  # noqa: E402


def vt(f):
    s = int(f) // C.FPS
    return "{}:{:02d}:{:02d}".format(s // 3600, s // 60 % 60, s % 60)


def longest(mask):
    r = deadball.runs(mask)
    return max((b - a for a, b in r), default=0)


def main():
    feat = features.build(load.load(), GAMES[GAME]["pitch"])
    x, y, sp = feat["ball_x"], feat["ball_y"], feat["ball_speed"]
    length, width = feat["pitch"]
    with np.errstate(invalid="ignore"):
        dc = np.hypot(x - length / 2, y - width / 2)
        players = deadball._rolling_mean(feat["player_speed"], C.STOP_WINDOW_FRAMES)
        ballm = deadball._rolling_mean(sp, C.STOP_WINDOW_FRAMES)
        playing = (players >= C.PLAY_SPEED_MS) & (ballm >= C.PLAY_BALL_MS) & ~feat["ball_out"]
    print("cand at    first<5m  first<2m  tracked<2m  quiet<2m  max open play   (frames; 25 = 1 s)")
    for a, _ in deadball.runs(deadball.goal_candidates(feat)):
        b = min(len(x), a + C.GOAL_MAX_FRAMES)
        w = dc[a:b]
        with np.errstate(invalid="ignore"):
            near5, near2 = np.flatnonzero(w <= 5), np.flatnonzero(w <= C.KICKOFF_RADIUS_M)
            quiet2 = (w <= C.KICKOFF_RADIUS_M) & (sp[a:b] < C.RESTART_QUIET_MS)
        print("{}   {:>8}  {:>8}  {:>10}  {:>8}  {:>13}".format(
            vt(a),
            vt(a + near5[0]) if len(near5) else "-",
            vt(a + near2[0]) if len(near2) else "-",
            longest(w <= C.KICKOFF_RADIUS_M), longest(quiet2), longest(playing[a:b])))


if __name__ == "__main__":
    main()
