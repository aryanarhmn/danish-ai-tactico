"""Read the tactical ATD once and cache it as numpy arrays.

The raw file is ~199k lines of text and takes a minute or two to parse. The cache (cache/atd.npz)
makes every later run take seconds. Delete the cache to force a re-read.

Arrays, all indexed by VIDEO frame (0 .. 199318), positions in metres:
    ball      (n, 2)      float32, NaN where the ball is not tracked
    players   (n, S, 2)   float32, NaN where a slot is empty
    speed     (n, S)      float32, m/s as reported by the tracker
    team      (S,)        int8, 0 = home (Barcelona), 1 = away (Athletic)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "atd.npz")


def _parse(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                continue

            b = parts[2]
            if b and b[0] != "N":
                values = b.split(",")
                ball[frame] = (float(values[0]) * length, float(values[1]) * width)

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                values = group.split(",")
                players[frame, slot] = (float(values[0]) * length, float(values[1]) * width)
                if len(values) > 2 and values[2][:1] != "N":
                    speed[frame, slot] = float(values[2])
    return ball, players, speed


def load(game=GAME, data_dir=None, refresh=False):
    """dict of arrays (see module docstring). Parses the raw ATD on first use, then reads cache."""
    if not refresh and os.path.isfile(CACHE_PATH):
        with np.load(CACHE_PATH) as cached:
            return {key: cached[key] for key in cached.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)
    print("parsing {} (first run only, takes a minute or two) ...".format(
        os.path.basename(raw_path)))
    ball, players, speed = _parse(raw_path, n_frames, len(team), GAMES[game]["pitch"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE_PATH, ball=ball, players=players, speed=speed, team=team)
    return {"ball": ball, "players": players, "speed": speed, "team": team}
