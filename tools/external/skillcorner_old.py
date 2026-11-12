"""SkillCorner's first open data release: 9 European matches of 2019/20 broadcast tracking.

EXTERNAL DATA for training (MIT licence, the same repository as tools/external/skillcorner_open.py;
SkillCorner asks to be credited). The files were removed from the repository in October 2025 and
stay in its git history, commit 1b99d40:

    git clone --filter=blob:none --no-checkout https://github.com/SkillCorner/opendata.git C:\\mtah-ext\\sc-old-repo
    python tools/external/skillcorner_old.py --repo C:\\mtah-ext\\sc-old-repo --out C:\\mtah-ext\\sc-old

Each 10 fps frame carries SkillCorner's possession group and, while a player controls the ball,
that player. The group alone stays set through some stoppages, and there are no stoppage events,
so dead balls cannot be labelled. We therefore label only frames where a named player has the ball
(H or A, about a fifth of the match); every other frame is left unlabelled ("X") and is ignored in
training. These matches are
used for training only, never as a test set.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

COMMIT = "1b99d40"
MATCHES = (2068, 2269, 2417, 2440, 2841, 3442, 3518, 3749, 4039)


def git_json(repo, path):
    out = subprocess.run(["git", "-C", repo, "show", "{}:{}".format(COMMIT, path)],
                         capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8"))


def convert(repo, match_id):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from skillcorner_open import FPS_IN, FPS_OUT, _resample
    meta = git_json(repo, "data/matches/{}/match_data.json".format(match_id))
    frames = git_json(repo, "data/matches/{}/structured_data.json".format(match_id))
    home_id = meta["home_team"]["id"]
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    ball_obj = meta["ball"]["trackable_object"]
    team_of = {p["trackable_object"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}
    ids = sorted(team_of)
    slot = {obj: i for i, obj in enumerate(ids)}

    n10 = len(frames)
    ball = np.full((n10, 2), np.nan)
    players = np.full((n10, len(ids), 2), np.nan)
    period = np.zeros(n10, dtype=np.int8)
    lab10 = np.full(n10, "X", dtype="<U1")
    for k, f in enumerate(frames):
        period[k] = f.get("period") or 0
        for d in f.get("data") or []:
            obj = d.get("trackable_object")
            if obj == ball_obj:
                ball[k] = (d["x"] + length / 2, d["y"] + width / 2)
            elif obj in slot:
                players[k, slot[obj]] = (d["x"] + length / 2, d["y"] + width / 2)
        poss = f.get("possession") or {}
        g = poss.get("group")
        # only frames where a named player has the ball: the group alone stays set through some
        # stoppages, and these matches have no events to mark them
        if period[k] and poss.get("trackable_object") is not None and g in ("home team", "away team"):
            lab10[k] = "H" if g == "home team" else "A"

    n = int((n10 - 1) * FPS_OUT / FPS_IN) + 1
    near = np.round(np.arange(n) * FPS_IN / FPS_OUT).astype(int).clip(0, n10 - 1)
    per25 = period[near]
    halves = []
    for p in (1, 2):
        idx = np.flatnonzero(per25 == p)
        if len(idx):
            halves.append((p, int(idx[0]), int(idx[-1])))
    sx, sy = 105.0 / length, 68.0 / width
    return {"ball": (_resample(ball, n) * [sx, sy]).astype(np.float32),
            "players": (_resample(players, n) * [sx, sy]).astype(np.float32),
            "speed": None, "team": np.array([team_of[o] for o in ids], dtype=np.int8),
            "halves": np.array(halves), "labels": lab10[near]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="clone of github.com/SkillCorner/opendata")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrica_sample import _speed
    os.makedirs(args.out, exist_ok=True)
    for m in MATCHES:
        d = convert(args.repo, m)
        d["speed"] = _speed(d["players"])
        np.savez_compressed(os.path.join(args.out, "SCO-{}.npz".format(m)), **d)
        lab = d["labels"]
        inside = np.zeros(len(lab), bool)
        for _, a, b in d["halves"]:
            inside[a:b + 1] = True
        print("{}  {} frames  labelled {:.0%} (H {:.0%} A {:.0%})  ball missing {:.0%}".format(
            m, len(lab), np.mean(lab[inside] != "X"), np.mean(lab[inside] == "H"),
            np.mean(lab[inside] == "A"), np.mean(np.isnan(d["ball"][inside, 0]))), flush=True)


if __name__ == "__main__":
    main()
