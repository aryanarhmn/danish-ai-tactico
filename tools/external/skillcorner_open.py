"""SkillCorner open data: 20 A-League 2024/25 matches of automatic broadcast tracking.

EXTERNAL DATA for training (MIT licence; SkillCorner asks to be credited).
https://github.com/SkillCorner/opendata

Why it helps: this is automatic tracking from video, like the competition's, so players and the
ball really go missing, instead of being removed by our simulation. Each frame carries SkillCorner's
own possession group (home team / away team / none), and the dynamic events mark every stoppage
(throw-in, goal kick, free kick, corner, goal) before and after a player possession.

Labels per frame:
  D  from the end of a player possession followed by an interruption, to the start of the next
     player possession that comes from an interruption;
  H/A  SkillCorner's possession group; frames without a group (loose ball) keep the last team.
Positions SkillCorner extrapolated (is_detected false) are dropped, so gaps stay gaps. The 10 fps
data is resampled to 25 fps by linear interpolation between detected frames 0.1 s apart.

    git clone --depth 1 https://github.com/SkillCorner/opendata.git C:\\mtah-ext\\skillcorner
    python tools/external/skillcorner_open.py --src C:\\mtah-ext\\skillcorner\\data\\matches --out C:\\mtah-ext\\sc

The tracking files are stored with Git LFS (about 1.8 GB); Git for Windows fetches them during the
clone. If a file is still a pointer, the script prints the direct download link.
"""

import argparse
import csv
import json
import os

import numpy as np

FPS_IN, FPS_OUT = 10, 25
RAW = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{m}/{m}_tracking_extrapolated.jsonl"


def _resample(arr10, n_out):
    """(n10, ...) -> (n_out, ...) linear interpolation; NaN unless both neighbours are known."""
    t = np.arange(n_out) * FPS_IN / FPS_OUT
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(arr10) - 1)
    w = (t - i0).reshape((-1,) + (1,) * (arr10.ndim - 1))
    a, b = arr10[i0], arr10[i1]
    out = a * (1 - w) + b * w
    exact = (w == 0)
    return np.where(exact, a, out)


def convert(folder, match_id):
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(match_id)), encoding="utf-8"))
    home_id = meta["home_team"]["id"]
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    team_of = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}
    ids = sorted(team_of)
    slot = {pid: i for i, pid in enumerate(ids)}

    frames = []
    with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id)), encoding="utf-8") as handle:
        for line in handle:
            frames.append(json.loads(line))
    n10 = len(frames)
    ball = np.full((n10, 2), np.nan)
    players = np.full((n10, len(ids), 2), np.nan)
    group = np.full(n10, -1, dtype=np.int8)
    period = np.zeros(n10, dtype=np.int8)
    for k, f in enumerate(frames):
        period[k] = f["period"] or 0
        b = f["ball_data"]
        if b and b.get("is_detected") and b.get("x") is not None:
            ball[k] = (b["x"] + length / 2, b["y"] + width / 2)
        g = (f.get("possession") or {}).get("group")
        group[k] = 0 if g == "home team" else 1 if g == "away team" else -1
        for p in f["player_data"]:
            if p.get("is_detected") and p["player_id"] in slot:
                players[k, slot[p["player_id"]]] = (p["x"] + length / 2, p["y"] + width / 2)

    # dead-ball intervals from the dynamic events (frame numbers are 10 fps frame indices)
    dead10 = np.zeros(n10, dtype=bool)
    poss = []
    with open(os.path.join(folder, "{}_dynamic_events.csv".format(match_id)), encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            if r["event_type"] == "player_possession":
                poss.append((int(r["frame_start"]), int(r["frame_end"]), int(r["period"] or 0),
                             r.get("game_interruption_before") or "", r.get("game_interruption_after") or ""))
    poss.sort()
    for (s0, e0, p0, _, after), (s1, e1, p1, before, _) in zip(poss, poss[1:]):
        if p0 == p1 and (after or before) and s1 > e0:
            dead10[e0 + 1:s1] = True

    # labels at 10 fps: dead, else group, else carry the last group forward within the period
    lab10 = np.full(n10, "X", dtype="<U1")
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        last = -1
        for k in idx:
            if dead10[k]:
                lab10[k] = "D"
                last = -1
                continue
            if group[k] >= 0:
                last = group[k]
            if last >= 0:
                lab10[k] = "HA"[last]

    n = int((n10 - 1) * FPS_OUT / FPS_IN) + 1
    ball25 = _resample(ball, n)
    players25 = _resample(players, n)
    near = np.round(np.arange(n) * FPS_IN / FPS_OUT).astype(int).clip(0, n10 - 1)
    labels = lab10[near]
    per25 = period[near]
    halves = []
    for p in (1, 2):
        idx = np.flatnonzero(per25 == p)
        if len(idx):
            halves.append((p, int(idx[0]), int(idx[-1])))
    # scale to the 105 x 68 pitch the features assume
    sx, sy = 105.0 / length, 68.0 / width
    ball25 = ball25 * [sx, sy]
    players25 = players25 * [sx, sy]
    team = np.array([team_of[pid] for pid in ids], dtype=np.int8)
    return {"ball": ball25.astype(np.float32), "players": players25.astype(np.float32),
            "speed": None, "team": team, "halves": np.array(halves), "labels": labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="opendata/data/matches folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metrica_sample import _speed
    os.makedirs(args.out, exist_ok=True)
    for m in sorted(os.listdir(args.src)):
        folder = os.path.join(args.src, m)
        track = os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(m))
        if not os.path.isfile(track):
            continue
        with open(track, encoding="utf-8") as handle:
            if handle.read(40).startswith("version https://git-lfs"):
                raise SystemExit("{} is a Git LFS pointer; download it from {}".format(track, RAW.format(m=m)))
        d = convert(folder, m)
        d["speed"] = _speed(d["players"])
        np.savez_compressed(os.path.join(args.out, "SC-{}.npz".format(m)), **d)
        lab = d["labels"]
        inside = np.zeros(len(lab), bool)
        for _, a, b in d["halves"]:
            inside[a:b + 1] = True
        print("{}  {} frames  H {:.0%} A {:.0%} D {:.0%}  ball missing {:.0%}".format(
            m, len(lab), np.mean(lab[inside] == "H"), np.mean(lab[inside] == "A"),
            np.mean(lab[inside] == "D"), np.mean(np.isnan(d["ball"][inside, 0]))), flush=True)


if __name__ == "__main__":
    main()
