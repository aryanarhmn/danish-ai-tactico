"""DFL / Sportec open data: seven Bundesliga matches with official position data.

EXTERNAL DATA for tuning and training (CC BY 4.0). Bassek, M., Rein, R., Weber, H., Memmert, D.
(2025). An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data.
https://doi.org/10.6084/m9.figshare.28196177

Each ball frame carries the provider's own BallPossession (1 home, 2 away) and BallStatus (1 alive,
0 dead), so these matches give a per-frame possession and dead-ball truth from tracking -- the same
kind of definition as the competition (CHALLENGE.md) -- and the ball stays tracked while it is dead,
as in the competition tracking.

    python tools/external/dfl_open.py --out C:\\mtah-ext\\dfl        (download + convert, ~2.6 GB
                                                                   of XML, kept only while parsing)

Writes one <match>.npz per match with the arrays of tactico/load.py plus halves and labels.
"""

import argparse
import json
import os
import re
import sys
import urllib.request

import numpy as np

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
PITCH = (105.0, 68.0)
FRAME_RE = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?[^>]*? S="([-\d.]+)"'
                      r'(?:[^>]*? BallPossession="(\d)")?(?:[^>]*? BallStatus="(\d)")?')
SET_RE = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')


def _files():
    with urllib.request.urlopen(ARTICLE) as r:
        files = json.load(r)["files"]
    matches = {}
    for f in files:
        m = re.search(r"(DFL-MAT-\w+)\.xml", f["name"])
        kind = "info" if "matchinformation" in f["name"] else "pos" if "positions" in f["name"] else None
        if m and kind:
            matches.setdefault(m.group(1), {})[kind] = f["download_url"]
    return matches


def _home_team(info_xml):
    m = re.search(r'TeamId="([^"]+)"[^>]*Role="home"', info_xml)
    return m.group(1)


def convert(pos_path, home_id):
    """Stream the positions XML into arrays. Returns the dict described in the module doc."""
    sets = {}           # (section, team, person) -> list of (N, x, y, s, poss, status)
    key = None
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = FRAME_RE.match(line)
                if m:
                    sets[key].append(m.groups())
            elif line.startswith("<FrameSet"):
                m = SET_RE.match(line)
                key = (m.group(1), m.group(2), m.group(3))
                sets[key] = []
    sections = [s for s in ("firstHalf", "secondHalf") if any(k[0] == s for k in sets)]
    ball_key = {s: next(k for k in sets if k[0] == s and k[1] == "BALL") for s in sections}
    persons = sorted({k[2] for k in sets if k[1] != "BALL"})
    team_of = {k[2]: (0 if k[1] == home_id else 1) for k in sets if k[1] != "BALL"}
    slot = {p: i for i, p in enumerate(persons)}

    offsets, halves, total = {}, [], 0
    for i, s in enumerate(sections):
        ns = [int(r[0]) for r in sets[ball_key[s]]]
        lo, hi = min(ns), max(ns)
        offsets[s] = total - lo
        halves.append((i + 1, total, total + hi - lo))
        total += hi - lo + 1

    ball = np.full((total, 2), np.nan, np.float32)
    players = np.full((total, len(persons), 2), np.nan, np.float32)
    speed = np.full((total, len(persons)), np.nan, np.float32)
    labels = np.full(total, "X", dtype="<U1")
    for (s, team, person), rows in sets.items():
        if not rows:
            continue
        arr = np.array([[float(r[0]), float(r[1]), float(r[2]), float(r[4])] for r in rows])
        idx = arr[:, 0].astype(int) + offsets[s]
        xy = np.stack([arr[:, 1] + PITCH[0] / 2, arr[:, 2] + PITCH[1] / 2], axis=1)
        if team == "BALL":
            ball[idx] = xy
            poss = np.array([r[5] or "0" for r in rows])
            status = np.array([r[6] or "0" for r in rows])
            lab = np.where(poss == "1", "H", np.where(poss == "2", "A", "X"))
            labels[idx] = np.where(status == "0", "D", lab)
        else:
            players[idx, slot[person]] = xy
            speed[idx, slot[person]] = arr[:, 3] / 3.6
    team = np.array([team_of[p] for p in persons], dtype=np.int8)
    return {"ball": ball, "players": players, "speed": speed, "team": team,
            "halves": np.array(halves), "labels": labels}


def load_match(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["halves"] = [tuple(int(v) for v in h) for h in d["halves"]]
    return d


def list_matches(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for match, urls in sorted(_files().items()):
        dst = os.path.join(args.out, match + ".npz")
        if os.path.exists(dst):
            print("have", match)
            continue
        with urllib.request.urlopen(urls["info"]) as r:
            home = _home_team(r.read().decode("utf-8"))
        tmp = os.path.join(args.out, match + ".xml")
        print("downloading", match, flush=True)
        urllib.request.urlretrieve(urls["pos"], tmp)
        d = convert(tmp, home)
        np.savez_compressed(dst, **d)
        os.remove(tmp)
        lab = d["labels"]
        print("  {} frames  H {:.0%}  A {:.0%}  D {:.0%}".format(
            len(lab), np.mean(lab == "H"), np.mean(lab == "A"), np.mean(lab == "D")), flush=True)


if __name__ == "__main__":
    main()
