"""Metrica Sports' public sample games, read into the same arrays as tactico/load.py.

EXTERNAL DATA for tuning. The organisers' rules (clarification e-mail, Nov 2026): parameters must be
tuned on external data, never on the provided game, its ground truth or its video. These three
anonymised games are published by Metrica Sports at https://github.com/metrica-sports/sample-data
("please be responsible with the use of this data ... acknowledge the source").

Clone that repository anywhere and pass its path, or set METRICA_SAMPLE_DIR:

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git C:\\mtah-ext\\sample-data

Arrays match tactico/load.py: 0-based frame index (= Metrica frame - 1), positions in metres on a
105 x 68 pitch, NaN where missing. The sample games have no tracker speed, so player speed is
computed from positions (5-frame centred mean).

Possession labels come from the event data (events_to_labels below), because these games have no
possession ground truth. They are an approximation of the official definition (CHALLENGE.md):
possession changes on an on-ball action by the other team, stays with the last team while the ball
is loose, and the ball is dead from a stoppage event (ball out, foul, offside, goal) until the next
restart event (set piece, goal kick, kick-off).
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

PITCH = (105.0, 68.0)
FPS = 25
GAMES = (1, 2, 3)


def sample_dir(path=None):
    path = path or os.environ.get("METRICA_SAMPLE_DIR")
    if not path:
        raise SystemExit("set METRICA_SAMPLE_DIR or pass --sample-dir (a clone of "
                         "github.com/metrica-sports/sample-data)")
    return os.path.join(path, "data") if os.path.isdir(os.path.join(path, "data")) else path


def _speed(players):
    """(n, S) m/s from (n, S, 2) positions, 5-frame centred mean."""
    step = np.full(players.shape[:2], np.nan, dtype=np.float32)
    step[1:] = np.sqrt(np.sum(np.diff(players, axis=0) ** 2, axis=2)) * FPS
    filled = np.where(np.isnan(step), 0.0, step)
    count = (~np.isnan(step)).astype(np.float32)
    k = np.ones(5, dtype=np.float32)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, filled)
    den = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, count)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (num / den).astype(np.float32)
    out[np.isnan(step)] = np.nan
    return out


def _halves(period):
    out = []
    for p in sorted(set(int(v) for v in period if v > 0)):
        idx = np.flatnonzero(period == p)
        out.append((p, int(idx[0]), int(idx[-1])))
    return out


def _load_csv_game(folder, g):
    sides = []
    for side in ("Home", "Away"):
        path = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(g, side))
        with open(path) as handle:
            rows = list(csv.reader(handle))
        header = rows[2]
        data = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r]
                         for r in rows[3:]], dtype=np.float64)
        cols = [i for i, h in enumerate(header) if h.startswith("Player")]
        sides.append((data, cols, header.index("Ball")))
    (home, hcols, hball), (away, acols, _) = sides
    n = min(len(home), len(away))
    period = home[:n, 0].astype(int)
    players = np.concatenate([
        np.stack([home[:n, [c, c + 1]] for c in hcols], axis=1),
        np.stack([away[:n, [c, c + 1]] for c in acols], axis=1)], axis=1)
    team = np.array([0] * len(hcols) + [1] * len(acols), dtype=np.int8)
    ball = home[:n, [hball, hball + 1]]
    return period, ball, players, team


def _load_fifa_game(folder):
    meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    local = meta.find(".//Score").get("idLocalTeam")
    team_of = {p.get("id"): (0 if p.get("teamId") == local else 1) for p in meta.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    ids = sorted(set(chan_player.values()))
    slot = {pid: i for i, pid in enumerate(ids)}
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        refs = [chan_player[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), [slot[p] for p in refs]))
    half_ends = {}
    for prm in meta.iter("ProviderParameter"):
        name, value = prm.findtext("Name"), prm.findtext("Value")
        if value and name.endswith(("_start", "_end")):
            half_ends[name] = int(value)
    n = max(e for _, e, _ in specs)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(ids), 2), np.nan)
    with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            f, plist, b = line.rstrip("\n").split(":")
            f = int(f)
            refs = next(r for s, e, r in specs if s <= f <= e)
            for k, grp in enumerate(plist.split(";")):
                x, y = grp.split(",")
                if x != "NaN":
                    players[f - 1, refs[k]] = (float(x), float(y))
            bx, by = b.split(",")
            if bx != "NaN":
                ball[f - 1] = (float(bx), float(by))
    period = np.zeros(n, dtype=int)
    for p, key in ((1, "first_half"), (2, "second_half")):
        period[half_ends[key + "_start"] - 1:half_ends[key + "_end"]] = p
    team = np.array([team_of[pid] for pid in ids], dtype=np.int8)
    return period, ball, players, team


def _events(folder, g):
    """[(team 0/1, type, subtype, period, start_frame0, end_frame0)] sorted by start."""
    out = []
    if g in (1, 2):
        with open(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(g))) as handle:
            for r in csv.DictReader(handle):
                out.append((0 if r["Team"] == "Home" else 1, r["Type"], r["Subtype"] or "",
                            int(r["Period"]), int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1))
    else:
        data = json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))
        for e in data["data"]:
            sub = e.get("subtypes")
            if isinstance(sub, list):
                sub = "-".join(s["name"] for s in sub)
            elif isinstance(sub, dict):
                sub = sub["name"]
            out.append((0 if e["team"]["id"] == "FIFATMA" else 1, e["type"]["name"], sub or "",
                        int(e["period"]), int(e["start"]["frame"]) - 1, int(e["end"]["frame"]) - 1))
    return sorted(out, key=lambda e: (e[4], e[5]))


OWNER_TYPES = ("PASS", "SHOT", "SET PIECE", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT")
MAX_DEAD_S = 180
MAX_GAP_S = 20


def _is_restart(e):
    return e[1] == "SET PIECE" or "GOAL KICK" in e[2] or "KICK OFF" in e[2]


def _stop_frame(e):
    """Frame the ball goes dead for a stoppage event, else None."""
    typ, sub, s, end = e[1], e[2], e[4], max(e[4], e[5])
    if typ == "BALL OUT" or (typ == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
        return end
    if typ == "FAULT RECEIVED":
        return s
    if typ == "BALL LOST" and ("OFFSIDE" in sub or "END HALF" in sub):
        return end
    return None


def events_to_labels(events, n, halves):
    """Per-frame 'H'/'A'/'D'/'X' from the event list (see module docstring).

    A stoppage with no restart event is ended by the next on-ball action instead. Stretches with no
    on-ball event for MAX_GAP_S, dead balls longer than MAX_DEAD_S and everything outside the
    first and last event of a half stay X (not scored).
    """
    labels = np.full(n, "X", dtype="<U1")
    for p, first, last in halves:
        owner, seg, prev, dead_from = None, None, None, None
        for e in (e for e in events if e[3] == p):
            team, typ, s = e[0], e[1], e[4]
            if dead_from is not None:
                restart = _is_restart(e) or typ in ("PASS", "CARRY", "SHOT")
                if not restart:
                    continue
                if s - dead_from <= MAX_DEAD_S * FPS:
                    labels[dead_from:s] = "D"
                dead_from, owner, seg, prev = None, team, s, s
            elif typ in OWNER_TYPES:
                if owner is not None and s - prev > MAX_GAP_S * FPS:
                    labels[seg:prev + 1] = "HA"[owner]
                    owner = None
                if owner is None:
                    owner, seg = team, s
                elif team != owner:
                    labels[seg:s] = "HA"[owner]
                    owner, seg = team, s
                prev = s
            stop = _stop_frame(e)
            if stop is not None and dead_from is None:
                if owner is not None:
                    labels[seg:stop] = "HA"[owner]
                dead_from, owner = stop, None
        if owner is not None:
            labels[seg:prev + 1] = "HA"[owner]
    return labels


IMPUTE_GAP_FRAMES = 25


def impute_ball(ball, min_gap=IMPUTE_GAP_FRAMES):
    """Fill ball gaps of at least min_gap frames by straight-line interpolation.

    In the sample games the ball is removed from the tracking whenever it is dead (84-100 % of
    dead frames, 2-3 % of live ones). The automatic tracking of the competition game keeps the
    ball while it is fetched and placed. Without this, a tuner learns "no ball = dead ball",
    which does not transfer. Short gaps are kept as tracker drop-outs.
    """
    out = ball.copy()
    miss = np.isnan(ball[:, 0])
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < min_gap or a == 0 or b >= len(ball):
            continue
        t = (np.arange(a, b) - (a - 1)) / float(b - (a - 1))
        out[a:b] = ball[a - 1] + t[:, None] * (ball[b] - ball[a - 1])
    return out


def load_game(g, path=None):
    """dict: ball, players, speed, team (as tactico/load.py), halves, labels (event-derived)."""
    folder = os.path.join(sample_dir(path), "Sample_Game_{}".format(g))
    if g in (1, 2):
        period, ball, players, team = _load_csv_game(folder, g)
    else:
        period, ball, players, team = _load_fifa_game(folder)
    length, width = PITCH
    scale = np.array([length, width])
    ball = (ball * scale).astype(np.float32)
    players = (players * scale).astype(np.float32)
    halves = _halves(period)
    labels = events_to_labels(_events(folder, g), len(ball), halves)
    raw_ball = ball
    ball = impute_ball(ball)
    return {"ball": ball, "raw_ball": raw_ball, "players": players, "speed": _speed(players), "team": team,
            "halves": halves, "labels": labels}
