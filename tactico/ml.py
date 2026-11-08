"""Learned possession model: per-frame classifier + Viterbi decoding.

Trained ONLY on external data (tools/train_ml.py): the DFL open matches, with the ball removed on
41 % of frames and position noise added so that they look like the competition's automatic
tracking. Never trained or tuned on the competition game, its ground truth or its video.

1. features(): one row per frame, built from the ball, the players and their teams. Most features
   still say something when the ball is missing: which team moves towards the other's goal, how
   spread out each team is, how fast everyone runs, where the ball was last seen.
   Features come in a home block and an away block, so swapping the blocks swaps the teams; training
   uses both orders, which makes the model blind to which side is "home".
2. A gradient-boosted classifier gives P(H), P(A), P(D) per frame.
3. viterbi(): the most likely label sequence given those probabilities and a cost for every change
   (H<->A and live<->dead), so the output is whole possessions, not flicker.
"""

import numpy as np

FPS = 25
HOME, AWAY = 0, 1
STATES = ("H", "A", "D")


# ----------------------------------------------------------------------------- helpers

def _ffill(a):
    """Forward-fill NaN along axis 0 (1-D or 2-D)."""
    a = a.copy()
    mask = np.isnan(a)
    idx = np.where(~mask, np.arange(len(a)).reshape((-1,) + (1,) * (a.ndim - 1)), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.take_along_axis(a, idx, axis=0) if a.ndim > 1 else a[idx]
    return out


def _roll(x, window, centre=True):
    """NaN-aware moving mean over `window` frames (centred, or trailing)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cc = np.concatenate([[0], np.cumsum(ok)])
    n = len(x)
    if centre:
        lo = np.clip(np.arange(n) - window // 2, 0, n)
        hi = np.clip(np.arange(n) + window // 2 + 1, 0, n)
    else:
        lo = np.clip(np.arange(n) - window + 1, 0, n)
        hi = np.arange(n) + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / (cc[hi] - cc[lo])
    return out.astype(np.float32)


def _shift_mean(x, window, ahead):
    """Mean over the next (ahead=True) or previous `window` frames, NaN-aware."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cc = np.concatenate([[0], np.cumsum(ok)])
    n = len(x)
    if ahead:
        lo = np.arange(n) + 1
        hi = np.clip(lo + window, 0, n)
        lo = np.clip(lo, 0, n)
    else:
        hi = np.arange(n)
        lo = np.clip(hi - window, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / (cc[hi] - cc[lo])
    return out.astype(np.float32)


def attack_sign(players, team, halves):
    """Per frame and team: +1 if the team attacks towards x = length, -1 otherwise.
    A team's mean x over a half is on its own side, so the team with the lower mean attacks right."""
    n = len(players)
    sign = np.zeros((n, 2), dtype=np.float32)
    for _, first, last in halves:
        sl = slice(first, last + 1)
        mean = [np.nanmean(players[sl][:, team == t, 0]) for t in (HOME, AWAY)]
        s_home = 1.0 if mean[HOME] < mean[AWAY] else -1.0
        sign[sl, HOME] = s_home
        sign[sl, AWAY] = -s_home
    return sign


# ----------------------------------------------------------------------------- features

TEAM_FEATURES = ("d1", "d2", "n2", "n5", "n10", "d1_last", "speed", "adv", "adv_v1", "adv_v3",
                 "adv_v10", "adv_m10", "adv_p10", "spread_y10",
                 "spread_x", "spread_y", "slow", "ball_att", "d1_m1", "d1_p1",
                 "d1_m3", "d1_p3", "n5_m1", "n5_p1")
SHARED_FEATURES = ("ball_seen", "ball_age", "ball_speed", "ball_speed_m1", "ball_speed_p1",
                   "edge_touch", "edge_goal", "out", "speed_all", "speed_all_m2", "speed_all_p2",
                   "speed_all_m5", "speed_all_p5", "speed_all_m15", "speed_all_p15",
                   "seen_m1", "seen_p1", "seen_p3", "gap_d1")


def features(d, pitch=(105.0, 68.0)):
    """(n, F) float32 feature matrix and the list of names. Home block, away block, shared."""
    ball, players, team = d["ball"], d["players"], d["team"]
    halves = d["halves"]
    length, width = pitch
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])

    last = _ffill(ball)
    idx = np.where(seen, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    age = np.where(idx >= 0, np.arange(n) - idx, 9999).astype(np.float32)
    age = np.minimum(age, 1500)

    bstep = np.full(n, np.nan, np.float32)
    bstep[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    bspeed = _roll(np.minimum(bstep, 40), 5)

    lx, ly = last[:, 0], last[:, 1]
    edge_touch = np.minimum(ly, width - ly)
    edge_goal = np.minimum(lx, length - lx)
    out = ((lx < -0.5) | (lx > length + 0.5) | (ly < -0.5) | (ly > width + 0.5)).astype(np.float32)

    if "speed" in d and d["speed"] is not None:
        pspeed = np.asarray(d["speed"], dtype=np.float32)
    else:
        pspeed = np.full(players.shape[:2], np.nan, np.float32)
    pspeed = np.minimum(pspeed, 12)
    with np.errstate(invalid="ignore"):
        speed_all = np.nanmean(pspeed, axis=1)
    sign = attack_sign(players, team, halves)

    diff_seen = players - ball[:, None, :]
    dist_seen = np.linalg.norm(diff_seen, axis=2)
    dist_last = np.linalg.norm(players - last[:, None, :], axis=2)

    blocks = {}
    for t in (HOME, AWAY):
        m = team == t
        dt = np.where(np.isnan(dist_seen[:, m]), np.inf, dist_seen[:, m])
        dl = np.where(np.isnan(dist_last[:, m]), np.inf, dist_last[:, m])
        srt = np.sort(dt, axis=1)
        d1 = np.where(np.isinf(srt[:, 0]), np.nan, srt[:, 0]).astype(np.float32)
        d2 = np.where(np.isinf(srt[:, 1]), np.nan, srt[:, 1]).astype(np.float32)
        d1_last = np.min(dl, axis=1)
        d1_last = np.where(np.isinf(d1_last), np.nan, d1_last).astype(np.float32)
        with np.errstate(invalid="ignore"):
            n2 = np.where(seen, np.sum(dt <= 2, axis=1), np.nan).astype(np.float32)
            n5 = np.where(seen, np.sum(dt <= 5, axis=1), np.nan).astype(np.float32)
            n10 = np.where(seen, np.sum(dt <= 10, axis=1), np.nan).astype(np.float32)
            px = players[:, m, 0]
            py = players[:, m, 1]
            cx = np.nanmean(px, axis=1)
            adv = (cx - length / 2) * sign[:, t]
            spread_x = np.nanstd(px, axis=1)
            spread_y = np.nanstd(py, axis=1)
            sp = np.nanmean(pspeed[:, m], axis=1)
            slow = np.nanmean((pspeed[:, m] < 1.0).astype(np.float32)
                              + np.where(np.isnan(pspeed[:, m]), np.nan, 0), axis=1)
        adv_v = np.full(n, np.nan, np.float32)
        adv_v[1:] = np.diff(adv) * FPS
        ball_att = ((lx - length / 2) * sign[:, t]).astype(np.float32)
        blocks[t] = {
            "d1": d1, "d2": d2, "n2": n2, "n5": n5, "n10": n10, "d1_last": d1_last,
            "speed": sp, "adv": adv, "adv_v1": _roll(adv_v, 25), "adv_v3": _roll(adv_v, 75),
            "adv_v10": _roll(adv_v, 250), "adv_m10": _shift_mean(adv, 250, False),
            "adv_p10": _shift_mean(adv, 250, True), "spread_y10": _roll(spread_y, 250),
            "spread_x": spread_x, "spread_y": spread_y, "slow": slow,
            "ball_att": ball_att,
            "d1_m1": _shift_mean(d1, 25, False), "d1_p1": _shift_mean(d1, 25, True),
            "d1_m3": _shift_mean(d1, 75, False), "d1_p3": _shift_mean(d1, 75, True),
            "n5_m1": _shift_mean(n5, 25, False), "n5_p1": _shift_mean(n5, 25, True),
        }
    seenf = seen.astype(np.float32)
    shared = {
        "ball_seen": seenf, "ball_age": age, "ball_speed": bspeed,
        "ball_speed_m1": _shift_mean(bspeed, 25, False), "ball_speed_p1": _shift_mean(bspeed, 25, True),
        "edge_touch": edge_touch.astype(np.float32), "edge_goal": edge_goal.astype(np.float32),
        "out": out, "speed_all": speed_all.astype(np.float32),
        "speed_all_m2": _shift_mean(speed_all, 50, False), "speed_all_p2": _shift_mean(speed_all, 50, True),
        "speed_all_m5": _shift_mean(speed_all, 125, False), "speed_all_p5": _shift_mean(speed_all, 125, True),
        "speed_all_m15": _shift_mean(speed_all, 375, False),
        "speed_all_p15": _shift_mean(speed_all, 375, True),
        "seen_m1": _shift_mean(seenf, 25, False), "seen_p1": _shift_mean(seenf, 25, True),
        "seen_p3": _shift_mean(seenf, 75, True),
        "gap_d1": blocks[HOME]["d1"] - blocks[AWAY]["d1"],
    }
    cols = [blocks[HOME][k] for k in TEAM_FEATURES] + [blocks[AWAY][k] for k in TEAM_FEATURES] \
        + [shared[k] for k in SHARED_FEATURES]
    names = ["h_" + k for k in TEAM_FEATURES] + ["a_" + k for k in TEAM_FEATURES] + list(SHARED_FEATURES)
    return np.stack(cols, axis=1).astype(np.float32), names


def swap_teams(X):
    """Same rows with the home and away blocks swapped (gap_d1 negated)."""
    k = len(TEAM_FEATURES)
    Y = X.copy()
    Y[:, :k], Y[:, k:2 * k] = X[:, k:2 * k], X[:, :k]
    Y[:, -1] = -X[:, -1]
    return Y


# ----------------------------------------------------------------------------- decoding

def predict_proba(model, X):
    """Average of the model on the rows and on the team-swapped rows (with H/A swapped back),
    so the output does not depend on which team is called home."""
    p1 = model.predict_proba(X)
    p2 = model.predict_proba(swap_teams(X))[:, [1, 0, 2]]
    return (p1 + p2) / 2


def viterbi(prob, switch_cost, dead_cost, prior=None, floor=1e-4):
    """Most likely H/A/D sequence. prob: (n, 3). Costs are in log units per change."""
    n = len(prob)
    lp = np.log(np.clip(prob, floor, 1.0))
    if prior is not None:
        lp = lp - np.log(np.asarray(prior))
    trans = np.array([[0.0, switch_cost, dead_cost],
                      [switch_cost, 0.0, dead_cost],
                      [dead_cost, dead_cost, 0.0]])
    score = lp[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - trans          # from i to j
        best = np.argmax(cand, axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + lp[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def label(prob, halves, switch_cost, dead_cost, prior=None):
    """Full-length label array, decoded half by half; X outside the halves."""
    out = np.full(len(prob), "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi(prob[first:last + 1], switch_cost, dead_cost, prior)
        out[first:last + 1] = np.array(STATES)[path]
    return out
