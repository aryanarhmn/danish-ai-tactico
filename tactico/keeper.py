"""Goalkeeper holding the ball after a save or a catch.

Observed in the public ground truth (video 08:34.0): after an Athletic shot, the Barcelona keeper
saves and holds the ball, and the ground truth marks those seconds as DEAD BALL until he releases
it. So in Metrica's corrected tracking a keeper with the ball in his hands is a stoppage.

The tracking has no player identities, so the keeper is inferred: per half, each team defends the
goal on the side where its players stand on average, and on each frame its keeper is the player
closest to that goal line, provided he is inside his own penalty area.

A hold is: that keeper within KEEPER_RADIUS_M of a slow ball, for KEEPER_MIN_FRAMES in a row, with
frames where the tracker loses the ball (common when it is in his hands) carried over for up to
KEEPER_MAX_GAP_FRAMES. Whether a hold becomes a dead ball is decided in possession.py: only when
the ball came from the OTHER team (a save or a catch), never after a back-pass from a team-mate,
which a keeper plays with his feet.
"""

import numpy as np

from . import config as C

BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16


def defended_side(players, team, first, last, length):
    """{team: 0 or 1}: 0 = defends the goal at x=0, 1 = the goal at x=length, for one half."""
    xs = players[first:last + 1, :, 0]
    out = {}
    for side in (0, 1):
        with np.errstate(all="ignore"):
            mean_x = np.nanmean(xs[:, team == side])
        out[side] = 0 if mean_x < length / 2 else 1
    return out


def hold_team(atd, feat, halves, pitch):
    """int8 array per frame: 0 or 1 = that team's keeper holds the ball, -1 = no hold."""
    length, width = pitch
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    n = len(ball)
    raw = np.full(n, -1, dtype=np.int8)

    for _, first, last in halves:
        sides = defended_side(players, team, first, last, length)
        sl = slice(first, last + 1)
        for side in (0, 1):
            px = players[sl, :, 0]
            py = players[sl, :, 1]
            mine = team == side
            goal_x = 0.0 if sides[side] == 0 else length
            depth = np.abs(px - goal_x)
            depth = np.where(mine[None, :] & ~np.isnan(depth), depth, np.inf)
            k = np.argmin(depth, axis=1)
            rows = np.arange(len(k))
            kx, ky, kd = px[rows, k], py[rows, k], depth[rows, k]
            in_box = (kd <= BOX_DEPTH_M) & (np.abs(ky - width / 2) <= BOX_HALF_WIDTH_M)

            bx, by = ball[sl, 0], ball[sl, 1]
            with np.errstate(invalid="ignore"):
                near = np.hypot(bx - kx, by - ky) <= C.KEEPER_RADIUS_M
                slow = feat["ball_speed"][sl] < C.KEEPER_BALL_MAX_MS
            hit = in_box & near & slow
            block = raw[sl]
            block[hit] = side

    # carry a hold across frames where the ball is not tracked
    visible = feat["ball_visible"]
    out = raw.copy()
    last_team, gap = -1, 0
    for t in range(n):
        if raw[t] != -1:
            last_team, gap = raw[t], 0
        elif not visible[t] and last_team != -1 and gap < C.KEEPER_MAX_GAP_FRAMES:
            out[t] = last_team
            gap += 1
        else:
            last_team, gap = -1, 0

    # keep only holds that last long enough
    for side in (0, 1):
        m = out == side
        padded = np.concatenate(([False], m, [False])).astype(np.int8)
        edges = np.diff(padded)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a < C.KEEPER_MIN_FRAMES:
                out[a:b] = -1
    return out
