"""Possession state machine with hysteresis.

Per frame a team is a "candidate" when it is clearly closest to the ball (config.py). Possession
only changes hands when the other team has been candidate for SWITCH_FRAMES in a row, and the
change is then dated back to where that run began -- spellF1 rewards a correct start frame far
more than macroF1 does. Loose balls, single touches and lost ball tracking keep the current owner.

Labels: "H", "A", "D", or "X" where the system has no owner yet (only right after a dead ball,
before anyone has control). The scorer forward-fills X, so it never splits a possession.
"""

import numpy as np

from . import config as C

HOME, AWAY, NONE = 0, 1, -1


def candidates(feat):
    d_h, d_a = feat["d_home"], feat["d_away"]
    h = np.where(np.isnan(d_h), np.inf, d_h)
    a = np.where(np.isnan(d_a), np.inf, d_a)
    cand = np.full(len(h), NONE, dtype=np.int8)
    cand[(h <= C.CONTROL_RADIUS_M) & (h + C.CONTROL_MARGIN_M <= a)] = HOME
    cand[(a <= C.CONTROL_RADIUS_M) & (a + C.CONTROL_MARGIN_M <= h)] = AWAY
    return cand


def kicker_team(feat):
    """Per frame: the team of the nearest player if within RESTART_RADIUS_M of the ball, else NONE.

    Used at the moment a dead ball ends: the restart is taken by the team that owns it, and the
    player taking it is the one standing at the ball. Waiting for a team to be "in control" for
    RESTART_FRAMES instead gives the ball away on long restarts (validation set 1:22:01: the
    Athletic keeper's long free kick was credited to the Barca player it landed on 7 s later).
    """
    d_h, d_a = feat["d_home"], feat["d_away"]
    h = np.where(np.isnan(d_h), np.inf, d_h)
    a = np.where(np.isnan(d_a), np.inf, d_a)
    out = np.full(len(h), NONE, dtype=np.int8)
    out[(h <= a) & (h <= C.RESTART_RADIUS_M)] = HOME
    out[(a < h) & (a <= C.RESTART_RADIUS_M)] = AWAY
    return out


def restart_owner(kicker, t, start, stop):
    """Team at the ball around frame t (the first live frame after a dead ball), or NONE."""
    lo, hi = max(start, t - C.RESTART_LOOK_FRAMES), min(stop, t + C.RESTART_LOOK_FRAMES + 1)
    window = kicker[lo:hi]
    home, away = int(np.sum(window == HOME)), int(np.sum(window == AWAY))
    if home == away == 0:
        return NONE
    return HOME if home >= away else AWAY


def label_block(cand, dead, start, stop, hold=None, kicker=None):
    """Labels for frames start..stop-1 (one half). Returns an array of '<U1'.

    hold (optional): per-frame team whose keeper holds the ball, -1 for none (keeper.py). A hold
    that begins while the OTHER team (or nobody) owns the ball is a save or a catch, and is dead
    ball until the keeper releases it. A hold that begins while his own team owns the ball is a
    back-pass and stays in play.
    """
    out = np.full(stop - start, "X", dtype="<U1")
    owner = NONE
    run_team, run_start, run_len = NONE, 0, 0
    alive_from = start  # first frame of the current live stretch
    hold_dead = False
    owner_since, prev_owner = start, NONE

    for t in range(start, stop):
        i = t - start
        if hold is not None:
            h = hold[t]
            if h == NONE:
                hold_dead = False
            elif t == start or hold[t - 1] != h:
                # a save or a catch: the ball came from the other team, either still owned by
                # them, or won by the keeper's team only moments ago (the keeper's own parry)
                just_won = (owner == h and prev_owner not in (NONE, h)
                            and t - owner_since <= C.KEEPER_RECENT_FRAMES)
                hold_dead = owner != h or just_won
        if dead[t] or (hold is not None and hold[t] != NONE and hold_dead):
            out[i] = "D"
            owner, run_team, run_len = NONE, NONE, 0
            prev_owner = NONE
            alive_from = t + 1
            continue

        if owner == NONE and kicker is not None and t == alive_from and t > start:
            k = restart_owner(kicker, t, start, stop)
            if k != NONE:
                owner, owner_since = k, t
                run_team, run_start, run_len = k, t, 0

        c = cand[t]
        if c != NONE and c == run_team:
            run_len += 1
        elif c != NONE:
            run_team, run_start, run_len = c, t, 1
        # c == NONE: keep the run counter as is -- a lost frame does not break control

        if owner == NONE and run_team != NONE and run_len >= C.RESTART_FRAMES:
            owner, owner_since = run_team, alive_from
            out[alive_from - start:i] = "H" if owner == HOME else "A"
        elif owner != NONE and run_team != owner and run_len >= C.SWITCH_FRAMES:
            prev_owner, owner, owner_since = owner, run_team, run_start
            out[run_start - start:i] = "H" if owner == HOME else "A"

        if owner != NONE:
            out[i] = "H" if owner == HOME else "A"
    return out


def label(feat, dead, halves, hold=None):
    """Full-length label array. halves = [(half, first_frame, last_frame_inclusive), ...]."""
    n = len(dead)
    cand = candidates(feat)
    kicker = kicker_team(feat) if C.USE_RESTART_KICKER else None
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in halves:
        labels[first:last + 1] = label_block(cand, dead, first, last + 1, hold, kicker)
    return labels
