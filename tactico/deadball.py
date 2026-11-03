"""Dead-ball mask: True on frames where the ball is not in play.

Three independent signals, OR-ed, then cleaned:
  1. the ball is outside the pitch
  2. play has visibly stopped (players slow, ball missing or still) for a while
  3. Smart Tagging set pieces: from the start of the tag window to the assumed restart
Plus the one fact that is published outright: the second half is graded from frame 110808 but
kicks off at 113742, so everything in between is dead.
"""

import numpy as np

from . import config as C

SECOND_HALF_GRADED_FROM = 110808
SECOND_HALF_KICKOFF = 113742


def runs(mask):
    """[(start, stop)] of True runs, stop exclusive."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _min_run(mask, min_len):
    out = mask.copy()
    for a, b in runs(mask):
        if b - a < min_len:
            out[a:b] = False
    return out


def _close_gaps(mask, max_gap):
    out = mask.copy()
    for a, b in runs(~mask):
        if b - a < max_gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def stopped(feat):
    """Players slow AND ball missing or slow, held for STILL_FRAMES."""
    slow_players = feat["player_speed"] < C.STILL_SPEED_MS
    with np.errstate(invalid="ignore"):
        ball_quiet = ~feat["ball_visible"] | (feat["ball_speed"] < C.BALL_STILL_MS)
    return _min_run(slow_players & ball_quiet, C.STILL_FRAMES)


def setpieces(n_frames, events):
    mask = np.zeros(n_frames, dtype=bool)
    for e in events:
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        a, b = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
        restart = a + int(round((b - a) * C.SETPIECE_RESTART_FRAC))
        mask[a:restart] = True
    return mask


def _rolling_mean(x, half):
    """Centred mean over +/- half frames, ignoring NaN."""
    v = np.where(np.isnan(x), 0.0, x).astype(np.float64)
    c = np.where(np.isnan(x), 0, 1).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((cv[hi] - cv[lo]) / (cc[hi] - cc[lo])).astype(np.float32)


def until_restart(feat, trigger, backdate=0):
    """Dead from each trigger until the restart is taken.

    Restart = the ball has been quiet (slower than RESTART_QUIET_MS, or not tracked) for
    RESTART_QUIET_FRAMES, then is struck at RESTART_KICK_MS or more, and is inside the pitch
    (within RESTART_INSIDE_M) within a second. Throw-ins are struck from outside the line, hence
    "within a second". Capped at MAX_OUT_DEAD_FRAMES in case no restart is recognised.
    After a restart, frames where the trigger is still on do not open a new dead stretch.
    """
    speed = feat["ball_speed"]
    visible = feat["ball_visible"]
    inside = feat["ball_inside_wide"]
    out = feat["ball_out"]
    n = len(trigger)
    dead = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        quiet = ~visible | (speed < C.RESTART_QUIET_MS)
        kick = visible & (speed >= C.RESTART_KICK_MS)
        # fallback release: play has clearly resumed (players running, ball moving on the pitch)
        players = _rolling_mean(feat["player_speed"], C.STOP_WINDOW_FRAMES)
        ballm = _rolling_mean(speed, C.STOP_WINDOW_FRAMES)
        playing = (players >= C.PLAY_SPEED_MS) & (ballm >= C.PLAY_BALL_MS) & ~out

    t = 0
    while t < n:
        if not trigger[t]:
            t += 1
            continue
        start, quiet_run, play_run = max(0, t - backdate), 0, 0
        t += 1
        while t < n and t - start < C.MAX_OUT_DEAD_FRAMES:
            play_run = play_run + 1 if playing[t] else 0
            if play_run >= C.PLAY_FRAMES:
                t = t - play_run + 1
                break
            if quiet[t]:
                quiet_run += 1
            elif kick[t] and quiet_run >= C.RESTART_QUIET_FRAMES and \
                    inside[t:t + C.FPS].any() and not out[t:t + C.FPS].all():
                break
            else:
                quiet_run = 0
            t += 1
        dead[start:t] = True
        while t < n and trigger[t]:
            t += 1
    return dead


def out_until_restart(feat):
    """Ball out of play stays dead until the restart (corner, goal kick, throw-in).

    Seen on the public window (corner at 11:35.2, goal kick at 12:01.1 video time): `ball_out`
    fires when the ball crosses the line, but the ball is then fetched and placed on or near the
    pitch, `ball_out` drops to 0, and the old mask called play alive long before the restart.
    """
    return until_restart(feat, _min_run(feat["ball_out"], C.OUT_MIN_FRAMES))


def lost_at_line(feat):
    """True over every run of >= LOST_FRAMES missing-ball frames whose last tracked position was
    within LOST_LINE_M of a pitch line (or beyond it)."""
    x, y, visible = feat["ball_x"], feat["ball_y"], feat["ball_visible"]
    length, width = feat["pitch"]
    m = C.LOST_LINE_M
    out = np.zeros(len(x), dtype=bool)
    for a, b in runs(~visible):
        if b - a < C.LOST_FRAMES or a == 0:
            continue
        lx, ly = x[a - 1], y[a - 1]
        if lx < m or lx > length - m or ly < m or ly > width - m:
            out[a:b] = True
    return out


def whistle_until_restart(feat):
    """Referee stops play with the ball on the pitch (offside, foul): dead until the free kick.

    Seen on the public window (free kick at 10:35.6, taken at ~10:56 video time): mean player
    speed falls to ~0.8-0.9 m/s but keeps bouncing above STILL_SPEED_MS, so the frame-by-frame
    rule never held long enough. Here both speeds are averaged over +/- STOP_WINDOW_FRAMES first;
    a stop is players slower than STOP_SPEED_MS with the ball slower than STOP_BALL_MS for
    STOP_FRAMES. Players need a moment to slow down after the whistle, so the dead stretch is
    dated back STOP_BACKDATE_FRAMES.
    """
    players = _rolling_mean(feat["player_speed"], C.STOP_WINDOW_FRAMES)
    ball = _rolling_mean(feat["ball_speed"], C.STOP_WINDOW_FRAMES)
    with np.errstate(invalid="ignore"):
        still = (players < C.STOP_SPEED_MS) & ~(ball >= C.STOP_BALL_MS)
    trigger = _min_run(still, C.STOP_FRAMES)
    return until_restart(feat, trigger, C.STOP_BACKDATE_FRAMES)


def goal_candidates(feat):
    """True where a goal may have happened: the ball tracked behind the goal line between the posts,
    or the ball vanishing for GOAL_LOST_FRAMES right in front of the goal mouth."""
    x, y, visible = feat["ball_x"], feat["ball_y"], feat["ball_visible"]
    length, width = feat["pitch"]
    with np.errstate(invalid="ignore"):
        mouth = np.abs(y - width / 2) <= C.GOAL_HALF_WIDTH_M
        behind = ((x < 0) & (x > -C.GOAL_DEPTH_M)) | ((x > length) & (x < length + C.GOAL_DEPTH_M))
        near = (x < C.GOAL_LOST_LINE_M) | (x > length - C.GOAL_LOST_LINE_M)
    cand = _min_run(behind & mouth, C.GOAL_MIN_FRAMES)
    for a, b in runs(~visible):
        if b - a >= C.GOAL_LOST_FRAMES and a > 0 and near[a - 1] and mouth[a - 1]:
            cand[a] = True
    return cand


def goal_until_kickoff(feat):
    """Ball over the goal line between the posts -> dead until the kick-off from the centre spot.

    Kick-off = the ball lies within KICKOFF_RADIUS_M of the centre spot, slower than
    RESTART_QUIET_MS, for KICKOFF_QUIET_FRAMES, and is then struck at RESTART_KICK_MS or more.
    No kick-off within GOAL_MAX_FRAMES, or GOAL_ABORT_PLAY_FRAMES of open play first -> not a goal,
    nothing is marked and the other rules decide.
    """
    x, y, speed = feat["ball_x"], feat["ball_y"], feat["ball_speed"]
    length, width = feat["pitch"]
    n = len(x)
    with np.errstate(invalid="ignore"):
        at_centre = np.hypot(x - length / 2, y - width / 2) <= C.KICKOFF_RADIUS_M
        quiet_centre = at_centre & (speed < C.RESTART_QUIET_MS)
        kick = speed >= C.RESTART_KICK_MS
        players = _rolling_mean(feat["player_speed"], C.STOP_WINDOW_FRAMES)
        ballm = _rolling_mean(speed, C.STOP_WINDOW_FRAMES)
        playing = (players >= C.PLAY_SPEED_MS) & (ballm >= C.PLAY_BALL_MS) & ~feat["ball_out"]
    trigger = goal_candidates(feat)

    dead = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        if not trigger[t]:
            t += 1
            continue
        start, quiet_run, play_run, kickoff = t, 0, 0, None
        u = t + 1
        while u < n and u - start < C.GOAL_MAX_FRAMES:
            play_run = play_run + 1 if playing[u] else 0
            if play_run >= C.GOAL_ABORT_PLAY_FRAMES:
                break
            if quiet_centre[u]:
                quiet_run += 1
            elif kick[u] and quiet_run >= C.KICKOFF_QUIET_FRAMES:
                kickoff = u
                break
            elif not np.isnan(x[u]):
                quiet_run = 0
            u += 1
        if kickoff is not None:
            dead[start:kickoff] = True
            t = kickoff
        else:
            t += 1
            while t < n and trigger[t]:
                t += 1
    return dead


def goals(feat):
    """(start, kick-off) frame pairs found by goal_until_kickoff, for printing."""
    return runs(goal_until_kickoff(feat))


def mask(feat, events=None, forced=((SECOND_HALF_GRADED_FROM, SECOND_HALF_KICKOFF),)):
    """forced: (start, stop) frame ranges known to be dead. The default is the Barca-Athletic
    second-half gap; pass () for other games."""
    n = len(feat["ball_visible"])
    dead = feat["ball_out"] | stopped(feat)
    if C.USE_OUT_LATCH:
        dead |= out_until_restart(feat)
    if C.USE_LOST_OUT:
        dead |= until_restart(feat, lost_at_line(feat))
    if C.USE_WHISTLE:
        dead |= whistle_until_restart(feat)
    if C.USE_SMART_TAGGING and events:
        dead |= setpieces(n, events)
    if C.USE_GOAL:
        goal = goal_until_kickoff(feat)
        dead |= goal
    dead = _close_gaps(dead, C.MERGE_GAP_FRAMES)
    dead = _min_run(dead, C.MIN_DEAD_FRAMES)
    for a, b in forced:
        dead[a:b] = True
    return dead
