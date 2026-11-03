"""Per-frame quantities the rules are built on. Pure numpy, no state."""

import numpy as np

from . import config as C


def nearest_by_team(ball, players, team):
    """(d_home, d_away): distance in metres from the ball to each team's nearest player.

    NaN where the ball is missing or a team has no tracked player on that frame.
    """
    diff = players - ball[:, None, :]                       # (n, S, 2)
    dist = np.sqrt(np.sum(diff * diff, axis=2))             # (n, S), NaN propagates
    out = []
    for side in (0, 1):
        d = np.where(team[None, :] == side, dist, np.nan)
        with np.errstate(all="ignore"):
            nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        nearest[np.isinf(nearest)] = np.nan
        out.append(nearest.astype(np.float32))
    return out[0], out[1]


def ball_speed(ball):
    """m/s from consecutive positions. NaN where either frame is missing."""
    v = np.full(len(ball), np.nan, dtype=np.float32)
    step = np.sqrt(np.sum(np.diff(ball, axis=0) ** 2, axis=1)) * C.FPS
    v[1:] = step
    return v


def ball_out(ball, pitch):
    """True where the tracked ball is outside the pitch lines by more than OUT_MARGIN_M."""
    length, width = pitch
    m = C.OUT_MARGIN_M
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -m) | (x > length + m) | (y < -m) | (y > width + m)


def ball_inside(ball, pitch, margin):
    """True where the tracked ball is inside the pitch, allowing `margin` metres beyond the lines."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x >= -margin) & (x <= length + margin) & (y >= -margin) & (y <= width + margin)


def mean_player_speed(speed):
    with np.errstate(all="ignore"):
        return np.nanmean(speed, axis=1).astype(np.float32)


def build(atd, pitch):
    ball = atd["ball"]
    d_home, d_away = nearest_by_team(ball, atd["players"], atd["team"])
    return {
        "ball_visible": ~np.isnan(ball[:, 0]),
        "d_home": d_home,
        "d_away": d_away,
        "ball_speed": ball_speed(ball),
        "ball_out": ball_out(ball, pitch),
        "ball_inside_wide": ball_inside(ball, pitch, C.RESTART_INSIDE_M),
        "player_speed": mean_player_speed(atd["speed"]),
        "ball_x": ball[:, 0],
        "ball_y": ball[:, 1],
        "pitch": pitch,
    }
