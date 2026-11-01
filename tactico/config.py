"""Every tunable number in one place.

Tune these against your own hand-labelled validation set (see tools/intervals_to_gt.py), never
against the public 5 minutes alone: 17 possessions is too few to choose between two settings.
"""

FPS = 25

# --- possession -----------------------------------------------------------------------------
# A team is a "candidate" on a frame when its nearest player is within CONTROL_RADIUS_M of the
# ball AND at least CONTROL_MARGIN_M closer than the nearest opponent.
CONTROL_RADIUS_M = 2.0
CONTROL_MARGIN_M = 0.5

# The other team must be candidate for this many frames in a row before possession switches.
# The switch is then dated back to the first of those frames. The public ground truth has
# possessions as short as 0.8 s, so do not set this much above 25.
SWITCH_FRAMES = 15

# After a dead ball, the first team that is candidate for this many frames in a row starts the
# new possession (dated back to the end of the dead ball).
RESTART_FRAMES = 5

# --- dead ball -------------------------------------------------------------------------------
# Ball counts as out of play when it is this far outside the pitch lines (metres).
OUT_MARGIN_M = 0.5

# "Play has stopped": mean player speed below STILL_SPEED_MS for at least STILL_FRAMES frames
# while the ball is missing or nearly still.
STILL_SPEED_MS = 1.2
STILL_FRAMES = 50
BALL_STILL_MS = 1.0

# Ball out of play stays dead until the restart is taken (deadball.out_until_restart).
# Restart = ball quiet (slower than RESTART_QUIET_MS, or not tracked) for RESTART_QUIET_FRAMES,
# then struck at RESTART_KICK_MS or more, and inside the pitch (+RESTART_INSIDE_M) within 1 s.
USE_OUT_LATCH = True
RESTART_QUIET_MS = 2.0
RESTART_QUIET_FRAMES = 5
RESTART_KICK_MS = 3.0
RESTART_INSIDE_M = 2.0
MAX_OUT_DEAD_FRAMES = 1500
# The ball must be out for OUT_MIN_FRAMES in a row to count (tracking noise near the lines).
OUT_MIN_FRAMES = 5
# Fallback release when no restart kick is recognised: players averaging PLAY_SPEED_MS or more and
# the ball averaging PLAY_BALL_MS or more, on the pitch, for PLAY_FRAMES -> play has resumed
# (dated back to where that run began).
PLAY_SPEED_MS = 1.5
PLAY_BALL_MS = 3.0
PLAY_FRAMES = 25

# Whistle with the ball on the pitch (offside, foul) -> dead until the free kick
# (deadball.whistle_until_restart). Speeds are averaged over +/- STOP_WINDOW_FRAMES first.
USE_WHISTLE = True
STOP_WINDOW_FRAMES = 25
STOP_SPEED_MS = 1.2
STOP_BALL_MS = 3.0
STOP_FRAMES = 25
STOP_BACKDATE_FRAMES = 50

# Smart Tagging set pieces. The tag window (~5 s) contains the restart, but where in the window is
# not known. SETPIECE_RESTART_FRAC = 0.5 assumes the middle; tune it on the public 5 minutes and
# your validation set. Frames from the window start to the restart are marked dead.
USE_SMART_TAGGING = False
SETPIECE_RESTART_FRAC = 0.5

# Clean-up of the dead-ball mask: drop dead runs shorter than MIN_DEAD_FRAMES, and merge dead runs
# separated by fewer than MERGE_GAP_FRAMES alive frames.
MIN_DEAD_FRAMES = 25
MERGE_GAP_FRAMES = 25

# --- goalkeeper holding the ball (tactico/keeper.py) -----------------------------------------
# Public GT, video 08:34.0 (Barca keeper) and 10:35.6 (Athletic keeper): keeper saves and holds
# the ball -> ground truth says dead ball until he releases it. A hold after a ball from the other
# team is marked D (also when his team won it less than KEEPER_RECENT_FRAMES earlier, e.g. his own
# parry before gathering); after a back-pass from a team-mate it is not.
USE_KEEPER = False  # off until we have confirmed it on the right scenes
KEEPER_RADIUS_M = 1.5
KEEPER_BALL_MAX_MS = 3.0
KEEPER_MIN_FRAMES = 25
KEEPER_MAX_GAP_FRAMES = 125
KEEPER_RECENT_FRAMES = 125
