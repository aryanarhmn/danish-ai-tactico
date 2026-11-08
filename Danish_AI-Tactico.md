# Danish AI-Tactico: frame-level possession from automatic tracking

More than a Hack 2027, Phase 1. Team: Aryan Rahman (lead), Allan Rahman, Ibrahim, Kiyan, William.
Repository: https://github.com/<user>/danish-ai-tactico

## Approach

We label every frame of FC Barcelona vs Athletic Club as `H`, `A` or `D` with a learned model:
a gradient-boosted classifier that gives a probability for each label per frame, followed by
Viterbi decoding that charges a cost for every change of label. The decoder turns flickering
per-frame guesses into whole possessions, which is what spellF1 rewards.

The model is trained only on external data, as the organisers' clarification of 3 November
requires. No parameter is fitted to the competition game, its ground truth or its video, and no
hand-labelled data is used anywhere.

We started with a hand-written rule system (nearest player with hysteresis, plus dead-ball rules for
ball out, whistles and restarts). It is still in the repository (`python run.py` without `--model`)
and serves as our baseline. On held-out external matches the learned model beats it by a wide
margin (table below), so the submission uses the model.

## Data

**Provided assets.** Only the automatic tactical tracking (ATD): ball and player positions, player
speed and the team of each track. We tried the Smart Tagging set pieces and dropped them: the tag
window does not say where the restart is, and using it lengthened dead balls by 10 to 12 seconds.
The video was watched only to understand what counts as possession and a dead ball.

**External training data.**

- DFL / Sportec open data: seven Bundesliga matches with official position data, CC BY 4.0
  (Bassek, Rein, Weber and Memmert, 2025, Scientific Data, doi:10.6084/m9.figshare.28196177). Every
  ball frame carries the provider's `BallPossession` and `BallStatus`. This is the same kind of
  truth as the competition ground truth, which CHALLENGE.md describes as derived mechanically from
  corrected tracking, and the ball stays tracked while it is dead.
- Metrica Sports sample data: three anonymised matches with tracking and events
  (github.com/metrica-sports/sample-data). They have no possession truth, so we derive labels from
  the events. We use these matches only as a second, independent test set.

**Making clean data look like the competition data.** External tracking is corrected; the
competition tracking is automatic. `tools/noise_match.py` measures the provided tracking with an
automatic check that prints summary numbers only (no ground truth, no video): the ball is missing
on 41 % of the frames inside the halves, in 150 gaps with a mean of 17 s and a median of 11 s;
positions are very smooth (about 1 cm of frame-to-frame jitter on the ball); about 10.2 of 11
players per team are in view; speeds are in m/s.

During training we degrade every external match to match those numbers: the ball is removed on
41 % of frames in gaps of mean 10 or 17 seconds, 1.5 to 3 cm of smooth position error is added, and
player tracks drop out for 7 to 12 % of the time. Each match is used under two such degradations.
Test matches get a third degradation the model has not seen.

## Features

One row per frame, 67 values. Each team gets the same block of features, so swapping the blocks
swaps the teams. Training uses both orders, and prediction averages both, so the model cannot
learn anything about which side is "home".

Per team: distance from the ball to the nearest and second-nearest player, players within 2, 5 and
10 m, the same distances measured from the last place the ball was seen, means of these over the
previous and next 1 and 3 seconds, mean running speed, share of players standing still, how far the
team's centre is pushed towards the opponent's goal (now and averaged over the previous and next
10 seconds) and how fast it moves that way over 1, 3 and 10 seconds, the team's
spread in length and width, and the ball's position along the team's attacking direction.

Shared: whether the ball is tracked and for how long it has been missing, ball speed now and over
the last and next second, distance to the touchline and goal line, ball outside the pitch, mean
player speed over 2, 5 and 15 second windows before and after the frame.

Many of these work without the ball. The team in possession pushes up and spreads out; the
defending team drops and narrows; at a stoppage everyone slows down. That matters when the ball is
missing on four frames in ten, often for 10 to 30 seconds at a time.

## Possession rules

The definitions come from the training truth (provider tracking), not from hand-written rules.
What the model learned matches CHALLENGE.md on the points we checked: possession stays with the last
team through loose balls and lost tracking, a touch without control does not change it, and every
restart starts with a clear possessor. Dead ball runs from the moment play stops until the restart
is taken. The second-half block before kick-off (frames 110808 to 113741) is always `D`.

The two decoding costs (switching team, entering or leaving a dead ball) were chosen by
leave-one-match-out cross-validation on the DFL matches. Results change little across a wide range
of costs.

## Results on external data

All numbers are spellF1 (the ranking metric) on matches the model was not trained on, degraded as
above.

| | Rules (v7) | Learned model |
|---|---|---|
| DFL, leave-one-match-out (7 matches) | 0.329 | 0.533 |
| Metrica sample games, model trained on DFL (3 matches) | 0.427 | 0.563 |

The model wins on every one of the ten matches. We report no score on the competition game: its
public five minutes were not used to choose or tune anything.

## Limitations

The external truth is not the competition truth. DFL marks about 26 possessions per five minutes,
while the public window of the competition game has 17, so the two providers probably differ in
how quickly they hand possession over. The decoding cost absorbs part of this, but not all.

Our degradation is a model of the automatic tracking, not a copy. It does not simulate identity
switches, false detections or a ball tracked in the wrong place, and real gaps are not random:
the ball is more likely lost in the air, in crowds and when it is dead.

With more time we would train a sequence model (a bidirectional recurrent network) on the same
features, model where the ball goes missing instead of placing gaps at random, and add more
external matches.

## Reproduction

```bash
pip install -r requirements.txt -r requirements-ml.txt
git clone --depth 1 https://github.com/metrica-sports/sample-data.git C:/mtah-ext/sample-data
python tools/external/dfl_open.py --out C:/mtah-ext/dfl            # downloads and converts the DFL matches
python tools/train_ml.py --dfl-dir C:/mtah-ext/dfl --cv            # cross-validation (optional)
python tools/train_ml.py --dfl-dir C:/mtah-ext/dfl --save models/possession.pkl --switch-cost 8 --dead-cost 16
python run.py --model models/possession.pkl               # -> evaluation/results/Danish_AI-Tactico.csv
python evaluation/validate_submission.py evaluation/results/Danish_AI-Tactico.csv
```

The trained model is committed in `models/`, so `python run.py --model models/possession.pkl`
reproduces the submitted file byte for byte without retraining.
