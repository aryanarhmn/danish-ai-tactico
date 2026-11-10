# Danish AI-Tactico: frame-level possession from automatic tracking

The formatted version of this document is `Danish_AI-Tactico.pdf`, submitted with the entry.

More than a Hack 2027, Phase 1. Team Danish AI-Tactico, all students:

- Aryan Abdel Rahman (team lead), MSc in Data Science, University of Southern Denmark (SDU)
- Allan Abdel Rahman, Global Management and Manufacturing, University of Southern Denmark (SDU)
- Ibraahim Dahir Xanaf, MSc in Data Science, University of Southern Denmark (SDU)
- William Luu, BSc in Health Data, Technical University of Denmark (DTU)
- Kiyan Rohanimanesh, PBA in Cybersecurity, UCL University College, Odense

Repository: https://github.com/aryanarhmn/danish-ai-tactico

## Approach

We label every frame of FC Barcelona vs Athletic Club as `H`, `A` or `D` with two learned models
and a decoder. A gradient-boosted classifier gives a probability for each label per frame from a
few seconds of context. A bidirectional recurrent network (GRU) reads the same features over a
whole half, so it can carry what happened 30 seconds ago through a long stretch without a tracked
ball. We average the two models' probabilities, and Viterbi decoding then charges a cost for every
change of label. The decoder turns flickering per-frame guesses into whole possessions, which is
what spellF1 rewards.

The model is trained only on external data, as the organisers' clarification of 3 November
requires. No parameter is fitted to the competition game, its ground truth or its video, and no
hand-labelled data is used anywhere. Before that clarification we had tuned our rules on the public
five minutes and on a few minutes we labelled by hand; when the rules were clarified we deleted the
hand labels and the tools that made them, and redid everything on external data. The git history
shows both stages.

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
- SkillCorner open data: twenty A-League 2024/25 matches of automatic broadcast tracking at
  10 fps, MIT licence (github.com/SkillCorner/opendata; data by SkillCorner). This is the same kind
  of tracking as the competition's: players and the ball really drop out (the ball is missing on
  35 to 48 % of frames). Each frame carries SkillCorner's possession group, and the dynamic events
  mark every throw-in, goal kick, free kick, corner and goal. We label a frame `D` from the end of a
  player possession that ends in a stoppage to the start of the next one, and otherwise give it
  the team in possession, carried forward through loose balls. Positions SkillCorner extrapolated
  are dropped, so gaps stay gaps, and the data is resampled to 25 fps.
- Metrica Sports sample data: three anonymised matches with tracking and events
  (github.com/metrica-sports/sample-data). They have no possession truth, so we derive labels from
  the events. We use these matches only as a second, independent test set.

**Making clean data look like the competition data.** External tracking is corrected; the
competition tracking is automatic. Two scripts measure the provided tracking with automatic checks
that print summary numbers only (no ground truth, no video). `tools/noise_match.py`: the ball is
missing on 41 % of the frames inside the halves, in 150 gaps with a mean of 17 s and a median of
11 s; positions are very smooth (about 1 cm of frame-to-frame jitter on the ball); about 10.2 of 11
players per team are in view; speeds are in m/s. `tools/gap_profile.py`: the ball is missing on
80 % of frames where the players' mean speed is below 1 m/s, 42 % at 1 to 2 m/s and 11 % above
2 m/s. The tracker loses the ball mostly around stoppages.

During training we degrade every DFL match to match those numbers (the SkillCorner matches are
used as they are). Ball gaps are placed where
a smooth random field falls below the measured missing rate for the players' current speed, which
gives 43 % missing in about 140 gaps per match with a mean of 17 s and a median of 10 s. We add
1.5 to 3 cm of smooth position error, and player tracks drop out for 7 to 12 % of the time. Each
match is used under two such degradations. Test matches get a third degradation the model has not
seen.

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
| DFL, leave-one-match-out (7 matches) | 0.383 | 0.568 |
| Metrica sample games, model trained on DFL (3 matches) | 0.492 | 0.582 |

The model wins on every one of the ten matches.

**Why we also train on SkillCorner.** DFL and Metrica are both corrected tracking, which we degrade
ourselves. When we tested the DFL-trained model on the SkillCorner matches, which come from a real
automatic tracker, it fell to 0.306. So we compared two models with five-fold cross-validation
over all thirty matches (`tools/cv_multi.py`; every fold holds out DFL and SkillCorner matches the
model has not seen):

| Held out | Trained on DFL | Trained on DFL + SkillCorner |
|---|---|---|
| DFL (7 matches, degraded) | 0.565 | 0.521 |
| SkillCorner (20 matches, automatic tracking) | 0.306 | 0.476 |
| Metrica (3 matches) | 0.585 | 0.574 |

The combined model loses a little on the corrected data and gains 0.17 on real automatic tracking.
The competition tracking is automatic, so we submit the combined model. Its worst case is far
better, and its mean over the three test sets is higher (0.524 against 0.485).

**Adding the sequence model.** The GRU (`tactico/seq.py`, `tools/train_seq.py`) runs at five frames
per second on 60-second training sequences, uses both team orders like the classifier, and is
trained on the same matches. Same five folds, same decoding costs (8 and 16):

| Held out | Classifier | GRU | Average of both |
|---|---|---|---|
| DFL (7 matches, degraded) | 0.521 | 0.545 | **0.551** |
| SkillCorner (20 matches) | 0.468 | 0.481 | **0.493** |

The average beats the classifier on 25 of the 27 held-out matches, so the submission uses it.

We also tried a second stage that sees the first stage's probabilities over 30 seconds before
and after each frame (`tools/stack_cv.py`). It scored the same on DFL (0.568) and 0.014 higher on
the Metrica games, too little to justify a second model, so we left it out. Earlier we tried, and
dropped, a rule that the ball must move with the player and rules for goals and for a ball lost at
the line; none helped on external data. We report no score on the competition game: its
public five minutes were not used to choose or tune anything.

## Limitations

The external truth is not the competition truth. DFL marks about 23 possessions per five minutes and
SkillCorner about 15, while CHALLENGE.md gives 17 for the public window of the competition game, so
providers differ in how quickly they hand possession over. Training on both puts the model between
the two, and the decoding cost absorbs part of the rest, but not all.

Our degradation is a model of the automatic tracking, not a copy. It places ball gaps by the players'
speed, as measured, but it does not simulate identity switches, false detections or a ball tracked
in the wrong place.

We did not test the GRU on the Metrica games. With more time we would add more external matches of
automatic tracking and average several GRUs trained from different seeds.

## Reproduction

```bash
pip install -r requirements.txt -r requirements-ml.txt
git clone --depth 1 https://github.com/metrica-sports/sample-data.git C:/mtah-ext/sample-data
git clone --depth 1 https://github.com/SkillCorner/opendata.git C:/mtah-ext/skillcorner
python tools/external/dfl_open.py --out C:/mtah-ext/dfl            # downloads and converts the DFL matches
python tools/external/skillcorner_open.py --src C:/mtah-ext/skillcorner/data/matches --out C:/mtah-ext/sc
python tools/cv_multi.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --sample-dir C:/mtah-ext/sample-data   # optional
python tools/train_ml.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --save models/possession.pkl --switch-cost 8 --dead-cost 16
python tools/train_seq.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --epochs 6 --save models/sequence.pt
python run.py --model models/possession.pkl --seq models/sequence.pt   # -> evaluation/results/Danish_AI-Tactico.csv
python evaluation/validate_submission.py evaluation/results/Danish_AI-Tactico.csv
```

Both trained models are committed in `models/`, so
`python run.py --model models/possession.pkl --seq models/sequence.pt` reproduces the submitted file
byte for byte without retraining. The GRU runs on one CPU thread, so its sums come out the same on
every run.
