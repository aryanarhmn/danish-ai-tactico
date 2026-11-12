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
what spellF1 rewards. In training, frames inside short possessions count up to three times as much,
because spellF1 gives a two-second possession the same weight as a forty-second one.

The model is trained only on external data, as the organisers' clarification of 3 November
requires. No parameter is fitted to the competition game, its ground truth or its video, and no
hand-labelled data is used anywhere. Before that clarification we had tuned our rules on the public
five minutes and on a few minutes we labelled by hand. When the rules were clarified we deleted the
hand labels and the tools that made them, and redid everything on external data. The submitted
system uses neither the hand labels nor the public five minutes. The git history
shows both stages.

We started with a hand-written rule system (nearest player with hysteresis, plus dead-ball rules for
ball out, whistles and restarts). It is still in the repository (`python run.py` without `--model`)
and serves as our baseline. On held-out external matches the learned model beats it by a wide
margin (table below), so the submission uses the model.

We used Claude (Anthropic) as a coding and writing assistant. Every number in this document comes
from the scripts in the repository.

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
- SkillCorner's first open release: nine European matches of 2019/20 broadcast tracking (Premier
  League, Bundesliga, Serie A, Ligue 1, LaLiga), in the same repository's git history under the
  same MIT licence (`tools/external/skillcorner_old.py`). They have no stoppage events, so we label
  only the frames where a named player has the ball (about a fifth of each match) and use them for
  training only.
- Metrica Sports sample data: three anonymised matches with tracking and events
  (github.com/metrica-sports/sample-data). They have no possession truth, so we derive labels from
  the events. We use these matches only as a second, independent test set.

**Training labels: who made them and how many.** Nobody on our team labelled anything by hand. No
label in training, validation or tuning comes from the provided game, and the public five minutes
were used only to check that the pipeline and file format are right.

| Dataset | Who produced the labels | Labelled frames (25 fps) | Use |
|---|---|---|---|
| DFL / Sportec, 7 matches | the provider: official `BallPossession` and `BallStatus` on every frame | 1,002,644 | training, cross-validation |
| SkillCorner 2024/25, 20 matches | SkillCorner: possession group and stoppage events; our code turns them into H, A, D | 2,948,890 | training, cross-validation |
| SkillCorner 2019/20, 9 matches | SkillCorner: the player in possession; we keep only those frames | 278,267 | training only |
| Metrica sample, 3 matches | our code, from Metrica's event data | – | test only |

**Making clean data look like the competition data.** External tracking is corrected, while the
competition tracking is automatic. Two scripts measure the provided tracking with automatic checks
that print summary numbers only (no ground truth, no video). `tools/noise_match.py`: the ball is
missing on 41 % of the frames inside the halves, in 150 gaps with a mean of 17 s and a median of
11 s. Positions are very smooth (about 1 cm of frame-to-frame jitter on the ball), about 10.2 of 11
players per team are in view, and speeds are in m/s. `tools/gap_profile.py`: the ball is missing on
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

Many of these work without the ball. The team in possession pushes up and spreads out, the
defending team drops and narrows, and at a stoppage everyone slows down. The ball is missing on four
frames in ten, often for 10 to 30 seconds at a time, so these signals carry a lot of the work.

## Possession rules

The definitions come from the training truth (provider tracking), not from hand-written rules.
What the model learned matches CHALLENGE.md on the points we checked: possession stays with the last
team through loose balls and lost tracking, a touch without control does not change it, and every
restart starts with a clear possessor. Dead ball runs from the moment play stops until the restart
is taken. The second-half block before kick-off (frames 110808 to 113741) is always `D`; the
organisers publish this in FAQ.md (graded from 110,808, kick-off at 113,742).

The two decoding costs (switching team, entering or leaving a dead ball) were chosen by
cross-validation on the external matches, together with the blend of the two models (see Results).
They are 4 and 12. Results change little across a wide range of costs.

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
over all thirty matches (`tools/cv_multi.py`). Every fold holds out DFL and SkillCorner matches the
model has not seen.

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

The average beats the classifier on 25 of the 27 held-out matches.

**Tuning the blend.** The same five folds were run again with three GRUs per fold (seeds 0, 1, 2),
and the held-out probabilities were saved (`tools/train_seq.py --dump`). `tools/blend_cv.py` then
decoded every combination of 1 or 3 GRUs, GRU weight 0.5 to 1.0 and decoding costs 2 to 12 (switch)
and 8 to 24 (dead ball). The criterion was fixed beforehand: the mean of the DFL mean and the
SkillCorner mean.

| Setting | DFL | SkillCorner | Criterion |
|---|---|---|---|
| 1 GRU, weight 0.5, costs 8/16 (previous) | 0.550 | 0.494 | 0.522 |
| 3 GRUs, weight 0.5, costs 4/12 | 0.556 | 0.498 | 0.527 |
| 3 GRUs, weight 0.8, costs 4/12 | 0.572 | 0.499 | 0.535 |
| same, plus the nine older SkillCorner matches in training | 0.573 | 0.504 | 0.538 |
| same, with short possessions weighted in training (submitted) | **0.633** | **0.516** | **0.575** |

The top of the table is flat: GRU weights 0.7 to 0.8 with switch costs 2 to 4 and dead-ball costs
8 to 12 all score 0.535 within 0.001, so we took a setting in the middle of that plateau. It beats
the previous setting on 20 of the 27 held-out matches. Three seeds matter once the GRU carries most
of the weight: with one GRU, weight 0.8 scores 0.524.

The nine older SkillCorner matches were added last, with the same folds, seeds, weight and costs.
They raise the criterion by 0.003 and improve 18 of the 27 held-out matches, mostly the automatic
tracking. Re-tuning the weight and costs with them gave at most 0.003 more and only on 17 matches,
which we read as noise, so we kept the settings above. Two decoding ideas did not help either: a
minimum duration per label (best +0.001, better on 9 of 27 matches) and writing `X` where the model
is unsure, which only lowered the score (`tools/decode_cv.py`).

**Weighting short possessions.** spellF1 counts every possession once, however long it is, but in a
per-frame loss a 40-second possession weighs twenty times as much as a 2-second one. The models
therefore found too few possessions: 276 per held-out DFL match against 437 in the truth, and 183
per SkillCorner match against 294. We now weight the loss inside short possessions. A frame in an
`H` or `A` possession of n frames gets weight min(√(250/n), 3) when the possession is shorter than
10 seconds, so a 2.5-second possession counts twice per frame and one under about 1.1 seconds three
times (`--spell-balance 1.0` in `tools/train_ml.py` and `tools/train_seq.py`). With the same folds,
seeds, weight and costs, the models find 336 and 214 possessions per match, and spellF1 rises on all
7 DFL matches and on 18 of the 20 SkillCorner matches (the worst falls by 0.022). The classifier
alone gains 0.018 on DFL and 0.002 on SkillCorner, so most of the gain comes through the GRU. We
ran the grid of weights and costs again on the new folds: 0.8 and 4/12 is still at the top, so the
settings did not change.

We also tried a second stage that sees the first stage's probabilities over 30 seconds before
and after each frame (`tools/stack_cv.py`). It scored the same on DFL (0.568) and 0.014 higher on
the Metrica games, too little to justify a second model, so we left it out. Earlier we tried, and
dropped, a rule that the ball must move with the player and rules for goals and for a ball lost at
the line. None of them helped on external data.

**Public five-minute check.** After the final model was fixed, we scored it once on the public five
minutes with the official script, as a check that the file and labels are right: spellF1 0.549,
macroF1 0.774, 18 predicted possessions against 17. The window has only 17 possessions, so one more
or one fewer moves spellF1 by about six points. Nothing was chosen or tuned on it.

## Limitations

The external truth is not the competition truth. DFL marks about 23 possessions per five minutes and
SkillCorner about 15, while CHALLENGE.md gives 17 for the public window of the competition game, so
providers differ in how quickly they hand possession over. Training on both puts the model between
the two, and the decoding cost absorbs part of the rest, but not all.

The SkillCorner labels come from SkillCorner's own algorithm, not from people, so the SkillCorner
scores measure how closely we reproduce that algorithm on automatic tracking. The DFL labels are
the provider's official ones.

Our degradation is an incomplete model of the automatic tracking. It places ball gaps by the players'
speed, as measured, but it does not simulate identity switches, false detections or a ball tracked
in the wrong place.

We did not test the GRU on the Metrica games. With more time we would add more external matches of
automatic tracking with full possession and stoppage labels, and try a larger GRU.

## Reproduction

```bash
pip install -r requirements.txt -r requirements-ml.txt
git clone --depth 1 https://github.com/metrica-sports/sample-data.git C:/mtah-ext/sample-data
git clone --depth 1 https://github.com/SkillCorner/opendata.git C:/mtah-ext/skillcorner
python tools/external/dfl_open.py --out C:/mtah-ext/dfl            # downloads and converts the DFL matches
python tools/external/skillcorner_open.py --src C:/mtah-ext/skillcorner/data/matches --out C:/mtah-ext/sc
git clone --filter=blob:none --no-checkout https://github.com/SkillCorner/opendata.git C:/mtah-ext/sc-old-repo
python tools/external/skillcorner_old.py --repo C:/mtah-ext/sc-old-repo --out C:/mtah-ext/sc-old
python tools/cv_multi.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --sample-dir C:/mtah-ext/sample-data   # optional
python tools/train_ml.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --extra-dir C:/mtah-ext/sc-old --save models/possession.pkl --switch-cost 8 --dead-cost 16 --spell-balance 1.0
python tools/train_seq.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --extra-dir C:/mtah-ext/sc-old --cv --seeds 3 --spell-balance 1.0 --dump C:/mtah-ext/dump   # optional
python tools/blend_cv.py C:/mtah-ext/dump                                                                       # optional
python tools/train_seq.py --dfl-dir C:/mtah-ext/dfl --sc-dir C:/mtah-ext/sc --extra-dir C:/mtah-ext/sc-old --epochs 6 --seeds 3 --save models/sequence.pt --spell-balance 1.0
python run.py --model models/possession.pkl --seq models/sequence.pt --seq-weight 0.8 --switch-cost 4 --dead-cost 12
python evaluation/validate_submission.py evaluation/results/Danish_AI-Tactico.csv
```

Both trained models are committed in `models/`, so
`python run.py --model models/possession.pkl --seq models/sequence.pt --seq-weight 0.8 --switch-cost 4 --dead-cost 12`
reproduces the submitted file
byte for byte without retraining. The GRU runs on one CPU thread, so its sums come out the same on
every run.
