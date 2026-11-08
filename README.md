# Danish AI-Tactico — More than a Hack 2027, Phase 1

Frame-level ball possession for FC Barcelona vs Athletic Club, built on Metrica Sports'
[challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027). Their files are
kept unchanged (`games.py`, `evaluation/`, `starter/`, docs; their README is `README_metrica.md`).

## Approach

A gradient-boosted classifier gives P(H), P(A), P(D) for every frame from tracking features, and
Viterbi decoding with a cost per change turns that into whole possessions (`tactico/ml.py`). It is
trained only on external data: the seven DFL open matches, degraded to look like the competition's
automatic tracking (`tools/train_ml.py`). The technical document is `Danish_AI-Tactico.md`.

The earlier rule system (nearest player with hysteresis, dead-ball rules) is kept as the baseline:
`python run.py` without `--model`.

## Reproduce

```bash
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data            # Windows: $env:MTAH_DATA_DIR = "C:\mtah-data"
pip install -r requirements-ml.txt
python run.py --model models/possession.pkl   # -> evaluation/results/Danish_AI-Tactico.csv
python evaluation/validate_submission.py evaluation/results/Danish_AI-Tactico.csv
python evaluation/run_evaluation.py
```

The output is deterministic: running it twice gives a byte-identical file.

## Tuning on external data

Every number in `tactico/config.py` is tuned on external data only, as the organisers' rules
require: Metrica Sports' three public sample games
([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), used with
acknowledgement of the source). Those games have no possession ground truth, so
`tools/external/metrica_sample.py` derives labels from their event data, and
`tools/tune_external.py` scores each parameter variant on all three games with the competition
scorer and prints the mean.

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../sample-data
python tools/tune_external.py --sample-dir ../sample-data --grid poss
python tools/tune_external.py --sample-dir ../sample-data --grid poss --noise 0.3
```

Two differences from the competition data are handled explicitly: the sample games drop the ball
from the tracking while it is dead, so long ball gaps are filled by interpolation (otherwise a
tuner learns "no ball = dead ball"); and their tracking is corrected, so `--noise` adds simulated
position error to check that a choice also holds on noisier tracking.

No hand labels are used anywhere, and nothing is tuned on the competition game, its ground truth
or its video. The video was only watched to understand what counts as possession and a dead ball,
which shaped which rules exist.

## Data

The match data is not in this repository and must not be: `data/` and `cache/` are
gitignored.
