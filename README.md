# Danish AI-Tactico — More than a Hack 2027, Phase 1

Frame-level ball possession for FC Barcelona vs Athletic Club, built on Metrica Sports'
[challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027). Their files are
kept unchanged (`games.py`, `evaluation/`, `starter/`, docs; their README is `README_metrica.md`).

## Approach (first draft)

1. **Load** the tactical ATD once and cache it as numpy arrays (`tactico/load.py`).
2. **Features** per frame: each team's nearest player to the ball, ball visibility and speed,
   ball outside the pitch, mean player speed (`tactico/features.py`).
3. **Dead ball** = ball out of the pitch, or play stopped (players slow and ball missing or still),
   or the lead-in to a Smart Tagging set piece; plus the published second-half gap before kickoff
   (`tactico/deadball.py`).
4. **Possession** = a state machine with hysteresis. A team becomes owner only after it has been
   clearly closest to the ball for `SWITCH_FRAMES` frames in a row, and the change is dated back to
   the start of that run. Loose balls and lost tracking keep the current owner
   (`tactico/possession.py`).

All parameters are in `tactico/config.py`.

## Reproduce

```bash
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data            # Windows: $env:MTAH_DATA_DIR = "C:\mtah-data"
python run.py                                 # -> evaluation/results/Danish_AI-Tactico.csv
python evaluation/validate_submission.py evaluation/results/Danish_AI-Tactico.csv
python evaluation/run_evaluation.py
```

The output is deterministic: running it twice gives a byte-identical file.

## Validation

`tools/intervals_to_gt.py` turns hand-labelled video intervals into a ground-truth CSV for
`run_evaluation.py --gt`. Hand labels are used for validation only; the submitted file comes
entirely from `run.py`.

## Data

The match data is not in this repository and must not be: `data/`, `cache/` and `val/` are
gitignored.
