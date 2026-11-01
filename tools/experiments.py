"""Run every candidate version and score them on both ground truths. Prints scores only.

    python tools/experiments.py

Each version is the current tactico/config.py plus a few overrides. Files go to
evaluation/results/exp_*.csv (gitignored). The output is scoreboards only -- no match data --
so it may be shared.
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "results")
VAL = os.path.join(ROOT, "val", "mit_facit.csv")

VERSIONS = [
    ("exp_v4", ["USE_RESTART_KICKER=false"]),
    ("exp_kicker", []),
    ("exp_kicker_keeper", ["USE_KEEPER=true"]),
    ("exp_kicker_m10", ["CONTROL_MARGIN_M=1.0"]),
    ("exp_kicker_m15", ["CONTROL_MARGIN_M=1.5"]),
    ("exp_kicker_m10_keeper", ["CONTROL_MARGIN_M=1.0", "USE_KEEPER=true"]),
]


def main():
    for old in glob.glob(os.path.join(RESULTS, "exp_*.csv")):
        os.remove(old)
    for name, sets in VERSIONS:
        cmd = [sys.executable, os.path.join(ROOT, "run.py"), "--out", os.path.join(RESULTS, name + ".csv")]
        for s in sets:
            cmd += ["--set", s]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if out.returncode:
            print(out.stderr)
            sys.exit(1)
        graded = [l for l in out.stdout.splitlines() if l.startswith(("graded", "possessions"))]
        print("{:<24} {}".format(name, "   ".join(x.split("(")[0].strip() for x in graded)))
    for gt, title in ((VAL, "OUR VALIDATION SET"), ("public", "METRICA PUBLIC 5 MINUTES")):
        if gt != "public" and not os.path.isfile(gt):
            print("\n(no {} -- skipped)".format(gt))
            continue
        print("\n=== {} ===".format(title))
        sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                        "--gt", gt, "--no-detail"], cwd=ROOT)


if __name__ == "__main__":
    main()
