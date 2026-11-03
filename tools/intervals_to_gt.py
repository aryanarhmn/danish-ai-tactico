"""Turn hand-labelled intervals into a full-length ground-truth CSV for validation.

Label a stretch of the tactical video in a spreadsheet, one row per change, and save it as CSV:

    start,label
    12:30.00,H          <- video time mm:ss.ff (ff = frame 0-24), or a plain frame number
    12:41.12,A
    12:55.03,D
    13:10.00,H
    ...
    20:00.00,END        <- where your labelled stretch stops

Then:

    python tools/intervals_to_gt.py val/my_labels.csv val/my_gt.csv
    python evaluation/run_evaluation.py --gt val/my_gt.csv --bootstrap 1000

Every frame outside your labelled stretches is X, so only what you labelled is scored. Several
stretches are fine: end each with an END row. Keep val/ out of git (it is in .gitignore).

Hand labels are allowed for VALIDATION and TRAINING only. The submitted file must come from code.
"""

import csv
import io
import re
import sys

N_FRAMES = 199319
FPS = 25


def to_frame(text):
    """'1:20:05' or '1:20:05.12' (h:mm:ss), '12:30' or '12:30.12' (mm:ss), or a frame number.
    The part after the dot is a frame (0-24); a comma works as the dot too."""
    text = text.strip().replace(",", ".")
    # Numbers exports durations as '30m 18s', '31m' or '1h 2m 3s'
    m = re.fullmatch(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?", text)
    if m and any(m.groups()):
        h, mi, se = (float(g) if g else 0 for g in m.groups())
        return int(round((h * 3600 + mi * 60 + se) * FPS))
    if ":" not in text:
        # Danish Numbers/Excel write times as 1.20.28 -- dots instead of colons
        parts = text.split(".")
        if len(parts) in (3, 4):
            text = ":".join(parts[:3]) + ("." + parts[3] if len(parts) == 4 else "")
        else:
            return int(text)
    clock, _, frames = text.partition(".")
    parts = [int(p) for p in clock.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds * FPS + int(frames or 0)


def fmt(frame):
    s = frame // FPS
    return "{}:{:02d}:{:02d}".format(s // 3600, s // 60 % 60, s % 60)


def main(src, dst):
    labels = ["X"] * N_FRAMES
    with open(src, newline="", encoding="utf-8-sig") as raw:
        lines = raw.read().splitlines()
    # Numbers puts the sheet and table name above the header: skip until the 'start' row
    first = next((i for i, line in enumerate(lines) if line.lower().startswith("start")), 0)
    with io.StringIO("\n".join(lines[first:])) as handle:
        sample = handle.read(2048)
        handle.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        rows = [(to_frame(r["start"]), r["label"].strip().upper())
                for r in csv.DictReader(handle, delimiter=delimiter) if r.get("start")]
    for (a, la), (b, lb) in zip(rows, rows[1:]):
        if b <= a:
            raise ValueError("times must increase: {} ({}) is followed by {} ({}). Fix that row.".format(
                fmt(a), la, fmt(b), lb))
    for (start, lab), (stop, _) in zip(rows, rows[1:]):
        if lab == "END":
            continue
        if lab not in ("H", "A", "D"):
            raise ValueError("label must be H, A, D or END, got {!r} at frame {}".format(lab, start))
        labels[start:stop] = [lab] * (stop - start)
    with open(dst, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote {} ({} labelled frames)".format(dst, sum(lab != "X" for lab in labels)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
