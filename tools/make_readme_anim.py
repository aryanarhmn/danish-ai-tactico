"""Animated SVG for the README: 40 seconds of a held-out Bundesliga match (DFL open data, CC BY 4.0)
with the truth and our model's call underneath. EXTERNAL DATA ONLY; never the competition match.

The match is degraded exactly as in the cross-validation test (ball removed where the competition
tracker loses it, some player tracks dropped), and the model's labels are the held-out predictions
saved by `tools/train_seq.py --cv --dump`. The clip was chosen to be typical, not the best: its
frame agreement is printed and should sit near the match average.

    python tools/make_readme_anim.py --dfl-dir C:\\mtah-ext\\dfl --dump C:\\mtah-ext\\dump --out docs/replay.svg
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "external"))
import blend_cv as B  # noqa: E402
import train_ml as T  # noqa: E402
from tune_external import degrade, raw_games  # noqa: E402

MATCH, START, SECONDS, FPS = "DJ03WOH", 103799, 40, 5
W, H = 1200, 940
SC = 9.6                       # px per metre
PX, PY = 75, 112               # pitch origin
COL = {"H": "#1f5c45", "A": "#c99a2e", "D": "#c4cbc7"}
INK, MUTED, LINE, BG, GRASS = "#16201b", "#6b7a73", "#ffffff", "#ffffff", "#e9efe9"


def fmt(v):
    return ";".join("{:.1f}".format(x) for x in v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dfl-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "replay.svg"))
    args = ap.parse_args()

    games = raw_games("dfl", None, args.dfl_dir)
    i = [n for n, _ in games].index(MATCH)
    d = games[i][1]
    spec = T.parse_mix(T.TEST_MIX)[0]       # the held-out degradation, same seed as the CV test
    dd = degrade(d, spec[0], spec[1], int(spec[2]), spec[3], seed=500 + 10 * (i + 1), by_speed=spec[4])
    matches = []
    for p in sorted(glob.glob(os.path.join(args.dump, "fold*.pkl"))):
        with open(p, "rb") as handle:
            matches += pickle.load(handle)
    m = [x for x in matches if x["name"] == MATCH][0]
    prob = 0.2 * m["p_hgb"] + 0.8 * m["p_gru"].mean(axis=0)
    pred = B.label(prob, m["halves"], 4.0, 12.0)
    truth = d["labels"]

    n = SECONDS * FPS
    idx = START + np.arange(n) * (25 // FPS)
    seg = slice(START, START + SECONDS * 25)
    inside = np.zeros(len(truth), bool)
    for _, a, b in m["halves"]:
        inside[a:b + 1] = True
    print("clip agreement {:.1%}, match agreement {:.1%}".format(
        np.mean(truth[seg] == pred[seg]), np.mean(truth[inside] == pred[inside])))

    dur = "{}s".format(SECONDS)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
           'aria-label="Forty seconds of a held-out Bundesliga match: player and ball tracking with the true '
           'possession and our model\'s call underneath.">'.format(w=W, h=H),
           '<title>Danish AI-Tactico: our model on a held-out match</title>',
           '<style>text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}'
           '.m{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}</style>',
           '<rect width="{}" height="{}" rx="18" fill="{}"/>'.format(W, H, BG),
           '<text x="{}" y="50" font-size="24" font-weight="700" fill="{}">Who has the ball?</text>'.format(PX, INK),
           '<text x="{}" y="80" font-size="15" fill="{}">A Bundesliga match the model never trained on. '
           'The ball is removed where the competition tracker loses it.</text>'.format(PX, MUTED)]

    # pitch
    pw, ph = 105 * SC, 68 * SC
    g = ['<rect x="{}" y="{}" width="{}" height="{}" rx="6" fill="{}"/>'.format(PX, PY, pw, ph, GRASS)]
    st = 'fill="none" stroke="{}" stroke-width="2.5"'.format(LINE)
    g += ['<rect x="{}" y="{}" width="{}" height="{}" {}/>'.format(PX, PY, pw, ph, st),
          '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" {3}/>'.format(PX + pw / 2, PY, PY + ph, st),
          '<circle cx="{}" cy="{}" r="{}" {}/>'.format(PX + pw / 2, PY + ph / 2, 9.15 * SC, st)]
    for x0, sgn in ((PX, 1), (PX + pw, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            x = x0 if sgn > 0 else x0 - depth * SC
            g.append('<rect x="{}" y="{}" width="{}" height="{}" {}/>'.format(
                x, PY + ph / 2 - half * SC, depth * SC, 2 * half * SC, st))
    out += g

    # players
    pl = dd["players"][idx]
    for k in range(pl.shape[1]):
        xy = pl[:, k]
        seen = ~np.isnan(xy[:, 0])
        if seen.sum() < n * 0.2:
            continue
        xs = np.where(seen, PX + xy[:, 0] * SC, np.nan)
        ys = np.where(seen, PY + xy[:, 1] * SC, np.nan)
        for arr in (xs, ys):                 # hold the last known place while hidden
            good = np.flatnonzero(~np.isnan(arr))
            arr[:] = np.interp(np.arange(n), good, arr[good])
        colour = COL["H"] if d["team"][k] == 0 else COL["A"]
        out.append('<circle r="9" fill="{}" stroke="#ffffff" stroke-width="2">'
                   '<animate attributeName="cx" dur="{d}" repeatCount="indefinite" values="{}"/>'
                   '<animate attributeName="cy" dur="{d}" repeatCount="indefinite" values="{}"/>'
                   '<animate attributeName="opacity" dur="{d}" repeatCount="indefinite" calcMode="discrete" values="{}"/>'
                   '</circle>'.format(colour, fmt(xs), fmt(ys), ";".join("1" if s else "0" for s in seen), d=dur))

    # ball (only where the degraded tracking has it)
    b = dd["ball"][idx]
    seen = ~np.isnan(b[:, 0])
    bx, by = PX + b[:, 0] * SC, PY + b[:, 1] * SC
    for arr in (bx, by):
        good = np.flatnonzero(~np.isnan(arr))
        arr[:] = np.interp(np.arange(n), good, arr[good])
    out.append('<circle r="6" fill="#ffffff" stroke="{}" stroke-width="2.5">'
               '<animate attributeName="cx" dur="{d}" repeatCount="indefinite" values="{}"/>'
               '<animate attributeName="cy" dur="{d}" repeatCount="indefinite" values="{}"/>'
               '<animate attributeName="opacity" dur="{d}" repeatCount="indefinite" calcMode="discrete" values="{}"/>'
               '</circle>'.format(INK, fmt(bx), fmt(by), ";".join("1" if s else "0" for s in seen), d=dur))

    # timelines
    ty = PY + ph + 34
    tx, tw = PX + 120, pw - 120
    rows = [("Truth", truth[seg]), ("Our model", pred[seg]),
            ("Ball tracked", np.where(np.isnan(dd["ball"][seg][:, 0]), "-", "B"))]
    for r, (name, lab) in enumerate(rows):
        y = ty + r * 30
        out.append('<text x="{}" y="{}" font-size="14" fill="{}">{}</text>'.format(PX, y + 15, INK, name))
        runs = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        for a, e in zip(runs, np.r_[runs[1:], len(lab)]):
            c = {"B": "#8c9a93", "-": "#eef1ef"}.get(lab[a], COL.get(lab[a], "#ffffff"))
            out.append('<rect x="{:.1f}" y="{}" width="{:.1f}" height="{}" fill="{}"/>'.format(
                tx + a / len(lab) * tw, y + (8 if r == 2 else 0), (e - a) / len(lab) * tw, 6 if r == 2 else 20, c))
    out.append('<line x1="{x}" x2="{x}" y1="{}" y2="{}" stroke="{}" stroke-width="2">'
               '<animate attributeName="x1" dur="{d}" repeatCount="indefinite" values="{x};{x2}"/>'
               '<animate attributeName="x2" dur="{d}" repeatCount="indefinite" values="{x};{x2}"/>'
               '</line>'.format(ty - 6, ty + 86, INK, x=tx, x2=tx + tw, d=dur))

    ly = H - 26
    items = [("H", "Home team has the ball"), ("A", "Away team has the ball"), ("D", "Dead ball")]
    x = PX
    for key, text in items:
        out.append('<rect x="{}" y="{}" width="14" height="14" rx="3" fill="{}"/>'.format(x, ly - 12, COL[key]))
        out.append('<text x="{}" y="{}" font-size="14" fill="{}">{}</text>'.format(x + 22, ly, INK, text))
        x += 230
    out.append('<text x="{}" y="{}" font-size="12" text-anchor="end" fill="{}" class="m">DFL open data, CC BY 4.0 '
               '· 40 s at 5 fps</text>'.format(PX + pw, ly, MUTED))
    out.append('</svg>')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out))
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
