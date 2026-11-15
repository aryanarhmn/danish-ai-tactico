"""Train / cross-validate the GRU sequence model (tactico/seq.py) on EXTERNAL data only.

    python tools/train_seq.py --dfl-dir C:\\mtah-ext\\dfl --sc-dir C:\\mtah-ext\\sc --cv
    python tools/train_seq.py --dfl-dir C:\\mtah-ext\\dfl --sc-dir C:\\mtah-ext\\sc --save models\\sequence.pt

--cv uses the same five folds as tools/cv_multi.py and reports the classifier alone, the GRU alone
and their average, on held-out DFL (degraded) and SkillCorner matches.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "external"))
from tactico import ml, seq  # noqa: E402
import train_ml as T  # noqa: E402
from cv_multi import sc_games  # noqa: E402

COSTS = [(4.0, 16.0), (8.0, 16.0), (12.0, 16.0), (8.0, 24.0)]
EPOCHS = 12


def to32(games):
    """float32 features to halve the memory (the classifier bins them anyway)."""
    for g in games:
        g["Xs"] = [x.astype(np.float32) for x in g["Xs"]]
        g["X"] = g["Xs"][0]
    return games


def chunks(games, stats, rng):
    """Training sequences (inputs, targets) at 5 fps, both team orders.

    Two passes so the output is allocated once (a list plus np.stack held everything twice). Same
    random offsets, same order and same values as the one-pass version."""
    k = len(ml.TEAM_FEATURES)
    plan, total = [], 0
    for g in games:
        for j in range(len(g["Xs"])):
            for _, first, last in g["halves"]:
                off = int(rng.integers(0, seq.STEP))
                n_steps = len(range(first + off, last + 1, seq.STEP))
                n_chunks = (n_steps - seq.CHUNK) // seq.CHUNK + 1 if n_steps >= seq.CHUNK else 0
                plan.append((g, j, first + off, last, n_chunks))
                total += 2 * n_chunks
    n_in = 2 * games[0]["Xs"][0].shape[1]
    X = np.empty((total, seq.CHUNK, n_in), dtype=np.float32)
    Y = np.empty((total, seq.CHUNK), dtype=np.int64)
    Wt = np.empty((total, seq.CHUNK), dtype=np.float32) if T.SPELL_BALANCE > 0 else None
    i = 0
    for g, j, start, last, n_chunks in plan:
        if not n_chunks:
            continue
        z, _ = seq.prepare_inputs(g["Xs"][j][start:last + 1:seq.STEP], stats)
        y = g["y"][start:last + 1:seq.STEP]
        wy = T.spell_weights(g["y"])[start:last + 1:seq.STEP] if Wt is not None else None
        for c in range(n_chunks):
            zz, yy = z[c * seq.CHUNK:(c + 1) * seq.CHUNK], y[c * seq.CHUNK:(c + 1) * seq.CHUNK]
            X[i], X[i + 1] = zz, seq.swap_inputs(zz, k)
            Y[i], Y[i + 1] = yy, np.where(yy == 2, 2, np.where(yy >= 0, 1 - yy, -1))
            if Wt is not None:
                Wt[i] = Wt[i + 1] = wy[c * seq.CHUNK:(c + 1) * seq.CHUNK]
            i += 2
    if Wt is not None:
        return X, Y, Wt
    return X, Y


DEVICE = "cpu"       # "cuda" trains on the graphics card; prediction always runs on the CPU
BOUNDARY_W = 0.0     # >0: train a fourth output, "the label changes within BOUNDARY_K steps"
BOUNDARY_K = 2       # steps at 5 fps, i.e. +-0.4 s


def boundary_targets(Y):
    """(chunks, steps) float: 1 within BOUNDARY_K steps of a label change, 0 elsewhere, -1 where
    the label is unknown."""
    import torch
    change = torch.zeros(Y.shape, dtype=torch.float32)
    known = (Y[:, 1:] >= 0) & (Y[:, :-1] >= 0)
    change[:, 1:] = ((Y[:, 1:] != Y[:, :-1]) & known).float()
    if BOUNDARY_K:
        k = 2 * BOUNDARY_K + 1
        change = torch.nn.functional.max_pool1d(change[:, None], k, stride=1, padding=BOUNDARY_K)[:, 0]
    return torch.where(Y >= 0, change, torch.full_like(change, -1.0))


def fit_seq(games, seed=0, log=None):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.concatenate([g["Xs"][0][g["y"] >= 0][::25] for g in games])
    _, stats = seq.prepare_inputs(allX)
    out = chunks(games, stats, rng)
    X, Y = out[0], out[1]
    W = torch.from_numpy(out[2]) if len(out) > 2 else None
    net = seq.build_net(X.shape[2], 4 if BOUNDARY_W > 0 else 3).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    n = len(Xt)
    if BOUNDARY_W > 0:
        Bt = boundary_targets(Yt)
        frac = float((Bt == 1).sum()) / float((Bt >= 0).sum())
        pos_w = torch.tensor((1 - frac) / max(frac, 1e-6), device=DEVICE)
        if log:
            log("    boundary frames {:.1%}, positive weight {:.1f}".format(frac, float(pos_w)))
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 32):
            idx = perm[b:b + 32]
            opt.zero_grad()
            xb, yb = Xt[idx].to(DEVICE), Yt[idx].to(DEVICE)
            logits = net(xb)
            cls = logits[..., :3]
            if W is None:
                loss = lossf(cls.reshape(-1, 3), yb.reshape(-1))
            else:                       # weighted mean over labelled steps
                yy = yb.reshape(-1)
                ww = W[idx].to(DEVICE).reshape(-1) * (yy >= 0)
                raw = torch.nn.functional.cross_entropy(cls.reshape(-1, 3), yy.clamp(min=0),
                                                        reduction="none")
                loss = (raw * ww).sum() / ww.sum().clamp(min=1e-6)
            if BOUNDARY_W > 0:
                bt = Bt[idx].to(DEVICE).reshape(-1)
                ok = bt >= 0
                bl = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[..., 3].reshape(-1)[ok], bt[ok], pos_weight=pos_w)
                loss = loss + BOUNDARY_W * bl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if log:
            log("    epoch {} loss {:.3f}".format(ep + 1, tot / n))
    net = net.cpu()
    net.eval()
    return {"net": net, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir")
    parser.add_argument("--sc-dir")
    parser.add_argument("--extra-dir", help="more .npz matches used for training only, in every fold "
                        "(tools/external/skillcorner_old.py)")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-folds", type=int, default=None, help="run only the first N folds")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--save")
    parser.add_argument("--spell-balance", type=float, default=0.0,
                        help="extra loss weight on short possessions (0 = off, as in v13; v14 uses 1)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (training only)")
    parser.add_argument("--boundary-weight", type=float, default=0.0,
                        help="train a change-point output with this loss weight (0 = off, as in v14)")
    parser.add_argument("--seeds", type=int, default=1, help="GRUs per fold / in the saved ensemble")
    parser.add_argument("--dump", help="with --cv: save each fold's held-out probabilities here "
                        "(one pickle per fold; folds already saved are skipped)")
    args = parser.parse_args()
    T.SPELL_BALANCE = args.spell_balance
    globals()["BOUNDARY_W"] = args.boundary_weight
    globals()["DEVICE"] = args.device
    if args.device == "cuda":
        import torch as _t
        if not _t.cuda.is_available():
            raise SystemExit("--device cuda, but PyTorch sees no graphics card (CPU-only build?)")
        print("training on", _t.cuda.get_device_name(0), flush=True)
    import torch
    globals()["EPOCHS"] = args.epochs
    t0 = time.time()
    log = lambda m: print(m + " ({:.0f} s)".format(time.time() - t0), flush=True)  # noqa: E731
    dfl_train = to32(T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TRAIN_MIX))[0])
    sc = sc_games(args.sc_dir, f32=True) if args.sc_dir else []
    extra = sc_games(args.extra_dir, f32=True) if args.extra_dir else []
    if args.cv:
        dfl_test = to32(T.prepare("dfl", None, args.dfl_dir, T.parse_mix(T.TEST_MIX), seed_base=500)[0])
        fold_d = [i % args.folds for i in range(len(dfl_train))]
        fold_s = [(i + 2) % args.folds for i in range(len(sc))]
        res = {k: {c: [] for c in COSTS} for k in ("hgb|D", "gru|D", "avg|D", "hgb|S", "gru|S", "avg|S")}
        for f in range(args.only_folds or args.folds):
            if args.dump:
                import pickle
                os.makedirs(args.dump, exist_ok=True)
                out = os.path.join(args.dump, "fold{}.pkl".format(f))
                if os.path.exists(out):
                    log("  fold {} already saved".format(f))
                    continue
            tr = [g for g, k in zip(dfl_train, fold_d) if k != f] + [g for g, k in zip(sc, fold_s) if k != f] + extra
            te = [("D", g) for g, k in zip(dfl_test, fold_d) if k == f] + [("S", g) for g, k in zip(sc, fold_s) if k == f]
            m1 = T.fit(tr)
            m2s = [fit_seq(tr, seed=k, log=log) for k in range(args.seeds)]
            dump = []
            for tag, g in te:
                p1 = ml.predict_proba(m1, g["X"])
                p2s = [seq.predict_proba(m, g["X"], g["halves"]) for m in m2s]
                p2 = np.mean(p2s, axis=0)
                if args.dump:
                    entry = {"tag": tag, "name": g["name"], "p_hgb": p1.astype(np.float32),
                             "p_gru": np.stack(p2s).astype(np.float32), "labels": g["labels"],
                             "halves": g["halves"]}
                    if BOUNDARY_W > 0:
                        entry["p_bnd"] = np.stack([seq.predict_boundary(m, g["X"], g["halves"])
                                                   for m in m2s]).astype(np.float16)
                    dump.append(entry)
                for name, p in (("hgb", p1), ("gru", p2), ("avg", (p1 + p2) / 2)):
                    for c in COSTS:
                        res[name + "|" + tag][c].append(T.score(g["labels"], ml.label(p, g["halves"], *c)))
            if args.dump:
                with open(out, "wb") as handle:
                    pickle.dump(dump, handle)
            log("  fold {} done".format(f))
        for key, r in res.items():
            T.report(r, key)
    if args.save:
        nets = [fit_seq(dfl_train + sc + extra, seed=k, log=log) for k in range(args.seeds)]
        torch.save({"states": [b["net"].state_dict() for b in nets], "stats": nets[0]["stats"],
                    "n_in": 2 * len(ml.FEATURE_NAMES), "n_out": 4 if BOUNDARY_W > 0 else 3}, args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
