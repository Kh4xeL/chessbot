"""
Fits the rating estimator (python/elo_estimate.py) on held-out Lichess games and reports
how well it does on a separate half of them.

  * fit half (even game ids): the loss table P(loss bin | rating), the weights a, b of the
    policy and loss evidence (best mean absolute error), and the width factor tau (80% of
    true ratings inside the 80% range);
  * test half (odd game ids): accuracy for single games and for 5-game sets of players with
    nearly the same rating.

    python training/fit_elo_estimator.py           # data/heldout_features.npz -> assets/elo_model.json
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from python.elo_estimate import LOSS_BINS, MODEL_PATH, RATINGS, loss_bin, posterior, summarize  # noqa: E402

R = np.array(RATINGS, dtype=np.float64)


def units(meta, skip):
    """Player-games: (game id, colour) -> row indices of that player's moves (after `skip` plies)."""
    keys = meta[:, 0].astype(np.int64) * 2 + meta[:, 2]
    keep = meta[:, 1] >= skip
    order = np.argsort(keys, kind="stable")
    k_sorted = keys[order]
    bounds = np.flatnonzero(np.diff(k_sorted)) + 1
    out = []
    for rows in np.split(order, bounds):
        rows = rows[keep[rows]]
        if len(rows) >= 10:
            out.append(rows)
    return out


def loss_table(truth, bins, width=100.0):
    """log P(bin | rating) on the grid, from moves weighted by how close their player's rating is."""
    tab = np.zeros((len(R), len(LOSS_BINS)))
    for j, r in enumerate(R):
        w = np.exp(-0.5 * ((truth - r) / width) ** 2)
        tab[j] = np.bincount(bins, weights=w, minlength=len(LOSS_BINS)) + 1.0
    tab /= tab.sum(axis=1, keepdims=True)
    return np.log(tab)


def evaluate(llp, lll, truth, a, b, tau):
    est, inside = [], []
    for p, l, t in zip(llp, lll, truth):
        ll = a * p + b * l
        s = summarize(posterior(ll, 1.0))
        rng = summarize(posterior(ll, tau))
        est.append(s["rating"])
        inside.append(rng["low"] <= t <= rng["high"])
    err = np.array(est) - truth
    return err, np.array(inside), np.array(est)


def report(name, err, inside, truth):
    ae = np.abs(err)
    print(f"{name}: n={len(err):,}  MAE {ae.mean():.0f}  median {np.median(ae):.0f}  "
          f"within 150: {np.mean(ae <= 150):.0%}  within 250: {np.mean(ae <= 250):.0%}  "
          f"80% range holds truth: {np.mean(inside):.0%}")
    bands = np.clip(truth // 200 * 200, 600, 2600)
    print("   bias by band: " + "  ".join(f"{int(b)}:{err[bands == b].mean():+.0f}" for b in np.unique(bands)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "heldout_features.npz"))
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--skip", type=int, default=0, help="ignore the first N plies (opening moves)")
    ap.add_argument("--group", type=int, default=5, help="games per multi-game estimate")
    args = ap.parse_args()

    d = np.load(args.data)
    logp, meta, loss = d["logp"].astype(np.float64), d["meta"], d["loss"]
    assert list(d["ratings"]) == RATINGS
    logp = np.maximum(logp, np.log(1e-4))
    bins = loss_bin(loss)
    truth_move = np.clip(meta[:, 3], R[0], R[-1]).astype(np.float64)

    us = units(meta, args.skip)
    fit = [u for u in us if meta[u[0], 0] % 2 == 0]
    test = [u for u in us if meta[u[0], 0] % 2 == 1]
    fit_rows = np.concatenate(fit)
    table = loss_table(truth_move[fit_rows], bins[fit_rows])

    def feats(us_):
        llp = [logp[u].sum(axis=0) for u in us_]
        lll = [table[:, bins[u]].sum(axis=1) for u in us_]
        truth = np.array([truth_move[u[0]] for u in us_])
        return llp, lll, truth

    fp, fl, ft = feats(fit)
    best = None
    for a, b in itertools.product([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5], [0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]):
        err, _, _ = evaluate(fp, fl, ft, a, b, 1.0)
        mae = np.abs(err).mean()
        if best is None or mae < best[0]:
            best = (mae, a, b)
    _, a, b = best
    print(f"weights: policy a={a}, loss b={b}  (fit MAE {best[0]:.0f})")
    tau = 1.0
    for t in [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]:
        _, inside, _ = evaluate(fp, fl, ft, a, b, t)
        tau = t
        if inside.mean() >= 0.80:
            break
    print(f"width factor tau={tau}")

    tp, tl, tt = feats(test)
    err, inside, est = evaluate(tp, tl, tt, a, b, tau)
    report("test, single games", err, inside, tt)
    err_p, _, _ = evaluate(tp, tl, tt, a, 0.0, tau)
    print(f"   (policy evidence alone: MAE {np.abs(err_p).mean():.0f})")

    # Sets of `group` games from players of nearly the same rating (consecutive by true rating).
    order = np.argsort(tt, kind="stable")
    gp, gl, gt = [], [], []
    for s in range(0, len(order) - args.group + 1, args.group):
        idx = order[s:s + args.group]
        gp.append(sum(tp[i] for i in idx))
        gl.append(sum(tl[i] for i in idx))
        gt.append(tt[idx].mean())
    err_g, inside_g, _ = evaluate(gp, gl, np.array(gt), a, b, tau)
    report(f"test, {args.group} games", err_g, inside_g, np.array(gt))

    model = {"a": a, "b": b, "tau": tau, "skip_plies": args.skip, "loss_bins": LOSS_BINS,
             "ratings": RATINGS, "loss_logp": np.round(table, 5).tolist(),
             "test": {"single_mae": round(float(np.abs(err).mean())),
                      "single_within_150": round(float(np.mean(np.abs(err) <= 150)), 3),
                      "group": args.group, "group_mae": round(float(np.abs(err_g).mean())),
                      "group_within_150": round(float(np.mean(np.abs(err_g) <= 150)), 3)}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
