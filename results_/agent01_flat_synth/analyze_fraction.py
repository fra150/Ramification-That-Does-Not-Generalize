"""Analyze the level1_fraction sweep for W1-13.

Reads results/redteam/level_frac_*_seeds*.csv, produced by
experiments.level_assignment, and reports paired statistics by fraction.

Stdlib only; no project source is modified.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics

T95 = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}


def describe(deltas):
    n = len(deltas)
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    crit = T95.get(n, 1.96)
    return {
        "n": n, "mean": mean, "std": std, "se": se, "t": t,
        "ci_low": mean - crit * se, "ci_high": mean + crit * se,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def fmt(s):
    return (f"mean={s['mean']:+.4f} sd={s['std']:.4f} t={s['t']:+.2f} "
            f"CI95=[{s['ci_low']:+.4f},{s['ci_high']:+.4f}] "
            f"W/L/T={s['wins']}/{s['losses']}/{s['ties']}")


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"level_(?:frac|fine)_(\d{3})_seeds", os.path.basename(path))
        if not m:
            continue
        fraction = int(m.group(1)) / 100.0
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["fraction"] = fraction
                r["seed"] = int(r["seed"])
                r["vocab"] = int(r["vocab"])
                for k in ("boundary_f1", "boundary_macro_precision",
                          "boundary_macro_recall", "boundary_micro_f1",
                          "bits_per_char", "tokens_per_char", "decode_word",
                          "decode_doc"):
                    r[k] = float(r[k])
                r["n_l1"] = int(r["n_l1"])
                for k in ("l1_intra", "l1_cross", "l2_intra", "l2_cross"):
                    r[k] = int(r[k]) if r[k] not in ("", None) else 0
                rows.append(r)
    return rows


def idx(rows):
    return {(r["fraction"], r["seed"], r["mode"]): r for r in rows}


def paired(ix, frac, a, b, metric, seeds):
    out = []
    for seed in seeds:
        ra, rb = ix.get((frac, seed, a)), ix.get((frac, seed, b))
        if ra is None or rb is None:
            continue
        out.append(ra[metric] - rb[metric])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/redteam/level_frac_*_seeds42-51.csv")
    ap.add_argument("--reference", default="results/level_assignment_v512_10seeds.csv")
    ap.add_argument("--out", default="results/redteam/fraction_summary.txt")
    args = ap.parse_args()

    rows = load(args.glob)
    if not rows:
        raise SystemExit(f"no rows matched {args.glob}")
    ix = idx(rows)
    fractions = sorted({r["fraction"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"fractions={fractions} seeds={seeds} modes={modes} rows={len(rows)}")

    # 0. frequency must be invariant to the requested fraction (prefix split).
    emit("\n== frequency invariance across fractions ==")
    for metric in ("boundary_f1", "bits_per_char", "tokens_per_char"):
        worst = 0.0
        for seed in seeds:
            vals = [ix[(f, seed, "frequency")][metric] for f in fractions
                    if (f, seed, "frequency") in ix]
            worst = max(worst, max(vals) - min(vals))
        emit(f"{metric}: max within-seed spread across fractions = {worst:.2e}")

    # 1. level sizes actually realized.
    emit("\n== realized level-1 size (requested vs actual, mean over seeds) ==")
    for f in fractions:
        cells = []
        for mode in ("frequency", "morph", "random_closed"):
            ns = [ix[(f, s, mode)]["n_l1"] for s in seeds if (f, s, mode) in ix]
            if ns:
                cells.append(f"{mode}={statistics.mean(ns):.1f} [{min(ns)}-{max(ns)}]")
        emit(f"fraction={f:.2f}  " + "  ".join(cells))

    # 1b. composition of the realized morph levels (the mechanism).
    emit("\n== morph level composition (mean over seeds) ==")
    emit(f"{'frac':>4} {'n_l1':>7} {'L1intra':>8} {'L1cross':>8} "
         f"{'L2intra':>8} {'intra_cov':>9} {'L1cross_share':>13}")
    for f in fractions:
        sub = [r for r in rows if r["fraction"] == f and r["mode"] == "morph"]
        if not sub:
            continue
        l1i = statistics.mean(r["l1_intra"] for r in sub)
        l1c = statistics.mean(r["l1_cross"] for r in sub)
        l2i = statistics.mean(r["l2_intra"] for r in sub)
        total_i = l1i + l2i
        emit(f"{f:>4.2f} {statistics.mean(r['n_l1'] for r in sub):>7.1f} "
             f"{l1i:>8.1f} {l1c:>8.1f} {l2i:>8.1f} "
             f"{(l1i / total_i if total_i else 0):>9.3f} "
             f"{(l1c / (l1i + l1c) if l1i + l1c else 0):>13.3f}")

    # 2. raw means.
    emit("\n== raw means by fraction ==")
    emit(f"{'frac':>4} {'mode':<14} {'F1':>8} {'bpc':>8} {'tok/ch':>8} {'n_l1':>6}")
    for f in fractions:
        for mode in modes:
            sub = [r for r in rows if r["fraction"] == f and r["mode"] == mode]
            emit(f"{f:>4.2f} {mode:<14} "
                 f"{statistics.mean([r['boundary_f1'] for r in sub]):>8.4f} "
                 f"{statistics.mean([r['bits_per_char'] for r in sub]):>8.4f} "
                 f"{statistics.mean([r['tokens_per_char'] for r in sub]):>8.4f} "
                 f"{statistics.mean([r['n_l1'] for r in sub]):>6.1f}")

    # 3. paired contrasts by fraction.
    pairs = [("morph", "frequency"), ("random_closed", "frequency"),
             ("morph", "random_closed")]
    emit("\n== paired contrasts by fraction ==")
    for metric in ("boundary_f1", "bits_per_char"):
        emit(f"\n-- {metric} --")
        for f in fractions:
            for a, b in pairs:
                d = paired(ix, f, a, b, metric, seeds)
                if d:
                    emit(f"frac={f:.2f} {a:<13}-{b:<10} {fmt(describe(d))}")

    # 3b. precision / recall decomposition and decode sanity.
    emit("\n== precision / recall decomposition (morph - frequency) ==")
    for metric in ("boundary_macro_precision", "boundary_macro_recall"):
        emit(f"-- {metric} --")
        for f in fractions:
            d = paired(ix, f, "morph", "frequency", metric, seeds)
            if d:
                emit(f"frac={f:.2f} {fmt(describe(d))}")
    emit("\n== micro-F1 morphology contrast (morph - frequency) ==")
    for f in fractions:
        d = paired(ix, f, "morph", "frequency", "boundary_micro_f1", seeds)
        if d:
            emit(f"frac={f:.2f} {fmt(describe(d))}")
    emit("\n== decode checks (should be 1.0 everywhere) ==")
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        emit(f"{mode}: min decode_word={min(r['decode_word'] for r in sub):.3f} "
             f"min decode_doc={min(r['decode_doc'] for r in sub):.3f}")

    # 4. does the morph advantage depend on fraction?
    emit("\n== fraction dependence of the morph-frequency advantage ==")
    for metric in ("boundary_f1", "bits_per_char"):
        emit(f"\n-- {metric}: per-seed delta by fraction --")
        deltas = {}
        for f in fractions:
            deltas[f] = paired(ix, f, "morph", "frequency", metric, seeds)
            emit(f"frac={f:.2f} " + " ".join(f"{x:+.4f}" for x in deltas[f]))
        emit("pairwise difference of deltas (adjacent fractions and vs 0.6):")
        for i, fa in enumerate(fractions):
            for fb in fractions[i + 1:]:
                if not (fb == fractions[i + 1] or fa == 0.6 or fb == 0.6):
                    continue
                if len(deltas[fa]) == len(deltas[fb]):
                    d = [x - y for x, y in zip(deltas[fa], deltas[fb])]
                    emit(f"  {fa:.2f} vs {fb:.2f}: {fmt(describe(d))}")

    # 5. best fraction, in-sample and leave-one-seed-out (LOSO).
    emit("\n== fraction selection: in-sample vs LOSO ==")
    for metric, sign in (("boundary_f1", 1.0), ("bits_per_char", -1.0)):
        emit(f"\n-- {metric} (higher score = better) --")
        perf = {(f, s): ix[(f, s, "morph")][metric] - ix[(f, s, "frequency")][metric]
                for f in fractions for s in seeds if (f, s, "morph") in ix}
        means = {f: statistics.mean([perf[(f, s)] for s in seeds]) for f in fractions}
        best = max(means, key=lambda f: sign * means[f])
        emit("in-sample mean delta by fraction: " +
             " ".join(f"{f:.2f}:{sign * means[f]:+.4f}" for f in fractions))
        emit(f"in-sample best fraction = {best:.2f} (delta {means[best]:+.4f})")
        loso_scores, picks = [], []
        for held in seeds:
            train = [s for s in seeds if s != held]
            mu = {f: statistics.mean([perf[(f, s)] for s in train]) for f in fractions}
            pick = max(mu, key=lambda f: sign * mu[f])
            picks.append(pick)
            loso_scores.append(sign * perf[(pick, held)])
        fixed = [sign * perf[(0.6, s)] for s in seeds if (0.6, s) in perf]
        emit("LOSO picks: " + " ".join(f"{p:.2f}" for p in picks))
        emit(f"LOSO-selected held-out score mean={statistics.mean(loso_scores):+.4f}; "
             f"fixed 0.6 mean={statistics.mean(fixed):+.4f}; "
             f"optimism={statistics.mean([sign * means[best]] * len(seeds)) - statistics.mean(loso_scores):+.4f}")

    # 6. reproduction of the published v512 run at fraction 0.6.
    if args.reference and os.path.exists(args.reference):
        emit("\n== reproduction vs published fraction=0.6 CSV ==")
        with open(args.reference, encoding="utf-8") as fh:
            ref = {(int(r["seed"]), r["mode"]): r for r in csv.DictReader(fh)}
        worst = {}
        for (f, s, m), r in ix.items():
            if abs(f - 0.6) > 1e-9:
                continue
            rr = ref.get((s, m))
            if rr is None:
                continue
            for k in ("n_l1", "boundary_f1", "bits_per_char", "tokens_per_char"):
                d = abs(float(r[k]) - float(rr[k]))
                worst[k] = max(worst.get(k, 0.0), d)
        emit("max |new - published| at fraction 0.6: " +
             " ".join(f"{k}={v:.3e}" for k, v in worst.items()))

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
