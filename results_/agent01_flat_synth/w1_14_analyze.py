"""W1-14 analysis: is morph outside the random_closed distribution?

Reads results/redteam/w1_14_random_closed_draws.csv (long format) and writes
per-seed summaries plus pooled tests. No source changes.

Usage:
    python -m results.redteam.w1_14_analyze
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

from experiments.analyze_levels import T95

METRICS = ("boundary_f1", "bits_per_char")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    return float(row[key])


def mean(xs):
    return statistics.mean(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def ci95(xs):
    n = len(xs)
    m = mean(xs)
    s = sd(xs)
    se = s / math.sqrt(n)
    crit = T95.get(n, 1.96)
    return m, s, se, m - crit * se, m + crit * se


def tstat(xs, null=0.0):
    n = len(xs)
    s = sd(xs)
    if n < 2 or s == 0:
        return float("inf") if mean(xs) != null else 0.0
    return (mean(xs) - null) / (s / math.sqrt(n))


def fisher_combined(pvals):
    """Fisher's method; p-values must be in (0,1]. df = 2k (even)."""
    x = -2.0 * sum(math.log(max(p, 1e-300)) for p in pvals)
    k = len(pvals)
    # Survival function of chi^2 with df=2k: exp(-x/2) sum_{j=0}^{k-1} (x/2)^j/j!
    half = x / 2.0
    term = 1.0
    total = 1.0
    for j in range(1, k):
        term *= half / j
        total += term
    return math.exp(-half) * total


def worse_than_morph(draw_value, morph_value, metric):
    """True if the random draw is worse than morph."""
    return draw_value < morph_value if metric == "boundary_f1" else draw_value > morph_value


def better_than_morph(draw_value, morph_value, metric):
    return draw_value > morph_value if metric == "boundary_f1" else draw_value < morph_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", default="results/redteam/w1_14_random_closed_draws.csv"
    )
    ap.add_argument(
        "--published", default="results/level_assignment_v512_10seeds.csv"
    )
    ap.add_argument("--out-prefix", default="results/redteam/w1_14")
    args = ap.parse_args()

    rows = load(args.csv)
    seeds = sorted({int(r["seed"]) for r in rows})
    vocab = int(rows[0]["vocab"])

    by_seed = defaultdict(dict)
    draws = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        seed = int(r["seed"])
        tag = r["tag"]
        if tag == "flat":
            by_seed[seed]["flat"] = r
        elif tag == "morph":
            by_seed[seed]["morph"] = r
        elif tag == "random_closed_canonical":
            by_seed[seed]["canonical"] = r
        elif tag.startswith("random_closed_d"):
            draws[seed][int(r["draw"])] = r

    # Cross-check canonical draw against the published single-draw control.
    pub = {}
    try:
        for r in load(args.published):
            pub[(int(r["seed"]), int(r["vocab"]), r["mode"])] = r
    except FileNotFoundError:
        pub = {}

    per_seed = []
    for seed in seeds:
        d = by_seed[seed]
        draw_rows = [draws[seed][k] for k in sorted(draws[seed])]
        assert len(draw_rows) == 10, (seed, len(draw_rows))
        rec = {
            "seed": seed,
            "vocab": vocab,
            "n_draws": len(draw_rows),
            "n_l1_morph": int(d["morph"]["n_l1"]),
            "n_l1_draw_min": min(int(r["n_l1"]) for r in draw_rows),
            "n_l1_draw_max": max(int(r["n_l1"]) for r in draw_rows),
            "n_distinct_l1_sets": len({r["level1_fingerprint"] for r in draw_rows}),
        }

        # pairwise Jaccard among the 10 level-1 sets, from recorded jaccard to
        # morph is not enough; compute from fingerprints only -> use n_l1 and
        # published J(morph) instead. Pairwise overlap needs the id lists; the
        # runner stores only the fingerprint, so report J to morph/frequency.
        rec["jaccard_morph_mean"] = mean([f(r, "jaccard_morph") for r in draw_rows])
        rec["jaccard_freq_mean"] = mean([f(r, "jaccard_frequency") for r in draw_rows])
        rec["l1_intra_morph"] = int(d["morph"]["l1_intra"])
        rec["l1_intra_draw_mean"] = mean([int(r["l1_intra"]) for r in draw_rows])
        rec["l1_intra_draw_min"] = min(int(r["l1_intra"]) for r in draw_rows)
        rec["l1_intra_draw_max"] = max(int(r["l1_intra"]) for r in draw_rows)
        rec["l1_intra_canonical"] = int(d["canonical"]["l1_intra"])
        rec["jaccard_morph_canonical"] = f(d["canonical"], "jaccard_morph")
        rec["n_draws_size_294"] = sum(int(r["n_l1"]) == 294 for r in draw_rows)

        for metric in METRICS:
            vals = [f(r, metric) for r in draw_rows]
            m = f(d["morph"], metric)
            flat = f(d["flat"], metric)
            canonical = f(d["canonical"], metric)
            rec[f"flat_{metric}"] = flat
            rec[f"morph_{metric}"] = m
            rec[f"mean_{metric}"] = mean(vals)
            rec[f"median_{metric}"] = statistics.median(vals)
            q = statistics.quantiles(vals, n=4)
            rec[f"iqr_{metric}"] = q[2] - q[0]
            rec[f"sd_{metric}"] = sd(vals)
            rec[f"min_{metric}"] = min(vals)
            rec[f"max_{metric}"] = max(vals)
            rec[f"range_{metric}"] = max(vals) - min(vals)
            rec[f"delta_{metric}"] = m - mean(vals)
            rec[f"delta_median_{metric}"] = m - statistics.median(vals)
            rec[f"delta_vs_flat_{metric}"] = m - flat
            rec[f"canonical_{metric}"] = canonical
            rec[f"canonical_delta_{metric}"] = m - canonical
            rec[f"canonical_z_{metric}"] = (
                (canonical - mean(vals)) / sd(vals) if sd(vals) > 0 else 0.0
            )
            rec[f"canonical_pct_{metric}"] = (
                sum(worse_than_morph(v, canonical, metric) for v in vals) / len(vals)
            )
            z = rec[f"delta_{metric}"] / rec[f"sd_{metric}"] if rec[f"sd_{metric}"] > 0 else 0.0
            # Positive z always means "morph better in SD units".
            rec[f"z_improvement_{metric}"] = z if metric == "boundary_f1" else -z
            wins = sum(worse_than_morph(v, m, metric) for v in vals)
            ties = sum(v == m for v in vals)
            losses = len(vals) - wins - ties
            rec[f"draw_wins_{metric}"] = wins
            rec[f"draw_ties_{metric}"] = ties
            rec[f"draw_losses_{metric}"] = losses
            rec[f"rank_{metric}"] = losses + 1  # 1 = morph best of 11
            rec[f"winrate_{metric}"] = (wins + 0.5 * ties) / len(vals)
            # One-sided Monte-Carlo p: probability a random draw is at least
            # as good as morph, with add-one smoothing.
            at_least = sum(better_than_morph(v, m, metric) or v == m for v in vals)
            p = (1 + at_least) / (1 + len(vals))
            rec[f"p_random_better_{metric}"] = p
            rec[f"morph_beyond_max_{metric}"] = 1 if all(
                worse_than_morph(v, m, metric) for v in vals
            ) else 0
            rec[f"margin_beyond_best_{metric}"] = (
                m - max(vals) if metric == "boundary_f1" else min(vals) - m
            )
            rec[f"margin_over_sd_{metric}"] = (
                rec[f"margin_beyond_best_{metric}"] / sd(vals) if sd(vals) > 0 else 0.0
            )

        # Published check (flat/morph/canonical should match to CSV rounding).
        for mode in ("flat", "morph"):
            key = (seed, vocab, mode)
            if key in pub:
                rec[f"published_{mode}_f1_diff"] = (
                    f(d[mode], "boundary_f1") - f(pub[key], "boundary_f1")
                )
                rec[f"published_{mode}_bpc_diff"] = (
                    f(d[mode], "bits_per_char") - f(pub[key], "bits_per_char")
                )
        if (seed, vocab, "random_closed") in pub:
            p = pub[(seed, vocab, "random_closed")]
            rec["published_canonical_f1_diff"] = (
                f(d["canonical"], "boundary_f1") - f(p, "boundary_f1")
            )
            rec["published_canonical_bpc_diff"] = (
                f(d["canonical"], "bits_per_char") - f(p, "bits_per_char")
            )
        per_seed.append(rec)

    # ---- per-seed CSV ----
    out_prefix = args.out_prefix
    per_path = f"{out_prefix}_per_seed.csv"
    with open(per_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_seed[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed)
    print(f"wrote {per_path}")

    # ---- print per-seed table ----
    print("\n=== per-seed: morph vs 10 random_closed draws (v512) ===")
    print(
        "seed  morphF1  randF1(mean+-sd)      [min,max]      zF1  "
        "wins  morphBPC randBPC(mean+-sd)     [min,max]     zBPC  wins"
    )
    for r in per_seed:
        print(
            f"{r['seed']}   {r['morph_boundary_f1']:.4f}  "
            f"{r['mean_boundary_f1']:.4f}+-{r['sd_boundary_f1']:.4f}  "
            f"[{r['min_boundary_f1']:.4f},{r['max_boundary_f1']:.4f}]  "
            f"{r['z_improvement_boundary_f1']:+.2f}  "
            f"{r['draw_wins_boundary_f1']:2d}/10  "
            f"{r['morph_bits_per_char']:.4f}  "
            f"{r['mean_bits_per_char']:.4f}+-{r['sd_bits_per_char']:.4f}  "
            f"[{r['min_bits_per_char']:.4f},{r['max_bits_per_char']:.4f}]  "
            f"{r['z_improvement_bits_per_char']:+.2f}  "
            f"{r['draw_wins_bits_per_char']:2d}/10"
        )

    # ---- pooled ----
    print("\n=== pooled paired tests across seeds (morph vs per-seed random mean) ===")
    pooled = {}
    for metric in METRICS:
        deltas = [r[f"delta_{metric}"] for r in per_seed]
        m, s, se, lo, hi = ci95(deltas)
        t = tstat(deltas)
        w = sum(x > 0 for x in deltas) if metric == "boundary_f1" else sum(x < 0 for x in deltas)
        l = sum(x < 0 for x in deltas) if metric == "boundary_f1" else sum(x > 0 for x in deltas)
        wr = [r[f"winrate_{metric}"] for r in per_seed]
        pvals = [r[f"p_random_better_{metric}"] for r in per_seed]
        fisher_p = fisher_combined(pvals)
        pooled[metric] = {
            "mean_delta": m,
            "sd_delta": s,
            "se": se,
            "ci_low": lo,
            "ci_high": hi,
            "t": t,
            "wins": w,
            "losses": l,
            "ties": len(deltas) - w - l,
            "winrate_mean": mean(wr),
            "winrate_t": tstat(wr, 0.5),
            "fisher_p": fisher_p,
            "n_beyond_best": sum(r[f"morph_beyond_max_{metric}"] for r in per_seed),
        }
        print(
            f"{metric}: delta(morph-randommean) mean={m:+.5f} "
            f"sd={s:.5f} t={t:+.2f} CI95=[{lo:+.5f},{hi:+.5f}] "
            f"W/L/T={w}/{l}/{len(deltas)-w-l}"
        )
        total_better = sum(r[f"draw_losses_{metric}"] for r in per_seed)
        total_ties = sum(r[f"draw_ties_{metric}"] for r in per_seed)
        ranks = [r[f"rank_{metric}"] for r in per_seed]
        print(
            f"    of 100 random draws, {total_better} beat morph and "
            f"{total_ties} tie; morph ranks {mean(ranks):.2f}/11 on average "
            f"(best {min(ranks)}, worst {max(ranks)})"
        )
        print(
            f"    winrate vs individual draws={mean(wr):.3f} "
            f"t(vs 0.5)={tstat(wr, 0.5):+.2f}  Fisher p={fisher_p:.4g}  "
            f"morph beyond best draw in {pooled[metric]['n_beyond_best']}/{len(seeds)} seeds"
        )
        # Median-draw sensitivity (robust to one extreme bad draw).
        dmed = [r[f"delta_median_{metric}"] for r in per_seed]
        mmed, smed, semed, lomed, himed = ci95(dmed)
        wmed = sum(x > 0 for x in dmed) if metric == "boundary_f1" else sum(x < 0 for x in dmed)
        print(
            f"    delta vs median random draw: mean={mmed:+.5f} "
            f"t={tstat(dmed):+.2f} CI95=[{lomed:+.5f},{himed:+.5f}] "
            f"W/L/T={wmed}/{len(dmed)-wmed}/0"
        )
        k = pooled[metric]["n_beyond_best"]
        n = len(seeds)
        # Under exchangeability, P(morph is best of 11) = 1/11 per seed.
        p_binom = sum(
            math.comb(n, j) * (1 / 11) ** j * (10 / 11) ** (n - j)
            for j in range(k, n + 1)
        )
        print(
            f"    exact binomial P(>= {k} of {n} seeds beyond max | p=1/11)"
            f"={p_binom:.4g}"
        )

    # ---- spread decomposition ----
    print("\n=== random_closed spread decomposition ===")
    for metric in METRICS:
        within = [r[f"sd_{metric}"] for r in per_seed]
        pooled_within = math.sqrt(mean([v * v for v in within]))
        means = [r[f"mean_{metric}"] for r in per_seed]
        morphs = [r[f"morph_{metric}"] for r in per_seed]
        all_draws = [
            f(r, metric)
            for seed in seeds
            for r in (draws[seed][k] for k in sorted(draws[seed]))
        ]
        d = pooled[metric]["mean_delta"]
        spread = pooled_within
        zs = [r[f"z_improvement_{metric}"] for r in per_seed]
        print(
            f"{metric}: within-seed SD(pooled)={spread:.5f}  "
            f"between-seed SD(random means)={sd(means):.5f}  "
            f"total SD(all 100 draws)={sd(all_draws):.5f}"
        )
        print(
            f"    mean standardized shift z=(morph-randommean)/random-SD: "
            f"mean={mean(zs):+.2f} t={tstat(zs):+.2f} "
            f"(per-seed range [{min(zs):+.2f},{max(zs):+.2f}])"
        )
        print(
            f"    |mean delta|/within-SD={abs(d)/spread:.2f}  "
            f"|mean delta|/total-SD={abs(d)/sd(all_draws):.2f}  "
            f"per-seed |z| min/median/max="
            f"{min(abs(r[f'z_improvement_{metric}']) for r in per_seed):.2f}/"
            f"{statistics.median(abs(r[f'z_improvement_{metric}']) for r in per_seed):.2f}/"
            f"{max(abs(r[f'z_improvement_{metric}']) for r in per_seed):.2f}"
        )
        # How much does morph move relative to the random range per seed?
        frac = [
            r[f"margin_beyond_best_{metric}"] / r[f"range_{metric}"]
            if r[f"range_{metric}"] > 0
            else 0.0
            for r in per_seed
        ]
        print(
            f"    margin beyond best draw / random range: "
            f"mean={mean(frac):+.2f} min={min(frac):+.2f} max={max(frac):+.2f}"
        )
        # Average per-seed percentile of morph within its own 10 draws
        # (oriented so higher=better).
        pcts = []
        for r in per_seed:
            seed = r["seed"]
            vals = [f(draws[seed][k], metric) for k in sorted(draws[seed])]
            m = r[f"morph_{metric}"]
            if metric == "boundary_f1":
                p = sum(1 for v in vals if v < m) / len(vals)
            else:
                p = sum(1 for v in vals if v > m) / len(vals)
            pcts.append(p)
        print(
            f"    mean per-seed percentile of morph among its 10 draws="
            f"{100*mean(pcts):.0f}th (higher=better)"
        )

    # ---- size-mismatch sensitivity ----
    print("\n=== size-294 sensitivity (random_closed can emit 292-294 L1 rules) ===")
    target_n1 = per_seed[0]["n_l1_morph"]
    for metric in METRICS:
        deltas = []
        skipped = []
        for seed in seeds:
            d = by_seed[seed]
            vals = [
                f(draws[seed][k], metric)
                for k in sorted(draws[seed])
                if int(draws[seed][k]["n_l1"]) == target_n1
            ]
            if len(vals) < 2:
                skipped.append((seed, len(vals)))
                continue
            deltas.append(f(d["morph"], metric) - mean(vals))
        m, s, se, lo, hi = ci95(deltas)
        print(
            f"{metric}: delta vs size-294 draws only mean={m:+.5f} "
            f"t={tstat(deltas):+.2f} CI95=[{lo:+.5f},{hi:+.5f}] "
            f"n_seeds={len(deltas)} skipped={skipped}"
        )
        print(
            f"    all-draw delta mean={pooled[metric]['mean_delta']:+.5f}; "
            f"draws with n_l1={target_n1}: "
            f"{sum(r['n_draws_size_294'] for r in per_seed)}/100"
        )

    # ---- level-1 size regression ----
    print("\n=== level-1 size vs metric (within-seed, across 10 draws) ===")
    for metric in METRICS:
        slope_num = 0.0
        slope_den = 0.0
        ss_tot = 0.0
        ss_res = 0.0
        per_seed_slope = []
        for seed in seeds:
            d = by_seed[seed]
            xs = [int(draws[seed][k]["n_l1"]) for k in sorted(draws[seed])]
            ys = [f(draws[seed][k], metric) for k in sorted(draws[seed])]
            xb, yb = mean(xs), mean(ys)
            sxx = sum((x - xb) ** 2 for x in xs)
            sxy = sum((x - xb) * (y - yb) for x, y in zip(xs, ys))
            if sxx > 0:
                slope = sxy / sxx
                per_seed_slope.append(slope)
                slope_num += sxy
                slope_den += sxx
                ss_tot += sum((y - yb) ** 2 for y in ys)
                ss_res += sum((y - yb - slope * (x - xb)) ** 2 for x, y in zip(xs, ys))
        if not slope_den:
            print(f"{metric}: no within-seed size variation")
            continue
        slope = slope_num / slope_den
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        print(
            f"{metric}: pooled within-seed slope per +1 L1 rule="
            f"{slope:+.6f}  R2={r2:.3f}  "
            f"per-seed slope median={statistics.median(per_seed_slope):+.6f} "
            f"range=[{min(per_seed_slope):+.6f},{max(per_seed_slope):+.6f}]"
        )
        residuals = []
        for seed in seeds:
            d = by_seed[seed]
            xs = [int(draws[seed][k]["n_l1"]) for k in sorted(draws[seed])]
            ys = [f(draws[seed][k], metric) for k in sorted(draws[seed])]
            xb, yb = mean(xs), mean(ys)
            predicted = yb + slope * (target_n1 - xb)
            residuals.append(f(d["morph"], metric) - predicted)
        print(
            f"    morph residual at n_l1={target_n1} vs size prediction "
            f"mean={mean(residuals):+.5f} t={tstat(residuals):+.2f} "
            f"(F1: positive=better than size predicts; "
            f"bpc: negative=better than size predicts)"
        )

    # ---- level-1 intra purity: position of morph and metric slope ----
    print("\n=== level-1 intra purity vs metric (within-seed, across 10 draws) ===")
    for metric in METRICS:
        slope_num = 0.0
        slope_den = 0.0
        ss_tot = 0.0
        ss_res = 0.0
        per_seed_slope = []
        for seed in seeds:
            d = by_seed[seed]
            xs = [int(draws[seed][k]["l1_intra"]) for k in sorted(draws[seed])]
            ys = [f(draws[seed][k], metric) for k in sorted(draws[seed])]
            xb, yb = mean(xs), mean(ys)
            sxx = sum((x - xb) ** 2 for x in xs)
            sxy = sum((x - xb) * (y - yb) for x, y in zip(xs, ys))
            if sxx > 0:
                slope = sxy / sxx
                per_seed_slope.append(slope)
                slope_num += sxy
                slope_den += sxx
                ss_tot += sum((y - yb) ** 2 for y in ys)
                ss_res += sum((y - yb - slope * (x - xb)) ** 2 for x, y in zip(xs, ys))
        slope = slope_num / slope_den if slope_den else 0.0
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        print(
            f"{metric}: pooled within-seed slope per +1 intra L1 rule="
            f"{slope:+.6f}  R2={r2:.3f}  "
            f"per-seed slope median={statistics.median(per_seed_slope):+.6f} "
            f"range=[{min(per_seed_slope):+.6f},{max(per_seed_slope):+.6f}]"
        )
        higher = sum(1 for r in per_seed if r["l1_intra_morph"] > r["l1_intra_draw_max"])
        morph_intra = [r["l1_intra_morph"] for r in per_seed]
        print(
            f"    morph l1_intra mean={mean(morph_intra):.1f} "
            f"range=[{min(morph_intra)},{max(morph_intra)}] "
            f"vs draw range "
            f"[{min(r['l1_intra_draw_min'] for r in per_seed)},"
            f"{max(r['l1_intra_draw_max'] for r in per_seed)}]; "
            f"morph above every draw in {higher}/{len(seeds)} seeds"
        )
        residuals = []
        for seed in seeds:
            d = by_seed[seed]
            xs = [int(draws[seed][k]["l1_intra"]) for k in sorted(draws[seed])]
            ys = [f(draws[seed][k], metric) for k in sorted(draws[seed])]
            xb, yb = mean(xs), mean(ys)
            rec = next(r for r in per_seed if r["seed"] == seed)
            predicted = yb + slope * (rec["l1_intra_morph"] - xb)
            residuals.append(f(d["morph"], metric) - predicted)
        print(
            f"    morph residual vs purity prediction mean={mean(residuals):+.5f} "
            f"t={tstat(residuals):+.2f} "
            f"(F1: positive=above purity prediction; "
            f"bpc: negative=below purity prediction)"
        )

    # ---- canonical draw bias ----
    print("\n=== published single random_closed draw vs 10-draw distribution ===")
    for metric in METRICS:
        zs = [r[f"canonical_z_{metric}"] for r in per_seed]
        deltas_pub = [r[f"canonical_delta_{metric}"] for r in per_seed]
        deltas_rand = [r[f"delta_{metric}"] for r in per_seed]
        diffs = [a - b for a, b in zip(deltas_pub, deltas_rand)]
        pcts = [r[f"canonical_pct_{metric}"] for r in per_seed]
        if metric == "boundary_f1":
            best = sum(1 for r in per_seed if r[f"canonical_{metric}"] >= r[f"max_{metric}"])
            worst = sum(1 for r in per_seed if r[f"canonical_{metric}"] <= r[f"min_{metric}"])
        else:
            best = sum(1 for r in per_seed if r[f"canonical_{metric}"] <= r[f"min_{metric}"])
            worst = sum(1 for r in per_seed if r[f"canonical_{metric}"] >= r[f"max_{metric}"])
        print(
            f"{metric}: canonical z within draws mean={mean(zs):+.2f} "
            f"t={tstat(zs):+.2f} min={min(zs):+.2f} max={max(zs):+.2f} | "
            f"published delta mean={mean(deltas_pub):+.5f} "
            f"vs 10-draw-mean delta={mean(deltas_rand):+.5f} "
            f"(difference={mean(diffs):+.5f}, t={tstat(diffs):+.2f})"
        )
        print(
            f"    canonical percentile among its 10 draws: mean={100*mean(pcts):.0f}th "
            f"t(vs 50th)={tstat(pcts, 0.5):+.2f}; "
            f"best-of-10 in {best}/{len(zs)} seeds, "
            f"worst-of-10 in {worst}/{len(zs)} seeds; "
            f"|z|>1 in {sum(1 for z in zs if abs(z) > 1)}/{len(zs)} seeds, "
            f"|z|>2 in {sum(1 for z in zs if abs(z) > 2)}/{len(zs)}"
        )

    z_intra = []
    z_jacc = []
    for r in per_seed:
        seed = r["seed"]
        intras = [int(draws[seed][k]["l1_intra"]) for k in sorted(draws[seed])]
        jaccs = [f(draws[seed][k], "jaccard_morph") for k in sorted(draws[seed])]
        z_intra.append(
            (r["l1_intra_canonical"] - mean(intras)) / sd(intras) if sd(intras) else 0.0
        )
        z_jacc.append(
            (r["jaccard_morph_canonical"] - mean(jaccs)) / sd(jaccs)
            if sd(jaccs)
            else 0.0
        )
    print(
        f"canonical level-1 properties vs the 10 draws: "
        f"l1_intra z mean={mean(z_intra):+.2f} t={tstat(z_intra):+.2f}; "
        f"Jaccard-to-morph z mean={mean(z_jacc):+.2f} t={tstat(z_jacc):+.2f}"
    )

    # ---- n_l1 sensitivity ----
    print("\n=== size check ===")
    print(
        "n_l1 morph:",
        sorted({r["n_l1_morph"] for r in per_seed}),
        "| n_l1 draws range per seed:",
        [(r["seed"], r["n_l1_draw_min"], r["n_l1_draw_max"]) for r in per_seed],
    )
    print(
        "distinct level-1 sets among 10 draws:",
        [(r["seed"], r["n_distinct_l1_sets"]) for r in per_seed],
    )
    print(
        "mean Jaccard(draw, morph):",
        f"{mean([r['jaccard_morph_mean'] for r in per_seed]):.3f}",
        "| mean Jaccard(draw, frequency):",
        f"{mean([r['jaccard_freq_mean'] for r in per_seed]):.3f}",
    )

    print("\n=== reproduction check vs results/level_assignment_v512_10seeds.csv ===")
    for mode in ("flat", "morph", "canonical"):
        f1_key = f"published_{mode}_f1_diff"
        bpc_key = f"published_{mode}_bpc_diff"
        print(
            f"{mode}: max |F1 diff|="
            f"{max(abs(r[f1_key]) for r in per_seed if f1_key in r):.2e} "
            f"max |bpc diff|="
            f"{max(abs(r[bpc_key]) for r in per_seed if bpc_key in r):.2e}"
        )


if __name__ == "__main__":
    main()
