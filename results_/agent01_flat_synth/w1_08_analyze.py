"""W1-08 analysis: tie-break stability tables from w1_08_*_full.csv."""

from __future__ import annotations

import csv
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
PUBLISHED = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "results", "level_assignment_v512_10seeds.csv"
)


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt(x):
    return f"{x:+.6f}"


def tcrit(n):
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(n, 1.96)


def describe(vals):
    n = len(vals)
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    crit = tcrit(n)
    return {
        "n": n,
        "mean": m,
        "sd": sd,
        "se": se,
        "t": t,
        "lo": m - crit * se,
        "hi": m + crit * se,
        "pos": sum(1 for v in vals if v > 0),
        "neg": sum(1 for v in vals if v < 0),
        "zero": sum(1 for v in vals if v == 0),
    }


def main():
    runs = load(f"w1_08_runs_{TAG}.csv")
    pairwise = load(f"w1_08_pairwise_{TAG}.csv")
    ties = load(f"w1_08_ties_{TAG}.csv")
    rules = load(f"w1_08_rules_{TAG}.csv")
    if not runs:
        print("no full-run CSVs yet")
        return

    seeds = sorted({int(r["seed"]) for r in runs})
    preferred = ["id", "rev_id", "exp_short", "exp_long", "rand101", "rand202", "rand303"]
    present = {r["variant"] for r in runs}
    variants = [v for v in preferred if v in present] + sorted(present - set(preferred))
    out = []
    pr = out.append

    # ---- validation against the published 10-seed CSV ----
    pub = {}
    for r in load_published():
        if int(r["vocab"]) == 512 and int(r["seed"]) in seeds:
            pub[(int(r["seed"]), r["mode"])] = r
    pr("=== validation: id vs published level_assignment_v512_10seeds.csv ===")
    max_df1 = max_dbpc = 0.0
    for r in runs:
        if r["variant"] != "id":
            continue
        key = (int(r["seed"]), r["mode"])
        if key not in pub:
            continue
        p = pub[key]
        df1 = abs(float(r["boundary_f1"]) - float(p["boundary_f1"]))
        dbpc = abs(float(r["bits_per_char"]) - float(p["bits_per_char"]))
        max_df1 = max(max_df1, df1)
        max_dbpc = max(max_dbpc, dbpc)
        if df1 > 0 or dbpc > 0:
            pr(f"  seed={key[0]} mode={key[1]:<13} dF1={df1:.2e} dbpc={dbpc:.2e}")
    pr(f"  max |dF1| = {max_df1:.3e}, max |dbpc| = {max_dbpc:.3e}")
    pr("")

    # ---- sanity: frequency is the flat tokenization, decode is exact ----
    idx = {(int(r["seed"]), r["variant"], r["mode"]): r for r in runs}
    max_ff = 0.0
    for seed in seeds:
        for variant in variants:
            f = idx[(seed, variant, "flat")]
            q = idx[(seed, variant, "frequency")]
            max_ff = max(
                max_ff,
                abs(float(f["boundary_f1"]) - float(q["boundary_f1"])),
                abs(float(f["bits_per_char"]) - float(q["bits_per_char"])),
            )
    bad_decode = [
        r for r in runs
        if float(r["decode_word"]) != 1.0 or float(r["decode_doc"]) != 1.0
    ]
    pr("=== sanity checks ===")
    pr(f"  max |flat - frequency| over F1/bpc: {max_ff:.3e} (expected 0)")
    pr(f"  runs with decode_word or decode_doc != 1: {len(bad_decode)}")
    l1 = sorted({int(r["n_l1"]) for r in runs})
    l2 = sorted({int(r["n_l2"]) for r in runs})
    pr(f"  level sizes observed: n_l1={l1}, n_l2={l2}")
    pr("")

    # ---- per-variant paired deltas ----
    for metric in ("boundary_f1", "bits_per_char"):
        for comp in ("morph_minus_frequency", "morph_minus_random"):
            pr(f"=== {comp.replace('_', ' ')} on {metric} ===")
            pr(f"{'variant':<10} {'mean':>11} {'sd':>9} {'t':>8} {'CI95':>25} {'W/L/T':>8}")
            for variant in variants:
                vals = [
                    float(r[comp])
                    for r in pairwise
                    if r["variant"] == variant and r["metric"] == metric
                ]
                st = describe(vals)
                ci = f"[{st['lo']:+.6f},{st['hi']:+.6f}]"
                pr(
                    f"{variant:<10} {fmt(st['mean']):>11} {st['sd']:>9.6f} "
                    f"{st['t']:>+8.2f} {ci:>25} "
                    f"{st['pos']}/{st['neg']}/{st['zero']:>8}"
                )
            # grand mean: average across variants, then t-test across seeds
            per_seed = []
            for seed in seeds:
                vals = [
                    float(r[comp])
                    for r in pairwise
                    if int(r["seed"]) == seed and r["metric"] == metric
                ]
                per_seed.append(statistics.mean(vals))
            st = describe(per_seed)
            variant_means = []
            for variant in variants:
                variant_means.append(
                    statistics.mean(
                        float(r[comp])
                        for r in pairwise
                        if r["variant"] == variant and r["metric"] == metric
                    )
                )
            sd_variant_means = (
                statistics.stdev(variant_means) if len(variant_means) > 1 else 0.0
            )
            pr(
                f"{'ALL':<10} {fmt(st['mean']):>11} {st['sd']:>9.6f} "
                f"{st['t']:>+8.2f} "
                f"{'[' + format(st['lo'], '+.6f') + ',' + format(st['hi'], '+.6f') + ']':>25} "
                f"{st['pos']}/{st['neg']}/{st['zero']:>8}"
            )
            pr(
                f"  sd of the {len(variants)} variant means = {sd_variant_means:.6f}"
            )
            if sd_variant_means > 1e-9:
                pr(
                    f"  |grand mean| / sd = "
                    f"{(abs(st['mean']) / sd_variant_means):.2f}x"
                )
            else:
                pr("  effect is invariant across variants to < 1e-9")
            rand_variants = [v for v in variants if v.startswith("rand")]
            if len(rand_variants) > 1:
                rmeans = [
                    statistics.mean(
                        float(r[comp])
                        for r in pairwise
                        if r["variant"] == v and r["metric"] == metric
                    )
                    for v in rand_variants
                ]
                pr(
                    f"  random tie-break only ({len(rmeans)} variants): "
                    f"sd={statistics.stdev(rmeans):.6f}, "
                    f"range={max(rmeans) - min(rmeans):.6f}"
                )
            pr("")

    # ---- sign stability over all variant x seed cells ----
    pr("=== cell-level sign stability (variant x seed) ===")
    for metric in ("boundary_f1", "bits_per_char"):
        for comp in ("morph_minus_frequency", "morph_minus_random"):
            vals = [
                float(r[comp]) for r in pairwise if r["metric"] == metric
            ]
            n = len(vals)
            good = sum(1 for v in vals if (v > 0 if metric == "boundary_f1" else v < 0))
            bad = sum(1 for v in vals if (v < 0 if metric == "boundary_f1" else v > 0))
            zero = n - good - bad
            pr(f"  {comp:<22} {metric:<14} beneficial {good}/{n}, harmful {bad}, zero {zero}")
    pr("")

    # ---- absolute tie-break noise ----
    pr("=== absolute metric movement across variants, within seed ===")
    pr(f"{'seed':>5} {'morph F1 range':>18} {'freq F1 range':>18} "
       f"{'rand F1 range':>18} {'d(F1) morph-freq':>20}")
    for seed in seeds:
        def vals(mode, metric="boundary_f1"):
            return [
                float(r[metric])
                for r in runs
                if int(r["seed"]) == seed and r["mode"] == mode
            ]
        m, f, rc = vals("morph"), vals("frequency"), vals("random_closed")
        d = [
            float(r["morph_minus_frequency"])
            for r in pairwise
            if int(r["seed"]) == seed and r["metric"] == "boundary_f1"
        ]
        pr(
            f"{seed:>5} {max(m) - min(m):>18.2e} {max(f) - min(f):>18.2e} "
            f"{max(rc) - min(rc):>18.2e} {max(d) - min(d):>20.2e}"
        )
    pr("")

    # ---- regime-mean spread for the raw metrics ----
    pr("=== raw metric spread across regimes ===")
    pr(
        f"{'metric':<14} {'mode':<14} {'mean':>10} "
        f"{'sd_regime_means':>16} {'max_within_seed_range':>22}"
    )
    for metric in ("boundary_f1", "bits_per_char", "tokens_per_char"):
        for mode in ("flat", "frequency", "morph", "random_closed"):
            regime_means = []
            within_ranges = []
            for variant in variants:
                vals = [
                    float(r[metric])
                    for r in runs
                    if r["variant"] == variant and r["mode"] == mode
                ]
                regime_means.append(statistics.mean(vals))
            for seed in seeds:
                vals = [
                    float(r[metric])
                    for r in runs
                    if int(r["seed"]) == seed and r["mode"] == mode
                ]
                within_ranges.append(max(vals) - min(vals))
            sd = (
                statistics.stdev(regime_means)
                if len(regime_means) > 1
                else 0.0
            )
            pr(
                f"{metric:<14} {mode:<14} {statistics.mean(regime_means):>10.6f} "
                f"{sd:>16.2e} {max(within_ranges):>22.2e}"
            )
    pr("")

    # ---- tie rates ----
    pr("=== BPE tie steps ===")
    for variant in variants:
        ts = [int(r["tie_steps"]) for r in ties if r["variant"] == variant]
        ms = [int(r["merges"]) for r in ties if r["variant"] == variant]
        rate = 100.0 * sum(ts) / max(1, sum(ms))
        pr(
            f"  {variant:<10} ties={ts} of {ms} "
            f"({rate:.1f}% overall, per-seed {min(ts)}-{max(ts)})"
        )
    pr("")

    # ---- rule-sequence divergence ----
    pr("=== flat rule-sequence divergence from the id tie-break ===")
    for variant in variants:
        rows = [r for r in rules if r["variant"] == variant]
        fd = [int(r["first_diff_rank"]) for r in rows if r["first_diff_rank"] != ""]
        pr(
            f"  {variant:<10} first differing rank per seed: "
            f"{fd if fd else 'none (identical sequence)'}"
        )
    pr("")

    text = "\n".join(out)
    path = os.path.join(HERE, f"w1_08_report_tables_{TAG}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nwrote {path}")


def load_published():
    with open(PUBLISHED, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    main()
