"""Compare held-out-provenance morph assignments with full-provenance morph
assignments truncated to the same level-1 size."""

from __future__ import annotations

import csv

from results.redteam.w1_15_levels_weighting import describe, write_csv


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    runs = load("results/redteam/w1_15_levels_weighting_runs.csv")
    topk = load("results/redteam/w1_15_topk_control.csv")

    def get_run(seed, variant):
        for r in runs:
            if (int(r["seed"]) == seed and r["variant"] == variant
                    and r["mode"] == "morph" and r["eval_set"] == "test_all"):
                return r
        return None

    def get_topk(seed, k):
        for r in topk:
            if int(r["seed"]) == seed and int(r["target_k"]) == k:
                return r
        return None

    seeds = sorted({int(r["seed"]) for r in topk})
    out = []
    print("Held-out provenance vs full provenance truncated to the same L1 size")
    for variant in ("freq_halfA", "freq_halfB", "unique_halfA", "unique_halfB",
                    "root_cv"):
        for metric, topk_key in (("boundary_f1", "boundary_f1"),
                                 ("bits_per_char", "bits_per_char")):
            deltas = []
            for seed in seeds:
                r = get_run(seed, variant)
                if r is None:
                    continue
                k = int(r["n_l1"])
                t = get_topk(seed, k)
                if t is None:
                    continue
                deltas.append(float(r[metric]) - float(t[topk_key]))
            d = describe(deltas)
            out.append({"comparison": f"{variant} vs full_topk (same L1 size)",
                        "metric": metric, **d})
            print(
                f"  {variant:<15} vs full-topk {metric:<13} "
                f"mean={d['mean']:+.4f} t={d['t']:+.2f} "
                f"CI95=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] "
                f"W/L/T={d['wins']}/{d['losses']}/{d['ties']}"
            )

    print("\nFull-provenance morph truncated to held-out L1 sizes, vs frequency")
    for metric, key in (("boundary_f1", "delta_vs_frequency"),
                        ("bits_per_char", "delta_bpc_vs_frequency")):
        by_seed = {}
        for r in topk:
            by_seed.setdefault(int(r["seed"]), []).append(float(r[key]))
        deltas = [sum(v) / len(v) for _, v in sorted(by_seed.items())]
        d = describe(deltas)
        out.append({"comparison": "full_topk (held-out L1 size) vs frequency",
                    "metric": metric, **d})
        print(
            f"  per-seed mean over k {metric:<13} "
            f"mean={d['mean']:+.4f} t={d['t']:+.2f} "
            f"CI95=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] "
            f"W/L/T={d['wins']}/{d['losses']}/{d['ties']} (n={d['n']})"
        )

    print("\nFull-provenance morph vs size-matched random-closed (same k)")
    for metric, mkey, rkey in (
        ("boundary_f1", "boundary_f1", "random_closed_f1"),
        ("bits_per_char", "bits_per_char", "random_closed_bpc"),
    ):
        by_seed = {}
        for r in topk:
            by_seed.setdefault(int(r["seed"]), []).append(
                float(r[mkey]) - float(r[rkey])
            )
        deltas = [sum(v) / len(v) for _, v in sorted(by_seed.items())]
        d = describe(deltas)
        out.append({"comparison": "full_topk morph vs random_closed (same k)",
                    "metric": metric, **d})
        print(
            f"  morph - random {metric:<13} "
            f"mean={d['mean']:+.4f} t={d['t']:+.2f} "
            f"CI95=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] "
            f"W/L/T={d['wins']}/{d['losses']}/{d['ties']} (n={d['n']})"
        )

    print("\nSize-matched random-closed vs frequency")
    for metric, key in (("boundary_f1", "delta_random_vs_frequency"),
                        ("bits_per_char", None)):
        by_seed = {}
        for r in topk:
            if key is None:
                # random bpc - frequency bpc
                d0 = float(r["random_closed_bpc"]) - float(r["frequency_bpc"])
            else:
                d0 = float(r[key])
            by_seed.setdefault(int(r["seed"]), []).append(d0)
        deltas = [sum(v) / len(v) for _, v in sorted(by_seed.items())]
        d = describe(deltas)
        out.append({"comparison": "random_closed (same k) vs frequency",
                    "metric": metric, **d})
        print(
            f"  random - frequency {metric:<13} "
            f"mean={d['mean']:+.4f} t={d['t']:+.2f} "
            f"CI95=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] "
            f"W/L/T={d['wins']}/{d['losses']}/{d['ties']} (n={d['n']})"
        )

    write_csv("results/redteam/w1_15_topk_comparison.csv", out)
    print("\nwrote results/redteam/w1_15_topk_comparison.csv")


if __name__ == "__main__":
    main()
