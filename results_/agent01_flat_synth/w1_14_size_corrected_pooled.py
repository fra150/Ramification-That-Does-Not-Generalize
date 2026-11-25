"""W1-14: pooled estimate with a size-corrected random control.

For every seed use the buggy-sampler draws that have exactly 294 unique L1
rules when available; for seeds with no 294-rule draw (44, 45, 50), use the
deduplicated sampler's 294-rule draws. Writes a compact table and pooled t.
"""

import csv
import math
import statistics

T95 = {9: 2.262, 10: 2.228}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(xs):
    return statistics.mean(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def tstat(xs, null=0.0):
    if len(xs) < 2 or sd(xs) == 0:
        return float("inf") if mean(xs) != null else 0.0
    return (mean(xs) - null) / (sd(xs) / math.sqrt(len(xs)))


main = load("results/redteam/w1_14_random_closed_draws.csv")
corr = load("results/redteam/w1_14_corrected_sampler.csv")
seeds = sorted({int(r["seed"]) for r in main})
morph = {}
draws294 = {}
for r in main:
    seed = int(r["seed"])
    if r["tag"] == "morph":
        morph[seed] = r
    elif r["tag"].startswith("random_closed_d") and int(r["n_l1"]) == 294:
        draws294.setdefault(seed, []).append(r)
corr_draws = {}
for r in corr:
    if r["draw"] != "summary":
        corr_draws.setdefault(int(r["seed"]), []).append(r)

rows = []
for seed in seeds:
    if seed in draws294 and len(draws294[seed]) >= 2:
        src = "buggy-294"
        rs = draws294[seed]
    else:
        src = "corrected-294"
        rs = corr_draws.get(seed, [])
    if len(rs) < 2:
        print(f"seed {seed}: no size-294 source ({len(rs)} draws), skipped")
        continue
    f1s = [float(r["boundary_f1"]) for r in rs]
    bpcs = [float(r["bits_per_char"]) for r in rs]
    rows.append(
        {
            "seed": seed,
            "source": src,
            "n": len(rs),
            "morph_f1": float(morph[seed]["boundary_f1"]),
            "rand_f1": mean(f1s),
            "delta_f1": float(morph[seed]["boundary_f1"]) - mean(f1s),
            "morph_bpc": float(morph[seed]["bits_per_char"]),
            "rand_bpc": mean(bpcs),
            "delta_bpc": float(morph[seed]["bits_per_char"]) - mean(bpcs),
        }
    )

print("seed source            n  deltaF1   deltaBPC")
for r in rows:
    print(
        f"{r['seed']}   {r['source']:<15} {r['n']:>2}  "
        f"{r['delta_f1']:+.5f}  {r['delta_bpc']:+.5f}"
    )

for metric, sign in (("delta_f1", 1), ("delta_bpc", -1)):
    ds = [r[metric] for r in rows]
    m, s = mean(ds), sd(ds)
    se = s / math.sqrt(len(ds))
    crit = T95.get(len(ds), 2.262)
    w = sum(1 for d in ds if d * sign > 0)
    print(
        f"pooled {metric}: mean={m:+.5f} sd={s:.5f} t={m/se:+.2f} "
        f"CI95=[{m-crit*se:+.5f},{m+crit*se:+.5f}] W/L={w}/{len(ds)-w} "
        f"n={len(ds)}"
    )

with open("results/redteam/w1_14_size_corrected_pooled.csv", "w", newline="",
          encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("wrote results/redteam/w1_14_size_corrected_pooled.csv")
