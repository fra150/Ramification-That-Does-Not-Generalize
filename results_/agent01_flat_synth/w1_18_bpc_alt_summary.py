"""W1-18: combine the parallel alt-run shards and print 10-seed paired stats."""
from __future__ import annotations

import csv
import glob
import math
import statistics

SHARDS = [
    "results/redteam/w1_18_bpc_alt_42_43.csv",
    "results/redteam/w1_18_bpc_alt_44_45.csv",
    "results/redteam/w1_18_bpc_alt_46_47.csv",
    "results/redteam/w1_18_bpc_alt_48_49.csv",
    "results/redteam/w1_18_bpc_alt_50_51.csv",
]
OUT = "results/redteam/w1_18_bpc_alt.csv"
MAIN = "results/redteam/w1_18_bpc_audit.csv"

rows = []
for path in SHARDS:
    with open(path, encoding="utf-8") as fh:
        rows.extend(csv.DictReader(fh))
rows.sort(key=lambda r: (int(r["seed"]), r["mode"]))
assert len(rows) == 40, len(rows)
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# cross-check the official column against the main audit
main = {
    (int(r["seed"]), r["mode"]): float(r["bpc_official_a0.1"])
    for r in csv.DictReader(open(MAIN, encoding="utf-8"))
}
diffs = [
    abs(float(r["bpc_official_uniform_heldout"]) - main[(int(r["seed"]), r["mode"])])
    for r in rows
]
print(f"official cross-check: max |alt - main| = {max(diffs):.3e}")


def describe(deltas):
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(n)
    if se == 0:
        t = 0.0 if mean == 0 else float("inf")
    else:
        t = mean / se
    return mean, sd, t, mean - 2.262 * se, mean + 2.262 * se


metrics = [k for k in rows[0] if k.startswith("bpc_") or k.startswith("tpc_")]
seeds = sorted({int(r["seed"]) for r in rows})
idx = {(int(r["seed"]), r["mode"]): r for r in rows}
for pair in (("morph", "frequency"), ("random_closed", "frequency")):
    a, b = pair
    print(f"\n== {a} - {b}, v512, 10 seeds ==")
    for metric in metrics:
        d = [
            float(idx[(s, a)][metric]) - float(idx[(s, b)][metric])
            for s in seeds
        ]
        mean, sd, t, lo, hi = describe(d)
        print(
            f"{metric:<32} mean={mean:+.5f} sd={sd:.5f} t={t:+7.2f} "
            f"CI95=[{lo:+.5f},{hi:+.5f}] "
            f"W/L/T={sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}/{sum(x == 0 for x in d)}"
        )

# mean ratio of in-distribution to uniform-heldout gain
d_uni, d_ind = [], []
for s in seeds:
    d_uni.append(
        float(idx[(s, "morph")]["bpc_official_uniform_heldout"])
        - float(idx[(s, "frequency")]["bpc_official_uniform_heldout"])
    )
    d_ind.append(
        float(idx[(s, "morph")]["bpc_indist_zipf_90_10"])
        - float(idx[(s, "frequency")]["bpc_indist_zipf_90_10"])
    )
print(
    f"\nratio mean(indist)/mean(uniform) = "
    f"{statistics.mean(d_ind) / statistics.mean(d_uni):.3f} "
    f"(uniform {statistics.mean(d_uni):+.5f}, indist {statistics.mean(d_ind):+.5f})"
)

# in-distribution OOV shards
oov_paths = [
    "results/redteam/w1_18_indist_oov_a.csv",
    "results/redteam/w1_18_indist_oov_b.csv",
]
oov = []
for path in oov_paths:
    with open(path, encoding="utf-8") as fh:
        oov.extend(csv.DictReader(fh))
oov.sort(key=lambda r: (int(r["seed"]), r["mode"]))
assert len(oov) == 20, len(oov)
with open("results/redteam/w1_18_indist_oov.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(oov[0].keys()))
    w.writeheader()
    w.writerows(oov)
oidx = {(int(r["seed"]), r["mode"]): r for r in oov}
print("\n== in-distribution 90/10 OOV, 10 seeds ==")
for metric in ("unseen_tokens", "unseen_types", "bpc_add0.1", "bpc_seen_only", "tpc"):
    d = [
        float(oidx[(s, "morph")][metric]) - float(oidx[(s, "frequency")][metric])
        for s in seeds
    ]
    mean, sd, t, lo, hi = describe(d)
    print(
        f"{metric:<16} mean={mean:+.6f} sd={sd:.6f} t={t:+7.2f} "
        f"CI95=[{lo:+.6f},{hi:+.6f}] "
        f"W/L/T={sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}/{sum(x == 0 for x in d)}"
    )
