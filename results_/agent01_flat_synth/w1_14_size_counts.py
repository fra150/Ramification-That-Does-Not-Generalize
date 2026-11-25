import csv
from collections import Counter

rows = [
    r
    for r in csv.DictReader(open("results/redteam/w1_14_random_closed_draws.csv"))
    if r["tag"].startswith("random_closed_d")
]
by_seed = {}
for r in rows:
    by_seed.setdefault(int(r["seed"]), Counter())[int(r["n_l1"])] += 1
for seed in sorted(by_seed):
    print(seed, dict(sorted(by_seed[seed].items())))
print("total", dict(sorted(Counter(int(r["n_l1"]) for r in rows).items())))
print("not294", sum(1 for r in rows if int(r["n_l1"]) != 294))
print("distinct sets", len({(r["seed"], r["level1_fingerprint"]) for r in rows}))
