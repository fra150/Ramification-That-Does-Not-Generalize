import csv
import statistics

rows = list(csv.DictReader(open("results/redteam/w1_14_per_seed.csv")))
print("seed  rand-flatF1  morph-flatF1  rand-flatBPC  morph-flatBPC")
for r in rows:
    rf1 = float(r["mean_boundary_f1"]) - float(r["flat_boundary_f1"])
    mf1 = float(r["morph_boundary_f1"]) - float(r["flat_boundary_f1"])
    rb = float(r["mean_bits_per_char"]) - float(r["flat_bits_per_char"])
    mb = float(r["morph_bits_per_char"]) - float(r["flat_bits_per_char"])
    print(f"{r['seed']}  {rf1:+.4f}       {mf1:+.4f}       {rb:+.4f}       {mb:+.4f}")

print(
    "mean morph-flat F1",
    f"{statistics.mean(float(r['morph_boundary_f1'])-float(r['flat_boundary_f1']) for r in rows):+.5f}",
    "bpc",
    f"{statistics.mean(float(r['morph_bits_per_char'])-float(r['flat_bits_per_char']) for r in rows):+.5f}",
)
