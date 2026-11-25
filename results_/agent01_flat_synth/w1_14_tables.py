import csv

rows = list(csv.DictReader(open("results/redteam/w1_14_per_seed.csv")))
print("seed  dF1      dF1med   zF1   rankF1 beyondF1 dF1canon  nL1")
for r in rows:
    print(
        f"{r['seed']}  {float(r['delta_boundary_f1']):+.5f}  "
        f"{float(r['delta_median_boundary_f1']):+.5f}  "
        f"{float(r['z_improvement_boundary_f1']):+.2f}  "
        f"{r['rank_boundary_f1']:>2}     {r['morph_beyond_max_boundary_f1']}    "
        f"{float(r['canonical_delta_boundary_f1']):+.5f}  "
        f"{r['n_l1_draw_min']}-{r['n_l1_draw_max']}"
    )
print()
print("seed  dBPC     dBPCmed  zBPC  rankBPC beyondBPC dBPCcanon")
for r in rows:
    print(
        f"{r['seed']}  {float(r['delta_bits_per_char']):+.5f}  "
        f"{float(r['delta_median_bits_per_char']):+.5f}  "
        f"{float(r['z_improvement_bits_per_char']):+.2f}  "
        f"{r['rank_bits_per_char']:>2}     {r['morph_beyond_max_bits_per_char']}    "
        f"{float(r['canonical_delta_bits_per_char']):+.5f}"
    )
print()
print("seed  flatF1  morphF1  randF1   canonF1  | flatBPC morphBPC randBPC  canonBPC")
for r in rows:
    print(
        f"{r['seed']}  {float(r['flat_boundary_f1']):.4f}  "
        f"{float(r['morph_boundary_f1']):.4f}  "
        f"{float(r['mean_boundary_f1']):.4f}  "
        f"{float(r['canonical_boundary_f1']):.4f}  | "
        f"{float(r['flat_bits_per_char']):.4f}  "
        f"{float(r['morph_bits_per_char']):.4f}  "
        f"{float(r['mean_bits_per_char']):.4f}  "
        f"{float(r['canonical_bits_per_char']):.4f}"
    )
print()
print("seed  l1intra_morph  draw_mean  range   canon  jacc_canon")
for r in rows:
    print(
        f"{r['seed']}  {r['l1_intra_morph']:>3}  {float(r['l1_intra_draw_mean']):.1f}  "
        f"[{r['l1_intra_draw_min']},{r['l1_intra_draw_max']}]  "
        f"{r['l1_intra_canonical']:>3}  {float(r['jaccard_morph_canonical']):.3f}"
    )
