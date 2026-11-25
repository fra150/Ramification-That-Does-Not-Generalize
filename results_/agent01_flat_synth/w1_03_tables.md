# W1-03 tables (5 seeds: [np.int64(42), np.int64(43), np.int64(44), np.int64(45), np.int64(46)])

## Corpus statistics (mean over seeds)

| condition | word tokens | periods | types | coverage | top-1 share | top-10 share | entropy (bits/word) | alphabet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 120000 | 8446 | 7468 | 0.810 | 0.1520 | 0.4083 | 8.25 | 19 |
| uniform | 120000 | 8355 | 9216 | 1.000 | 0.0002 | 0.0022 | 13.11 | 19 |
| noperiod | 120000 | 0 | 7468 | 0.810 | 0.1520 | 0.4083 | 8.25 | 19 |
| uniform_noperiod | 120000 | 0 | 9216 | 1.000 | 0.0002 | 0.0022 | 13.11 | 19 |

## Mean metrics by condition and mode (mean over seeds)

### boundary_f1

| condition | flat | frequency | morph | random_closed | morph_type |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8556 (0.0518) | 0.8556 (0.0518) | 0.8624 (0.0541) | 0.8505 (0.0480) | 0.8624 (0.0541) |
| uniform | 0.8954 (0.0735) | 0.8954 (0.0735) | 0.8954 (0.0735) | 0.8751 (0.0643) | 0.8954 (0.0735) |
| noperiod | 0.8560 (0.0521) | 0.8560 (0.0521) | 0.8627 (0.0543) | 0.8464 (0.0480) | 0.8627 (0.0543) |
| uniform_noperiod | 0.8952 (0.0735) | 0.8952 (0.0735) | 0.8952 (0.0735) | 0.8770 (0.0637) | 0.8952 (0.0735) |

### bits_per_char

| condition | flat | frequency | morph | random_closed | morph_type |
|---|---:|---:|---:|---:|---:|
| baseline | 2.3217 (0.0691) | 2.3217 (0.0691) | 2.3049 (0.0773) | 2.3540 (0.0623) | 2.3049 (0.0773) |
| uniform | 2.0002 (0.0593) | 2.0002 (0.0593) | 2.0002 (0.0593) | 2.0919 (0.0417) | 2.0002 (0.0593) |
| noperiod | 2.3006 (0.0681) | 2.3006 (0.0681) | 2.2843 (0.0753) | 2.3355 (0.0499) | 2.2843 (0.0753) |
| uniform_noperiod | 1.9895 (0.0590) | 1.9895 (0.0590) | 1.9895 (0.0590) | 2.0697 (0.0440) | 1.9895 (0.0590) |

## Flat (frequency-level) metric by seed and condition

### boundary_f1

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | 0.8665 | 0.9494 | 0.8682 | 0.9491 |
| 43 | 0.8011 | 0.7882 | 0.8007 | 0.7878 |
| 44 | 0.9142 | 0.8486 | 0.9143 | 0.8486 |
| 45 | 0.8935 | 0.9464 | 0.8938 | 0.9461 |
| 46 | 0.8029 | 0.9446 | 0.8030 | 0.9444 |

### bits_per_char

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | 2.3539 | 1.9655 | 2.3231 | 1.9546 |
| 43 | 2.3920 | 2.0823 | 2.3736 | 2.0713 |
| 44 | 2.2403 | 2.0435 | 2.2224 | 2.0325 |
| 45 | 2.2553 | 1.9463 | 2.2354 | 1.9362 |
| 46 | 2.3671 | 1.9633 | 2.3487 | 1.9530 |

## Per-seed morph - frequency deltas

### boundary_f1

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | +0.0041 | +0.0000 | +0.0043 | +0.0000 |
| 43 | +0.0097 | +0.0000 | +0.0097 | +0.0000 |
| 44 | +0.0096 | +0.0000 | +0.0096 | +0.0000 |
| 45 | +0.0100 | +0.0000 | +0.0097 | +0.0000 |
| 46 | +0.0001 | +0.0000 | +0.0001 | +0.0000 |

### bits_per_char

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | -0.0054 | +0.0000 | -0.0056 | +0.0000 |
| 43 | -0.0241 | +0.0000 | -0.0240 | +0.0000 |
| 44 | -0.0299 | +0.0000 | -0.0297 | +0.0000 |
| 45 | -0.0237 | +0.0000 | -0.0211 | +0.0000 |
| 46 | -0.0010 | +0.0000 | -0.0009 | +0.0000 |

## Paired deltas (5 seeds; CI is 95% t)

### boundary_f1

| condition | comparison | mean delta | 95% CI | t | W/L/T |
|---|---|---:|---|---:|---:|
| baseline | morph - frequency | +0.0067 | [+0.0012, +0.0122] | 3.40 | 5/0/0 |
| baseline | morph - random_closed | +0.0119 | [-0.0064, +0.0302] | 1.81 | 4/1/0 |
| baseline | random_closed - frequency | -0.0052 | [-0.0214, +0.0111] | -0.88 | 2/3/0 |
| baseline | morph_type - frequency | +0.0067 | [+0.0012, +0.0122] | 3.40 | 5/0/0 |
| baseline | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform | morph - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform | morph - random_closed | +0.0204 | [+0.0070, +0.0337] | 4.24 | 5/0/0 |
| uniform | random_closed - frequency | -0.0204 | [-0.0337, -0.0070] | -4.24 | 0/5/0 |
| uniform | morph_type - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| noperiod | morph - frequency | +0.0067 | [+0.0013, +0.0121] | 3.46 | 5/0/0 |
| noperiod | morph - random_closed | +0.0163 | [+0.0064, +0.0262] | 4.57 | 5/0/0 |
| noperiod | random_closed - frequency | -0.0096 | [-0.0153, -0.0039] | -4.70 | 0/5/0 |
| noperiod | morph_type - frequency | +0.0067 | [+0.0013, +0.0121] | 3.46 | 5/0/0 |
| noperiod | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform_noperiod | morph - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform_noperiod | morph - random_closed | +0.0181 | [+0.0010, +0.0353] | 2.94 | 5/0/0 |
| uniform_noperiod | random_closed - frequency | -0.0181 | [-0.0353, -0.0010] | -2.94 | 0/5/0 |
| uniform_noperiod | morph_type - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform_noperiod | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |

### bits_per_char

| condition | comparison | mean delta | 95% CI | t | W/L/T |
|---|---|---:|---|---:|---:|
| baseline | morph - frequency | -0.0168 | [-0.0327, -0.0010] | -2.95 | 5/0/0 |
| baseline | morph - random_closed | -0.0491 | [-0.1031, +0.0049] | -2.52 | 5/0/0 |
| baseline | random_closed - frequency | +0.0323 | [-0.0170, +0.0816] | 1.82 | 1/4/0 |
| baseline | morph_type - frequency | -0.0168 | [-0.0327, -0.0010] | -2.95 | 5/0/0 |
| baseline | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform | morph - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform | morph - random_closed | -0.0917 | [-0.1594, -0.0240] | -3.76 | 5/0/0 |
| uniform | random_closed - frequency | +0.0917 | [+0.0240, +0.1594] | 3.76 | 0/5/0 |
| uniform | morph_type - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| noperiod | morph - frequency | -0.0163 | [-0.0316, -0.0009] | -2.95 | 5/0/0 |
| noperiod | morph - random_closed | -0.0512 | [-0.0893, -0.0130] | -3.72 | 5/0/0 |
| noperiod | random_closed - frequency | +0.0349 | [+0.0098, +0.0600] | 3.86 | 0/5/0 |
| noperiod | morph_type - frequency | -0.0163 | [-0.0316, -0.0009] | -2.95 | 5/0/0 |
| noperiod | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform_noperiod | morph - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform_noperiod | morph - random_closed | -0.0802 | [-0.1484, -0.0119] | -3.26 | 5/0/0 |
| uniform_noperiod | random_closed - frequency | +0.0802 | [+0.0119, +0.1484] | 3.26 | 0/5/0 |
| uniform_noperiod | morph_type - frequency | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |
| uniform_noperiod | morph_type - morph | +0.0000 | [+0.0000, +0.0000] | nan | 0/0/5 |

## Did the effect shift between corpora? (paired by seed)

### boundary_f1

| comparison of morph-frequency effect | mean shift | 95% CI | t | p |
|---|---:|---|---:|---:|
| baseline - uniform | +0.0067 | [+0.0012, +0.0122] | 3.40 | 0.0273 |
| baseline - noperiod | +0.0000 | [-0.0002, +0.0003] | 0.36 | 0.7396 |
| baseline - uniform_noperiod | +0.0067 | [+0.0012, +0.0122] | 3.40 | 0.0273 |
| uniform - uniform_noperiod | +0.0000 | [+0.0000, +0.0000] | nan | nan |
| noperiod - uniform_noperiod | +0.0067 | [+0.0013, +0.0121] | 3.46 | 0.0258 |

### bits_per_char

| comparison of morph-frequency effect | mean shift | 95% CI | t | p |
|---|---:|---|---:|---:|
| baseline - uniform | -0.0168 | [-0.0327, -0.0010] | -2.95 | 0.0420 |
| baseline - noperiod | -0.0006 | [-0.0019, +0.0008] | -1.10 | 0.3338 |
| baseline - uniform_noperiod | -0.0168 | [-0.0327, -0.0010] | -2.95 | 0.0420 |
| uniform - uniform_noperiod | +0.0000 | [+0.0000, +0.0000] | nan | nan |
| noperiod - uniform_noperiod | -0.0163 | [-0.0316, -0.0009] | -2.95 | 0.0421 |

## Level-1 composition and tokenization change (mean over seeds)

| condition | mode | L1 rules | L1 intra | L1 cross | L1 intra share | L2 intra | L2 cross | test words changed vs frequency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | frequency | 294 | 193 | 100 | 0.658 | 63 | 133 | 0.000 |
| baseline | morph | 294 | 257 | 36 | 0.873 | 0 | 196 | 0.014 |
| baseline | random_closed | 294 | 207 | 85 | 0.706 | 49 | 147 | 0.039 |
| uniform | frequency | 294 | 264 | 29 | 0.897 | 1 | 195 | 0.000 |
| uniform | morph | 294 | 265 | 28 | 0.900 | 0 | 196 | 0.000 |
| uniform | random_closed | 294 | 208 | 85 | 0.707 | 57 | 139 | 0.072 |
| noperiod | frequency | 294 | 194 | 100 | 0.661 | 63 | 134 | 0.000 |
| noperiod | morph | 294 | 257 | 37 | 0.875 | 0 | 197 | 0.014 |
| noperiod | random_closed | 294 | 209 | 84 | 0.713 | 48 | 149 | 0.033 |
| uniform_noperiod | frequency | 294 | 264 | 30 | 0.898 | 1 | 196 | 0.000 |
| uniform_noperiod | morph | 294 | 265 | 29 | 0.900 | 0 | 197 | 0.000 |
| uniform_noperiod | random_closed | 293 | 204 | 90 | 0.694 | 61 | 137 | 0.051 |
