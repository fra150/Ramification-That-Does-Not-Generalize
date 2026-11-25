# W1-03 tables (1 seeds: [np.int64(42)])

## Corpus statistics (mean over seeds)

| condition | word tokens | periods | types | coverage | top-1 share | top-10 share | entropy (bits/word) | alphabet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 120000 | 8408 | 7478 | 0.811 | 0.1519 | 0.4078 | 8.25 | 19 |
| uniform | 120000 | 8306 | 9216 | 1.000 | 0.0002 | 0.0022 | 13.11 | 19 |
| noperiod | 120000 | 0 | 7478 | 0.811 | 0.1519 | 0.4078 | 8.25 | 19 |
| uniform_noperiod | 120000 | 0 | 9216 | 1.000 | 0.0002 | 0.0022 | 13.11 | 19 |

## Mean metrics by condition and mode (mean over seeds)

### boundary_f1

| condition | flat | frequency | morph | random_closed | morph_type |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8665 (nan) | 0.8665 (nan) | 0.8706 (nan) | 0.8757 (nan) | n/a |
| uniform | 0.9494 (nan) | 0.9494 (nan) | 0.9494 (nan) | 0.9191 (nan) | n/a |
| noperiod | 0.8682 (nan) | 0.8682 (nan) | 0.8726 (nan) | 0.8611 (nan) | n/a |
| uniform_noperiod | 0.9491 (nan) | 0.9491 (nan) | 0.9491 (nan) | 0.9156 (nan) | n/a |

### bits_per_char

| condition | flat | frequency | morph | random_closed | morph_type |
|---|---:|---:|---:|---:|---:|
| baseline | 2.3539 (nan) | 2.3539 (nan) | 2.3484 (nan) | 2.3496 (nan) | n/a |
| uniform | 1.9655 (nan) | 1.9655 (nan) | 1.9655 (nan) | 2.1414 (nan) | n/a |
| noperiod | 2.3231 (nan) | 2.3231 (nan) | 2.3174 (nan) | 2.3424 (nan) | n/a |
| uniform_noperiod | 1.9546 (nan) | 1.9546 (nan) | 1.9546 (nan) | 2.0899 (nan) | n/a |

## Flat (frequency-level) metric by seed and condition

### boundary_f1

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | 0.8665 | 0.9494 | 0.8682 | 0.9491 |

### bits_per_char

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | 2.3539 | 1.9655 | 2.3231 | 1.9546 |

## Per-seed morph - frequency deltas

### boundary_f1

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | +0.0041 | +0.0000 | +0.0043 | +0.0000 |

### bits_per_char

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | -0.0054 | +0.0000 | -0.0056 | +0.0000 |

## Paired deltas (5 seeds; CI is 95% t)

### boundary_f1

| condition | comparison | mean delta | 95% CI | t | W/L/T |
|---|---|---:|---|---:|---:|
| baseline | morph - frequency | +0.0041 | [n/a, n/a] | nan | 1/0/0 |
| baseline | morph - random_closed | -0.0051 | [n/a, n/a] | nan | 0/1/0 |
| baseline | random_closed - frequency | +0.0092 | [n/a, n/a] | nan | 1/0/0 |
| uniform | morph - frequency | +0.0000 | [n/a, n/a] | nan | 0/0/1 |
| uniform | morph - random_closed | +0.0303 | [n/a, n/a] | nan | 1/0/0 |
| uniform | random_closed - frequency | -0.0303 | [n/a, n/a] | nan | 0/1/0 |
| noperiod | morph - frequency | +0.0043 | [n/a, n/a] | nan | 1/0/0 |
| noperiod | morph - random_closed | +0.0114 | [n/a, n/a] | nan | 1/0/0 |
| noperiod | random_closed - frequency | -0.0071 | [n/a, n/a] | nan | 0/1/0 |
| uniform_noperiod | morph - frequency | +0.0000 | [n/a, n/a] | nan | 0/0/1 |
| uniform_noperiod | morph - random_closed | +0.0335 | [n/a, n/a] | nan | 1/0/0 |
| uniform_noperiod | random_closed - frequency | -0.0335 | [n/a, n/a] | nan | 0/1/0 |

### bits_per_char

| condition | comparison | mean delta | 95% CI | t | W/L/T |
|---|---|---:|---|---:|---:|
| baseline | morph - frequency | -0.0054 | [n/a, n/a] | nan | 1/0/0 |
| baseline | morph - random_closed | -0.0011 | [n/a, n/a] | nan | 1/0/0 |
| baseline | random_closed - frequency | -0.0043 | [n/a, n/a] | nan | 1/0/0 |
| uniform | morph - frequency | +0.0000 | [n/a, n/a] | nan | 0/0/1 |
| uniform | morph - random_closed | -0.1760 | [n/a, n/a] | nan | 1/0/0 |
| uniform | random_closed - frequency | +0.1760 | [n/a, n/a] | nan | 0/1/0 |
| noperiod | morph - frequency | -0.0056 | [n/a, n/a] | nan | 1/0/0 |
| noperiod | morph - random_closed | -0.0249 | [n/a, n/a] | nan | 1/0/0 |
| noperiod | random_closed - frequency | +0.0193 | [n/a, n/a] | nan | 0/1/0 |
| uniform_noperiod | morph - frequency | +0.0000 | [n/a, n/a] | nan | 0/0/1 |
| uniform_noperiod | morph - random_closed | -0.1353 | [n/a, n/a] | nan | 1/0/0 |
| uniform_noperiod | random_closed - frequency | +0.1353 | [n/a, n/a] | nan | 0/1/0 |

## Did the effect shift between corpora? (paired by seed)

### boundary_f1

| comparison of morph-frequency effect | mean shift | 95% CI | t | p |
|---|---:|---|---:|---:|
| baseline - uniform | +0.0041 | [n/a, n/a] | nan | nan |
| baseline - noperiod | -0.0002 | [n/a, n/a] | nan | nan |
| baseline - uniform_noperiod | +0.0041 | [n/a, n/a] | nan | nan |
| uniform - uniform_noperiod | +0.0000 | [n/a, n/a] | nan | nan |
| noperiod - uniform_noperiod | +0.0043 | [n/a, n/a] | nan | nan |

### bits_per_char

| comparison of morph-frequency effect | mean shift | 95% CI | t | p |
|---|---:|---|---:|---:|
| baseline - uniform | -0.0054 | [n/a, n/a] | nan | nan |
| baseline - noperiod | +0.0002 | [n/a, n/a] | nan | nan |
| baseline - uniform_noperiod | -0.0054 | [n/a, n/a] | nan | nan |
| uniform - uniform_noperiod | +0.0000 | [n/a, n/a] | nan | nan |
| noperiod - uniform_noperiod | -0.0056 | [n/a, n/a] | nan | nan |

## Level-1 composition and tokenization change (mean over seeds)

| condition | mode | L1 rules | L1 intra | L1 cross | L1 intra share | L2 intra | L2 cross | test words changed vs frequency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | frequency | 294 | 197 | 96 | 0.670 | 53 | 143 | 0.000 |
| baseline | morph | 294 | 250 | 43 | 0.850 | 0 | 196 | 0.009 |
| baseline | random_closed | 294 | 214 | 79 | 0.728 | 36 | 160 | 0.034 |
| uniform | frequency | 294 | 275 | 18 | 0.935 | 0 | 196 | 0.000 |
| uniform | morph | 294 | 275 | 18 | 0.935 | 0 | 196 | 0.000 |
| uniform | random_closed | 294 | 216 | 77 | 0.735 | 59 | 137 | 0.204 |
| noperiod | frequency | 294 | 198 | 96 | 0.673 | 53 | 144 | 0.000 |
| noperiod | morph | 294 | 251 | 43 | 0.854 | 0 | 197 | 0.009 |
| noperiod | random_closed | 294 | 204 | 90 | 0.694 | 47 | 150 | 0.031 |
| uniform_noperiod | frequency | 294 | 275 | 19 | 0.935 | 0 | 197 | 0.000 |
| uniform_noperiod | morph | 294 | 275 | 19 | 0.935 | 0 | 197 | 0.000 |
| uniform_noperiod | random_closed | 294 | 209 | 85 | 0.711 | 66 | 131 | 0.085 |
