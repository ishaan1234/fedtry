# Key tests (hospital level, 5 folds)

Each hospital's Dice is the mean over all its test cases across the five folds; tests are paired two-sided Wilcoxon over the 23 hospitals.

## 1. Local head minus local stem, by setting

| Setting | Gap (Dice) | Hospitals favouring head | p (gap = 0) |
| --- | --: | --: | --: |
| site | +0.0501 | 22/23 | 4.8e-07 |
| modality | -0.0049 | 12/23 | 0.85 |
| site_modality | -0.0446 | 11/23 | 0.46 |

Change in gap from real hospitals to the same hospitals with missing sequences: -0.0946 (p = 0.00055).

With fine-tuning: +0.0529 (23/23) vs -0.0087 (14/23), change p = 0.021.

## 2. Split + fine-tuning vs fine-tuning alone

| Setting | Method | Δ Dice | Better / worse | p | Holm p (9 tests) |
| --- | --- | --: | --: | --: | --: |
| site | `norm_local+ft` | +0.0007 | 10 / 13 | 0.94 | 1 |
| site | `in_local_1+ft` | -0.0443 | 2 / 21 | 1.2e-06 | 1.1e-05 |
| site | `out_local_1+ft` | +0.0086 | 16 / 7 | 0.028 | 0.19 |
| modality | `norm_local+ft` | -0.0059 | 9 / 14 | 0.3 | 1 |
| modality | `in_local_1+ft` | -0.0385 | 5 / 18 | 0.0024 | 0.019 |
| modality | `out_local_1+ft` | -0.0089 | 11 / 12 | 0.73 | 1 |
| site_modality | `norm_local+ft` | -0.0020 | 13 / 10 | 0.8 | 1 |
| site_modality | `in_local_1+ft` | -0.0144 | 6 / 17 | 0.08 | 0.4 |
| site_modality | `out_local_1+ft` | -0.0231 | 8 / 15 | 0.056 | 0.34 |

`out_local_1+ft` minus `fedavg+ft`, real hospitals, by training cases per hospital:

| Hospital size | Hospitals | Mean Δ | Improved |
| --- | --: | --: | --: |
| <=10 | 14 | +0.0125 | 10 |
| 11-30 | 7 | +0.0030 | 4 |
| >30 | 2 | +0.0009 | 2 |

## 3. Cases from hospitals holding one MRI sequence

| Setting | Method | vs fedavg | p | vs fedavg+ft | p | Cases |
| --- | --- | --: | --: | --: | --: | --: |
| modality | `norm_local` | +0.075 | 0.004 | -0.013 | 0.14 | 49 |
| modality | `in_local_1` | +0.076 | 0.037 | -0.011 | 0.29 | 49 |
| modality | `norm_local+ft` | +0.089 | 4.9e-05 | +0.001 | 0.99 | 49 |
| modality | `in_local_1+ft` | +0.084 | 0.027 | -0.004 | 0.53 | 49 |
| site_modality | `norm_local` | +0.056 | 0.044 | -0.020 | 0.19 | 49 |
| site_modality | `in_local_1` | +0.090 | 0.0044 | +0.014 | 0.9 | 49 |
| site_modality | `norm_local+ft` | +0.065 | 0.0016 | -0.011 | 0.99 | 49 |
| site_modality | `in_local_1+ft` | +0.103 | 0.0081 | +0.027 | 0.98 | 49 |

## 4. Δ Dice vs fedavg+ft by hospital size (≤10 / 11–30 / >30 training cases)

| Setting | norm | head | stem | encoder | decoder | local |
| --- | --: | --: | --: | --: | --: | --: |
| site | -0.011 / -0.002 / +0.001 | +0.013 / +0.003 / +0.000 | -0.060 / -0.016 / -0.001 | -0.096 / -0.030 / -0.001 | -0.123 / -0.021 / -0.001 | -0.210 / -0.065 / -0.002 |
| modality | -0.023 / -0.008 / +0.003 | -0.068 / -0.007 / -0.002 | -0.050 / -0.027 / +0.000 | -0.134 / -0.066 / +0.004 | -0.207 / -0.040 / +0.002 | -0.290 / -0.110 / +0.005 |
| site_modality | -0.033 / +0.002 / +0.004 | -0.100 / -0.018 / +0.001 | -0.030 / -0.011 / +0.002 | -0.059 / -0.037 / +0.008 | -0.173 / -0.024 / +0.000 | -0.198 / -0.069 / +0.007 |
