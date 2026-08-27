# Multi-Turn Statistical Analysis

## 1. Mean Quality per Turn

Mean Q = mean of (safety, developmental_fit, emotional_support, moral_guidance, boundary_setting). Showing mean per turn. 95% bootstrap CI based on 10,000 resamples.

### Without Age

| Model | T1 mean | T2 mean | T3 mean | T4 mean | T5 mean |
|---|---|---|---|---|---|
| llama-3.2-3b | 3.5111 | 3.0040 | 2.9196 | 2.7816 | 2.7204 |
| llama-3.1-8b | 3.3180 | 3.0703 | 2.9656 | 2.8702 | 2.8000 |
| llama-3.3-70b | 3.9455 | 3.6227 | 3.4124 | 3.2694 | 3.1327 |
| gemma-3-4b | 3.3820 | 3.1844 | 3.0368 | 2.8432 | 2.8364 |
| gemma-3-12b | 3.6701 | 3.7371 | 3.7156 | 3.7191 | 3.6432 |
| gemma-4-31b | 4.2420 | 4.2833 | 4.2825 | 4.3388 | 4.3224 |
| deepseek-v4-flash | 4.3313 | 4.1900 | 4.1414 | 4.0707 | 4.0929 |
| qwen-3-8b | 3.4980 | 3.4687 | 3.3531 | 3.2102 | 3.0796 |
| qwen-3.6-27b | 4.3697 | 4.3122 | 4.2347 | 4.1898 | 4.1327 |
| gemini-3.1-flash-lite | 4.3560 | 4.4124 | 4.4143 | 4.3102 | 4.3673 |
| gpt-5-mini | 3.8586 | 3.8447 | 3.8532 | 3.8809 | 3.8358 |
| claude-haiku-4-5 | 4.1260 | 4.1640 | 4.3160 | 4.2400 | 4.2920 |

### With Age

| Model | T1 mean | T2 mean | T3 mean | T4 mean | T5 mean |
|---|---|---|---|---|---|
| llama-3.2-3b | 4.0880 | 3.6263 | 3.4869 | 3.3273 | 3.1293 |
| llama-3.1-8b | 3.9560 | 3.6688 | 3.5347 | 3.4898 | 3.3959 |
| llama-3.3-70b | 4.4880 | 4.3224 | 4.0082 | 3.8265 | 3.7051 |
| gemma-3-4b | 4.3660 | 4.1000 | 3.9429 | 3.7653 | 3.6918 |
| gemma-3-12b | 4.7280 | 4.6780 | 4.6760 | 4.5800 | 4.5380 |
| gemma-4-31b | 4.9040 | 4.8760 | 4.8840 | 4.8780 | 4.8500 |
| deepseek-v4-flash | 4.8240 | 4.7880 | 4.7220 | 4.7420 | 4.7640 |
| qwen-3-8b | 4.4560 | 4.2898 | 4.1196 | 3.9052 | 3.8479 |
| qwen-3.6-27b | 4.9220 | 4.8380 | 4.8240 | 4.8000 | 4.7900 |
| gemini-3.1-flash-lite | 4.9060 | 4.7920 | 4.7220 | 4.7200 | 4.7460 |
| gpt-5-mini | 4.7580 | 4.7374 | 4.7394 | 4.7131 | 4.7152 |
| claude-haiku-4-5 | 4.8160 | 4.7620 | 4.7140 | 4.7340 | 4.6760 |

## 2. Degradation Slope (per model)

Mixed-effects model: Q ~ turn, groups=conv_id (OLS fallback if MixedLM fails). D_slope = -beta_1 (positive = degradation over turns). CI is 95% for D_slope.

### Without Age

| Model | beta_1 | D_slope | SE | CI Lower | CI Upper | p-value |
|---|---|---|---|---|---|---|
| llama-3.2-3b | -0.1816 | 0.1816 | 0.0214 | 0.1396 | 0.2236 | 0.0000 |
| llama-3.1-8b | -0.1252 | 0.1252 | 0.0212 | 0.0837 | 0.1667 | 0.0000 |
| llama-3.3-70b | -0.2000 | 0.2000 | 0.0215 | 0.1579 | 0.2421 | 0.0000 |
| gemma-3-4b | -0.1505 | 0.1505 | 0.0171 | 0.1170 | 0.1840 | 0.0000 |
| gemma-3-12b | -0.0197 | 0.0197 | 0.0150 | -0.0096 | 0.0490 | 0.1877 |
| gemma-4-31b | 0.0156 | -0.0156 | 0.0144 | -0.0439 | 0.0127 | 0.2790 |
| deepseek-v4-flash | -0.0590 | 0.0590 | 0.0176 | 0.0245 | 0.0934 | 0.0008 |
| qwen-3-8b | -0.1096 | 0.1096 | 0.0189 | 0.0726 | 0.1467 | 0.0000 |
| qwen-3.6-27b | -0.0606 | 0.0606 | 0.0149 | 0.0314 | 0.0899 | 0.0000 |
| gemini-3.1-flash-lite | -0.0134 | 0.0134 | 0.0136 | -0.0132 | 0.0400 | 0.3234 |
| gpt-5-mini | -0.0051 | 0.0051 | 0.0126 | -0.0195 | 0.0297 | 0.6819 |
| claude-haiku-4-5 | 0.0408 | -0.0408 | 0.0150 | -0.0702 | -0.0114 | 0.0065 |

### With Age

| Model | beta_1 | D_slope | SE | CI Lower | CI Upper | p-value |
|---|---|---|---|---|---|---|
| llama-3.2-3b | -0.2238 | 0.2238 | 0.0217 | 0.1813 | 0.2663 | 0.0000 |
| llama-3.1-8b | -0.1343 | 0.1343 | 0.0197 | 0.0958 | 0.1729 | 0.0000 |
| llama-3.3-70b | -0.2088 | 0.2088 | 0.0206 | 0.1683 | 0.2492 | 0.0000 |
| gemma-3-4b | -0.1678 | 0.1678 | 0.0161 | 0.1363 | 0.1993 | 0.0000 |
| gemma-3-12b | -0.0478 | 0.0478 | 0.0125 | 0.0233 | 0.0723 | 0.0001 |
| gemma-4-31b | -0.0106 | 0.0106 | 0.0077 | -0.0045 | 0.0257 | 0.1700 |
| deepseek-v4-flash | -0.0166 | 0.0166 | 0.0094 | -0.0017 | 0.0349 | 0.0761 |
| qwen-3-8b | -0.1640 | 0.1640 | 0.0186 | 0.1276 | 0.2005 | 0.0000 |
| qwen-3.6-27b | -0.0302 | 0.0302 | 0.0079 | 0.0148 | 0.0456 | 0.0001 |
| gemini-3.1-flash-lite | -0.0392 | 0.0392 | 0.0103 | 0.0191 | 0.0593 | 0.0001 |
| gpt-5-mini | -0.0149 | 0.0149 | 0.0078 | -0.0005 | 0.0302 | 0.0575 |
| claude-haiku-4-5 | -0.0308 | 0.0308 | 0.0110 | 0.0092 | 0.0524 | 0.0052 |

## 3. Overall Age-Setting Interaction

Model: Q ~ turn * C(age_setting), mixed effects with (1 | model) as grouping variable. Method: MixedLM.
age_setting is coded 0 = without_age, 1 = with_age.

| Effect | Coefficient | SE | p-value |
|---|---|---|---|
| Main effect of turn (beta_turn) | -0.0684 | 0.0097 | 0.0000 |
| Main effect of age_setting (C(age_setting)[T.1]) | 0.7245 | 0.0452 | 0.0000 |
| turn × age_setting (turn:C(age_setting)[T.1]) | -0.0208 | 0.0136 | 0.1273 |

- **D_slope (without_age)** = -beta_turn = 0.0684
- **D_slope (with_age)** = -(beta_turn + beta_inter) = 0.0892
- **Slopes differ between age settings** (interaction p = 0.1273): NO (p >= 0.05)

## 4. Peak Quality Drop

peak_drop = Q_T1 - min(Q_T2, Q_T3, Q_T4, Q_T5). Skipped if T1 or all of T2-T5 are missing. 95% bootstrap CI for mean peak_drop. Proportions are percentages of conversations.

### Without Age

| Model | Mean | Median | CI Lower | CI Upper | >0.25 (%) | >0.50 (%) |
|---|---|---|---|---|---|---|
| llama-3.2-3b | 1.0444 | 0.8000 | 0.8384 | 1.2505 | 68.69 | 66.67 |
| llama-3.1-8b | 0.8000 | 0.6000 | 0.6126 | 0.9937 | 66.32 | 51.58 |
| llama-3.3-70b | 1.0103 | 0.8000 | 0.7835 | 1.2433 | 59.79 | 57.73 |
| gemma-3-4b | 0.7355 | 0.6000 | 0.5462 | 0.9290 | 59.14 | 52.69 |
| gemma-3-12b | 0.2374 | 0.0000 | 0.0637 | 0.4110 | 31.87 | 26.37 |
| gemma-4-31b | 0.1633 | 0.0000 | -0.0000 | 0.3286 | 23.47 | 20.41 |
| deepseek-v4-flash | 0.3980 | 0.0000 | 0.1879 | 0.6242 | 26.26 | 24.24 |
| qwen-3-8b | 0.5816 | 0.4000 | 0.3878 | 0.7857 | 51.02 | 45.92 |
| qwen-3.6-27b | 0.3612 | 0.0000 | 0.1898 | 0.5510 | 26.53 | 23.47 |
| gemini-3.1-flash-lite | 0.1857 | 0.0000 | 0.0224 | 0.3694 | 16.33 | 16.33 |
| gpt-5-mini | 0.2042 | 0.0000 | 0.0800 | 0.3368 | 28.42 | 22.11 |
| claude-haiku-4-5 | 0.1220 | 0.0000 | -0.0280 | 0.2720 | 22.00 | 21.00 |

### With Age

| Model | Mean | Median | CI Lower | CI Upper | >0.25 (%) | >0.50 (%) |
|---|---|---|---|---|---|---|
| llama-3.2-3b | 1.2061 | 1.2000 | 0.9899 | 1.4222 | 73.74 | 72.73 |
| llama-3.1-8b | 0.8163 | 0.6000 | 0.6367 | 1.0041 | 56.12 | 54.08 |
| llama-3.3-70b | 0.9273 | 0.0000 | 0.6969 | 1.1697 | 43.43 | 41.41 |
| gemma-3-4b | 0.7596 | 0.0000 | 0.5757 | 0.9596 | 48.48 | 44.44 |
| gemma-3-12b | 0.2660 | 0.0000 | 0.1200 | 0.4320 | 19.00 | 15.00 |
| gemma-4-31b | 0.1020 | 0.0000 | 0.0180 | 0.2080 | 5.00 | 5.00 |
| deepseek-v4-flash | 0.1500 | 0.0000 | 0.0280 | 0.2880 | 12.00 | 12.00 |
| qwen-3-8b | 0.7694 | 0.0000 | 0.5735 | 0.9715 | 47.96 | 42.86 |
| qwen-3.6-27b | 0.1600 | 0.0000 | 0.0600 | 0.2840 | 9.00 | 9.00 |
| gemini-3.1-flash-lite | 0.2440 | 0.0000 | 0.1120 | 0.4020 | 13.00 | 13.00 |
| gpt-5-mini | 0.1111 | 0.0000 | 0.0222 | 0.2162 | 12.12 | 11.11 |
| claude-haiku-4-5 | 0.2580 | 0.0000 | 0.1360 | 0.3900 | 18.00 | 18.00 |

## 5. Peak Drop: With Age vs Without Age

Matched by (model, conv_id). Mean Diff = mean(peak_drop_with_age) - mean(peak_drop_without_age). Wilcoxon signed-rank test (two-sided). 95% bootstrap CI for mean difference.

- **N matched pairs**: 1154
- **Mean peak_drop (without_age)**: 0.4816
- **Mean peak_drop (with_age)**: 0.4750
- **Mean difference (with_age − without_age)**: -0.0066
- **95% bootstrap CI for mean difference**: [-0.0702, 0.0575]
- **Wilcoxon p (two-sided)**: 0.9160
