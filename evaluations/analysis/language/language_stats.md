# Language Statistical Analysis

## Per-Language Summary

| Language | n | Mean | Median | SD | CI Lower | CI Upper |
|---|---|---|---|---|---|---|
| English | 6487 | 2.9631 | 2.8000 | 0.8411 | 2.9427 | 2.9832 |
| Mandarin | 6486 | 2.9973 | 2.8000 | 0.8762 | 2.9760 | 3.0186 |
| Hindi | 6485 | 2.9410 | 2.8000 | 0.9746 | 2.9175 | 2.9647 |
| Urdu | 6483 | 2.6936 | 2.6000 | 1.0414 | 2.6676 | 2.7185 |

## Omnibus Test (Friedman)

Friedman chi-square test across 4 languages (English, Mandarin, Hindi, Urdu), df = 3, N = 6467 matched (model, question) groups.

| Statistic | p-value | df | N groups |
|---|---|---|---|
| 715.8320 | 0.0000 | 3 | 6467 |

## Pairwise Comparisons

| Comparison | n pairs | Mean A | Mean B | Mean Diff | CI Lower | CI Upper | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|---|---|
| Mandarin vs English | 6473 | 2.9640 | 2.9984 | 0.0344 | 0.0140 | 0.0554 | 0.0000 | 0.0001 | 0.0645 |
| Hindi vs English | 6472 | 2.9640 | 2.9424 | -0.0216 | -0.0442 | 0.0014 | 0.3199 | 0.3199 | -0.0154 |
| Urdu vs English | 6470 | 2.9644 | 2.6948 | -0.2696 | -0.2945 | -0.2446 | 0.0000 | 0.0000 | -0.3003 |
| Hindi vs Mandarin | 6484 | 2.9971 | 2.9412 | -0.0559 | -0.0781 | -0.0340 | 0.0000 | 0.0000 | -0.0747 |
| Urdu vs Mandarin | 6482 | 2.9967 | 2.6938 | -0.3029 | -0.3266 | -0.2796 | 0.0000 | 0.0000 | -0.3857 |
| Urdu vs Hindi | 6481 | 2.9415 | 2.6934 | -0.2481 | -0.2681 | -0.2282 | 0.0000 | 0.0000 | -0.3740 |

## Model-Level Deltas (vs English)

| Model | Mandarin Δ | Hindi Δ | Urdu Δ | Avg |Δ| |
|---|---|---|---|---|
| llama-3.2-3b | -0.5318  | -0.7980  | -1.5106 | 0.9468 |
| llama-3-8b | -0.2333  | -0.5631  | -1.2012 | 0.6659 |
| llama-3.3-70b | -0.0749  | -0.1880  | -0.5126 | 0.2585 |
| gemma-3-4b | 0.1335  | 0.2465  | -0.4092 | 0.2631 |
| gemma-3-12b | 0.3295  | 0.4762  | 0.1347 | 0.3134 |
| gemma-4-31b | 0.2878  | 0.5888  | 0.5222 | 0.4663 |
| deepseek-v4-flash | -0.0216  | 0.0204  | -0.0485 | 0.0302 |
| qwen-3.1-8b | -0.0665  | -0.4333  | -0.6970 | 0.3989 |
| qwen-3.5-4b | 0.2465  | -0.2163  | -0.2433 | 0.2354 |
| qwen-3.6-27b | 0.0321  | 0.1463  | 0.1327 | 0.1037 |
| gemini-3.1-flash-lite | 0.1355  | 0.4184  | 0.3380 | 0.2973 |
| gpt-5-mini | -0.0004  | -0.1547  | -0.1551 | 0.1034 |
| claude-haiku-4.5 | 0.2068  | 0.1687  | 0.1475 | 0.1743 |

## Descriptive Highlights

The largest Urdu performance drop (relative to English) is observed for llama-3.2-3b (Δ = -1.5106). The largest Urdu gain is for gemma-4-31b (Δ = 0.5222). The most stable model across languages is deepseek-v4-flash (avg |Δ| = 0.0302). The largest average cross-lingual drop is seen for llama-3.2-3b (avg Δ = -0.9468). The largest average cross-lingual gain is seen for gemma-4-31b (avg Δ = 0.4663).
