# Cultural Alignment Statistical Analysis

## Per-Country Summary

| Country | n | Mean | Median | SD | CI Lower | CI Upper |
|---|---|---|---|---|---|---|
| Pakistan | 6414 | 3.4486 | 3.0000 | 1.1840 | 3.4188 | 3.4772 |
| India | 6441 | 3.4394 | 3.0000 | 1.1098 | 3.4114 | 3.4664 |
| China | 6432 | 3.6015 | 4.0000 | 1.1931 | 3.5728 | 3.6309 |
| Nigeria | 6398 | 4.2454 | 5.0000 | 0.9959 | 4.2210 | 4.2701 |

## Omnibus Test (Friedman)

Friedman test across all 4 countries (['Pakistan', 'India', 'China', 'Nigeria']), grouped by (model, qid) requiring all countries present.

- N blocks (matched quadruples): 6252
- Degrees of freedom: 3
- Statistic: 4025.9685
- p-value: 0.0000

## Pairwise Comparisons

Mean Diff = Mean B − Mean A. Holm correction applied across all 6 pairwise p-values.

| Comparison | n pairs | Mean A | Mean B | Mean Diff | CI Lower | CI Upper | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|---|---|
| India vs Pakistan | 6374 | 3.4437 | 3.4327 | -0.0110 | -0.0353 | 0.0135 | 0.2172 | 0.2172 | -0.0243 |
| China vs Pakistan | 6363 | 3.4479 | 3.5948 | 0.1469 | 0.1182 | 0.1756 | 0.0000 | 0.0000 | 0.1831 |
| China vs India | 6390 | 3.4393 | 3.5991 | 0.1598 | 0.1311 | 0.1881 | 0.0000 | 0.0000 | 0.2050 |
| Nigeria vs Pakistan | 6333 | 3.4478 | 4.2413 | 0.7935 | 0.7688 | 0.8190 | 0.0000 | 0.0000 | 0.8549 |
| Nigeria vs India | 6356 | 3.4399 | 4.2428 | 0.8029 | 0.7780 | 0.8277 | 0.0000 | 0.0000 | 0.8419 |
| Nigeria vs China | 6345 | 3.6017 | 4.2449 | 0.6432 | 0.6147 | 0.6714 | 0.0000 | 0.0000 | 0.7215 |

## Model-Level Summary

| Model | Pakistan | India | China | Nigeria | Overall |
|---|---|---|---|---|---|
| llama-3.2-3b | 2.5571 | 2.6573 | 3.1210 | 3.3931 | 2.9312 |
| llama-3-8b | 2.6426 | 2.7550 | 3.2646 | 3.4930 | 3.0387 |
| llama-3.3-70b | 3.0120 | 3.0763 | 3.4467 | 3.9177 | 3.3630 |
| gemma-3-4b | 3.2526 | 3.2618 | 3.2790 | 4.0063 | 3.4463 |
| gemma-3-12b | 4.0365 | 3.7773 | 3.9715 | 4.6189 | 4.0996 |
| gemma-4-31b | 3.8136 | 3.8252 | 3.7134 | 4.6265 | 3.9960 |
| deepseek-v4-flash | 3.7864 | 3.6714 | 3.7344 | 4.5918 | 3.9434 |
| qwen-3.1-8b | 2.9840 | 3.1222 | 3.4357 | 3.9095 | 3.3623 |
| qwen-3.5-4b | 3.1761 | 3.2923 | 3.4472 | 4.0569 | 3.4924 |
| qwen-3.6-27b | 4.2265 | 4.4116 | 4.2465 | 4.8778 | 4.4406 |
| gemini-3.1-flash-lite | 3.9363 | 3.8354 | 3.8139 | 4.6876 | 4.0694 |
| gpt-5-mini | 3.8818 | 3.6373 | 3.8297 | 4.5511 | 3.9749 |
| claude-haiku-4.5 | 3.5460 | 3.4008 | 3.5131 | 4.4718 | 3.7287 |

## Descriptive Highlights

- **Best model overall**: qwen-3.6-27b (mean cultural_alignment = 4.4406)
- **Models scoring ≥ 4.0 in ALL 4 countries**: qwen-3.6-27b
- **Lowest-performing model for Pakistan**: llama-3.2-3b (mean = 2.5571)
- **Lowest-performing model for India**: llama-3.2-3b (mean = 2.6573)
- **Country where each model scores highest**:
  - llama-3.2-3b: Nigeria (mean = 3.3931)
  - llama-3-8b: Nigeria (mean = 3.4930)
  - llama-3.3-70b: Nigeria (mean = 3.9177)
  - gemma-3-4b: Nigeria (mean = 4.0063)
  - gemma-3-12b: Nigeria (mean = 4.6189)
  - gemma-4-31b: Nigeria (mean = 4.6265)
  - deepseek-v4-flash: Nigeria (mean = 4.5918)
  - qwen-3.1-8b: Nigeria (mean = 3.9095)
  - qwen-3.5-4b: Nigeria (mean = 4.0569)
  - qwen-3.6-27b: Nigeria (mean = 4.8778)
  - gemini-3.1-flash-lite: Nigeria (mean = 4.6876)
  - gpt-5-mini: Nigeria (mean = 4.5511)
  - claude-haiku-4.5: Nigeria (mean = 4.4718)
