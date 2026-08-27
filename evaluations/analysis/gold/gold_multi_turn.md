# Gold Model — Multi-Turn Analysis

## Quality Degradation Slope (β₁) — Per Metric — Without Age

| Model | Safety | Dev. Fit | Emotional | Moral | Boundary | Total |
|---|---|---|---|---|---|---|
| LlamaPlushie SFT-1 | 0.048 | 0.088 | 0.068 | 0.062 | 0.056 | 0.065 |
| LlamaPlushie SFT-2 | 0.011 | 0.037 | 0.032 | 0.008 | 0.013 | 0.020 |
| LlamaPlushie SFT-3 | 0.026 | 0.038 | 0.030 | 0.027 | 0.018 | 0.028 |
| LlamaPlushie GRPO | 0.022 | 0.051 | 0.035 | 0.025 | 0.017 | 0.030 |

## Quality Degradation Slope (β₁) — Per Category — Without Age

| Model | Sexual | Self-Harm | Aggression | Moral | Health | School | Family | Online | Hate/Bias | Benign |
|---|---|---|---|---|---|---|---|---|---|---|
| LlamaPlushie SFT-1 | 0.240 | 0.082 | 0.026 | -0.020 | 0.012 | -0.000 | 0.068 | 0.006 | 0.160 | 0.074 |
| LlamaPlushie SFT-2 | 0.077 | 0.036 | 0.020 | -0.058 | -0.000 | -0.000 | 0.018 | 0.074 | 0.010 | 0.038 |
| LlamaPlushie SFT-3 | 0.044 | 0.006 | 0.030 | 0.014 | 0.064 | -0.000 | 0.086 | -0.000 | 0.034 | -0.000 |
| LlamaPlushie GRPO | 0.122 | 0.062 | -0.000 | -0.024 | 0.052 | -0.008 | -0.000 | -0.000 | 0.050 | 0.046 |

## Peak Degradation (T1 − min T2..T5) — Without Age

| Model | T1 | Lowest Turn | Lowest Score | Delta |
|---|---|---|---|---|
| LlamaPlushie SFT-1 | 4.880 | 5 | 4.628 | 0.252 |
| LlamaPlushie SFT-2 | 4.910 | 4 | 4.828 | 0.082 |
| LlamaPlushie SFT-3 | 4.924 | 5 | 4.796 | 0.128 |
| LlamaPlushie GRPO | 4.908 | 5 | 4.786 | 0.122 |

## Quality Degradation Slope (β₁) — Per Metric — With Age

| Model | Safety | Dev. Fit | Emotional | Moral | Boundary | Total |
|---|---|---|---|---|---|---|
| LlamaPlushie SFT-1 | 0.024 | 0.079 | 0.062 | 0.044 | 0.033 | 0.048 |
| LlamaPlushie SFT-2 | 0.028 | 0.057 | 0.040 | 0.043 | 0.032 | 0.040 |
| LlamaPlushie SFT-3 | 0.010 | 0.026 | 0.024 | 0.010 | 0.005 | 0.015 |
| LlamaPlushie GRPO | 0.015 | 0.033 | 0.038 | 0.008 | 0.016 | 0.022 |

## Quality Degradation Slope (β₁) — Per Category — With Age

| Model | Sexual | Self-Harm | Aggression | Moral | Health | School | Family | Online | Hate/Bias | Benign |
|---|---|---|---|---|---|---|---|---|---|---|
| LlamaPlushie SFT-1 | 0.138 | 0.056 | 0.008 | 0.074 | 0.086 | 0.060 | -0.016 | 0.024 | 0.014 | 0.040 |
| LlamaPlushie SFT-2 | 0.100 | 0.058 | -0.000 | 0.016 | 0.062 | 0.008 | 0.052 | 0.044 | 0.020 | 0.040 |
| LlamaPlushie SFT-3 | 0.058 | 0.030 | -0.000 | -0.000 | -0.012 | -0.000 | -0.000 | -0.004 | 0.026 | 0.052 |
| LlamaPlushie GRPO | 0.108 | 0.076 | 0.012 | -0.000 | 0.024 | -0.020 | 0.012 | -0.000 | -0.000 | 0.008 |

## Peak Degradation (T1 − min T2..T5) — With Age

| Model | T1 | Lowest Turn | Lowest Score | Delta |
|---|---|---|---|---|
| LlamaPlushie SFT-1 | 4.932 | 5 | 4.718 | 0.214 |
| LlamaPlushie SFT-2 | 4.930 | 5 | 4.772 | 0.158 |
| LlamaPlushie SFT-3 | 4.954 | 2 | 4.862 | 0.092 |
| LlamaPlushie GRPO | 4.936 | 5 | 4.828 | 0.108 |

