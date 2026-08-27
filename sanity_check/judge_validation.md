# Judge Validation
**Judges:** claude-opus-4-7, deepseek-v4-pro, gemini-3.1-pro-preview, gemma-4-31b, gpt-5.5, llama-3.3-70b, qwen-3.6-27b  
**Metrics:** safety, developmental_fit, emotional_support, moral_guidance, boundary_setting  
**Valid evaluations per judge:**
- claude-opus-4-7: 100/100
- deepseek-v4-pro: 100/100
- gemini-3.1-pro-preview: 100/100
- gemma-4-31b: 100/100
- gpt-5.5: 100/100
- llama-3.3-70b: 100/100
- qwen-3.6-27b: 100/100

## Pairwise Summary
| Judge A | Judge B | Spearman | κ (wt.) | Bias (B−A) | Direction |
|---------|---------|----------|---------|------------|----------|
| gemini-3.1-pro-preview | gemma-4-31b | 0.801 | 0.743 | +0.264 | B > A |
| gemini-3.1-pro-preview | qwen-3.6-27b | 0.789 | 0.753 | +0.226 | B > A |
| gemma-4-31b | qwen-3.6-27b | 0.782 | 0.788 | -0.038 | ~equal |
| claude-opus-4-7 | gpt-5.5 | 0.739 | 0.741 | +0.046 | ~equal |
| claude-opus-4-7 | gemma-4-31b | 0.719 | 0.696 | +0.240 | B > A |
| claude-opus-4-7 | qwen-3.6-27b | 0.719 | 0.697 | +0.202 | B > A |
| deepseek-v4-pro | gemma-4-31b | 0.718 | 0.647 | +0.340 | B > A |
| claude-opus-4-7 | gemini-3.1-pro-preview | 0.713 | 0.699 | -0.024 | ~equal |
| deepseek-v4-pro | qwen-3.6-27b | 0.710 | 0.657 | +0.302 | B > A |
| deepseek-v4-pro | gemini-3.1-pro-preview | 0.709 | 0.704 | +0.076 | ~equal |
| gemini-3.1-pro-preview | gpt-5.5 | 0.700 | 0.671 | +0.070 | ~equal |
| llama-3.3-70b | qwen-3.6-27b | 0.681 | 0.643 | -0.170 | A > B |
| deepseek-v4-pro | llama-3.3-70b | 0.665 | 0.535 | +0.472 | B > A |
| gemma-4-31b | llama-3.3-70b | 0.665 | 0.666 | +0.132 | B > A |
| claude-opus-4-7 | deepseek-v4-pro | 0.664 | 0.660 | -0.100 | A > B |
| deepseek-v4-pro | gpt-5.5 | 0.661 | 0.635 | +0.146 | B > A |
| gemma-4-31b | gpt-5.5 | 0.660 | 0.642 | -0.194 | A > B |
| gpt-5.5 | qwen-3.6-27b | 0.657 | 0.633 | +0.156 | B > A |
| claude-opus-4-7 | llama-3.3-70b | 0.651 | 0.574 | +0.372 | B > A |
| gpt-5.5 | llama-3.3-70b | 0.614 | 0.566 | +0.326 | B > A |
| gemini-3.1-pro-preview | llama-3.3-70b | 0.614 | 0.528 | +0.396 | B > A |

## Avg Spearman Matrix

| | claude-opus-4-7 | deepseek-v4-pro | gemini-3.1-pro-preview | gemma-4-31b | gpt-5.5 | llama-3.3-70b | qwen-3.6-27b |
|---|---|---|---|---|---|---|---|
| **claude-opus-4-7** | — | 0.664 | 0.713 | 0.719 | 0.739 | 0.651 | 0.719 |
| **deepseek-v4-pro** | 0.664 | — | 0.709 | 0.718 | 0.661 | 0.665 | 0.710 |
| **gemini-3.1-pro-preview** | 0.713 | 0.709 | — | 0.801 | 0.700 | 0.614 | 0.789 |
| **gemma-4-31b** | 0.719 | 0.718 | 0.801 | — | 0.660 | 0.665 | 0.782 |
| **gpt-5.5** | 0.739 | 0.661 | 0.700 | 0.660 | — | 0.614 | 0.657 |
| **llama-3.3-70b** | 0.651 | 0.665 | 0.614 | 0.665 | 0.614 | — | 0.681 |
| **qwen-3.6-27b** | 0.719 | 0.710 | 0.789 | 0.782 | 0.657 | 0.681 | — |

## Avg Weighted κ Matrix

| | claude-opus-4-7 | deepseek-v4-pro | gemini-3.1-pro-preview | gemma-4-31b | gpt-5.5 | llama-3.3-70b | qwen-3.6-27b |
|---|---|---|---|---|---|---|---|
| **claude-opus-4-7** | — | 0.660 | 0.699 | 0.696 | 0.741 | 0.574 | 0.697 |
| **deepseek-v4-pro** | 0.660 | — | 0.704 | 0.647 | 0.635 | 0.535 | 0.657 |
| **gemini-3.1-pro-preview** | 0.699 | 0.704 | — | 0.743 | 0.671 | 0.528 | 0.753 |
| **gemma-4-31b** | 0.696 | 0.647 | 0.743 | — | 0.642 | 0.666 | 0.788 |
| **gpt-5.5** | 0.741 | 0.635 | 0.671 | 0.642 | — | 0.566 | 0.633 |
| **llama-3.3-70b** | 0.574 | 0.535 | 0.528 | 0.666 | 0.566 | — | 0.643 |
| **qwen-3.6-27b** | 0.697 | 0.657 | 0.753 | 0.788 | 0.633 | 0.643 | — |

## Avg Bias Matrix (+ means col scores higher)

| | claude-opus-4-7 | deepseek-v4-pro | gemini-3.1-pro-preview | gemma-4-31b | gpt-5.5 | llama-3.3-70b | qwen-3.6-27b |
|---|---|---|---|---|---|---|---|
| **claude-opus-4-7** | — | -0.100 | -0.024 | +0.240 | +0.046 | +0.372 | +0.202 |
| **deepseek-v4-pro** | +0.100 | — | +0.076 | +0.340 | +0.146 | +0.472 | +0.302 |
| **gemini-3.1-pro-preview** | +0.024 | -0.076 | — | +0.264 | +0.070 | +0.396 | +0.226 |
| **gemma-4-31b** | -0.240 | -0.340 | -0.264 | — | -0.194 | +0.132 | -0.038 |
| **gpt-5.5** | -0.046 | -0.146 | -0.070 | +0.194 | — | +0.326 | +0.156 |
| **llama-3.3-70b** | -0.372 | -0.472 | -0.396 | -0.132 | -0.326 | — | -0.170 |
| **qwen-3.6-27b** | -0.202 | -0.302 | -0.226 | +0.038 | -0.156 | +0.170 | — |

## Per-Metric Breakdown

### claude-opus-4-7 vs deepseek-v4-pro

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.469 | 0.466 | -0.150 | 100 |
| developmental_fit | 0.842 | 0.829 | -0.120 | 100 |
| emotional_support | 0.683 | 0.702 | +0.010 | 100 |
| moral_guidance | 0.662 | 0.660 | -0.140 | 100 |
| boundary_setting | 0.664 | 0.645 | -0.100 | 100 |

### claude-opus-4-7 vs gemini-3.1-pro-preview

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.591 | 0.527 | -0.060 | 100 |
| developmental_fit | 0.820 | 0.804 | -0.140 | 100 |
| emotional_support | 0.724 | 0.740 | +0.020 | 100 |
| moral_guidance | 0.739 | 0.758 | -0.040 | 100 |
| boundary_setting | 0.688 | 0.666 | +0.100 | 100 |

### claude-opus-4-7 vs gemma-4-31b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.501 | 0.509 | +0.190 | 100 |
| developmental_fit | 0.838 | 0.800 | +0.230 | 100 |
| emotional_support | 0.757 | 0.692 | +0.360 | 100 |
| moral_guidance | 0.757 | 0.772 | +0.200 | 100 |
| boundary_setting | 0.744 | 0.709 | +0.220 | 100 |

### claude-opus-4-7 vs gpt-5.5

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.635 | 0.665 | +0.000 | 100 |
| developmental_fit | 0.819 | 0.790 | +0.090 | 100 |
| emotional_support | 0.800 | 0.791 | +0.120 | 100 |
| moral_guidance | 0.738 | 0.757 | -0.020 | 100 |
| boundary_setting | 0.704 | 0.704 | +0.040 | 100 |

### claude-opus-4-7 vs llama-3.3-70b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.580 | 0.539 | +0.220 | 100 |
| developmental_fit | 0.722 | 0.589 | +0.470 | 100 |
| emotional_support | 0.727 | 0.635 | +0.410 | 100 |
| moral_guidance | 0.646 | 0.658 | +0.280 | 100 |
| boundary_setting | 0.582 | 0.451 | +0.480 | 100 |

### claude-opus-4-7 vs qwen-3.6-27b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.500 | 0.455 | +0.220 | 100 |
| developmental_fit | 0.854 | 0.847 | +0.120 | 100 |
| emotional_support | 0.754 | 0.720 | +0.280 | 100 |
| moral_guidance | 0.740 | 0.758 | +0.160 | 100 |
| boundary_setting | 0.747 | 0.703 | +0.230 | 100 |

### deepseek-v4-pro vs gemini-3.1-pro-preview

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.552 | 0.555 | +0.090 | 100 |
| developmental_fit | 0.783 | 0.783 | -0.020 | 100 |
| emotional_support | 0.813 | 0.815 | +0.010 | 100 |
| moral_guidance | 0.685 | 0.683 | +0.100 | 100 |
| boundary_setting | 0.711 | 0.684 | +0.200 | 100 |

### deepseek-v4-pro vs gemma-4-31b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.515 | 0.453 | +0.340 | 100 |
| developmental_fit | 0.813 | 0.720 | +0.350 | 100 |
| emotional_support | 0.776 | 0.704 | +0.350 | 100 |
| moral_guidance | 0.769 | 0.698 | +0.340 | 100 |
| boundary_setting | 0.717 | 0.658 | +0.320 | 100 |

### deepseek-v4-pro vs gpt-5.5

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.479 | 0.466 | +0.150 | 100 |
| developmental_fit | 0.742 | 0.685 | +0.210 | 100 |
| emotional_support | 0.688 | 0.683 | +0.110 | 100 |
| moral_guidance | 0.730 | 0.711 | +0.120 | 100 |
| boundary_setting | 0.664 | 0.628 | +0.140 | 100 |

### deepseek-v4-pro vs llama-3.3-70b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.606 | 0.505 | +0.370 | 100 |
| developmental_fit | 0.734 | 0.524 | +0.590 | 100 |
| emotional_support | 0.717 | 0.620 | +0.400 | 100 |
| moral_guidance | 0.698 | 0.612 | +0.420 | 100 |
| boundary_setting | 0.567 | 0.414 | +0.580 | 100 |

### deepseek-v4-pro vs qwen-3.6-27b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.512 | 0.441 | +0.370 | 100 |
| developmental_fit | 0.830 | 0.790 | +0.240 | 100 |
| emotional_support | 0.750 | 0.715 | +0.270 | 100 |
| moral_guidance | 0.723 | 0.671 | +0.300 | 100 |
| boundary_setting | 0.735 | 0.667 | +0.330 | 100 |

### gemini-3.1-pro-preview vs gemma-4-31b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.726 | 0.634 | +0.250 | 100 |
| developmental_fit | 0.787 | 0.688 | +0.370 | 100 |
| emotional_support | 0.818 | 0.736 | +0.340 | 100 |
| moral_guidance | 0.756 | 0.748 | +0.240 | 100 |
| boundary_setting | 0.919 | 0.907 | +0.120 | 100 |

### gemini-3.1-pro-preview vs gpt-5.5

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.703 | 0.650 | +0.060 | 100 |
| developmental_fit | 0.766 | 0.697 | +0.230 | 100 |
| emotional_support | 0.719 | 0.706 | +0.100 | 100 |
| moral_guidance | 0.667 | 0.687 | +0.020 | 100 |
| boundary_setting | 0.646 | 0.613 | -0.060 | 100 |

### gemini-3.1-pro-preview vs llama-3.3-70b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.530 | 0.459 | +0.280 | 100 |
| developmental_fit | 0.631 | 0.452 | +0.610 | 100 |
| emotional_support | 0.735 | 0.652 | +0.390 | 100 |
| moral_guidance | 0.630 | 0.609 | +0.320 | 100 |
| boundary_setting | 0.542 | 0.468 | +0.380 | 100 |

### gemini-3.1-pro-preview vs qwen-3.6-27b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.670 | 0.610 | +0.280 | 100 |
| developmental_fit | 0.864 | 0.810 | +0.260 | 100 |
| emotional_support | 0.796 | 0.748 | +0.260 | 100 |
| moral_guidance | 0.759 | 0.751 | +0.200 | 100 |
| boundary_setting | 0.858 | 0.848 | +0.130 | 100 |

### gemma-4-31b vs gpt-5.5

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.519 | 0.509 | -0.190 | 100 |
| developmental_fit | 0.729 | 0.703 | -0.140 | 100 |
| emotional_support | 0.720 | 0.705 | -0.240 | 100 |
| moral_guidance | 0.695 | 0.690 | -0.220 | 100 |
| boundary_setting | 0.638 | 0.602 | -0.180 | 100 |

### gemma-4-31b vs llama-3.3-70b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.484 | 0.505 | +0.030 | 100 |
| developmental_fit | 0.677 | 0.662 | +0.240 | 100 |
| emotional_support | 0.798 | 0.816 | +0.050 | 100 |
| moral_guidance | 0.740 | 0.775 | +0.080 | 100 |
| boundary_setting | 0.624 | 0.572 | +0.260 | 100 |

### gemma-4-31b vs qwen-3.6-27b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.530 | 0.535 | +0.030 | 100 |
| developmental_fit | 0.838 | 0.824 | -0.110 | 100 |
| emotional_support | 0.877 | 0.872 | -0.080 | 100 |
| moral_guidance | 0.778 | 0.815 | -0.040 | 100 |
| boundary_setting | 0.888 | 0.896 | +0.010 | 100 |

### gpt-5.5 vs llama-3.3-70b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.574 | 0.510 | +0.220 | 100 |
| developmental_fit | 0.636 | 0.583 | +0.380 | 100 |
| emotional_support | 0.660 | 0.636 | +0.290 | 100 |
| moral_guidance | 0.667 | 0.663 | +0.300 | 100 |
| boundary_setting | 0.533 | 0.437 | +0.440 | 100 |

### gpt-5.5 vs qwen-3.6-27b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.551 | 0.510 | +0.220 | 100 |
| developmental_fit | 0.715 | 0.678 | +0.030 | 100 |
| emotional_support | 0.703 | 0.704 | +0.160 | 100 |
| moral_guidance | 0.676 | 0.676 | +0.180 | 100 |
| boundary_setting | 0.642 | 0.597 | +0.190 | 100 |

### llama-3.3-70b vs qwen-3.6-27b

| Metric | Spearman | κ (wt.) | Bias | n |
|--------|----------|---------|------|---|
| safety | 0.540 | 0.496 | +0.000 | 100 |
| developmental_fit | 0.714 | 0.635 | -0.350 | 100 |
| emotional_support | 0.777 | 0.755 | -0.130 | 100 |
| moral_guidance | 0.743 | 0.749 | -0.120 | 100 |
| boundary_setting | 0.633 | 0.580 | -0.250 | 100 |
