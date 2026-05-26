# Hypotheses

Because the benchmark has only five CV folds, p-values are exploratory. I report
effect sizes and confidence intervals together with test statistics.

## Main Hypothesis

**H0-main.** Pretrained audio encoders do not provide a reliable basis for whale
sound detection in long passive-acoustic recordings.

**H1-main.** Specialized pretrained audio encoders can provide useful whale
sound detectors, but reliable event-level detection requires more than good
window-level classification.

## Summary

Recompute all numbers from the saved report CSVs and long-file predictions:

```powershell
python -m filtering.analysis.hypothesis_tests
```

| ID | Sub-hypothesis | Statistical test | Main statistic | Decision |
|---|---|---|---:|---|
| H1 | Specialized encoders outperform the general Wav2Vec2 baseline in window ranking. | paired one-sided Student t-test and sign-permutation test on fold Window AP | t = 7.433, p = 0.0009 | Accepted |
| H2 | Specialized encoders reduce false positives compared with Wav2Vec2. | paired one-sided Student t-test on fold FPR | t = -2.911, p = 0.0218 | Accepted |
| H3 | High window AP is not sufficient for strong event-level detection. | Spearman correlation between Window AP and event metrics across model-folds | rho = 0.308, p = 0.186 | Accepted as limitation |
| H4 | MLP2 heads improve event F1 over linear heads. | paired Wilcoxon signed-rank test over matched folds | W = 107, p = 0.315 | Rejected |
| H5 | Benchmark model ranking transfers to the long-file review. | Spearman rank correlation and McNemar paired error test | rho = 1.000; chi2 = 86.118 | Accepted |

## H1: Specialized Encoders Improve Window Ranking

**Hypothesis.** Specialized bioacoustic/acoustic encoders should outperform the
general speech encoder Wav2Vec2 on whale `sound/noise` window ranking.

**Data.** Fold-level Window AP on `annotations_all`. For each fold, Wav2Vec2
was compared with the mean AP of Perch, Voxaboxen BEATs, and animal2vec.

**Test.** Paired one-sided t-test:

```text
d_i = AP_specialized_mean,i - AP_wav2vec2,i
t = mean(d) / (sd(d) / sqrt(n))
H0: mean(d) <= 0
H1: mean(d) > 0
```

An exact sign-permutation test was also computed because `n = 5` is small.

**Values.**

```text
n = 5 folds
d = [0.207, 0.166, 0.195, 0.346, 0.249]
mean(d) = 0.232
95% bootstrap CI = [0.186, 0.290]
t = 7.433
p_one-sided_t = 0.0009
p_one-sided_permutation = 0.0606
```

**Decision.** Accepted. The paired effect is positive in every fold. The
specialized encoders clearly provide stronger window-level ranking than the
general Wav2Vec2 baseline.

## H2: Specialized Encoders Reduce False Positives

**Hypothesis.** For a filtering task, specialized encoders should not only rank
sound windows higher, but also reduce false-positive noise windows relative to
Wav2Vec2.

**Data.** Fold-level false positive rate on `annotations_all`. For each fold, I
compared Wav2Vec2 with the mean FPR of the two strongest specialized baselines:
Perch and Voxaboxen BEATs.

**Test.** Paired one-sided t-test:

```text
d_i = FPR_specialized_mean,i - FPR_wav2vec2,i
t = mean(d) / (sd(d) / sqrt(n))
H0: mean(d) >= 0
H1: mean(d) < 0
```

**Values.**

```text
n = 5 folds
d = [-0.104, -0.002, -0.331, -0.147, -0.221]
mean(d) = -0.161
t = -2.911
p_one-sided_t = 0.0218
```

**Decision.** Accepted. Perch and Voxaboxen reduce false positives compared
with Wav2Vec2, which is important for long-recording filtering where false
alarms can dominate manual review time.

## H3: Window Ranking Is Not Enough for Event Detection

**Hypothesis.** Strong window-level ranking should not be treated as sufficient
evidence of strong event-level detection.

**Data.** All model-fold results on `annotations_all`, comparing Window AP with
event metrics after thresholding and temporal event matching.

**Test.** Spearman rank correlation:

```text
rho = corr(rank(Window AP), rank(Event metric))
H0: rho = 0
H1: rho != 0
```

**Values.**

```text
Window AP vs Event AP@0.5:
rho = 0.308
p_two-sided = 0.186

Window AP vs Event F1:
rho = -0.439
p_two-sided = 0.0528
```

**Decision.** Accepted as a limitation. The correlations do not support a clean
positive transfer from window ranking to event detection. This explains the main
tuning result: models can rank windows reasonably well while still failing to
produce reliable thresholded events.

## H4: MLP2 Heads Beat Linear Heads

**Hypothesis.** A two-layer MLP head should improve event detection over a
linear head on frozen embeddings.

**Data.** Matched fold-level event F1@0.5 for `linear` and `MLP2` heads across
four encoders and five folds each, giving 20 paired comparisons.

**Test.** One-sided paired Wilcoxon signed-rank test:

```text
d_i = F1_MLP2,i - F1_linear,i
H0: median(d) <= 0
H1: median(d) > 0
```

Wilcoxon is used because event F1 is bounded, small, and not obviously normal.

**Values.**

```text
n = 20 matched model-fold pairs
mean(d) = 0.0004
median(d) = 0.0032
95% bootstrap CI for mean(d) = [-0.0047, 0.0052]
Wilcoxon W = 107
p_one-sided = 0.315

mean difference by model:
Perch: +0.0031
Voxaboxen BEATs: +0.0020
animal2vec: -0.0018
Wav2Vec2: -0.0016
```

**Decision.** Rejected. The larger head does not materially improve event F1.
The failure mode is not solved by adding classifier capacity on top of frozen
embeddings.

## H5: Benchmark Ranking Transfers to Long-File Review

**Hypothesis.** Models that rank well in the CV benchmark should also rank well
on the 30-minute Orcasound inference review.

**Data.** Mean benchmark Window AP on `annotations_all` and long-file AP for
the same four baseline models. I also compared Perch and Voxaboxen paired
window correctness on the same long-file windows.

**Test 1.** Spearman rank correlation:

```text
rho = corr(rank(AP_benchmark), rank(AP_long_file))
H0: rho = 0
H1: rho != 0
```

**Values for rank transfer.**

```text
benchmark AP: Perch 0.806, Voxaboxen 0.801, animal2vec 0.623, Wav2Vec2 0.511
long-file AP: Perch 0.861, Voxaboxen 0.806, animal2vec 0.723, Wav2Vec2 0.587
Spearman rho = 1.000
p_two-sided = 0.000
```

**Test 2.** McNemar test for paired window errors between Perch and Voxaboxen:

```text
b = windows correct only for Perch
c = windows correct only for Voxaboxen
chi2 = (|b - c| - 1)^2 / (b + c)
H0: both models have the same paired error rate
```

**Values for paired window errors.**

```text
shared windows = 1784
b = 509
c = 252
chi2 = 86.118
p = 1.69e-20
```

**Decision.** Accepted for the baseline frozen models. The benchmark AP ranking
transfers to the long-file review, and Perch makes significantly fewer paired
long-file window errors than Voxaboxen at the chosen cutoff.

## Conclusion for the Main Hypothesis

The main hypothesis is supported with an important caveat. Specialized
pretrained encoders are useful for whale sound detection: they improve window
ranking, reduce false positives, and their benchmark ranking transfers to the
long-file review. However, event-level detection remains the main bottleneck:
strong window AP does not automatically produce strong event F1, and a larger
head on frozen embeddings does not solve this.
