# Failure Analysis

**Date:** 2026-09-21 (updated as experiments complete)  
**Project:** Adaptive Sample Selection for Efficient Neural Network Training

---

## Overview

This document pre-registers anticipated failure modes before experiments run,
and will be updated with actual observations once E007–E009 complete.

A failure analysis is mandatory scientific practice:
finding where a method breaks down is as valuable as finding where it works.

---

## Pre-Registered Failure Modes

### F1 — Label Noise Amplification

**Mechanism:**  
Difficulty selectors (and the difficulty component of adaptive selectors) score
mislabelled samples highly because the model cannot learn them correctly.
Repeatedly selecting these noisy examples wastes budget on uninformative gradients
that corrupt the model rather than improving it.

**Prediction:**  
At 10–20% label noise, the difficulty selector should degrade significantly
relative to random. The adaptive selector may partially mitigate this via
diversity, which prevents repeatedly selecting the same noise examples.

**Experiment:** E007 — 10% and 20% random label flips on Fashion-MNIST pool.

**Actual observation:** TBD

---

### F2 — Class Imbalance Over-Selection

**Mechanism:**  
In imbalanced datasets, the minority class is harder to classify correctly
(higher loss, higher entropy). A difficulty or uncertainty selector may
disproportionately select minority class examples, effectively creating a
biased training distribution even within the selection budget.

**Prediction:**  
Under 5:1 class imbalance, the difficulty selector over-selects the minority
class, potentially improving minority recall but degrading macro F1 due to
majority class under-representation. Diversity selector is more robust.

**Experiment:** E008 — 5:1 class imbalance.

**Actual observation:** TBD

---

### F3 — Redundant Sample Collapse

**Mechanism:**  
If a significant fraction of the pool contains exact or near-duplicate samples,
the difficulty selector will repeatedly select the same difficult examples
(they remain consistently hard regardless of duplication). The diversity selector
should detect near-identical features and avoid repeated selection.

**Prediction:**  
At 30% duplicates, diversity selector maintains performance; difficulty selector
sees reduced performance due to redundant gradient updates.

**Experiment:** E009 — 30% exact duplicates added to pool.

**Actual observation:** TBD

---

### F4 — Early Training Instability

**Mechanism:**  
During the first few epochs, model predictions are near-random.
Scoring by loss or entropy is meaningless — all examples look roughly
equally uncertain. Aggressive difficulty selection in early epochs selects
near-random (or adversarially bad) subsets.

**Mitigation:** warmup_epochs parameter (default: 1) uses random selection
for the first epoch before model predictions become meaningful.

**Evidence:** If removing the warmup degrades performance significantly,
this failure mode is confirmed.

**Experiment:** Ablation variant in E006 (no warmup vs with warmup).

**Actual observation:** TBD

---

### F5 — Late Training Stagnation

**Mechanism:**  
In late training, most examples are correctly classified with high confidence.
Difficulty and uncertainty scores are very low across the entire pool.
The selection signal becomes near-random, providing no benefit over random
selection. The diversity signal may still help by ensuring coverage.

**Prediction:**  
At 75–100% budget, all methods converge to similar accuracy because most samples
are informative at this point. The advantage of adaptive selection shrinks.

**Evidence:** Learning curves at 75–100% should show reduced gap between methods.

**Actual observation:** TBD

---

### F6 — Selection Overhead Cost

**Mechanism:**  
Scoring requires one forward pass over the entire pool (50k samples).
For small budgets (10%), this overhead may be large relative to the actual
training time, making adaptive selection slower in wall-clock terms
even if it's more efficient in samples-seen terms.

**Prediction:**  
Accuracy-vs-time plots will show adaptive selection starting later than random
due to scoring overhead, potentially negating budget advantages.

**Mitigation:** score_every parameter reduces overhead by recomputing less
frequently. Ablation tests score_every ∈ {1, 3, 5}.

**Actual observation:** TBD

---

## Summary Table (to be filled after experiments)

| Failure Mode | Expected | Observed | Confirmed? |
|---|---|---|---|
| F1 Label noise | Difficulty < random at 20% noise | TBD | TBD |
| F2 Class imbalance | Diversity > difficulty on macro F1 | TBD | TBD |
| F3 Redundant data | Diversity maintains; difficulty degrades | TBD | TBD |
| F4 Early instability | Warmup critical at 10% budget | TBD | TBD |
| F5 Late stagnation | Methods converge at 100% budget | TBD | TBD |
| F6 Selection overhead | Adaptive slower in wall-clock at 10% | TBD | TBD |

---

## Research Notes

A **negative result** is a valid scientific result if properly documented.

If adaptive selection is never significantly better than random on Fashion-MNIST,
the conclusion is:
> "For small, balanced, well-curated datasets, the overhead of adaptive selection
> is not justified. The advantage of adaptive selection likely requires
> larger, noisier, or more complex datasets."

This is a useful finding that would help practitioners decide when to apply
these techniques.
