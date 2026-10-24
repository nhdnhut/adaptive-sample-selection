# Research Question and Hypotheses

**Date:** 2026-09-21  
**Project:** Adaptive Sample Selection for Efficient Neural Network Training

---

## Primary Research Question

> **Can lightweight adaptive sample selection — using only forward-pass signals
> (loss, softmax entropy, embedding-based diversity) — improve training efficiency
> compared to uniform random sampling, when all methods operate under the same
> fixed data budget?**

"Efficiency" is defined as:
- Higher validation accuracy given the same number of training samples seen, or
- Fewer samples required to reach a target validation accuracy.

---

## Candidate Hypotheses

### H1 — Budget Efficiency
**Statement:** Adaptive sample selection achieves a higher validation accuracy than
uniform random sampling when given the same number of training samples (fixed data budget).

**Rationale:** Not all samples are equally informative. Selecting samples based
on difficulty and uncertainty should focus gradient updates on informative examples,
leading to better accuracy per sample.

**How to test:** Compare accuracy at each budget level (10%, 25%, 50%, 75%)
across all methods. A paired comparison at each budget.

**Expected finding:** Advantage of adaptive selection should be largest at the
smallest budget (10%) and shrink as budget grows toward 100%.

**Counter-argument to test:** If the dataset is well-balanced and most examples
are similarly informative, random selection may perform just as well.

---

### H2 — Difficulty Collapse
**Statement:** Pure difficulty-based selection (always selecting highest-loss
examples) does not consistently outperform random selection, and may perform worse,
because it amplifies noise and reduces sample diversity.

**Rationale:** Literature (Sorscher 2022, Jiang 2014) suggests that hard-example-
only strategies suffer from:
(a) Noise amplification: hard examples may be mislabeled or outliers.
(b) Diversity collapse: repeatedly selecting the same hard examples ignores
    the distribution of the rest of the dataset.

**How to test:** Compare difficulty selector vs random selector. If difficulty
performs < random at any budget, H2 has evidence.

**Expected finding:** Difficulty selector underperforms random at the smallest
budget (10%) but improves at medium budgets.

---

### H3 — Diversity × Difficulty Synergy
**Statement:** Combining difficulty/uncertainty with a diversity measure produces
a better performance-vs-budget trade-off than using either signal alone.

**Rationale:** Directly supported by SPLD (Jiang 2014), CRAIG (Mirzasoleiman 2020),
and the active learning survey (Ren 2021). Diversity prevents redundant selection.

**How to test:** Ablation study comparing:
- Difficulty only
- Uncertainty only
- Diversity only
- Difficulty + Diversity
- Uncertainty + Diversity
- All three (full adaptive)

If the combined method consistently outperforms any single-signal method, H3 is supported.

---

### H4 — Stage Dependence
**Statement:** The usefulness of adaptive selection varies with training phase.
Early training benefits from diverse, moderately-difficult examples. Late training
benefits from harder, boundary-clarifying examples.

**Rationale:** Curriculum learning research suggests that the optimal difficulty
level changes during training. This motivates a phase-adaptive selection policy.

**How to test:** Compare a static selection policy (fixed weights throughout training)
vs a schedule that transitions from diversity-weighted to difficulty-weighted.

**Expected finding:** Phase adaptation helps at small budgets where every epoch counts.

**Note:** This hypothesis is secondary; testing it requires more experimental runs.
We test it as an ablation variant if compute permits.

---

## Selected Primary Hypothesis

**Primary: H1** — Adaptive selection is more efficient than random sampling under
a fixed data budget.

**Secondary: H3** — Diversity + difficulty/uncertainty combination is the key driver
of this advantage.

**Exploratory: H2** — Pure difficulty selection may not help or may hurt at very small budgets.

H4 is a future-work direction if H1 and H3 are confirmed.

---

## Null Hypothesis

**H0:** Under the same fixed data budget and training protocol, no sample-selection
strategy produces statistically significantly higher validation accuracy than
uniform random sampling on Fashion-MNIST with a compact CNN.

If H0 is not rejected, the result is still scientifically valuable: it would
suggest that for small, balanced datasets like Fashion-MNIST, sample selection
provides no measurable advantage, and the effort of adaptive selection is not
warranted.

---

## Contribution Statement (pre-registered)

Regardless of outcome, we contribute:

1. A reusable, modular sample-selection benchmark framework.
2. A controlled comparison under fixed data budgets with multiple seeds.
3. An honest characterisation of when adaptive selection helps and when it does not.

We do NOT pre-claim that adaptive selection will outperform random sampling.
We treat this as a hypothesis to test, not a conclusion to illustrate.
