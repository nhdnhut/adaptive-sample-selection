# Research Gap Analysis

**Date:** 2026-09-21  
**Project:** Adaptive Sample Selection for Efficient Neural Network Training

---

## 1. What Is Already Well Studied

### A. Data ordering for convergence
Curriculum learning (Bengio 2009) and self-paced learning (Kumar 2010) have
established that training order matters. Easy-to-hard ordering improves convergence
and generalization. This is well replicated.

### B. Hard example mining
Online Hard Example Mining (OHEM, Shrivastava 2016) and Focal Loss (Lin 2017)
demonstrate that down-weighting or removing easy examples can improve training
efficiency in detection settings. These are widely adopted.

### C. Coreset selection
CRAIG (Mirzasoleiman 2020) and related methods show that representative subsets
of training data can match full-dataset performance. Results are reproducible on
standard vision benchmarks.

### D. Uncertainty-based active learning
The active learning literature (Ren 2021 survey) has extensively characterized
uncertainty sampling strategies (least confidence, margin, entropy) and their
comparison to random sampling under annotation budgets.

### E. Data Shapley and valuation
Ghorbani and Zou (2019) established principled data valuation frameworks.
These correctly identify noisy and high-value samples.

### F. Data pruning at scale
Sorscher et al. (2022) showed that prototype/difficulty-based pruning can outperform
random sampling under fixed data fractions, especially at low fractions.

---

## 2. What Is Still Interesting (Underexplored)

### A. Fixed compute / fixed data-budget evaluation — the primary gap

Most existing literature evaluates methods by comparing final accuracy after
full training. Few papers systematically answer:

> **Which strategy extracts the most model quality per unit of data seen?**

Specifically, the evaluation framework of:
- same model, same optimizer, same hardware
- same maximum training budget (N samples or T time)
- varying selection strategy

is underexplored at the scale accessible to an undergraduate researcher on a personal machine.

### B. Interaction between training phase and selection strategy

Evidence exists (curriculum learning, self-paced learning) that training phase
affects which samples are useful. However, a concrete, lightweight mechanism
for adapting selection signals across training phases — without full gradient
computation — has not been systematically evaluated in a budget-constrained setting
on standard small benchmarks.

### C. Diversity vs difficulty trade-off at different data budgets

The degradation curve of difficulty-only vs diversity-only vs combined approaches
across budget levels (10%, 25%, 50%, 75%, 100%) has not been cleanly characterized
in a controlled single-architecture setting. Most papers report only selected budget
points.

### D. Lightweight adaptive methods for CPU-constrained training

Methods like CRAIG require gradient computation over the full sample pool.
This is expensive on CPU. A lightweight adaptive selection method that uses
only forward-pass signals (loss, softmax entropy, embedding norms) and a simple
diversity measure (k-means in low-dimensional space) has not been formally
benchmarked against strong baselines under a CPU-time budget.

---

## 3. What Can Realistically Be Tested

Given:
- 1 researcher
- CPU-only Intel laptop, 8 cores, 8.3 GB RAM
- PyTorch, scikit-learn, matplotlib
- Target: each run < 15 minutes, full sweep in < 8 hours

Realistic experiments:
- Dataset: Fashion-MNIST (70k samples, 28×28 grayscale, 10 classes)
- Model: compact CNN, ~200k parameters
- Selectors: random, difficulty, uncertainty, diversity, adaptive (combination)
- Budgets: 10%, 25%, 50%, 75%, 100%
- Seeds: 3 per configuration
- Metrics: accuracy, macro F1, samples seen, wall-clock time, gradient updates

NOT realistic on this hardware:
- ImageNet experiments
- Full gradient coreset computation on large pools
- Monte Carlo dropout uncertainty (expensive with CPU)
- Data Shapley (exponential retraining cost)

---

## 4. The Specific Gap We Address

**Gap statement:**

> There is no controlled, publicly reproducible study that evaluates
> lightweight adaptive sample selection (combining difficulty, uncertainty,
> and diversity signals from forward-pass information only) against meaningful
> baselines under multiple fixed data budgets on a standard small vision
> benchmark using a CPU-feasible experimental setup.

This gap is real because:

1. Most papers use expensive selection methods (full gradient coreset, full active learning loop).
2. Most papers compare final accuracy after unconstrained training.
3. No reference benchmark exists for CPU-scale adaptive selection on Fashion-MNIST.
4. The budget-vs-performance *curve* across multiple budget levels is rarely fully reported.

**What we contribute (engineering + empirical):**

1. A modular, configurable framework for comparing sample-selection strategies.
2. A controlled benchmark under multiple fixed data budgets with multiple seeds.
3. A lightweight adaptive selector that combines loss-based difficulty, softmax entropy, and k-means diversity — all computable from forward passes only.
4. A Pareto analysis of performance vs training cost across strategies.
5. A failure analysis identifying when adaptive selection breaks down.

We do **not** claim algorithmic novelty in the sense of a theoretically new method.
We claim: a rigorous empirical contribution + a reusable research framework.

---

## 5. Framing Diagram

```
Same model (compact CNN)
Same optimizer (Adam)
Same hardware (CPU)
Same maximum budget (N samples)
           ↓
┌──────────────────────────────────┐
│ Strategy A: Random               │
│ Strategy B: Difficulty-only      │
│ Strategy C: Uncertainty-only     │
│ Strategy D: Diversity-only       │
│ Strategy E: Adaptive (combined)  │
└──────────────────────────────────┘
           ↓
Which strategy reaches validation accuracy T
using fewest examples?

Which strategy has the best accuracy given
a fixed count of N seen examples?
```

---

## 6. Threat to Validity (Acknowledged Upfront)

1. **Fashion-MNIST may be too easy.** Differences may be small. Mitigation: evaluate at multiple budgets; include 10% budget where differences should be largest.
2. **CPU experiments are slow.** Time measurements mix CPU efficiency with selection overhead. Mitigation: report both samples-seen and wall-clock time separately.
3. **Single model architecture.** Mitigation: briefly test on small MLP for robustness check.
4. **3 seeds may be insufficient for statistical power.** Acknowledged as a limitation; we report mean ± std and note this.
