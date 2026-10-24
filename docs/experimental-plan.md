# Experimental Plan

**Date:** 2026-09-21  
**Project:** Adaptive Sample Selection for Efficient Neural Network Training

---

## 1. Dataset

**Primary:** Fashion-MNIST
- 60,000 training images, 10,000 test images
- 28×28 grayscale, 10 balanced classes
- Normalised to mean=0.2860, std=0.3530

**Validation split:** 10,000 of training held out (stratified), leaving 50,000 for training pool.

**Debug dataset:** 5,000-sample subset of Fashion-MNIST for smoke testing.

---

## 2. Model

**Primary architecture:** `SmallCNN`
- Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)
- Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
- FC(64×5×5 → 256) → ReLU → Dropout(0.5)
- FC(256 → 10)
- ~450k parameters

**Secondary architecture (robustness check):** `SmallMLP`
- FC(784 → 256) → ReLU → Dropout(0.3)
- FC(256 → 128) → ReLU
- FC(128 → 10)
- ~235k parameters

---

## 3. Training Protocol (Fixed Across All Methods)

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 30 (full-budget baseline) |
| LR schedule | None (fixed LR for reproducibility) |
| Weight decay | 1e-4 |
| Loss | Cross-entropy |
| Device | CPU |

**Budget control:** For a budget of B% of the training pool:
- Each epoch presents exactly B% × 50,000 = N samples
- Total samples seen = epochs × N
- Comparison is made at equal `total_samples_seen` not equal epochs

---

## 4. Selectors

### A — Random
Uniform random sampling from the pool. Baseline.

### B — Difficulty
Select top-k highest-loss samples from pool (requires one forward pass over pool).

### C — Uncertainty
Select top-k highest-entropy samples from pool (softmax entropy).

### D — Diversity
K-means clustering in final feature space; sample proportionally from each cluster,
prefer underrepresented ones.

### E — Adaptive (proposed)
Scoring function:

```
S(x_i, t) = α · D_i(t) + β · U_i(t) + γ · V_i(t)
```

where:
- D_i(t) = normalised loss (difficulty) at epoch t
- U_i(t) = normalised softmax entropy (uncertainty) at epoch t
- V_i(t) = diversity score (distance to nearest selected centroid)
- α, β, γ = mixing weights (from ablation; default: 0.4, 0.4, 0.2)
- Selection: top-k by score S

Phase schedule:
- Epochs 1–10: α=0.2, β=0.2, γ=0.6 (diversity-heavy early)
- Epochs 11–20: α=0.4, β=0.4, γ=0.2 (balanced)
- Epochs 21–30: α=0.5, β=0.4, γ=0.1 (difficulty-heavy late)

---

## 5. Experiment Matrix

### Primary sweep (E001–E005)

| Exp | Method | Budget | Seeds |
|---|---|---|---|
| E001 | Random | 10%, 25%, 50%, 75%, 100% | 42, 123, 777 |
| E002 | Difficulty | 10%, 25%, 50%, 75%, 100% | 42, 123, 777 |
| E003 | Uncertainty | 10%, 25%, 50%, 75%, 100% | 42, 123, 777 |
| E004 | Diversity | 10%, 25%, 50%, 75%, 100% | 42, 123, 777 |
| E005 | Adaptive | 10%, 25%, 50%, 75%, 100% | 42, 123, 777 |

Total: 5 methods × 5 budgets × 3 seeds = **75 runs**

### Ablation (E006)

| Variant | Description |
|---|---|
| E006a | Adaptive: α only (difficulty only via adaptive code path) |
| E006b | Adaptive: β only (uncertainty only) |
| E006c | Adaptive: γ only (diversity only) |
| E006d | Adaptive: α + β (no diversity) |
| E006e | Adaptive: α + γ (no uncertainty) |
| E006f | Adaptive: β + γ (no difficulty) |
| E006g | Full adaptive (all three) |

Budget: 25% only, seeds: 42, 123, 777. Total: 7 × 3 = **21 runs**

### Failure analysis (E007–E009)

| Exp | Condition | Method | Budget |
|---|---|---|---|
| E007 | Label noise (10%, 20%) | All | 25% |
| E008 | Class imbalance (5:1 ratio) | All | 25% |
| E009 | Redundant data (duplicates) | All | 25% |

Each condition: 5 methods × 2 levels × 3 seeds ≈ **30 runs each**

### Robustness (E010)

MLP architecture: Adaptive vs Random vs Difficulty, budget 25%, 3 seeds = **9 runs**

---

## 6. Metrics Collected Per Run

```python
{
  "experiment_id": str,
  "method": str,
  "budget_fraction": float,
  "seed": int,
  "epoch": int,
  "samples_seen": int,
  "gradient_updates": int,
  "wall_clock_seconds": float,
  "val_accuracy": float,
  "val_loss": float,
  "val_f1_macro": float,
  "train_accuracy": float,
  "train_loss": float,
  "selection_overhead_seconds": float
}
```

All metrics saved per epoch. Raw files: `results/raw/{experiment_id}_{seed}.csv`.

---

## 7. Primary Evaluation

**Central figure:** Validation accuracy vs samples seen (all methods overlaid).

**Budget table:** For each budget level, mean ± std accuracy across seeds.

**Pareto frontier:** Validation accuracy vs total wall-clock time.

---

## 8. Statistical Analysis

- Report mean ± std across 3 seeds for each configuration.
- For significant claims, apply paired t-test (adaptive vs random) at each budget.
- Report effect size (Cohen's d) where appropriate.
- Explicitly distinguish statistical significance (p < 0.05) from practical significance (Δacc > 1%).

---

## 9. Execution Order

```
Phase 1: setup + tests (1 hour)
Phase 2: smoke tests on debug dataset (30 min)
Phase 3: E001 random baseline, all budgets (1 hour)
Phase 4: E002-E004 baselines (2-3 hours)
Phase 5: E005 adaptive method (1-2 hours)
Phase 6: E006 ablation (1 hour)
Phase 7: E007-E009 failure analysis (2-3 hours)
Phase 8: E010 robustness (30 min)
Phase 9: aggregate + figures (30 min)
Phase 10: paper write (ongoing)
```

---

## 10. Definition of Done

- [ ] All 75 primary runs complete with results saved
- [ ] Ablation 21 runs complete
- [ ] Failure analysis runs complete
- [ ] All figures generated from raw data (no manual editing)
- [ ] All tables generated programmatically
- [ ] Paper sections 1-11 drafted
- [ ] References.bib populated with verified citations
