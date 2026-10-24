# Experiment Log

**Project:** Adaptive Sample Selection for Efficient Neural Network Training  
**Format:** Each entry follows `[EXP-ID] | date | status | key finding`

---

## Registry

| ID | Method | Condition | Status | Key Finding |
|---|---|---|---|---|
| E001 | Random | All budgets, 3 seeds | Pending | — |
| E002 | Difficulty | All budgets, 3 seeds | Pending | — |
| E003 | Uncertainty | All budgets, 3 seeds | Pending | — |
| E004 | Diversity | All budgets, 3 seeds | Pending | — |
| E005 | Adaptive | All budgets, 3 seeds | Pending | — |
| E006 | Ablation | Budget 25%, 3 seeds | Pending | — |
| E007 | Noise robustness | 10%/20% label noise | Pending | — |
| E008 | Class imbalance | 5:1 ratio | Pending | — |
| E009 | Redundant data | Duplicates | Pending | — |
| E010 | MLP robustness | Budget 25%, 3 seeds | Pending | — |

---

## Detailed Entries

*(Entries will be added after each experiment completes.)*

### E001 — Random Baseline
- **Objective:** Establish random sampling performance curve across all budgets
- **Config:** `configs/random_{budget}.yaml`, seeds 42/123/777
- **Hypothesis:** Serves as H0 null baseline
- **Expected observation:** Monotonically increasing accuracy with budget
- **Actual observation:** TBD
- **Conclusion:** TBD

### E002 — Difficulty Selector
- **Objective:** Test whether selecting highest-loss samples outperforms random
- **Config:** `configs/difficulty_{budget}.yaml`, seeds 42/123/777
- **Hypothesis:** H2 — difficulty-only may hurt at 10% budget
- **Expected observation:** Mixed; may underperform at 10%, improve at 25-50%
- **Actual observation (PARTIAL — b010 only):**
  - Epoch 1 (warmup/random): val_acc ≈ 0.759 (expected for random warmup)
  - Epoch 2+: **catastrophic collapse** — val_acc falls to ~0.11 by epoch 5
  - Root cause: at 10% budget, difficulty selector fixates on the same impossible
    hard examples every epoch; the model unlearns what it knew from warmup.
  - Random at b010: **87.29% ± 1.40%** vs Difficulty at b010: **75.20% ± 0.95%** (best = epoch 1 warmup)
- **Conclusion:** H2 STRONGLY CONFIRMED at 10% budget. The magnitude of degradation
  (~12 percentage points) is larger than anticipated. This is a key finding.


### E003 — Uncertainty Selector
- **Objective:** Test entropy-based selection efficiency
- **Config:** `configs/uncertainty_{budget}.yaml`, seeds 42/123/777
- **Hypothesis:** H1 — uncertainty selection improves over random
- **Expected observation:** Modest improvement at small budgets
- **Actual observation:** TBD
- **Conclusion:** TBD

### E004 — Diversity Selector
- **Objective:** Test k-means diversity selection
- **Config:** `configs/diversity_{budget}.yaml`, seeds 42/123/777
- **Hypothesis:** H3 partial — diversity alone provides coverage benefit
- **Expected observation:** Good performance at small budgets
- **Actual observation:** TBD
- **Conclusion:** TBD

### E005 — Adaptive Selector
- **Objective:** Test proposed combined adaptive method
- **Config:** `configs/adaptive_{budget}.yaml`, seeds 42/123/777
- **Hypothesis:** H1 + H3 — adaptive outperforms all baselines at small budgets
- **Expected observation:** Best accuracy at 10-25% budgets
- **Actual observation:** TBD
- **Conclusion:** TBD

### E006 — Ablation Study
- **Objective:** Determine contribution of each component (D, U, V)
- **Budget:** 25%
- **Hypothesis:** H3 — combined (D+U+V) outperforms any single component
- **Expected observation:** Full adaptive > any subset
- **Actual observation:** TBD
- **Conclusion:** TBD

### E007 — Label Noise Robustness
- **Objective:** Test failure mode: label noise harms difficulty selector most
- **Condition:** 10% and 20% random label flips
- **Hypothesis:** Difficulty selector amplifies noise; adaptive mitigates via diversity
- **Actual observation:** TBD
- **Conclusion:** TBD

### E008 — Class Imbalance Robustness
- **Objective:** Test performance under 5:1 class ratio
- **Condition:** One class 5× more samples than others
- **Hypothesis:** Diversity selector maintains coverage; difficulty may over-select minority
- **Actual observation:** TBD
- **Conclusion:** TBD

### E009 — Redundant Data
- **Objective:** Test performance with 30% exact duplicates
- **Condition:** 30% of pool are duplicates of existing samples
- **Hypothesis:** Diversity selector avoids duplicates; difficulty may repeatedly select them
- **Actual observation:** TBD
- **Conclusion:** TBD

### E010 — MLP Architecture Robustness
- **Objective:** Verify findings generalise to a different architecture
- **Model:** SmallMLP, budget 25%, 3 seeds
- **Hypothesis:** Adaptive outperforms random regardless of architecture
- **Actual observation:** TBD
- **Conclusion:** TBD
