# Project Audit

**Date:** 2026-09-21  
**Auditor:** AI Research Agent  
**Project:** Learning What to Learn — Adaptive Sample Selection for Efficient Neural Network Training

---

## 1. Repository Overview

| Field | Value |
|---|---|
| Repository path | `c:\Users\nguye\Desktop\paper` |
| State at audit | **Empty directory** (no pre-existing files) |
| Repository type | **New project** — clean slate |
| Version control | Not yet initialised |

The repository was completely empty at audit time. No prior ML experiments, notebooks, configs, or source code exist.

---

## 2. Existing Useful Components

None. The repository is empty.

---

## 3. Hardware Environment

| Property | Value |
|---|---|
| Operating System | Windows 10 (Build 26200) |
| CPU | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (8 logical cores) |
| RAM | ~8.3 GB total |
| GPU | **None detected** — PyTorch 2.14.0+cpu only |
| CUDA | Not available |
| Default compute device | **CPU only** |

**Implication:** All experiments must run efficiently on CPU. Model sizes, dataset sizes, and experiment complexity must be chosen accordingly.

---

## 4. Software Environment

| Package | Version |
|---|---|
| Python | 3.14.7 |
| pip | 26.2.1 |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0 |
| NumPy | 2.5.3 |
| pandas | 3.0.5 |
| matplotlib | 3.11.2 |
| Pillow | 12.3.0 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| tqdm | 4.70.1 |
| PyYAML | 6.0.3 |

**Missing packages to install:**
- `seaborn` — enhanced statistical plots
- `pytest` — testing framework

---

## 5. Proposed Project Structure

```
paper/  (root)
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── data/
│   ├── models/
│   ├── selectors/
│   │   ├── base_selector.py
│   │   ├── random_selector.py
│   │   ├── difficulty_selector.py
│   │   ├── uncertainty_selector.py
│   │   ├── diversity_selector.py
│   │   └── adaptive_selector.py
│   ├── training/
│   ├── metrics/
│   ├── profiling/
│   └── utils/
├── configs/
├── experiments/
├── benchmarks/
├── scripts/
├── tests/
├── notebooks/
├── results/raw, processed, summaries/
├── figures/
├── docs/
└── paper/
```

---

## 6. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| CPU-only training | Medium | Models < 500k params; Fashion-MNIST primary dataset |
| 8.3 GB RAM | Medium | DataLoader workers=0; small batch sizes |
| Python 3.14 pre-release | Low | Stick to stable 3.12-compatible APIs |
| Experiment wall-clock time | High | Target < 15 min per run; mini-epoch scheduling |
| Reproducibility on Windows | Low | Set all seeds; deterministic algorithms |

---

## 7. Assumptions

1. Sam has internet access for one-time dataset download.
2. All experiments: single-process CPU.
3. Target runtime: each individual run < 15 minutes.
4. Primary dataset: **Fashion-MNIST** (justified in `dataset.md`).
5. Primary architecture: compact CNN (~200k parameters).
6. Data budgets: 10%, 25%, 50%, 75%, 100%.
7. Seeds: 3 seeds minimum (42, 123, 777) per configuration.
8. Paper uses `[TBD: ExperimentXXX]` placeholders until results exist.

---

## 8. Conclusion

Repository is a clean slate. Environment is well-equipped for CPU-based ML research. The primary constraint — CPU-only — shapes every design decision: small models, moderate dataset, and time-bounded experiments.

**Next step:** Literature review → `docs/literature-review.md`
