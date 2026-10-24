# Learning What to Learn: Adaptive Sample Selection for Efficient Neural Network Training

> A rigorous, reproducible research project investigating whether adaptive sample selection can make neural network training more data- and compute-efficient than uniform random sampling.

[![Python](https://img.shields.io/badge/Python-3.12+-brightgreen?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-49%20Passed-brightgreen?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

**Research question:** Can lightweight adaptive sample selection improve training efficiency
(validation accuracy per training sample seen) compared to random sampling under a fixed data budget?

## Quick Start

```bash
# 1. Clone / enter the project directory
cd paper

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all tests (verifies the pipeline)
python -m pytest tests/ -v

# 4. Run a single experiment (downloads Fashion-MNIST on first run)
python scripts/train.py --config configs/random_025.yaml --seed 42

# 5. Run the adaptive method at 25% budget
python scripts/train.py --config configs/adaptive_025.yaml --seed 42

# 6. Run the full experiment suite (75 runs, ~2-6 hours on CPU)
python scripts/run_experiment_suite.py --quiet

# 7. Aggregate results
python scripts/aggregate_results.py

# 8. Generate figures
python scripts/generate_figures.py
```

## Project Structure

```
paper/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/           Dataset loading and splitting
â”‚   â”œâ”€â”€ models/         SmallCNN, SmallMLP
â”‚   â”œâ”€â”€ selectors/      Random, Difficulty, Uncertainty, Diversity, Adaptive
â”‚   â”œâ”€â”€ training/       Main training loop (Trainer)
â”‚   â”œâ”€â”€ metrics/        Accuracy, F1, per-class evaluation
â”‚   â””â”€â”€ utils/          Config, logging, reproducibility
â”œâ”€â”€ configs/            YAML configs for all 25 methodÃ—budget combinations
â”œâ”€â”€ scripts/            train.py, run_experiment_suite.py, aggregate_results.py, generate_figures.py
â”œâ”€â”€ tests/              49 tests covering all modules
â”œâ”€â”€ results/            raw CSVs, processed summaries, experiment tables
â”œâ”€â”€ figures/            Publication figures (generated from results)
â”œâ”€â”€ docs/               Literature review, research gap, hypotheses, experimental plan
â””â”€â”€ paper/              LaTeX paper (main.tex, references.bib)
```

## Methods

| Method | Description | Selection Signal |
|---|---|---|
| **Random** | Uniform random sampling | None |
| **Difficulty** | Top-k highest-loss samples | Cross-entropy loss |
| **Uncertainty** | Top-k highest-entropy samples | Softmax entropy |
| **Diversity** | K-means in feature space | Penultimate-layer features |
| **Adaptive** | Weighted combination (proposed) | Loss + Entropy + Diversity |

## Experimental Design

- **Dataset:** Fashion-MNIST (50k training pool, 10k validation, 10k test)
- **Model:** SmallCNN (~450k parameters, CPU-feasible)
- **Budgets:** 10%, 25%, 50%, 75%, 100%
- **Seeds:** 42, 123, 777 (3 per configuration)
- **Total runs:** 75 primary + 21 ablation + ~90 failure analysis
- **Hardware:** CPU-only (Intel laptop, 8 cores, 8.3 GB RAM)

## Hypotheses

- **H1:** Adaptive selection achieves higher accuracy than random at the same budget
- **H2:** Difficulty-only selection may underperform random at very small budgets (noise amplification)
- **H3:** Combining difficulty + diversity outperforms either signal alone
- **H0:** No significant difference (falsifiable null)

## Documentation

| Document | Description |
|---|---|
| [Project Audit](docs/project-audit.md) | Repository state, hardware, software |
| [Literature Review](docs/literature-review.md) | 13 key papers reviewed |
| [Research Gap](docs/research-gap.md) | What is missing and what we address |
| [Research Question](docs/research-question.md) | Hypotheses and null hypothesis |
| [Experimental Plan](docs/experimental-plan.md) | Full experiment matrix |
| [Dataset](docs/dataset.md) | Fashion-MNIST details and splits |
| [Experiment Log](docs/experiment-log.md) | Per-experiment records |

## Reproducibility

Every experiment saves:
- Configuration JSON
- Per-epoch metrics CSV
- Run summary JSON
- Random seed used

Results are always generated from raw data â€” no manually edited values.

## Paper

The arXiv-style LaTeX paper is in `paper/main.tex`.
Sections with experimental results use `[TBD: ExperimentXXX]` placeholders
until the corresponding experiment is complete.

## Hardware Requirements

- Python 3.12+
- PyTorch 2.0+ (CPU version sufficient)
- ~8 GB RAM
- ~30 MB for Fashion-MNIST download
- No GPU required

## License

MIT


## Author and Attribution

- Author: Nguyen Huynh Dang Nhut (@nhdnhut)
- Faculty of Computer Science, University of Information Technology (UIT, VNU-HCM)

