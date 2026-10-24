# Literature Review

**Date:** 2026-09-21  
**Project:** Adaptive Sample Selection for Efficient Neural Network Training

---

## Overview

This review covers the key areas relevant to the research project:
sample selection, curriculum learning, active learning, hard example mining,
diversity-aware sampling, coreset selection, data valuation, and data-centric AI.

---

## Paper Summaries

---

### P01 — Curriculum Learning

| Field | Value |
|---|---|
| **Paper** | Curriculum Learning |
| **Authors** | Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston |
| **Year** | 2009 |
| **Venue** | ICML 2009 |
| **URL** | https://dl.acm.org/doi/10.1145/1553374.1553380 |
| **Problem** | Non-convex optimization in neural network training benefits from structured data ordering |
| **Method** | Present training examples in order from easy to hard, inspired by cognitive science |
| **Selection Signal** | Pre-defined difficulty metric (e.g., example cleanliness or task complexity) |
| **Training Setting** | Supervised classification and NLP |
| **Dataset** | Shape recognition, language modeling |
| **Metrics** | Test error, convergence speed |
| **Main Result** | Training with curriculum (easy-to-hard) converges faster and to better local minima than random ordering |
| **Limitations** | Requires a pre-defined, task-specific difficulty measure; curriculum is fixed and not adaptive |
| **Relation to Our Project** | Foundational concept: ordering by difficulty. Our work differs by making the selection *dynamic* and *data-budget-aware* rather than just sequencing |

---

### P02 — Self-Paced Learning

| Field | Value |
|---|---|
| **Paper** | Self-Paced Learning for Latent Variable Models |
| **Authors** | M. P. Kumar, Benjamin Packer, Daphne Koller |
| **Year** | 2010 |
| **Venue** | NIPS 2010 |
| **URL** | https://papers.nips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html |
| **Problem** | Pre-defined curricula require external knowledge; model should determine difficulty itself |
| **Method** | Adds a regularization term to the objective that jointly selects easy examples and learns model parameters; threshold annealed over time |
| **Selection Signal** | Training loss relative to a learned threshold λ |
| **Training Setting** | Latent variable models; later generalised to deep nets |
| **Dataset** | Object localization, coreference resolution |
| **Metrics** | Task accuracy |
| **Main Result** | Self-determined difficulty ordering avoids poor local optima without requiring external labels of difficulty |
| **Limitations** | Focuses on easy samples first; may under-exploit hard/novel examples that are informative |
| **Relation to Our Project** | Our difficulty selector is inspired by SPL. Key difference: we operate under a *fixed data budget*, not a temporal annealing schedule |

---

### P03 — Self-Paced Curriculum Learning

| Field | Value |
|---|---|
| **Paper** | Self-Paced Curriculum Learning |
| **Authors** | Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, Alexander Hauptmann |
| **Year** | 2015 |
| **Venue** | AAAI 2015 |
| **URL** | https://ojs.aaai.org/index.php/AAAI/article/view/9608 |
| **Problem** | How to combine the benefits of pre-defined curriculum with adaptive self-pacing |
| **Method** | SPCL: incorporates a prior (curriculum) into the SPL regularisation term |
| **Selection Signal** | Weighted combination of loss and a curriculum-derived prior |
| **Training Setting** | Supervised classification |
| **Dataset** | Various vision datasets |
| **Metrics** | Classification accuracy |
| **Main Result** | SPCL outperforms both pure curriculum learning and pure SPL |
| **Limitations** | Still requires some prior knowledge; complexity of tuning two schedules |
| **Relation to Our Project** | Motivates our adaptive method: combining multiple signals (loss + diversity) is better than either alone |

---

### P04 — An Empirical Study of Example Forgetting

| Field | Value |
|---|---|
| **Paper** | An Empirical Study of Example Forgetting during Deep Neural Network Learning |
| **Authors** | Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, Geoffrey Gordon |
| **Year** | 2019 |
| **Venue** | ICLR 2019 |
| **URL** | https://arxiv.org/abs/1901.12350 |
| **Problem** | Which training examples are truly necessary for good generalisation? |
| **Method** | Track per-example classification transitions (correct→incorrect) across epochs; count "forgetting events" |
| **Selection Signal** | Forgetting count — examples never forgotten can be pruned |
| **Training Setting** | Supervised image classification |
| **Dataset** | MNIST, CIFAR-10, permuted MNIST |
| **Metrics** | Test accuracy after pruning |
| **Main Result** | A large fraction (> 30%) of training examples can be removed with minimal accuracy loss; unforgettable examples are redundant |
| **Limitations** | Requires a full training pass to compute forgetting statistics; scores not available online |
| **Relation to Our Project** | Directly motivates our difficulty / redundancy analysis. Our difficulty selector uses per-epoch loss as a proxy for forgettability |

---

### P05 — CRAIG: Coresets for Accelerating Incremental Gradient Descent

| Field | Value |
|---|---|
| **Paper** | Coresets for Data-Efficient Training of Machine Learning Models |
| **Authors** | Baharan Mirzasoleiman, Jeff Bilmes, Jure Leskovec |
| **Year** | 2020 |
| **Venue** | ICML 2020 |
| **URL** | https://proceedings.mlr.press/v119/mirzasoleiman20a.html |
| **Problem** | Training on large datasets is expensive; can we find a small representative subset? |
| **Method** | CRAIG: treats coreset selection as weighted submodular optimisation to approximate full gradient |
| **Selection Signal** | Gradient similarity / submodular facility location |
| **Training Setting** | Supervised classification |
| **Dataset** | MNIST, CIFAR-10, CIFAR-100, ImageNet |
| **Metrics** | Accuracy vs fraction of data used |
| **Main Result** | CRAIG achieves near full-dataset accuracy using 10–40% of data; 6× speedup on CIFAR-10 |
| **Limitations** | Submodular optimisation itself is expensive; requires gradient computation over full pool |
| **Relation to Our Project** | Benchmark for what is achievable. We implement a lighter diversity selector rather than full submodular optimisation |

---

### P06 — Online Hard Example Mining (OHEM)

| Field | Value |
|---|---|
| **Paper** | Training Region-based Object Detectors with Online Hard Example Mining |
| **Authors** | Abhinav Shrivastava, Abhinav Gupta, Ross Girshick |
| **Year** | 2016 |
| **Venue** | CVPR 2016 |
| **URL** | https://arxiv.org/abs/1604.03540 |
| **Problem** | Class imbalance in detection: easy negatives dominate gradient |
| **Method** | Sort examples by loss in each mini-batch; backpropagate only on top-k highest-loss examples |
| **Selection Signal** | Per-example loss magnitude |
| **Training Setting** | Region-based object detection |
| **Dataset** | PASCAL VOC |
| **Metrics** | mAP |
| **Main Result** | OHEM significantly outperforms random sampling baselines; removes need for hand-tuned mining heuristics |
| **Limitations** | Double forward pass overhead; designed for detection, not general classification |
| **Relation to Our Project** | Our difficulty selector is conceptually similar to OHEM; we compare difficulty-only selection to combined strategies |

---

### P07 — Focal Loss

| Field | Value |
|---|---|
| **Paper** | Focal Loss for Dense Object Detection (RetinaNet) |
| **Authors** | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár |
| **Year** | 2017 |
| **Venue** | ICCV 2017 |
| **URL** | https://arxiv.org/abs/1708.02002 |
| **Problem** | Easy negatives overwhelm gradient in dense detection |
| **Method** | Modulate cross-entropy by (1−p_t)^γ to down-weight easy examples continuously |
| **Selection Signal** | Prediction confidence p_t |
| **Training Setting** | One-stage object detection |
| **Dataset** | MS COCO |
| **Metrics** | mAP |
| **Main Result** | Focal loss matches or exceeds two-stage detectors; γ=2 works well |
| **Limitations** | Implicit rather than explicit selection; does not increase sample diversity |
| **Relation to Our Project** | Inspiration for uncertainty/difficulty weighting; our uncertainty selector uses entropy, analogous to focal loss |

---

### P08 — Active Learning with Uncertainty Sampling

| Field | Value |
|---|---|
| **Paper** | A Survey of Deep Active Learning |
| **Authors** | Pengzhen Ren et al. |
| **Year** | 2021 |
| **Venue** | ACM Computing Surveys |
| **URL** | https://arxiv.org/abs/2009.00236 |
| **Problem** | How to minimise annotation cost while maximising model performance |
| **Method** | Survey of uncertainty-based (least confidence, margin, entropy) and diversity-based (coreset, BADGE) query strategies |
| **Selection Signal** | Softmax entropy, margin, confidence, gradient embeddings |
| **Training Setting** | Pool-based active learning |
| **Dataset** | Various vision + NLP |
| **Metrics** | Accuracy vs labelling budget |
| **Main Result** | Uncertainty + diversity combined strategies consistently outperform uncertainty-only; calibration matters |
| **Limitations** | Most methods assume oracle labelling; not directly applicable to fixed-label scenarios |
| **Relation to Our Project** | Key insight: our setting differs from active learning (labels are all available); we adapt uncertainty scoring to a *training efficiency* context rather than annotation efficiency |

---

### P09 — Data Shapley

| Field | Value |
|---|---|
| **Paper** | Data Shapley: Equitable Valuation of Data for Machine Learning |
| **Authors** | Amirata Ghorbani, James Zou |
| **Year** | 2019 |
| **Venue** | ICML 2019 |
| **URL** | https://arxiv.org/abs/1904.02868 |
| **Problem** | How to quantify the value of individual training examples? |
| **Method** | Apply Shapley value from cooperative game theory to data valuation; approximated via Monte Carlo and gradient methods |
| **Selection Signal** | Data Shapley value (marginal contribution averaged over subsets) |
| **Training Setting** | Classification with various models |
| **Dataset** | Various |
| **Metrics** | Value score vs performance impact |
| **Main Result** | Data Shapley correctly identifies high- and low-value (mislabeled, noisy) examples; superior to leave-one-out |
| **Limitations** | Computationally very expensive; O(n log n) model retrains needed for exact computation |
| **Relation to Our Project** | Motivates data valuation; our approach uses tractable proxies (loss, entropy) rather than full Shapley |

---

### P10 — Self-Paced Learning with Diversity

| Field | Value |
|---|---|
| **Paper** | Self-Paced Learning with Diversity |
| **Authors** | Lu Jiang, Deyu Meng, Teruko Mitamura, Alexander Hauptmann |
| **Year** | 2014 |
| **Venue** | NIPS 2014 |
| **URL** | https://papers.nips.cc/paper/2014/hash/c60d060b946d6dd6145dcbad5c4ccf6f-Abstract.html |
| **Problem** | Pure self-paced learning selects redundant easy examples |
| **Method** | SPLD: adds a group sparsity regulariser to enforce diversity in selected examples |
| **Selection Signal** | Loss magnitude + inter-sample diversity penalty |
| **Training Setting** | Supervised classification |
| **Dataset** | TRECVID, YouTube |
| **Metrics** | AP |
| **Main Result** | Diversity constraint prevents selection of redundant examples and improves performance |
| **Limitations** | Group sparsity computationally expensive; hyperparameter sensitivity |
| **Relation to Our Project** | **Most directly related.** H3 (difficulty + diversity) is supported by SPLD. Our adaptive method is a lightweight operationalisation of this idea |

---

### P11 — Dataset Condensation / Distillation

| Field | Value |
|---|---|
| **Paper** | Dataset Distillation |
| **Authors** | Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, Alexei A. Efros |
| **Year** | 2018 |
| **Venue** | arXiv 2018 |
| **URL** | https://arxiv.org/abs/1811.10959 |
| **Problem** | Can we compress a dataset into a tiny set of synthetic examples that contains all useful knowledge? |
| **Method** | Meta-learning: optimise synthetic examples so that one gradient step on them yields similar model to full training |
| **Selection Signal** | Meta-gradient w.r.t. synthetic data |
| **Training Setting** | Few-shot distillation |
| **Dataset** | MNIST, CIFAR-10 |
| **Metrics** | Accuracy with 1, 10, 50 distilled images per class |
| **Main Result** | 10 images per class → ~94% of full training performance on MNIST |
| **Limitations** | Distilled data is not interpretable; expensive outer optimisation |
| **Relation to Our Project** | Different approach to data efficiency; we select from *real* examples (no synthesis). Provides a theoretical upper bound on what data compression can achieve |

---

### P12 — Importance Sampling for SGD

| Field | Value |
|---|---|
| **Paper** | Stochastic Optimization with Importance Sampling for Regularized Loss Minimization |
| **Authors** | Peilin Zhao, Tong Zhang |
| **Year** | 2015 |
| **Venue** | ICML 2015 |
| **URL** | https://proceedings.mlr.press/v37/zhaoc15.html |
| **Problem** | Uniform SGD ignores per-sample informativeness |
| **Method** | Non-uniform sampling proportional to gradient norm (Lipschitz constant); reduces variance of gradient estimate |
| **Selection Signal** | Lipschitz constant / gradient norm |
| **Training Setting** | Regularized loss minimization |
| **Dataset** | Various classification |
| **Metrics** | Convergence rate |
| **Main Result** | Importance sampling yields provably faster convergence than uniform sampling under mild conditions |
| **Limitations** | Computing exact Lipschitz constants is expensive; approximation degrades guarantees |
| **Relation to Our Project** | Theoretical justification for non-uniform sampling. Our difficulty and uncertainty selectors act as practical proxies for importance sampling |

---

### P13 — Beyond neural scaling: data quality vs quantity

| Field | Value |
|---|---|
| **Paper** | Beyond neural scaling laws: beating power law scaling via data pruning |
| **Authors** | Ben Sorscher, Robert Geirhos, Shashank Vyas, Surya Ganguli, Ari Morcos |
| **Year** | 2022 |
| **Venue** | NeurIPS 2022 |
| **URL** | https://arxiv.org/abs/2206.14486 |
| **Problem** | Neural scaling laws assume random data; can data quality break the power-law? |
| **Method** | Prototype-based pruning: remove examples closest to class centroids (easy) to retain only challenging, diverse examples |
| **Selection Signal** | Distance to class prototype (self-supervised embedding) |
| **Training Setting** | Large-scale image classification |
| **Dataset** | ImageNet |
| **Metrics** | Accuracy vs data fraction |
| **Main Result** | Hard-example retention achieves super-linear scaling improvement over random pruning; quality > quantity at small fractions |
| **Limitations** | Requires self-supervised pretraining for good embeddings; large-scale setting |
| **Relation to Our Project** | Directly relevant to our budget-constrained setting. Supports H2: difficulty-only is not always optimal; embedding-based distance (diversity) matters |

---

## Summary Table

| Paper | Year | Method Type | Selection Signal | Key Finding | Relation |
|---|---|---|---|---|---|
| Bengio 2009 | 2009 | Curriculum | Pre-defined difficulty | Easy-first improves convergence | Foundation |
| Kumar 2010 | 2010 | Self-paced | Training loss threshold | Model-determined easy-first works | Difficulty selector |
| Jiang 2015 SPCL | 2015 | Combined | Loss + prior | Combining signals beats either alone | Motivation for adaptive |
| Toneva 2019 | 2019 | Pruning | Forgetting events | 30%+ data is redundant | Difficulty/redundancy |
| Mirzasoleiman 2020 | 2020 | Coreset | Gradient similarity | 10–40% subset ≈ full dataset | Benchmark |
| Shrivastava 2016 | 2016 | Hard mining | Loss magnitude | OHEM improves detection | Difficulty baseline |
| Lin 2017 | 2017 | Loss reweight | Prediction confidence | Focal loss improves easy/hard balance | Uncertainty weighting |
| Ren 2021 | 2021 | Active learning | Entropy, margin | Uncertainty + diversity > uncertainty alone | Our H3 |
| Ghorbani 2019 | 2019 | Data valuation | Shapley value | Principled data importance measurable | Tractable proxies |
| Jiang 2014 SPLD | 2014 | SPL + Diversity | Loss + group sparsity | Diversity prevents redundancy | **Most direct** |
| Wang 2018 | 2018 | Distillation | Meta-gradient | Synthetic compression possible | Upper bound |
| Zhao 2015 | 2015 | Importance sampling | Gradient norm | Non-uniform sampling ≈ faster convergence | Theoretical basis |
| Sorscher 2022 | 2022 | Data pruning | Embedding distance | Quality > quantity at low data fractions | Supports H2, H3 |

---

## Key Insights from Literature

1. **Difficulty-only selection is not sufficient.** Multiple papers (Sorscher 2022, Jiang 2014) show that pure hard-example selection can backfire due to redundancy and noise amplification.

2. **Combining difficulty with diversity consistently helps.** SPLD (Jiang 2014), CRAIG (Mirzasoleiman 2020), and the active learning survey (Ren 2021) all confirm this.

3. **The training-phase matters.** Curriculum learning suggests easy-first early; some evidence for hard-focus later. This motivates a *stage-adaptive* strategy.

4. **Fixed-budget evaluation is underexplored.** Most papers compare methods at final accuracy after full training. Fewer systematically compare *efficiency at the same compute budget*.

5. **Tractable proxies work.** Data Shapley and CRAIG are theoretically grounded but computationally expensive. Simpler proxies (loss, entropy, embedding distance) achieve most of the benefit.
