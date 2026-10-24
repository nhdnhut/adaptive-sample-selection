# Dataset Documentation

**Date:** 2026-09-21  
**Project:** Adaptive Sample Selection for Efficient Neural Network Training

---

## 1. Dataset Choice and Justification

**Primary dataset: Fashion-MNIST**

### Why not plain MNIST?
MNIST is trivially solvable by small CNNs (>99% accuracy). This ceiling makes it
unsuitable for studying sample selection, because:
- Almost all samples are easy at every budget level
- Differences between methods compress to within noise
- No meaningful learning efficiency differences are observable

### Why Fashion-MNIST?
- Same format as MNIST (28×28, grayscale, 10 classes) — easy to work with
- Significantly harder (typical CNN accuracy: 91–93%) — room to measure differences
- Balanced classes (6,000 training samples per class) — controlled baseline
- Well-studied benchmark — results are contextualised by community norms
- Downloadable via torchvision — one-line reproducible download
- CPU-feasible — one epoch on 50k samples < 2 minutes on 8-core Intel

### Why not CIFAR-10?
CIFAR-10 is a valid choice but:
- 3-channel RGB images → 3× more computation per sample
- Training to meaningful accuracy requires > 30 epochs on CPU (slow)
- Fashion-MNIST allows more experiment iterations in the same time budget

**CIFAR-10 is reserved as a secondary validation if Fashion-MNIST results are strong.**

---

## 2. Dataset Details

| Property | Value |
|---|---|
| Name | Fashion-MNIST |
| Original source | https://github.com/zalandoresearch/fashion-mnist |
| License | MIT License |
| Image dimensions | 28 × 28 pixels, grayscale (1 channel) |
| Total samples | 70,000 |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Number of classes | 10 (balanced) |
| Samples per class (train) | 6,000 |
| Samples per class (test) | 1,000 |

### Class Labels
| Index | Label |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 3. Preprocessing

### Normalisation
```
mean = 0.2860
std  = 0.3530
```
Values computed over the training split. Applied as `Normalize((0.2860,), (0.3530,))`.

### Augmentation (training only)
- Random horizontal flip (p=0.5)
- No crop, no colour jitter (keep it simple for reproducibility)

### Validation and test
- Normalisation only (no augmentation)

---

## 4. Train / Validation / Test Split

```
Original training set: 60,000
  └── Our training pool: 50,000  (stratified sample, 5,000 per class)
  └── Our validation set: 10,000 (stratified sample, 1,000 per class)

Original test set: 10,000  →  used as-is for final evaluation
```

The validation set is held out throughout training. Only the test set is
used for final model evaluation. Validation accuracy drives all learning
curves and method comparisons.

---

## 5. Data Budgets

For each budget fraction B ∈ {0.10, 0.25, 0.50, 0.75, 1.00}:
- The selector draws N = B × 50,000 samples per epoch from the training pool
- At B=1.00 (100%): all 50,000 samples are seen each epoch (standard training)
- At B=0.10 (10%): only 5,000 samples are seen per epoch

Each method receives exactly the same N samples per epoch for fair comparison.

---

## 6. Download Procedure

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```

Data downloads to `data/FashionMNIST/` (~30 MB, one-time download).

---

## 7. Reproducibility Procedure

1. `download=True` fetches dataset if not present.
2. Split into pool/validation uses `torch.Generator(seed=GLOBAL_SEED)`.
3. All random operations (shuffling, subsampling) seeded with experiment seed.
4. Dataset hash checked after download to confirm integrity:
   - Training images SHA256: stored in `data/checksums.json`
5. `data/` directory added to `.gitignore` (not committed to repo).

---

## 8. Potential Issues

| Issue | Status |
|---|---|
| Class imbalance | None in base dataset (balanced by design) |
| Noisy labels | Not present in base dataset; added synthetically in E007 |
| Duplicate samples | Not present in base dataset; added synthetically in E009 |
| Download failure | Fallback: manual download from GitHub |
| Normalisation shift | Recomputed from scratch if split changes |

---

## 9. Debug Dataset

For smoke tests: 5,000-sample stratified subset (500 per class from training pool).
Used only for:
- Confirming the pipeline runs end-to-end
- Verifying selector correctness
- Fast CI tests
