Breast Cancer Molecular Subtype Classification

## Problem Overview

The objective of this challenge was to classify histopathological images of human breast tissue into one of four **molecular subtypes**:

| Label | Subtype |
|-------|---------|
| 0 | Triple Negative |
| 1 | Luminal A |
| 2 | Luminal B |
| 3 | HER2(+) |

Each sample consists of an RGB tissue image paired with a binary mask highlighting the regions most likely to contain diseased tissue. The evaluation metric was the **weighted F1-score**.

---

## Dataset

- **Training set**: 691 image–mask pairs
- **Test set**: 477 image–mask pairs
- Images have variable aspect ratios with the shorter side fixed at 1024 px

### Key Challenges

- Presence of **Shrek-like artifacts** mislabeled as tissue classes
- **Artificially added boogers** (green blobs) highlighted by masks as high-importance regions
- **Duplicate patches** of the same tissue within single images
- Large areas devoid of informative content
- Marker traces on images

---

## Approach

### 1. Data Cleaning

- **Shrek removal**: similarity-based filtering using a pretrained InceptionV3; manual inspection was required to avoid discarding scarce tissue samples
- **Booger filtering**: HSV color-space analysis to identify and remove images containing artificial green blobs (density threshold on the H∈[35,60], S∈[80,255], V∈[160,255] range)

### 2. Preprocessing & Patching

To handle the high-resolution images (1024 px) without losing fine-grained detail, a **mask-guided patching** strategy was adopted:

1. Compute the tight bounding box of the mask
2. Pad to the minimum patch size if needed
3. Sample patch centers weighted by mask-positive pixels
4. Extract square patches (300×300 or 380×380) and reject patches with insufficient mask coverage

This ensures extracted patches are predominantly informative tissue regions.

### 3. Data Augmentation

- Random horizontal and vertical flips
- Slight random rotations (±4°)
- Class oversampling to address label imbalance

### 4. Architectures Explored

All models were trained with a **Multiple Instance Learning (MIL)** pooling strategy: patches from the same image share a label, and patch-level logits are averaged before computing the loss.

| Notebook | Architecture | Pretraining | Patch Size |
|----------|-------------|-------------|------------|
| `resnet18_patch_finetuning` | ResNet18 | ImageNet1k | 255×255 |
| `efficientnet_b3` | EfficientNet-B3 | ImageNet1k | 300×300 |
| `efficientnet_b4_300` | EfficientNet-B4 | ImageNet1k | 300×300 |
| `efficientnet_b4_380` | EfficientNet-B4 | ImageNet1k | 380×380 |
| `parallel_path_cnn` | Custom dual-path CNN | — | 255×255 |
| `wideresnet50` | WideResNet50 | PatchCamelyon | 96×96 |

Transfer learning was applied with an initial frozen-backbone phase followed by fine-tuning with differential learning rates. Layer Normalization was preferred over Batch Normalization given the small batch sizes.

Optimizers tested: **Adam**, **AdamW**, **Ranger**, **Lion**.

---

## Repository Structure

```
.
├── notebooks/
│   ├── resnet18_patch_finetuning.ipynb
│   ├── efficientnet_b3.ipynb
│   ├── efficientnet_b4_300.ipynb
│   ├── efficientnet_b4_380.ipynb
│   ├── parallel_path_cnn.ipynb
│   └── wideresnet50.ipynb
├── report/
│   └── AN2DL_Challenge2.pdf
└── README.md
```

---

## Tech Stack

- **Framework**: PyTorch + torchvision
- **Model zoo**: `timm`
- **Optimizers**: `pytorch-optimizer`, `rangerlite`
- **Data**: OpenCV, Pillow, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Environment**: Kaggle (GPU T4/P100)

---

## Requirements

```
torch>=2.0
torchvision
timm
opencv-python
pillow
pandas
scikit-learn
matplotlib
seaborn
pytorch-optimizer
rangerlite
torchview
```

> **Note**: notebooks were designed to run on Kaggle with the dataset mounted at `/kaggle/input/an2dl2526c2v2/`.
