# LoRA Film Multitask CycleGAN 🎨

A powerful framework for image-to-image translation combining **CycleGAN**, **Low-Rank Adaptation (LoRA)**, and **FiLM (Feature-wise Linear Modulation)** for efficient multitask learning. This repository explores advanced techniques for domain adaptation and style transfer with minimal parameter overhead.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Models & Architectures](#models--architectures)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Testing & Inference](#testing--inference)
- [Configuration](#configuration)
- [Results & Checkpoints](#results--checkpoints)
- [Citation & References](#citation--references)

---

## Overview

This repository implements state-of-the-art image translation techniques combining:

- **CycleGAN**: Unpaired image-to-image translation without paired training data
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning using low-rank decomposition
- **FiLM (Feature-wise Linear Modulation)**: Adaptive instance normalization for conditional image generation

The framework supports **multitask learning**, allowing a single model to handle multiple translation tasks simultaneously with task-specific adaptation.

### Use Cases
- 🐴 Domain adaptation (e.g., horse ↔ zebra translation)
- 🎨 Style transfer across multiple domains
- 🖼️ Artistic style applications
- 📸 Photo-realistic image synthesis

---

## Key Features

✨ **Multiple Model Variants**
- CycleGAN with ResNet generators
- CycleGAN with U-Net architectures
- FiLM-based conditional generation
- LoRA-adapted models for efficient fine-tuning

⚡ **Efficiency**
- Reduced memory footprint with LoRA
- Parameter-efficient multitask learning
- FiLM layers for per-sample conditioning

📊 **Training & Evaluation**
- Comprehensive loss tracking and visualization
- Weights & Biases (W&B) integration for experiment monitoring
- Multiple GAN loss objectives (vanilla, least-square, Hinge)
- Built-in evaluation metrics

🔄 **Flexibility**
- Customizable network architectures
- Task-specific model branching
- Resume/continue training capabilities

---

## Installation

### Prerequisites
- Python 3.7+
- CUDA 11.0+ (for GPU support)
- ~8GB VRAM (varies with batch size and architecture)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/lora-film-multitask-cyclegan.git
cd lora-film-multitask-cyclegan
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cyclegan python=3.9
conda activate cyclegan
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Requirements:**
- torch >= 0.4.0
- torchvision
- numpy
- scipy
- pillow
- scikit-image
- matplotlib
- wandb (optional, for experiment tracking)

---

## Quick Start

### 1️⃣ Basic CycleGAN Training

```bash
# Train on a single task
python models/STT/train.py \
    --dataroot_general ./data \
    --tasks horse2zebra \
    --name horse2zebra_cyclegan \
    --model cycle_gan \
    --batch_size 4 \
    --n_epochs 200
```

### 2️⃣ Multitask Training with FiLM Conditioning

```bash
python models/FiLM-MTT/train-per-batch.py \
    --dataroot_general ./data \
    --tasks horse2zebra day2night summer2winter_yosemite \
    --name multitask_film \
    --netG resnet_9blocks_film \
    --netD basic_film \
    --n_epochs 100 \
    --n_epochs_decay 100
```

### 3️⃣ Multitask Training with LoRA

```bash
python models/LoRA-MTT/train-per-batch.py \
    --dataroot_general ./data \
    --tasks horse2zebra day2night summer2winter_yosemite \
    --name multitask_lora \
    --netG resnet_9blocks_lora \
    --netD basic_film \
    --use_lora \
    --lora_rank 4 \
    --n_epochs 100 \
    --n_epochs_decay 100
```

### 4️⃣ Fine-tune LoRA on a New Task

```bash
python models/FT-LoRA-MTT/train-per-batch.py \
    --dataroot_general ./data \
    --name my_finetuned_model \
    --netG resnet_9blocks_lora \
    --netD basic_film \
    --use_lora \
    --lora_rank 4 \
    --pretrained_name my_pretrained_lora_model \
    --finetune_lora new_task \
    --n_epochs 50 \
    --n_epochs_decay 0
```
> `--tasks` can be omitted — it is auto-loaded from `results/{pretrained_name}/details/hyperparams.json`.

### 5️⃣ Inference

`infer.py` reads the architecture and weights directly from a checkpoint folder — no options file needed.

**Required folder structure for input images:**
```
input_images/        # folder you pass to --input
├── img001.jpg
├── img002.png
└── ...
```
Output is written automatically to `results/{checkpoint_name}/` (or a custom path via `--out`).

```bash
# Translate A -> B for a specific task
python models/LoRA-MTT/infer.py \
    --input path/to/input_images \
    --checkpoint checkpoints/my_model \
    --task horse2zebra

# Translate B -> A
python models/LoRA-MTT/infer.py \
    --input path/to/input_images \
    --checkpoint checkpoints/my_model \
    --task horse2zebra \
    --direction BtoA

# Use a specific epoch and custom output folder
python models/LoRA-MTT/infer.py \
    --input path/to/input_images \
    --checkpoint checkpoints/my_model \
    --task horse2zebra \
    --epoch 50 \
    --out results/my_run
```

---

## Project Structure

```
lora-film-multitask-cyclegan/
├── models/                        # All model implementations
│   ├── STT/                       # Single-Task Training (baseline CycleGAN)
│   ├── FiLM-MTT/                  # Multitask with FiLM conditioning
│   ├── LoRA-MTT/                  # Multitask with LoRA adapters
│   ├── FT-LoRA-MTT/               # Fine-tune a new LoRA task on a pretrained LoRA-MTT
│   └── data/                      # Shared dataset storage
│       ├── horse2zebra/
│       ├── day2night/
│       ├── summer2winter_yosemite/
│       └── ...
├── checkpoints/                   # Saved model weights (shared across models)
│   └── {experiment_name}/
│       ├── latest_net_G_A.pth
│       ├── latest_net_G_B.pth
│       ├── latest_net_D_A.pth
│       └── latest_net_D_B.pth
├── results/                       # Training results and logs (shared across models)
│   └── {experiment_name}/
│       ├── details/
│       │   ├── hyperparams.json   # Full options snapshot (used by FT-LoRA-MTT)
│       │   └── run_summary.json
│       ├── images/                # Sample translated images
│       ├── loss_log.csv
│       └── fid_kid.csv
├── datasets/                      # Raw downloaded datasets (before preprocessing)
├── download_datasets.sh           # Dataset download helper
├── main.ipynb                     # Main experiment notebook
└── README.md                      # This file
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| `models/` | All four model variants with their own train/test/infer scripts |
| `models/data/` | Preprocessed datasets used during training |
| `checkpoints/` | Saved `.pth` weight files, shared by all model variants |
| `results/` | Loss logs, FID/KID scores, sample images, and `hyperparams.json` |
| `datasets/` | Raw downloaded data (populated by `download_datasets.sh`) |

---

## Dataset Setup

### Structure

Each task dataset lives under `models/data/` and must follow the unpaired CycleGAN layout:

```
models/data/
├── horse2zebra/
│   ├── train/
│   │   ├── A/       # Domain A training images
│   │   └── B/       # Domain B training images
│   └── test/
│       ├── A/       # Domain A test images
│       └── B/       # Domain B test images
├── day2night/
│   └── ...
└── summer2winter_yosemite/
    └── ...
```

The `--dataroot_general` argument points to the `models/data/` root; each task name is then resolved as a subfolder automatically.

### Download Datasets

From the repo root, run:

```bash
# Linux / macOS / WSL
bash download_datasets.sh

# Windows (WSL)
wsl bash download_datasets.sh
```

The script uses `gdown` to fetch the Google Drive folder and copies only missing files into `models/data/`, leaving existing data untouched.

**Included datasets:**
- `horse2zebra`
- `day2night`
- `summer2winter_yosemite`
- `monet2photo`

### Custom Datasets

1. Create `models/data/{task_name}/train/A`, `train/B`, `test/A`, `test/B`
2. Pass the task name in `--tasks` and point `--dataroot_general` at `models/data/`

---

## Training

### Important Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataroot_general` | — | Root data folder (e.g. `./models/data`) |
| `--tasks` | — | Space-separated task names (e.g. `horse2zebra day2night`) |
| `--batch_size` | 1 | Batch size for training |
| `--n_epochs` | 100 | Epochs at constant learning rate |
| `--n_epochs_decay` | 100 | Epochs to linearly decay LR to zero |
| `--lr` | 0.0002 | Initial learning rate |
| `--lambda_A` / `--lambda_B` | 10 | Cycle consistency weights |
| `--lambda_identity` | 0.5 | Identity loss weight |
| `--lora_rank` | 4 | LoRA adapter rank (LoRA-MTT / FT-LoRA-MTT only) |
| `--pretrained_name` | — | Pretrained model name for FT-LoRA-MTT |
| `--finetune_lora` | — | New task to fine-tune (FT-LoRA-MTT only) |
| `--load_size` | 286 | Resize images to this size before crop |
| `--crop_size` | 256 | Crop images to this size |
| `--continue_train` | — | Resume training from latest checkpoint |

### Resume Training

```bash
python models/LoRA-MTT/train-per-batch.py \
    --dataroot_general ./models/data \
    --tasks horse2zebra day2night summer2winter_yosemite \
    --name my_experiment \
    --netG resnet_9blocks_lora \
    --netD basic_film \
    --use_lora \
    --continue_train \
    --epoch_count 51
```

### Log to Weights & Biases

```bash
python models/LoRA-MTT/train-per-batch.py \
    --dataroot_general ./models/data \
    --tasks horse2zebra day2night \
    --name my_experiment \
    --netG resnet_9blocks_lora \
    --netD basic_film \
    --use_lora \
    --use_wandb --wandb_project_name my_project
```

### Training Outputs

After training, outputs are written to two shared locations:

```
checkpoints/{experiment_name}/
├── latest_net_G_A.pth
├── latest_net_G_B.pth
├── latest_net_D_A.pth
├── latest_net_D_B.pth
└── {epoch}_net_*.pth      # Periodic saves

results/{experiment_name}/
├── details/
│   ├── hyperparams.json   # Full options snapshot
│   └── run_summary.json
├── images/                # Sample translated images
├── loss_log.csv
├── loss_plot.png
├── fid_kid.csv
└── fid.png
```

---

## Testing & Inference

All model variants include `infer.py`, which reconstructs the architecture automatically from `hyperparams.json` inside the checkpoint folder — no need to pass architecture flags manually.

### Basic Inference (AtoB)

```bash
python models/LoRA-MTT/infer.py \
    --input path/to/images \
    --checkpoint checkpoints/my_experiment \
    --task horse2zebra
```

### Inference (BtoA)

```bash
python models/LoRA-MTT/infer.py \
    --input path/to/images \
    --checkpoint checkpoints/my_experiment \
    --task horse2zebra \
    --direction BtoA
```

### Specific Epoch & Custom Output

```bash
python models/LoRA-MTT/infer.py \
    --input path/to/images \
    --checkpoint checkpoints/my_experiment \
    --task horse2zebra \
    --epoch 50 \
    --out results/my_run
```

> Use `models/FT-LoRA-MTT/infer.py` for fine-tuned models.

Output images are saved to `results/{checkpoint_name}/` by default (one output image per input).

---

## Configuration

### Model Selection

| Model | `--netG` | `--netD` | Use with |
|-------|----------|----------|----------|
| STT (baseline) | `resnet_9blocks` | `basic` | `STT/train.py` |
| FiLM-MTT | `resnet_9blocks_film` | `basic_film` | `FiLM-MTT/train-per-batch.py` |
| LoRA-MTT | `resnet_9blocks_lora` | `basic_film` | `LoRA-MTT/train-per-batch.py` |
| FT-LoRA-MTT | `resnet_9blocks_lora` | `basic_film` | `FT-LoRA-MTT/train-per-batch.py` |

### Loss Functions

```bash
--gan_mode lsgan     # Least-square GAN (default)
--gan_mode vanilla   # Standard cross-entropy GAN
--gan_mode wgangp    # Wasserstein GAN with gradient penalty
--gan_mode hinge     # Hinge loss
```

### Key Defaults (all models)

| Parameter | Default |
|-----------|---------|
| `--lr` | `0.0002` |
| `--beta1` | `0.5` |
| `--lambda_A` / `--lambda_B` | `10.0` |
| `--lambda_identity` | `0.5` |
| `--pool_size` | `50` |
| `--norm` | `instance` |
| `--load_size` | `286` |
| `--crop_size` | `256` |

---

## Results & Checkpoints

Checkpoints and results are stored at the repo root and shared across all model variants:

```
checkpoints/{experiment_name}/
├── latest_net_G_A.pth
├── latest_net_G_B.pth
├── latest_net_D_A.pth
├── latest_net_D_B.pth
└── {epoch}_net_*.pth       # E.g. 10_net_G_A.pth

results/{experiment_name}/
├── details/
│   ├── hyperparams.json    # Recreate architecture / used by FT-LoRA-MTT
│   └── run_summary.json
├── images/                 # Sample translated images
├── loss_log.csv
├── loss_plot.png
├── fid_kid.csv
└── fid.png
```

### Run Inference from a Checkpoint

```bash
# Latest weights
python models/LoRA-MTT/infer.py \
    --input path/to/images \
    --checkpoint checkpoints/my_experiment \
    --task horse2zebra

# Specific epoch
python models/LoRA-MTT/infer.py \
    --input path/to/images \
    --checkpoint checkpoints/my_experiment \
    --task horse2zebra \
    --epoch 10
```

---

## Citation & References

### Original Papers

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}

@inproceedings{perez2018film,
  title={FiLM: Visual reasoning with a general conditioning layer},
  author={Perez, Ethan and Strub, Florian and De Vries, Harm and Dumoulin, Vincent and Courville, Aaron C},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zili and Li, Yuanxuan and Wang, Shean and Wang, Lu and Zeng, Weizhe},
  journal={arXiv preprint arXiv:2106.09714},
  year={2021}
}
```

### Related Work

- [Original CycleGAN Repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)

---

## Troubleshooting

### GPU Memory Issues
- Reduce `--batch_size` (default: 1)
- Use `--crop_size 128` for smaller images
- Enable mixed precision: `--use_amp`

### Poor Quality Results
- Train longer: increase `--n_epochs`
- Adjust loss weights: `--lambda_A`, `--lambda_B`
- Try different GAN modes: `--gan_mode wgangp`

### Training Divergence
- Lower learning rate: `--lr 0.00001`
- Increase discriminator updates: `--n_critic 5`
- Use spectral normalization: `--use_spec_norm`

### Dataset Errors
- Verify image format (PNG, JPG, JPEG)
- Check directory structure matches requirements
- Ensure sufficient disk space for cache

---

## License

[Specify your license here - e.g., MIT, Apache 2.0]

---

## Acknowledgments

Built upon:
- [PyTorch CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Junyanz Zhou
- FiLM conditioning mechanism
- LoRA adaptation techniques

---

**Last Updated:** April 2026  

