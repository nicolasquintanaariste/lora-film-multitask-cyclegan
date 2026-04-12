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
- Support for paired (pix2pix) and unpaired (CycleGAN) datasets
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
cd Dissertation/pytorch-CycleGAN-and-pix2pix-film-lora

# Train on Horse-to-Zebra dataset
python train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_cyclegan \
    --model cycle_gan \
    --batch_size 4 \
    --n_epochs 200
```

### 2️⃣ Train with FiLM Conditioning

```bash
python train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_film \
    --model cycle_gan \
    --netG resnet_9blocks_film \
    --batch_size 4
```

### 3️⃣ Multitask Learning

```bash
python train.py \
    --dataroot ./datasets/multitask \
    --name multitask_model \
    --model cycle_gan \
    --netG resnet_9blocks_film \
    --tasks horse2zebra,photo2painting
```

### 4️⃣ Test/Inference

```bash
python test.py \
    --dataroot ./datasets/horse2zebra/testA \
    --name horse2zebra_cyclegan \
    --model test
```

---

## Project Structure

```
lora-film-multitask-cyclegan/
├── Dissertation/                           # Main research implementations
│   ├── pytorch-CycleGAN-and-pix2pix/       # Baseline CycleGAN
│   ├── pytorch-CycleGAN-and-pix2pix-film/  # FiLM integration
│   ├── pytorch-CycleGAN-and-pix2pix-lora/  # LoRA adaptation
│   ├── pytorch-CycleGAN-and-pix2pix-film-lora/  # LoRA + FiLM 🌟
│   ├── CycleGAN_ResNet/                    # ResNet architecture variants
│   ├── CycleGAN_ResNet_FiLM/               # ResNet + FiLM
│   ├── CycleGAN_Unet/                      # U-Net architecture variants
│   ├── CycleGAN_Unet_FiLM/                 # U-Net + FiLM
│   ├── CycleGAN_Unet_FiLM_Hinge/           # U-Net + FiLM + Hinge Loss
│   ├── CycleGAN_erik_linder/               # Original referenced implementation
│   ├── data/                               # Dataset storage
│   ├── generate_samples/                   # Sample generation scripts
│   └── checkpoints/                        # Saved model weights
│       ├── horse2zebra_020326/
│       ├── horse2zebra_040326/
│       └── ...
├── checkpoints/                            # Top-level model checkpoints
├── test/                                   # Testing utilities & results
├── wandb/                                  # W&B experiment logs
└── README.md                               # This file
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| `Dissertation/` | Multiple architecture implementations for comparison |
| `checkpoints/` | Pre-trained models and training checkpoints |
| `data/` | Image datasets (horses, zebras, photos, paintings, etc.) |
| `wandb/` | Experiment tracking logs and metrics |
| `test/` | Test sets and inference results |

---

## Models & Architectures

### 1️⃣ CycleGAN (Baseline)
Standard unpaired image-to-image translation with dual generators and discriminators.

**Configuration:**
```bash
--model cycle_gan --netG resnet_9blocks --netD basic
```

### 2️⃣ CycleGAN + FiLM
Adds FiLM layers for task-specific instance normalization.

**Configuration:**
```bash
--netG resnet_9blocks_film
```

**Benefits:**
- Task-aware feature modulation
- Improved multitask learning
- Fewer parameters than full fine-tuning

### 3️⃣ CycleGAN + LoRA
Low-rank adaptation of generator weights for efficient fine-tuning.

**Configuration:**
```bash
--lora_rank 8 --lora_alpha 16
```

**Benefits:**
- 90%+ parameter reduction
- Fast adaptation to new tasks
- Memory-efficient training

### 4️⃣ CycleGAN + LoRA + FiLM (Recommended) ⭐
Combines both techniques for maximum efficiency and performance.

**Configuration:**
```bash
--netG resnet_9blocks_film --lora_rank 8 --use_lora
```

### Architecture Variants

| Variant | Generator | Discriminator | Best For |
|---------|-----------|---------------|---------  |
| ResNet | ResNet blocks | PatchGAN | General purpose |
| U-Net | U-Net encoder-decoder | PatchGAN | Detailed features |
| Hybrid | ResNet + FiLM | PatchGAN | Conditional generation |
| Lightweight | ResNet + LoRA | PatchGAN | Resource-constrained |

---

## Dataset Setup

### Structure

Datasets should be organized in one of two formats:

#### Unpaired Format (CycleGAN)
```
datasets/horse2zebra/
├── trainA/          # Domain A training images
├── trainB/          # Domain B training images
├── testA/           # Domain A test images
└── testB/           # Domain B test images
```

#### Paired Format (pix2pix)
```
datasets/facades/
├── train/           # Paired images {A,B}
└── test/            # Paired test images {A,B}
```

### Download Datasets

Use the provided script:

```bash
cd Dissertation/pytorch-CycleGAN-and-pix2pix-film-lora
python ./util/get_data.py --dataset_name horse2zebra
```

**Available Datasets:**
- `horse2zebra`: 🐴 ↔ 🦓
- `photo2painting`: 📸 ↔ 🎨
- `cityscapes`: 🌆 (Street scenes)
- `coco`: Various scenes
- `facades`: 🏢 (Architectural)
- `sat2map`: 🛰️ (Satellite to map)

### Custom Datasets

1. Organize images into `trainA`, `trainB`, `testA`, `testB` directories
2. Specify dataset path:
```bash
--dataroot /path/to/your/dataset --dataset_mode unaligned
```

---

## Training

### Basic Training Command

```bash
python train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_exp1 \
    --model cycle_gan \
    --batch_size 4 \
    --n_epochs 200 \
    --lr 0.0002 \
    --lambda_A 10 \
    --lambda_B 10
```

### Important Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 1 | Batch size for training |
| `--n_epochs` | 200 | Number of training epochs |
| `--lr` | 0.0002 | Initial learning rate |
| `--lambda_A` / `--lambda_B` | 10 | Cycle consistency weights |
| `--lambda_identity` | 0.5 | Identity loss weight |
| `--pool_size` | 50 | Image buffer size for discriminators |
| `--load_size` | 286 | Load image to this size |
| `--crop_size` | 256 | Crop image to this size |
| `--continue_train` | - | Resume from checkpoint |

### Advanced Training

#### Enable Continuing Training
```bash
python train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_exp1 \
    --continue_train \
    --epoch_count 101
```

#### Distributed Training (Multi-GPU)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_distributed
```

#### Log to Weights & Biases
```bash
python train.py \
    --dataroot ./datasets/horse2zebra \
    --name horse2zebra_wandb \
    --use_wandb --wandb_project_name cyclegan_experiments
```

### Monitoring Training

Training outputs are saved to:
```
checkpoints/{experiment_name}/
├── web/                              # Visualizations (HTML)
├── latest_net_G_A.pth               # Generator A weights
├── latest_net_D_A.pth               # Discriminator A weights
├── loss_log.txt                     # Training losses
└── train_opt.txt                    # Training configuration
```

View loss plots:
```bash
python util/plot_losses.py checkpoints/horse2zebra_exp1/loss_log.txt
```

---

## Testing & Inference

### Generate Results

```bash
python test.py \
    --dataroot ./datasets/horse2zebra/testA \
    --name horse2zebra_exp1 \
    --model test \
    --phase test
```

Results saved to:
```
results/{experiment_name}/images/
```

### Use Pretrained Models

```bash
python test.py \
    --dataroot ./datasets/horse2zebra/testA \
    --name horse2zebra_pretrained \
    --model test \
    --checkpoints_dir ./checkpoints/horse2zebra_040326
```

### Custom Image Translation

```python
import torch
from models import create_model
from util.image_pool import ImagePool

# Load trained model
model = create_model(opt)
model.load_networks('latest')
model.eval()

# Translate image
with torch.no_grad():
    model.set_input({'A': image_tensor})
    model.forward()
    output = model.fake_B
```

### Batch Processing

```bash
python test.py \
    --dataroot ./custom_images \
    --name my_model \
    --phase test \
    --num_test 1000
```

---

## Configuration

### Training Options (`options/train_options.py`)

```python
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr_decay_iters', type=int, default=50)
parser.add_argument('--lambda_A', type=float, default=10.0)
parser.add_argument('--lambda_identity', type=float, default=0.5)
parser.add_argument('--gan_mode', type=str, default='lsgan')
parser.add_argument('--norm', type=str, default='instance')
parser.add_argument('--init_type', type=str, default='normal')
parser.add_argument('--init_gain', type=float, default=0.02)
```

### Model Selection

By **generator type**:
```bash
--netG resnet_9blocks      # 9-block ResNet
--netG resnet_6blocks      # 6-block ResNet
--netG unet_256            # U-Net (256x256)
--netG unet_128            # U-Net (128x128)
--netG resnet_9blocks_film # ResNet + FiLM
```

By **discriminator type**:
```bash
--netD basic               # Standard PatchGAN
--netD n_layers            # Multi-scale PatchGAN
--netD pixel               # Pixel-level discriminator
```

### Loss Functions

```bash
--gan_mode vanilla         # Standard GAN loss
--gan_mode lsgan          # Least-square GAN
--gan_mode wgangp         # Wasserstein GAN with gradient penalty
--gan_mode hinge          # Hinge loss
```

---

## Results & Checkpoints

### Pre-trained Models

Available checkpoints in `checkpoints/`:

| Model | Date | Dataset | Accuracy | Size |
|-------|------|---------|----------|------|
| `horse2zebra_020326` | 2026-03-02 | horse2zebra | - | ~200MB |
| `horse2zebra_040326` | 2026-03-04 | horse2zebra | - | ~200MB |

### Loading Checkpoints

```bash
# Use specific epoch
python test.py \
    --name horse2zebra_040326 \
    --epoch 15 \
    --load_iter 0
```

### Checkpoint Structure

```
checkpoints/experiment_name/
├── 5_net_G_A.pth          # Generator A at epoch 5
├── 5_net_D_A.pth          # Discriminator A at epoch 5
├── latest_net_G_A.pth     # Latest weights
├── latest_net_D_A.pth
├── latest_net_G_B.pth
├── latest_net_D_B.pth
├── train_opt.txt          # Training configuration
├── loss_log.txt           # Training metrics
└── web/                   # HTML visualizations
    └── index.html
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
- [pix2pix: Image-to-Image Translation with Conditional Adversarial Nets](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
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

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Additional model architectures
- [ ] Performance optimizations
- [ ] Extended documentation
- [ ] Example notebooks
- [ ] Unit tests

---

## License

[Specify your license here - e.g., MIT, Apache 2.0]

---

## Contact & Support

For questions, issues, or feedback:
- 📧 Email: [your.email@university.edu]
- 🐛 Issues: [GitHub Issues]
- 💬 Discussions: [GitHub Discussions]

---

## Acknowledgments

Built upon:
- [PyTorch CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Junyanz Zhou
- Erik Linder's CycleGAN implementation
- FiLM conditioning mechanism
- LoRA adaptation techniques

---

**Last Updated:** April 2026  
**Status:** Active Development 🚀

