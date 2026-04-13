"""Standalone script to generate a sample grid from pre-saved image folders.

Reads images from:
  A_real/       - real domain A images
  B_real/       - real domain B images
  A_generated/  - generated (fake) domain A images  (G_BA(real_B))
  B_generated/  - generated (fake) domain B images  (G_AB(real_A))

Images in each folder are sorted by filename and must be in 1-to-1
correspondence (same count, same order).

Output grid rows (top to bottom):
  real_A  |  fake_A  |  real_B  |  fake_B

Usage:
  python generate_grid.py                        # saves sample_grid.png next to this script
  python generate_grid.py --out my_grid.png      # custom output path
  python generate_grid.py --n 8                  # use first 8 images
"""

import argparse
import glob
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image


def load_images_sorted(folder: str, n: int | None = None) -> torch.Tensor:
    """Load all images from a folder sorted by filename, return (N, C, H, W) tensor in [-1, 1]."""
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    if not paths:
        raise FileNotFoundError(f"No images found in '{folder}'")
    if n is not None:
        paths = paths[:n]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))

    return torch.stack(tensors)  # (N, 3, H, W)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Generate a CycleGAN sample grid from image folders.")
    parser.add_argument("--root", default=script_dir, help="Root folder containing A_real/, B_real/, A_generated/, B_generated/")
    parser.add_argument("--out", default=os.path.join(script_dir, "sample_grid.png"), help="Output image path")
    parser.add_argument("--n", type=int, default=None, help="Number of images to use (default: all)")
    args = parser.parse_args()

    real_A    = load_images_sorted(os.path.join(args.root, "A_real"),      args.n)
    fake_A    = load_images_sorted(os.path.join(args.root, "A_generated"), args.n)
    real_B    = load_images_sorted(os.path.join(args.root, "B_real"),      args.n)
    fake_B    = load_images_sorted(os.path.join(args.root, "B_generated"), args.n)

    n = min(len(real_A), len(fake_A), len(real_B), len(fake_B))
    real_A, fake_A, real_B, fake_B = real_A[:n], fake_A[:n], real_B[:n], fake_B[:n]
    print(f"Using {n} images per row.")

    # Build one wide grid per row, then stack rows vertically
    real_A_grid = make_grid(real_A, nrow=n, normalize=True, padding=2)
    fake_A_grid = make_grid(fake_A, nrow=n, normalize=True, padding=2)
    real_B_grid = make_grid(real_B, nrow=n, normalize=True, padding=2)
    fake_B_grid = make_grid(fake_B, nrow=n, normalize=True, padding=2)

    image_grid = torch.cat((real_A_grid, fake_A_grid, real_B_grid, fake_B_grid), dim=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_image(image_grid, args.out, normalize=False)
    print(f"Saved grid to: {args.out}")


if __name__ == "__main__":
    main()
