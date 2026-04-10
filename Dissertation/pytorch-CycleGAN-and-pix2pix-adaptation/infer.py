"""Standalone inference script for the adaptation CycleGAN model.

Runs a trained generator on specific images without needing the full
dataset/options pipeline.

Usage:
  # Translate a folder of images A -> B (horse -> zebra)
  python infer.py --input path/to/images --checkpoint checkpoints/horse2zebra_040326_1 --direction AtoB

  # Translate a single image
  python infer.py --input path/to/horse.jpg --checkpoint checkpoints/horse2zebra_040326_1 --direction AtoB

  # Translate B -> A and pick a specific epoch's weights
  python infer.py --input path/to/images --checkpoint checkpoints/horse2zebra_040326_1 --direction BtoA --epoch 15

  # Custom output folder
  python infer.py --input path/to/images --checkpoint checkpoints/horse2zebra_040326_1 --out results/my_run
"""

import argparse
import glob
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Make sure imports resolve relative to this file's directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.networks import define_G


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def build_generator(checkpoint_path: str, direction: str, epoch: str, device: torch.device) -> torch.nn.Module:
    """Build and load a ResNet-9blocks generator from a checkpoint file."""
    suffix = "G_A" if direction == "AtoB" else "G_B"
    weight_file = os.path.join(checkpoint_path, f"{epoch}_net_{suffix}.pth")
    if not os.path.isfile(weight_file):
        raise FileNotFoundError(f"Weights not found: {weight_file}")

    net = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="resnet_9blocks",
        norm="instance",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
    )
    state = torch.load(weight_file, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    print(f"Loaded weights from: {weight_file}")
    return net


def collect_image_paths(input_path: str) -> list[str]:
    """Return sorted list of image paths from a file or directory."""
    if os.path.isfile(input_path):
        return [input_path]
    paths = sorted([
        p for ext in SUPPORTED_EXTS
        for p in glob.glob(os.path.join(input_path, f"*{ext}"))
    ])
    if not paths:
        raise FileNotFoundError(f"No supported images found in '{input_path}'")
    return paths


def preprocess(image_path: str) -> torch.Tensor:
    """Load an image and preprocess to a (1, 3, 256, 256) tensor in [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, C, H, W)


def postprocess(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1] for saving."""
    return (tensor.clamp(-1, 1) + 1) / 2


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained adaptation CycleGAN generator.")
    parser.add_argument("--input",      required=True, help="Path to an image file or folder of images")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint folder (e.g. checkpoints/horse2zebra_040326_1)")
    parser.add_argument("--direction",  default="AtoB", choices=["AtoB", "BtoA"], help="Translation direction")
    parser.add_argument("--epoch",      default="latest", help="Which epoch weights to load (e.g. latest, 5, 10, 15)")
    parser.add_argument("--out",        default=None, help="Output folder (default: <checkpoint>/infer_<direction>_<epoch>)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = args.out or os.path.join(args.checkpoint, f"infer_{args.direction}_{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    net = build_generator(args.checkpoint, args.direction, args.epoch, device)

    image_paths = collect_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s). Saving output to: {out_dir}")

    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        inp = preprocess(path).to(device)
        with torch.no_grad():
            out = net(inp)
        out_path = os.path.join(out_dir, f"{name}_translated.png")
        save_image(postprocess(out), out_path)
        print(f"  {os.path.basename(path)} -> {os.path.basename(out_path)}")

    print("Done.")


if __name__ == "__main__":
    main()
