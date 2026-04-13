"""Standalone inference script for the FT-LoRA multi-task CycleGAN model.

Reads hyperparams.json from the checkpoint to reconstruct the architecture
automatically (including per-task LoRA ranks for the finetune adapter),
then conditions the generator on a specific task via its LoRA adapter.

Usage:
  # Translate a folder of images A -> B for task 'horse2zebra'
  python infer.py --input path/to/images --checkpoint checkpoints/FTLoRA_... --task horse2zebra

  # Translate B -> A
  python infer.py --input path/to/images --checkpoint checkpoints/FTLoRA_... --task horse2zebra --direction BtoA

  # Pick a specific epoch's weights
  python infer.py --input path/to/images --checkpoint checkpoints/FTLoRA_... --task horse2zebra --epoch 50

  # Custom output folder
  python infer.py --input path/to/images --checkpoint checkpoints/FTLoRA_... --task horse2zebra --out results/my_run
"""

import argparse
import glob
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.networks import define_G


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_hyperparams(checkpoint_path: str) -> dict:
    """Load hyperparams.json from checkpoint folder (searches root and details/)."""
    candidates = [
        os.path.join(checkpoint_path, "hyperparams.json"),
        os.path.join(checkpoint_path, "details", "hyperparams.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"hyperparams.json not found in '{checkpoint_path}' or '{checkpoint_path}/details/'"
    )


def build_lora_ranks(hp: dict) -> list:
    """Reconstruct the per-task LoRA rank list matching training-time logic."""
    tasks = hp["tasks"]
    lora_rank = hp.get("lora_rank", 4)
    ft_task = hp.get("finetune_lora")
    ft_rank = hp.get("finetune_lora_rank") or lora_rank
    return [
        ft_rank if (ft_task and task == ft_task) else lora_rank
        for task in tasks
    ]


def build_generator(checkpoint_path: str, direction: str, epoch: str,
                    device: torch.device, hp: dict) -> torch.nn.Module:
    """Build and load a FT-LoRA generator from a checkpoint file."""
    tasks = hp["tasks"]
    num_tasks = len(tasks)
    lora_ranks = build_lora_ranks(hp)

    suffix = "G_A" if direction == "AtoB" else "G_B"
    weight_file = os.path.join(checkpoint_path, f"{epoch}_net_{suffix}.pth")
    if not os.path.isfile(weight_file):
        raise FileNotFoundError(f"Weights not found: {weight_file}")

    net = define_G(
        input_nc=hp.get("input_nc", 3),
        output_nc=hp.get("output_nc", 3),
        ngf=hp.get("ngf", 64),
        netG=hp.get("netG", "resnet_9blocks_lora"),
        norm=hp.get("norm", "instance"),
        use_dropout=not hp.get("no_dropout", True),
        init_type=hp.get("init_type", "normal"),
        init_gain=hp.get("init_gain", 0.02),
        num_tasks=num_tasks,
        lora_ranks=lora_ranks,
    )
    state = torch.load(weight_file, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    print(f"Loaded weights from: {weight_file}")
    print(f"LoRA ranks per task: { {t: r for t, r in zip(tasks, lora_ranks)} }")
    return net


def collect_image_paths(input_path: str) -> list:
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
    return transform(img).unsqueeze(0)


def postprocess(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1] for saving."""
    return (tensor.clamp(-1, 1) + 1) / 2


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained FT-LoRA multi-task CycleGAN generator.")
    parser.add_argument("--input",      required=True, help="Path to an image file or folder of images")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint folder (e.g. checkpoints/FTLoRA_...)")
    parser.add_argument("--task",       required=True, help="Task name to condition on (must be in the checkpoint's task list)")
    parser.add_argument("--direction",  default="AtoB", choices=["AtoB", "BtoA"], help="Translation direction")
    parser.add_argument("--epoch",      default="latest", help="Which epoch weights to load (e.g. latest, 50)")
    parser.add_argument("--out",        default=None, help="Output folder (default: <checkpoint>/infer_<task>_<direction>_<epoch>)")
    args = parser.parse_args()

    hp = load_hyperparams(args.checkpoint)
    tasks = hp["tasks"]
    if args.task not in tasks:
        raise ValueError(f"Task '{args.task}' not found in checkpoint tasks: {tasks}")
    tid = tasks.index(args.task)
    print(f"Task '{args.task}' -> tid={tid}  (all tasks: {tasks})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = args.out or os.path.join(args.checkpoint, f"infer_{args.task}_{args.direction}_{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    net = build_generator(args.checkpoint, args.direction, args.epoch, device, hp)

    image_paths = collect_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s). Saving output to: {out_dir}")

    tid_tensor = torch.tensor(tid, device=device)
    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        inp = preprocess(path).to(device)
        with torch.no_grad():
            out = net(inp, tid=tid_tensor)
        out_path = os.path.join(out_dir, f"{name}_translated.png")
        save_image(postprocess(out), out_path)
        print(f"  {os.path.basename(path)} -> {os.path.basename(out_path)}")

    print("Done.")


if __name__ == "__main__":
    main()
