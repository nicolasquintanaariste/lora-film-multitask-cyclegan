# metrics_utils.py
import os
import csv
import matplotlib.pyplot as plt

import torch
from torch_fidelity import calculate_metrics

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import ImageDatasetMetrics



class MetricLogger:
    def __init__(self, csv_path=None):
        self.epochs = []
        self.fid = []
        self.kid = []
        self.csv_path = csv_path
        self._wrote_header = False

    def log(self, epoch: int, fid: float, kid: float):
        self.epochs.append(int(epoch))
        self.fid.append(float(fid))
        self.kid.append(float(kid))

        if self.csv_path is not None:
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "fid", "kid"])
                writer.writerow([epoch, fid, kid])

def plot_fid(metric_logger: MetricLogger, out_path=None, show=False):
    if len(metric_logger.epochs) == 0:
        print("No FID data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(metric_logger.epochs, metric_logger.fid, marker="o", label="FID")
    ax.set_title("FID per epoch")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    
def plot_kid(metric_logger: MetricLogger, out_path=None, show=False):
    if len(metric_logger.epochs) == 0:
        print("No KID data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(metric_logger.epochs, metric_logger.kid, marker="o", label="KID")
    ax.set_title("KID per epoch")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    
def compute_fid_kid(real_dir, fake_dir, kid_default=1000):
    def _count_images(d):
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    n_real = _count_images(real_dir)
    n_fake = _count_images(fake_dir)
    n_samples = min(n_real, n_fake)

    if n_samples == 0:
        raise ValueError(f"No images found in real_dir='{real_dir}' or fake_dir='{fake_dir}'")

    kid_subset = min(kid_default, n_samples)

    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        fid=True,
        kid=True,
        kid_subset_size=kid_subset,
        cuda=torch.cuda.is_available(),
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"]), float(metrics["kernel_inception_distance_mean"])

def fid_save_real(in_dir, out_dir, transforms_, max_images=250, batch_size=1, n_cpu=0):
    """
    Saves normalised real images for FID/KID computation.
    Runs once before training.
    """

    os.makedirs(out_dir, exist_ok=True)

    dataset = ImageDatasetMetrics(
        root=in_dir,
        transforms_=transforms_,
        unaligned=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        drop_last=False
    )

    saved = 0
    for batch in loader:
        real_B = batch["B"]

        for i in range(real_B.size(0)):
            if saved >= max_images:
                return

            img = real_B[i]

            # denormalise from [-1,1] → [0,1] before saving
            img = img * 0.5 + 0.5

            save_image(
                img,
                os.path.join(out_dir, f"real_{saved:04d}.png"),
                normalize=False
            )

            saved += 1
