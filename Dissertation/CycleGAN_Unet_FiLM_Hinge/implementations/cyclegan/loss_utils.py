# loss_utils.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F

# Hinge for Adversarial loss
def d_hinge_loss(pred_real, pred_fake):
    return 0.5 * (torch.mean(F.relu(1.0 - pred_real)) + torch.mean(F.relu(1.0 + pred_fake)))

def g_hinge_loss(pred_fake):
    return -torch.mean(pred_fake)

class LossLogger:
    """
    Stores step-wise losses in memory, can also append to CSV.
    """
    def __init__(self, csv_path=None):
        self.data = defaultdict(list)
        self.csv_path = csv_path
        self._csv_inited = False

    def log(self, step: int, **metrics):
        self.data["step"].append(step)

        for k, v in metrics.items():
            if v is None:
                self.data[k].append(np.nan)
            else:
                self.data[k].append(float(v))

        if self.csv_path is not None:
            self._append_csv(step, metrics)

    def _append_csv(self, step, metrics):
        fieldnames = ["step"] + list(metrics.keys())

        mode = "a"
        if not os.path.exists(self.csv_path):
            mode = "w"

        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerow({"step": step, **{k: (None if metrics[k] is None else float(metrics[k])) for k in metrics}})

def ema(values, alpha=0.1):
    out = []
    m = None
    for x in values:
        if np.isnan(x):
            out.append(np.nan)
            continue
        m = x if m is None else (alpha * x + (1 - alpha) * m)
        out.append(m)
    return out

def plot_losses(logger: LossLogger, out_path=None, smooth_alpha=0.1, last_n=None, show=False):
    steps = np.array(logger.data.get("step", []), dtype=float)
    if len(steps) == 0:
        print("No loss data to plot.")
        return

    sl = slice(None)
    if last_n is not None and len(steps) > last_n:
        sl = slice(-last_n, None)

    def series(name):
        vals = np.array(logger.data.get(name, []), dtype=float)[sl]
        if smooth_alpha is None:
            return vals
        return np.array(ema(vals, alpha=smooth_alpha), dtype=float)

    x = steps[sl]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Generator
    for key, label in [("loss_G", "G total"), ("loss_GAN", "G adv"), ("loss_cycle", "Cycle"), ("loss_identity", "Identity")]:
        if key in logger.data:
            axes[0].plot(x, series(key), label=label)
    axes[0].set_title("Generator losses")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Discriminators
    for key, label in [("loss_D", "D total"), ("loss_D_A", "D_A"), ("loss_D_B", "D_B")]:
        if key in logger.data:
            axes[1].plot(x, series(key), label=label)
    axes[1].set_title("Discriminator losses")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Optional signals
    plotted_any = False
    for key, label in [
        ("dA_real_mean", "D_A(real) mean"),
        ("dA_fake_mean", "D_A(fake) mean"),
        ("dB_real_mean", "D_B(real) mean"),
        ("dB_fake_mean", "D_B(fake) mean"),
    ]:
        if key in logger.data:
            axes[2].plot(x, series(key), label=label)
            plotted_any = True

    axes[2].set_title("Discriminator signals")
    if plotted_any:
        axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Training step")

    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
