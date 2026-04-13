# metrics_utils.py
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import gc

import torch
from torch_fidelity import calculate_metrics
from torchvision import transforms

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import ImageDatasetMetrics
from utils import infer
from datasets import ImageDataset


class MetricLogger:
    def __init__(self, csv_path=None):
        self.epochs = []
        self.task = []
        self.fid = []
        self.kid = []
        self.csv_path = csv_path
        self._wrote_header = False

    def log(self, epoch: int, fid: float, kid: float, task=None):
        self.epochs.append(int(epoch))
        self.fid.append(float(fid))
        self.kid.append(float(kid))
        self.task.append(task)

        if self.csv_path is not None:
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "task", "fid", "kid"])
                writer.writerow([epoch, task, fid, kid])
class FIDEvaluator:
    def __init__(self, opt, transforms_, Tensor, base_results_dir, task2id=None):
        self.opt = opt
        self.transforms_ = transforms_
        self.task2id = task2id
        self.Tensor = Tensor
        self.metric_logger = MetricLogger(
            csv_path=os.path.join(base_results_dir, "fid_kid.csv")
        )
        self.base_results_dir = base_results_dir
        self.real_dirs = {}  # populated by prep_real()

    def prep_real(self, timer=None):
        """Save real images once before training starts."""
        ctx = timer.track("prep/fid_save_real") if timer else contextlib.nullcontext()
        with ctx:
            for task in self.opt.tasks:
                real_dir = os.path.join(self.opt.dataroot_general, task, "testB_normalised")
                self.fid_save_real(
                    in_dir=os.path.join(self.opt.dataroot_general, task),
                    out_dir=real_dir,
                    max_images=250,
                )
                self.real_dirs[task] = real_dir

    def evaluate(self, epoch, netG_A, netG_B, timer=None):
        """Run FID/KID for all tasks and save plots."""
        ctx = timer.track("fid/compute") if timer else contextlib.nullcontext()
        with ctx:
            for task in self.opt.tasks:
                fid_image_dir_A = f"{self.base_results_dir}/fake/{task}/A"
                fid_image_dir_B = f"{self.base_results_dir}/fake/{task}/B"
                self.fid_inference(epoch, task, netG_A, netG_B, fid_image_dir_A, fid_image_dir_B)

                fake_dir = os.path.join(fid_image_dir_B, f"epoch{epoch:03d}")
                fid, kid = self.compute_fid_kid(self.real_dirs[task], fake_dir)

                self.metric_logger.log(epoch=epoch, fid=fid, kid=kid, task=task)

            self.plot_fid(out_path=os.path.join(self.base_results_dir, "fid.png"), show=False)
            self.plot_kid(out_path=os.path.join(self.base_results_dir, "kid.png"), show=False)
            
    def plot_fid(self, out_path=None, show=False):
        if self.metric_logger.csv_path is None or not os.path.exists(self.metric_logger.csv_path):
            print("No CSV path or file not found.")
            return

        df = pd.read_csv(self.metric_logger.csv_path)

        fig, ax = plt.subplots(figsize=(10, 4))

        if 'task' in df.columns and df['task'].notna().any():
            for task in df['task'].unique():
                subset = df[df['task'] == task]
                ax.plot(subset['epoch'], subset['fid'], marker="o", label=f"FID {task}")
        else:
            ax.plot(df['epoch'], df['fid'], marker="o", label="FID")

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
        
    def plot_kid(self, out_path=None, show=False):
        if self.metric_logger.csv_path is None or not os.path.exists(self.metric_logger.csv_path):
            print("No CSV path or file not found.")
            return

        df = pd.read_csv(self.metric_logger.csv_path)

        fig, ax = plt.subplots(figsize=(10, 4))

        if 'task' in df.columns and df['task'].notna().any():
            for task in df['task'].unique():
                subset = df[df['task'] == task]
                ax.plot(subset['epoch'], subset['kid'], marker="o", label=f"KID {task}")
        else:
            ax.plot(df['epoch'], df['kid'], marker="o", label="KID")

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
    
    def fid_inference(self, epoch, task, G_AB, G_BA, fid_image_dir_A, fid_image_dir_B):
        fid_max_imgs = 250
        transforms_list = self.transforms_.transforms if isinstance(self.transforms_, transforms.Compose) else self.transforms_
        fid_dataset = ImageDataset(
            root=os.path.join(self.opt.dataroot_general, task),
            transforms_=transforms_list,
            unaligned=True,
            mode="test"
        )
        fid_loader = DataLoader(fid_dataset, batch_size=fid_max_imgs, shuffle=True, num_workers=0)

        gc.collect()
        torch.cuda.empty_cache()

        if self.task2id:
            tid = torch.tensor([self.task2id[task]], device=next(G_AB.parameters()).device, dtype=torch.long)
        else:
            tid = None
        fake_A, fake_B, real_A, real_B = infer(fid_loader, G_AB, G_BA, self.Tensor, tid)

        epoch_dir_A = os.path.join(fid_image_dir_A, f"epoch{epoch:03d}")
        epoch_dir_B = os.path.join(fid_image_dir_B, f"epoch{epoch:03d}")
        os.makedirs(epoch_dir_A, exist_ok=True)
        os.makedirs(epoch_dir_B, exist_ok=True)

        fake_A = (fake_A.detach().cpu() * 0.5 + 0.5)
        fake_B = (fake_B.detach().cpu() * 0.5 + 0.5)

        for i in range(fake_A.size(0)):
            save_image(fake_A[i], os.path.join(epoch_dir_A, f"{i:04d}.png"), normalize=False)
            save_image(fake_B[i], os.path.join(epoch_dir_B, f"{i:04d}.png"), normalize=False)

        
    def compute_fid_kid(self, real_dir, fake_dir, kid_default=1000):
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

    def fid_save_real(self, in_dir, out_dir, max_images=250):
        """
        Saves normalised real images for FID/KID computation.
        Runs once before training.
        """
        os.makedirs(out_dir, exist_ok=True)

        transforms_list = self.transforms_.transforms if isinstance(self.transforms_, transforms.Compose) else self.transforms_
        dataset = ImageDatasetMetrics(
            root=in_dir,
            transforms_=transforms_list,
            unaligned=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=0,
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