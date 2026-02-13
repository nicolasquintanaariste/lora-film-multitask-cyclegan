import random
import time
import datetime
import sys
import os
import shutil
import json

from torch.autograd import Variable
import torch
import numpy as np

from contextlib import contextmanager


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def copy_missing(src, dst):
    os.makedirs(dst, exist_ok=True)

    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dst_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(dst_root, exist_ok=True)

        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_root, f)

            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)


def save_hyperparameters(opt, save_path):
    """Save hyperparameters to JSON file."""
    details_dir = os.path.join(save_path, "details")
    os.makedirs(details_dir, exist_ok=True)

    out_path = os.path.join(details_dir, "hyperparams.json")
    with open(out_path, "w") as f:
        json.dump(vars(opt), f, indent=4)


class PhaseTimer:
    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync
        self.totals = {}  # name -> seconds
        self.counts = {}  # name -> number of times recorded

    def _sync(self):
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def track(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def as_dict(self):
        out = {}
        for k, v in self.totals.items():
            out[k] = {
                "seconds": float(v),
                "count": int(self.counts.get(k, 0)),
                "avg_seconds": float(v / max(1, self.counts.get(k, 1))),
            }
        return out


def save_run_summary(save_path: str, summary: dict):
    details_dir = os.path.join(save_path, "details")
    os.makedirs(details_dir, exist_ok=True)    

    with open(os.path.join(details_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
def seed_everything(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(f"Seed set to {seed}")
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
