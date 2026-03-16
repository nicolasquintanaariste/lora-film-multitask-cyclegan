import random
import time
import datetime
import sys
import os
import shutil
from statistics import mean

from torch.autograd import Variable
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image, make_grid
from datasets import ImageDataset
import torchvision.transforms as transforms
from torch.autograd import Variable

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

def sample_images(epoch, opt, G_AB, G_BA, task2id, image_folder, Tensor, seed=None):
    """Saves a generated sample from the validation loaders for all tasks."""
    # Test data loader
    val_loaders = {}
    n_samples = 5
    
    val_transforms_ = [
        transforms.Resize(int(256 * 1.12)),  # Resize shortest side
        transforms.CenterCrop((256, 256)),  # Center crop to square (deterministic)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    for task in opt.tasks:
        val_dataset = ImageDataset(
            root=os.path.join(opt.dataroot_general, task),
            transforms_=val_transforms_,
            unaligned=False if seed else True,
            mode="test"
        )
        # Determine how many samples we can take
        dataset_len = len(val_dataset)
        if dataset_len == 0:
            continue  # nothing to sample
        rng = np.random.default_rng(seed=seed)
        take = min(n_samples, dataset_len)
        # when dataset shorter than desired, just sample all without replacement
        indices = rng.choice(dataset_len, size=take, replace=False)
        fixed_subset = Subset(val_dataset, indices)
        
        val_loaders[task] = DataLoader(fixed_subset, batch_size=take, shuffle=False, num_workers=0)
    
    G_AB.eval()
    G_BA.eval()
    
    for task, loader in val_loaders.items():
        tid = torch.tensor([task2id[task]], device=next(G_AB.parameters()).device, dtype=torch.long)
        fake_A, fake_B, real_A, real_B = infer(loader, G_AB, G_BA, Tensor, tid)
        
        # Arrange images along x-axis
        real_A_grid = make_grid(real_A, nrow=5, normalize=True)
        fake_B_grid = make_grid(fake_B, nrow=5, normalize=True)
        real_B_grid = make_grid(real_B, nrow=5, normalize=True)
        fake_A_grid = make_grid(fake_A, nrow=5, normalize=True)
        
        # Arrange images along y-axis: [real_A | fake_B | real_B | fake_A]
        image_grid = torch.cat((real_A_grid, fake_B_grid, real_B_grid, fake_A_grid), 1)
        
        # Save image with task name included
        os.makedirs(image_folder, exist_ok=True)
        save_image(
            image_grid,
            os.path.join(image_folder, f"{task}_{epoch}_standard.png" if seed else f"{task}_{epoch}.png"),
            normalize=False
        )
    
def infer(loader, G_AB, G_BA, Tensor, tid=None):
    G_AB.eval()
    G_BA.eval()
    
    try:
        imgs = next(iter(loader))
    except StopIteration:
        # Re-create iterator if exhausted
        loader_iter = iter(loader)
        imgs = next(loader_iter)
    
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    with torch.no_grad():
        fake_B = G_AB(real_A, tid)
        fake_A = G_BA(real_B, tid)
        
    return fake_A, fake_B, real_A, real_B    
        
def inspect_trainable(model, name="model"):
    total = 0
    trainable = 0

    lora = 0
    film = 0
    base = 0

    # Support both plain nn.Module and wrapper-style models (e.g., BaseModel with net* attributes).
    if hasattr(model, "named_parameters"):
        param_iter = model.named_parameters()
    elif hasattr(model, "model_names"):
        pairs = []
        for net_name in getattr(model, "model_names", []):
            if not isinstance(net_name, str):
                continue
            net = getattr(model, f"net{net_name}", None)
            if net is None:
                continue
            for n, p in net.named_parameters():
                pairs.append((f"net{net_name}.{n}", p))
        param_iter = pairs
    else:
        raise TypeError(
            "inspect_trainable expects either a torch.nn.Module or a model wrapper with `model_names` and `net*` attributes."
        )

    for n, p in param_iter:
        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()

            n_lower = n.lower()
            if "lora_" in n_lower:
                lora += p.numel()
            elif "film" in n_lower or "embed" in n_lower or "to_gamma_beta" in n_lower:
                film += p.numel()
            else:
                base += p.numel()

    print(f"\n[{name}]")
    print(f"Total params:      {total:,}")
    print(f"Trainable params:  {trainable:,}")
    print(f"  LoRA params:     {lora:,}")
    print(f"  FiLM params:     {film:,}")
    print(f"  Base params:     {base:,}")
    
class MultiTaskDataLoader:
    """Cycles through tasks round-robin, yielding (data, task_id) pairs."""
    
    def __init__(self, task_datasets: dict, max_iters_mode: str = "min"):
        """
        task_datasets: {task_id: dataset}
        max_iters_mode: "min" uses smallest dataset length, "max" uses largest
        """
        self.task_datasets = task_datasets
        self.max_iters_mode = max_iters_mode
        self._tid_cycle = list(task_datasets.keys())
        self._tid_index = -1

    @property
    def iters_per_epoch(self):
        sizes = [len(ds) for ds in self.task_datasets.values()]
        if self.max_iters_mode == "min":
            return min(sizes)
        elif self.max_iters_mode == "max":
            return max(sizes)
        elif self.max_iters_mode == "avg":
            return int(mean(sizes))

    def next_tid(self):
        self._tid_index = (self._tid_index + 1) % len(self._tid_cycle)
        return self._tid_cycle[self._tid_index]

    def set_epoch(self, epoch):
        for dataset in self.task_datasets.values():
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)