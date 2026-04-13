import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

from PIL import Image
import random
from statistics import mean


class MultitaskUnalignedDataset(BaseDataset):

    def __init__(self, opt, model):
        super().__init__(opt)
        # Multitask logic
        self.model = model
        self.task2id = opt.task2id
        self.task_datasets = {}
        for task in opt.tasks:
            self.task_datasets[self.task2id[task]] = UnalignedDatasetTask(opt, task)
        
        self.max_iters = self.iters_per_epoch(opt.max_iters_mode)
        self._tid_cycle = list(self.task_datasets.keys())
        
    def iters_per_epoch(self, max_iters_mode):
        self.sizes = [len(ds) for ds in self.task_datasets.values()]
        if max_iters_mode == "min":
            return min(self.sizes)
        elif max_iters_mode == "max":
            return max(self.sizes)
        elif max_iters_mode == "avg":
            return int(mean(self.sizes))

    def __getitem__(self, index):
        index = index * self.max_iters * 99 # very hand wavy way of ensuring that entire datset can be utilised
        tid_idx = index % len(self._tid_cycle)
        tid = self._tid_cycle[tid_idx]
        task_dataset = self.task_datasets[tid]
        task_idx = index // len(self._tid_cycle) 
        A_path = task_dataset.A_paths[task_idx % task_dataset.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = task_idx % task_dataset.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, task_dataset.B_size - 1)
        B_path = task_dataset.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = task_dataset.transform_A(A_img)
        B = task_dataset.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path, "tid": tid}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.max_iters

class UnalignedDatasetTask(BaseDataset):
    def __init__(self, opt, task):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot_general, task, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot_general, task, opt.phase + "B")  # create a path '/path/to/data/trainB'

        task_limits = getattr(opt, "max_dataset_size_by_task_map", {})
        task_max_dataset_size = task_limits.get(task, opt.max_dataset_size)

        self.A_paths = sorted(make_dataset(self.dir_A, task_max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, task_max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
