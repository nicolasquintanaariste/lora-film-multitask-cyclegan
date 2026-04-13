import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class FiLMDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, task):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
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




# """Dataset class template

# This module provides a template for users to implement custom datasets.
# You can specify '--dataset_mode template' to use this dataset.
# The class name should be consistent with both the filename and its dataset_mode option.
# The filename should be <dataset_mode>_dataset.py
# The class name should be <Dataset_mode>Dataset.py
# You need to implement the following functions:
#     -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
#     -- <__init__>: Initialize this dataset class.
#     -- <__getitem__>: Return a data point and its metadata information.
#     -- <__len__>: Return the number of images.
# """

# from data.base_dataset import BaseDataset, get_transform
# import os

# from data.image_folder import make_dataset
# from PIL import Image
# import random


# class FiLMDataset(BaseDataset):
#     """A template dataset class for you to implement custom datasets."""

#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         """Add new dataset-specific options, and rewrite default values for existing options.

#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

#         Returns:
#             the modified parser.
#         """
#         parser.add_argument("--new_dataset_option", type=float, default=1.0, help="new dataset option")
#         parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
#         return parser

#     def __init__(self, opt, tasks):
#         """Initialize this dataset class.

#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

#         A few things can be done here.
#         - save the options (have been done in BaseDataset)
#         - get image paths and meta information of the dataset.
#         - define the image transformation.
#         """
#         # save the option and dataset root
#         BaseDataset.__init__(self, opt)
#         # get the image paths of your dataset;
#         self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
#         # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
#         self.transform = get_transform(opt)
        
#         self.taskId = random.randint(0, len(opt.tasks-1)) # Random interger to select the task performed for every batch
#         task2id = {task: i for i, task in enumerate(opt.tasks)} if opt.tasks else None
#         self.task_dataset = {}
        
#         for task in opt.tasks:
#             dir_A = os.path.join(opt.dataroot_general, task, opt.phase + "A")  # create a path '/path/to/data/trainA'
#             dir_B = os.path.join(opt.dataroot_general, task, opt.phase + "B")  # create a path '/path/to/data/trainB'

#             A_paths = sorted(make_dataset(dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
#             B_paths = sorted(make_dataset(dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
#             A_size = len(A_paths)  # get the size of dataset A
#             B_size = len(B_paths)  # get the size of dataset B
#             btoA = self.opt.direction == "BtoA"
#             input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
#             output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
#             transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
#             transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
#             tid = task2id[task]
            
#             self.tasks_dataset[tid] = {'dir_A':dir_A, 'dir_B':dir_B, 'A_paths':A_paths, 'B_paths':B_paths, 'A_size':A_size, 'B_size':B_size, 
#                                       'btoA':btoA, 'input_nc':input_nc, 'output_nc':output_nc, 'transform_A':transform_A, 'transform_B':transform_B}
            

#     def __getitem__(self, index):
#         """Return a data point and its metadata information.

#         Parameters:
#             index -- a random integer for data indexing

#         Returns:
#             a dictionary of data with their names. It usually contains the data itself and its metadata information.

#         Step 1: get a random image path: e.g., path = self.image_paths[index]
#         Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
#         Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
#         Step 4: return a data point as a dictionary.
#         """
#         # path = "temp"  # needs to be a string
#         # data_A = None  # needs to be a tensor
#         # data_B = None  # needs to be a tensor
#         # return {"data_A": data_A, "data_B": data_B, "path": path}
    
#         if self.taskId + 1 >= len(self.opt.tasks):
#             self.taskId = 0
#         else:
#             taskId += 1
#         dataset = self.get_task_dataset(taskId, index)
    
#     def get_task_dataset(self, tid, index):
#         A_path = self.task_dataset[tid]['A_paths'][index % self.task_dataset[tid]['A_size']] # make sure index is within then range
#         if self.opt.serial_batches:  # make sure index is within then range
#             index_B = index % self.task_dataset[tid]['B_size']
#         else:  # randomize the index for domain B to avoid fixed pairs.
#             index_B = random.randint(0, self.task_dataset[tid]['B_size'] - 1)
#         B_path = self.task_dataset[tid]['B_paths'][index_B]
#         A_img = Image.open(A_path).convert("RGB")
#         B_img = Image.open(B_path).convert("RGB")
#         # apply image transformation
#         A = self.task_dataset[tid]['transform_A'](A_img)
#         B = self.task_dataset[tid]['transform_B'](B_img)

#         return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

#     def __len__(self):
#         """Return the total number of images."""
#         return len(self.image_paths)
