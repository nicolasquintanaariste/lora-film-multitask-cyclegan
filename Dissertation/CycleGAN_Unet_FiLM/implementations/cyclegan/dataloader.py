from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class Day2NightDatasetMetrics(Dataset):
    def __init__(self, root_day, root_night, transform=None, size=299):
        self.root_day = root_day
        self.root_night = root_night
        self.transform = transform
        self.size = size

        self.day_images = os.listdir(root_day)
        self.night_images = os.listdir(root_night)
        self.length_dataset = max(len(self.night_images), len(self.day_images))

        self.day_len = len(self.day_images)
        self.night_len = len(self.night_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        day_img = self.day_images[index % self.day_len]
        night_img = self.night_images[index % self.night_len]

        day_path = os.path.join(self.root_day, day_img)
        night_path = os.path.join(self.root_night, night_img)

        day_img = np.array(Image.open(day_path).convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS))
        night_img = np.array(Image.open(night_path).convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS))

        day_img = np.array(day_img)
        night_img = np.array(night_img)

        day_img = self.transform(image=day_img)["image"]
        night_img = self.transform(image=night_img)["image"]

        return night_img, day_img