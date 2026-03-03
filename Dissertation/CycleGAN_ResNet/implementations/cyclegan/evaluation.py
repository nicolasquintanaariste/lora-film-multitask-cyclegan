import argparse

from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

import os

# ------------------------------
# Parse command-line arguments
# ------------------------------
parser = argparse.ArgumentParser(description="Compute FID score between two image folders")
parser.add_argument(
    "--folder1",
    type=str,
    default=r"C:\Users\usuario\OneDrive - TCDUD.onmicrosoft.com\Documents\Nicolas\University\Academic\Year 5\Repositories\MAI\Dissertation\CycleGAN_erik_linder\data\dummy_data\test\A",
    help="Path to first folder (default is local)"
)
parser.add_argument(
    "--folder2",
    type=str,
    default=r"C:\Users\usuario\OneDrive - TCDUD.onmicrosoft.com\Documents\Nicolas\University\Academic\Year 5\Repositories\MAI\Dissertation\CycleGAN_erik_linder\data\dummy_data\train\A",
    help="Path to second folder (default is local)"
)
args = parser.parse_args()

folder_path1 = args.folder1
folder_path2 = args.folder2

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
])

# Load all images
img_tensors = []
for filename in os.listdir(folder_path1):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path1, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor_single = transform(img)  # (3, 299, 299)
        img_tensors.append(img_tensor_single)

img_tensors1 = torch.stack(img_tensors) # (N, 3, 299, 299)

img_tensors = []
for filename in os.listdir(folder_path2):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path2, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor_single = transform(img)  # (3, 299, 299)
        img_tensors.append(img_tensor_single)

img_tensors2 = torch.stack(img_tensors) # (N, 3, 299, 299)

# Create the metric object
fid = FrechetInceptionDistance(feature=2048)  # 2048 = Inception-V3 last pooling layer

fid.reset()
fid.update(img_tensors2, real=True)
fid.update(img_tensors1, real=False)

fid_score = fid.compute()
print("FID score:", fid_score.item())


# # Create random images for testing. Display them
# imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
# img = imgs_dist1[0]  # shape (3, 299, 299)
# img = img.permute(1, 2, 0)  # now shape is (299, 299, 3)
# img = img.float() / 255.0
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)