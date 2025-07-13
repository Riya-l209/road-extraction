# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        self.masks = [file for file in os.listdir(mask_dir) if file.endswith('.jpg')]

        # Keep only matching files
        self.images = [f for f in self.images if f in self.masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # grayscale
        except Exception as e:
            print(f"❌ Error loading {img_path} or {mask_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask