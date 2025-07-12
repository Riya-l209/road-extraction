import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataset import RoadDataset
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
EPOCHS = 20

# Google Drive paths (make sure these exist)
TRAIN_IMG_DIR = "/content/drive/MyDrive/data/train_images"
TRAIN_MASK_DIR = "/content/drive/MyDrive/data/train_masks"
VAL_IMG_DIR = "/content/drive/MyDrive/data/val_images"
VAL_MASK_DIR = "/content/drive/MyDrive/data/val_masks"
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"

# ===== TRANSFORMS =====
train_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# ===== LOAD DATASETS =====
train_ds = RoadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_ds = RoadDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# ===== MODEL, LOSS, OPTIMIZER =====
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== TRAINING LOOP =====
def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(data)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Batch [{batch_idx}]")
        loop.set_postfix(loss=loss.item())

# ===== MAIN FUNCTION =====
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch [{epoch+1}/{EPOCHS}] on {DEVICE}")
        train_fn(train_loader, model, optimizer, loss_fn)

        # Save checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{CHECKPOINT_DIR}/road_unet_epoch{epoch+1}.pth")
        print(f"✅ Checkpoint saved for epoch {epoch+1}")

if __name__ == "__main__":
    main()

