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
EPOCHS = 3                              # Keep it light for now
BATCH_SIZE = 4                          # Safe for 8GB RAM
IMAGE_HEIGHT = 128                     # Faster training, test-size
IMAGE_WIDTH = 128
LEARNING_RATE = 1e-4
SAVE_CHECKPOINTS = True

# ===== TRANSFORMS =====
train_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    # A.HorizontalFlip(p=0.5),          # Optional: Disable for speed
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ===== DATASET & DATALOADER =====
train_ds = RoadDataset("data/train_images", "data/train_masks", transform=train_transform)
val_ds = RoadDataset("data/val_images", "data/val_masks", transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# ===== MODEL, LOSS, OPTIMIZER =====
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== TRAINING FUNCTION =====
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
    for epoch in range(EPOCHS):
        print(f"\n📀 Epoch [{epoch+1}/{EPOCHS}]")
        train_fn(train_loader, model, optimizer, loss_fn)

        if SAVE_CHECKPOINTS:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(checkpoint, f"checkpoints/road_unet_epoch{epoch+1}.pth")
            print(f"✅ Checkpoint saved for epoch {epoch+1}")

if __name__ == "__main__":
    main()
