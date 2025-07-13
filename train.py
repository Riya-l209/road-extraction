import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import RoadDataset
from model import UNet
from tqdm import tqdm
import os, time

# Paths
TRAIN_IMG_DIR = "/content/drive/MyDrive/data/train_images_jpg"
TRAIN_MASK_DIR = "/content/train_masks_jpg_local"
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configs
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🧠 Initial VRAM usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset + Loader
train_ds = RoadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# Model, loss, optimizer
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, leave=False)
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        preds = model(data)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if DEVICE == "cuda":
            vram = torch.cuda.memory_allocated(0) / 1024**2
            loop.set_postfix(loss=loss.item(), vram=f"{vram:.1f} MB")
        else:
            loop.set_postfix(loss=loss.item())

# Train Loop
for epoch in range(EPOCHS):
    start_time = time.time()
    print(f"\n🎯 Epoch {epoch+1}/{EPOCHS} started...")

    train_fn(train_loader, model, optimizer, loss_fn)

    # Save checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, f"{CHECKPOINT_DIR}/road_unet_epoch{epoch+1}.pth")

    duration = time.time() - start_time
    print(f"✅ Epoch {epoch+1} done in {duration:.2f} seconds")

    if DEVICE == "cuda":
        vram = torch.cuda.memory_allocated(0) / 1024**2
        print(f"💾 VRAM used after epoch {epoch+1}: {vram:.2f} MB")


