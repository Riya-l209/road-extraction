import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import UNet
from PIL import Image

# ========= CONFIG =============
CHECKPOINT_PATH = "/content/checkpoints/road_unet_epoch5.pth"
TEST_IMAGE_PATH = "/content/drive/MyDrive/data/train_images_jpg/24479185_15.jpg"
IMG_SIZE = (128, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================

# 🧠 Load model and weights
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 🔄 Define image transform
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# 📸 Read and preprocess image
img_bgr = cv2.imread(TEST_IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {TEST_IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, H, W]

# 🔮 Run prediction
with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

# 🖼️ Show results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap='gray')
plt.title("Predicted Road Mask")
plt.axis('off')

plt.tight_layout()
plt.show()

# 💾 Save output
cv2.imwrite("output_mask.jpg", (pred_mask * 255).astype(np.uint8))
print("✅ Output mask saved as 'output_mask.jpg'")
