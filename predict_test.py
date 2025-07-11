from model import UNet
import torch, cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

model = UNet(in_channels=3, out_channels=1)
checkpoint = torch.load("checkpoints/road_unet_epoch3.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

transform = Compose([
    Resize(128, 128),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

img = cv2.imread("data/val_images/10078660_15.tiff")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = transform(image=img)["image"].unsqueeze(0)

with torch.no_grad():
    out = model(input_tensor)
    out = torch.sigmoid(out)
    mask = (out > 0.3).float().squeeze().numpy()

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted")

plt.show()
