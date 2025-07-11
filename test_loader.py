from dataset import RoadDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_dataset = RoadDataset("data/train_images", "data/train_masks")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

for i, (image, mask) in enumerate(train_loader):
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(image[0].permute(1, 2, 0))
    plt.title("Satellite Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask[0][0], cmap='gray')
    plt.title("Mask")

    plt.show()
    break
