import os
import cv2
from glob import glob

def convert_tiff_to_jpg(source_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for filepath in glob(f"{source_folder}/*.tiff"):
        img = cv2.imread(filepath)
        if img is not None:
            filename = os.path.basename(filepath).replace(".tiff", ".jpg")
            cv2.imwrite(os.path.join(dest_folder, filename), img)
        else:
            print(f"⚠️ Skipping unreadable file: {filepath}")

# Example usage:
convert_tiff_to_jpg("/content/drive/MyDrive/data/train_images", "/content/drive/MyDrive/data/train_images_jpg")
convert_tiff_to_jpg("/content/drive/MyDrive/data/train_masks", "/content/drive/MyDrive/data/train_masks_jpg")
convert_tiff_to_jpg("/content/drive/MyDrive/data/val_images", "/content/drive/MyDrive/data/val_images_jpg")
convert_tiff_to_jpg("/content/drive/MyDrive/data/val_masks", "/content/drive/MyDrive/data/val_masks_jpg")
