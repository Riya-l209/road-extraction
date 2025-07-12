from flask import Flask, render_template, request
import os
import torch
import cv2
import numpy as np
from model import UNet
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = UNet(in_channels=3, out_channels=1)
checkpoint = torch.load("checkpoints/road_unet_epoch3.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Preprocessing
transform = Compose([
    Resize(128, 128),  # Change if your training size is different
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.png")
            file.save(input_path)

            # Read and preprocess the image
            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)["image"].unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = model(transformed)
                output = torch.sigmoid(output)
                mask = (output > 0.3).float().squeeze().numpy()

            # Save output mask
            output_path = os.path.join(UPLOAD_FOLDER, "output.png")
            plt.imsave(output_path, mask, cmap="gray")
            return render_template("index.html", input_image="input.png", output_image="output.png")

    return render_template("index.html", input_image=None, output_image=None)

if __name__ == "__main__":
    app.run(debug=True)
