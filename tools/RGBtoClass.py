import os
from PIL import Image
import numpy as np

root_dir = r"E:\prov-gigapath\outputs\preprocessing\output\val_Lab\mask"

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            img_path = os.path.join(dirpath, filename)

            img = Image.open(img_path).convert("L")
            arr = np.array(img)

            arr = (arr > 0).astype("uint8")

            Image.fromarray(arr, mode="L").save(img_path)
