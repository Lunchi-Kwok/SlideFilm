# 🧬 Prov-GigaPath_HRSeg

Prov-GigaPath_HRSeg is a high-resolution segmentation framework **built upon** [prov-gigapath](https://github.com/prov-gigapath/prov-gigapath/tree/main).  
It enables efficient whole-slide image (WSI) tiling, model training, fine-tuning, inference, and stitching into full-resolution segmentation outputs.

You can check out the [notebook](https://colab.research.google.com/drive/1CZeEeDWHyMiRb0DMe_UNh7Qlj7Ov194t?usp=sharing) to get started!

---

## 📂 1. Preparing Datasets

1. **Use supported WSIs**
   - Ensure your WSIs can be read by `openslide`.
   - If the format is not supported, use  [process_img.py](tools/process_img.py) and [process_mask](tools/process_mask.py) to convert them into formats compatible with `prov-gigapath`.

2. **Determine appropriate level**
   - Select the correct WSI level based on your dataset’s **micron-per-pixel (MPP)** value.

3. **Break WSIs into tiles**
   - Breaking WSIs into tiles

4. **Align images and masks**
   - Run [imgmask_same.py](tools/imgmask_same.py) to ensure the `img` and `mask` folders contain **the same number of files** with **identical names**.

5. **Convert RGB masks to class maps (if needed)**
   - If your masks are in RGB format, convert them to class-wise integer labels using [RGBtoClass.py](tools/RGBtoClass.py)
---

## 🚀 2. Inference

To directly test a trained model, run: [inference.py](inference.py)

## 🧠 3. Training from Scratch

If you want to train the segmentation model from the beginning, run: [main.py](main.py)

## 🔧 4. Fine-tuning a Pretrained Model

If you have a pretrained model and want to fine-tune it on a new dataset, run: [finetune.py](finetune.py)

## 🧩 5. Stitching Tile-Level Predictions

After inference, use the following command to merge all tile-level predictions back into full-resolution WSIs: [stitches.py](tools/stitches.py)