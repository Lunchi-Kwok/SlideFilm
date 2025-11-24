import os
import re
from pathlib import Path
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


def stitch_tiles(input_path, output_prefix="stitched"):
    input_path = Path(input_path)
    work_dir = input_path

    m = re.search(r'(\d+)[x_](\d+)\s*$', work_dir.name, re.IGNORECASE)
    if m:
        orig_w, orig_h = map(int, m.groups())
    else:
        nums = re.findall(r'\d+', work_dir.name)
        orig_w, orig_h = int(nums[-2]), int(nums[-1])

    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    pattern = re.compile(r"(-?\d+)x_(-?\d+)y", re.IGNORECASE)
    tiles = []

    for root, _, files in os.walk(work_dir):
        for f in files:
            if Path(f).suffix.lower() in img_exts:
                m = pattern.search(f)
                if not m:
                    continue
                x, y = map(int, m.groups())
                p = Path(root) / f
                with Image.open(p) as im:
                    w, h = im.size
                tiles.append({"path": p, "x": x, "y": y, "w": w, "h": h})

    if not tiles:
        print("No tiles found.")
        return

    xs = [t["x"] for t in tiles]
    ys = [t["y"] for t in tiles]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(t["x"] + t["w"] for t in tiles)
    max_y = max(t["y"] + t["h"] for t in tiles)

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    print(f"Canvas logical range x: [{min_x}, {max_x}), y: [{min_y}, {max_y})")
    print(f"Canvas size: {canvas_w} x {canvas_h}")

    prob_acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight_acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for t in tiles:
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        offset_x = x - min_x
        offset_y = y - min_y

        with Image.open(t["path"]) as im:
            im = im.convert("L")
            tile_arr = np.array(im, dtype=np.float32) / 255.0

        x0 = max(0, offset_x)
        y0 = max(0, offset_y)
        x1 = min(offset_x + w, canvas_w)
        y1 = min(offset_y + h, canvas_h)

        if x1 <= x0 or y1 <= y0:
            continue

        tx0 = x0 - offset_x
        ty0 = y0 - offset_y
        tx1 = tx0 + (x1 - x0)
        ty1 = ty0 + (y1 - y0)

        patch = tile_arr[ty0:ty1, tx0:tx1]

        prob_acc[y0:y1, x0:x1] += patch
        weight_acc[y0:y1, x0:x1] += 1.0

    weight_acc[weight_acc == 0] = 1.0
    final_prob = prob_acc / weight_acc

    binary_mask = (final_prob >= 0.5).astype(np.uint8) * 255
    rgb = Image.fromarray(binary_mask, mode="L")

    left_crop_px = abs(min_x) if min_x < 0 else 0
    top_crop_px = abs(min_y) if min_y < 0 else 0

    crop_box = (
        left_crop_px,
        top_crop_px,
        left_crop_px + orig_w,
        top_crop_px + orig_h,
    )

    rgb = rgb.crop(crop_box)

    out_png = f"{output_prefix}.png"
    rgb.save(out_png, compress_level=6)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    stitch_tiles(
        r"D:\google download\pred_2\Pred\plaqueImage15_4873_6162",
        output_prefix="stitched_result"
    )
