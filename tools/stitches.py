import os
import re
from pathlib import Path
from PIL import Image, ImageDraw

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

    xs = sorted(set(t["x"] for t in tiles))
    ys = sorted(set(t["y"] for t in tiles))

    delta_list = [64,128,256,512,1024,2048]
    d = xs[1] - xs[0]

    Sx = min(delta_list, key=lambda x: abs(x - d))
    Sy = Sx

    tile_w = tiles[0]["w"]
    tile_h = tiles[0]["h"]

    gx0 = (min(xs) // Sx) * Sx
    gy0 = (min(ys) // Sy) * Sy

    print("-----------------")

    for t in tiles:
        t["ix"] = (t["x"] - gx0) // Sx
        t["iy"] = (t["y"] - gy0) // Sy

    max_ix = max(t["ix"] for t in tiles)
    max_iy = max(t["iy"] for t in tiles)

    canvas_w = (max_ix + 1) * tile_w
    canvas_h = (max_iy + 1) * tile_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    occupied = set()
    for t in tiles:
        with Image.open(t["path"]).convert("RGBA") as im:
            canvas.alpha_composite(im, (t["ix"] * tile_w, t["iy"] * tile_h))
        occupied.add((t["ix"], t["iy"]))

    rgb = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    rgb.paste(canvas, mask=canvas.split()[-1])

    draw = ImageDraw.Draw(rgb)
    for iy in range(max_iy + 1):
        for ix in range(max_ix + 1):
            if (ix, iy) not in occupied:
                x0, y0 = ix * tile_w, iy * tile_h
                x1, y1 = x0 + tile_w, y0 + tile_h
                draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

    left_crop_px = abs(min(xs))
    top_crop_px = abs(min(ys))

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
        r"/plaqueImage15_4873_6162",
        output_prefix="stitched_result"
    )

