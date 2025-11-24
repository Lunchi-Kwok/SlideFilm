import os
import numpy as np
from PIL import Image
import openslide


def pad5(n: int) -> str:
    n = int(n)
    if n < 0:
        return "-" + str(abs(n)).zfill(4)
    else:
        return str(n).zfill(5)


def save_tile(img: np.ndarray, x_coord: int, y_coord: int, out_dir: str):
    fname = f"{pad5(x_coord)}x_{pad5(y_coord)}y.png"
    Image.fromarray(img).save(os.path.join(out_dir, fname))


def extract_tiles_minpad_provgigapath(
    wsi_path: str,
    base_out_dir: str,
    level: int = 1,
    tile_size: int = 256,
    stride_px: int = 256,
    pad_value: int = 0
):

    slide = openslide.OpenSlide(wsi_path)

    W0, H0 = slide.level_dimensions[0]

    base_name = os.path.splitext(os.path.basename(wsi_path))[0]

    out_dir = os.path.join(base_out_dir, f"{base_name}_{W0}_{H0}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving tiles to: {out_dir}")

    W_l, H_l = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])

    def compute_minpad(length):
        n_steps = int(np.ceil((length - tile_size) / stride_px)) + 1
        total_cover = (n_steps - 1) * stride_px + tile_size
        pad_total = max(0, total_cover - length)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return pad_left, pad_right, n_steps

    pad_left_x, pad_right_x, n_steps_x = compute_minpad(W_l)
    pad_top_y, pad_bottom_y, n_steps_y = compute_minpad(H_l)

    print(f"Level {level}: size=({W_l}, {H_l}), downsample={downsample}")
    print(f"pad_x: left={pad_left_x}, right={pad_right_x}")
    print(f"pad_y: top={pad_top_y}, bottom={pad_bottom_y}")

    x_starts = [i * stride_px - pad_left_x for i in range(n_steps_x)]
    y_starts = [j * stride_px - pad_top_y for j in range(n_steps_y)]

    for y_l in y_starts:
        for x_l in x_starts:

            tile_np = np.full((tile_size, tile_size, 3), pad_value, dtype=np.uint8)

            src_x0_l = max(x_l, 0)
            src_y0_l = max(y_l, 0)
            src_x1_l = min(x_l + tile_size, W_l)
            src_y1_l = min(y_l + tile_size, H_l)

            if src_x1_l > src_x0_l and src_y1_l > src_y0_l:
                w_l = src_x1_l - src_x0_l
                h_l = src_y1_l - src_y0_l

                loc_x0 = int(round(src_x0_l * downsample))
                loc_y0 = int(round(src_y0_l * downsample))

                region = slide.read_region((loc_x0, loc_y0), level, (w_l, h_l)).convert("RGB")
                region_np = np.array(region)

                dst_x = src_x0_l - x_l
                dst_y = src_y0_l - y_l
                tile_np[dst_y:dst_y + h_l, dst_x:dst_x + w_l] = region_np

            logical_x = int(round(x_l * downsample))
            logical_y = int(round(y_l * downsample))

            save_tile(tile_np, logical_x, logical_y, out_dir)

    print(f"✔ Done. Tiles saved at: {out_dir}")

extract_tiles_minpad_provgigapath(
    wsi_path="plaqueImage1.svs",
    base_out_dir="tile_outputs",
    level=1,
    tile_size=256,
    stride_px=128   # overlap 50%
)
