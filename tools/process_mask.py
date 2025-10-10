import os
from pathlib import Path
import pyvips
import openslide

def save_mask_pyramid(
    img: pyvips.Image,
    out_path: str,
    base_mpp: float,
    tile: int = 256,
    prefer: tuple = ("deflate", "lzw", "packbits")
):
    xres = 25400.0 / float(base_mpp)

    common = dict(
        tile=True, pyramid=True, bigtiff=True,
        tile_width=tile, tile_height=tile,
        resunit="inch", xres=xres, yres=xres,
        strip=True
    )

    last_err = None
    for codec in prefer:
        try:
            img.tiffsave(str(out_path), compression=codec, **common)
            print(f"    -> saved with compression='{codec}'")
            return
        except Exception as e:
            last_err = e
            print(f"    !! compression='{codec}' failed: {e}")

    try:
        img.tiffsave(str(out_path), compression="none", **common)
        print(f"    -> saved with compression='none' (no compression)")
    except Exception as e:
        raise RuntimeError(f"All compression methods failed, last error: {last_err} ; none also failed: {e}")

def build_pyramid_keep_original_mask(
    input_path: str,
    output_path: str,
    base_mpp: float = 0.27,
    tile: int = 256
):
    img = pyvips.Image.new_from_file(input_path, access="sequential")

    if img.format not in ("uchar", "ushort"):
        img = img.cast("uchar")

    save_mask_pyramid(img, output_path, base_mpp=base_mpp, tile=tile)

def process_folder_masks(
    input_dir: str,
    output_dir: str,
    base_mpp: float = 0.27,
    tile: int = 256,
    overwrite: bool = False
):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = {".tif", ".tiff", ".TIF", ".TIFF"}
    files = [p for p in in_root.rglob("*") if p.suffix in exts]

    if not files:
        print(f"[INFO] No TIFF masks found in: {in_root}")
        return

    print(f"[INFO] Found {len(files)} mask file(s). Start converting...")
    for i, src in enumerate(files, 1):
        rel = src.relative_to(in_root)
        dst = (out_root / rel).with_suffix(".tif")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            print(f"[SKIP {i}/{len(files)}] {rel} -> exists, skip (overwrite=False).")
            continue

        try:
            print(f"[{i}/{len(files)}] Converting mask: {rel}")
            build_pyramid_keep_original_mask(
                input_path=str(src),
                output_path=str(dst),
                base_mpp=base_mpp,
                tile=tile
            )

            slide = openslide.OpenSlide(str(dst))
            downsamples = [float(d) for d in slide.level_downsamples]
            dims = slide.level_dimensions
            level_mpp = [base_mpp * d for d in downsamples]
            print(f"    -> level_count={slide.level_count}")
            print(f"    -> level_dimensions={dims}")
            print(f"    -> level_downsamples={downsamples}")
            print(f"    -> level_mpp(approx)={['%.4f' % m for m in level_mpp]}")
        except Exception as e:
            print(f"[ERROR] Failed: {rel} -> {e}")

    print("[DONE] All mask conversions finished.")

if __name__ == "__main__":
    process_folder_masks(
        input_dir=r"D:\google download\aerial\root\WSI\Train\Masks",
        output_dir=r"D:\google download\aerial\root\WSI\Train\mask",
        base_mpp=0.27,
        tile=256,
        overwrite=False
    )


