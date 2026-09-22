import os
from pathlib import Path
import pyvips
import openslide

def build_pyramid_keep_original(
    input_path: str,
    output_path: str,
    base_mpp: float = 0.27,
    tile: int = 256,
    jpeg_quality: int = 90
):

    img = pyvips.Image.new_from_file(input_path, access="sequential")

    xres = 25400.0 / float(base_mpp)

    img.tiffsave(
        str(output_path),
        tile=True, pyramid=True, bigtiff=True,
        tile_width=tile, tile_height=tile,
        compression="jpeg", Q=jpeg_quality,
        resunit="inch", xres=xres, yres=xres
    )

def process_folder(
    input_dir: str,
    output_dir: str,
    base_mpp: float = 0.27,
    tile: int = 256,
    jpeg_quality: int = 90,
    overwrite: bool = False
):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = {".tif", ".tiff", ".TIF", ".TIFF"}

    files = [p for p in in_root.rglob("*") if p.suffix in exts]
    if not files:
        print(f"[INFO] No TIFFs found in: {in_root}")
        return

    print(f"[INFO] Found {len(files)} file(s). Start converting...")
    for i, src in enumerate(files, 1):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        dst = dst.with_suffix(".tif")

        if dst.exists() and not overwrite:
            print(f"[SKIP {i}/{len(files)}] {rel} -> exists, skip (use overwrite=True to force).")
            continue

        try:
            print(f"[{i}/{len(files)}] Converting: {rel}")
            build_pyramid_keep_original(
                input_path=str(src),
                output_path=str(dst),
                base_mpp=base_mpp,
                tile=tile,
                jpeg_quality=jpeg_quality
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

    print("[DONE] All conversions finished.")

if __name__ == "__main__":
    process_folder(
        input_dir=r"D:\google download\NEP25",
        output_dir=r"D:\google download\NEP25_py",
        base_mpp=0.2525,
        tile=256,
        jpeg_quality=90,
        overwrite=False
    )
