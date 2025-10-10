import os
from pathlib import Path

wsi_root = Path(r"E:\prov-gigapath\outputs\preprocessing\output\img")
mask_root = Path(r"E:\prov-gigapath\outputs\preprocessing\output\mask")

exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for mask_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
    img_dir_name = mask_dir.name.replace("mask_", "", 1)
    img_dir = wsi_root / img_dir_name

    img_stems = {
        Path(f).stem.lower()
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    }

    mask_stems_mapped = {
        Path(f).stem.lower().removeprefix("mask_")
        for f in os.listdir(mask_dir)
        if f.lower().endswith(exts)
    }

    extra_mask_stems = mask_stems_mapped - img_stems

    for f in os.listdir(mask_dir):
        if not f.lower().endswith(exts):
            continue
        stem = Path(f).stem.lower().removeprefix("mask_")
        if stem in extra_mask_stems:
            fpath = mask_dir / f
            os.remove(fpath)
            print(f"[DEL] {fpath}")
