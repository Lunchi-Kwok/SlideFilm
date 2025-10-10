import os
from pathlib import Path

wsi_root = Path(r"E:\prov-gigapath\outputs\preprocessing\output\img")    # 图像目录
mask_root = Path(r"E:\prov-gigapath\outputs\preprocessing\output\mask")  # 掩膜目录

exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for mask_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
    # 对应的 img 目录名：去掉 mask_ 前缀
    img_dir_name = mask_dir.name.replace("mask_", "", 1)
    img_dir = wsi_root / img_dir_name

    if not img_dir.exists():
        print(f"[WARN] 跳过 {mask_dir.name}：未找到对应的 img 目录 -> {img_dir}")
        continue

    # ROI 文件名集合（不含扩展名）
    img_stems = {
        Path(f).stem.lower()
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    }

    # mask 文件名集合，去掉 mask_ 前缀后比对
    mask_stems_mapped = {
        Path(f).stem.lower().removeprefix("mask_")
        for f in os.listdir(mask_dir)
        if f.lower().endswith(exts)
    }

    # 找出 mask 有但 ROI 没有的
    extra_mask_stems = mask_stems_mapped - img_stems

    # 删除这些 mask 文件
    for f in os.listdir(mask_dir):
        if not f.lower().endswith(exts):
            continue
        stem = Path(f).stem.lower().removeprefix("mask_")
        if stem in extra_mask_stems:
            fpath = mask_dir / f
            os.remove(fpath)
            print(f"[DEL] {fpath}")
