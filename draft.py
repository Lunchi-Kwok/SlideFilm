from pathlib import Path
import openslide
from gigapath.pipeline import tile_one_slide

INPUT_DIR = r"D:\google download\KPIS\NEP25\img_py"
SAVE_DIR  = 'NEP25/preprocessing/'
level = 1

WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".svslide", ".bif"}

in_dir = Path(INPUT_DIR)
out_dir = Path(SAVE_DIR)


slides = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in WSI_EXTS]

for slide_path in slides:
    slide = openslide.OpenSlide(str(slide_path))
    width, height = slide.level_dimensions[0]
    img_size = f"{width}_{height}"
    tile_one_slide(
        str(slide_path),
        save_dir=SAVE_DIR,
        level=level,
        img_size=img_size
    )