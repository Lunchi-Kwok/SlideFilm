import os
from gigapath.pipeline import tile_one_slide


slide_dir = r"D:\google download\aerial\root\WSI\Train\mask"
tmp_dir = r"outputs/preprocessing/"

# 遍历文件夹
for fname in os.listdir(slide_dir):
    if fname.endswith(".tif") or fname.endswith(".svs"):
        slide_path = os.path.join(slide_dir, fname)
        print(f"Processing: {slide_path}")
        tile_one_slide(slide_path, save_dir=tmp_dir, level=1)
