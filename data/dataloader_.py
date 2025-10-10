from gigapath.pipeline import TileEncodingDataset
from torch.utils.data import Dataset, DataLoader
import os

class SlideTilesDataset(Dataset):
    def __init__(self, slide_dir, transform, mask_dir=None, mask_transform=None,paired_transform=None):
        self.slide_dirs = sorted([
            os.path.join(slide_dir, d)
            for d in os.listdir(slide_dir)
        ])
        self.transform = transform

        if mask_dir is not None:
            self.mask_dirs = sorted([
                os.path.join(mask_dir, d)
                for d in os.listdir(mask_dir)
            ])
        else:
            self.mask_dirs = None

        self.mask_transform = mask_transform
        self.paired_transform = paired_transform

    def __len__(self):
        return len(self.slide_dirs)

    def __getitem__(self, idx):
        slide_dir = self.slide_dirs[idx]
        tile_paths = sorted([
            os.path.join(slide_dir, f)
            for f in os.listdir(slide_dir)
            if f.endswith(".png")
        ])

        if self.mask_dirs is not None:
            mask_dir = self.mask_dirs[idx]
            mask_paths = sorted([
                os.path.join(mask_dir, f)
                for f in os.listdir(mask_dir)
                if f.endswith(".png")
            ])
        else:
            mask_paths = None

        tile_ds = TileEncodingDataset(
            image_paths=tile_paths,
            transform=self.transform,
            mask_dir=mask_paths,
            mask_transform=self.mask_transform,
            paired_transform=self.paired_transform
        )
        return tile_ds