import os
import shutil
from pathlib import Path

source_dir = Path("/Volumes/MacBackup/ml_datasets/catsdogs/dogs-vs-cats/train")
target_dir = Path("./data/cats_vs_dogs_small")


def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        destination_dir = target_dir / subset_name / category
        Path.mkdir(destination_dir, parents=True, exist_ok=True)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=source_dir / fname,
                            dst=destination_dir / fname)


if __name__ == '__main__':
    make_subset("train", start_index=0, end_index=1000)
    make_subset("validation", start_index=1000, end_index=1500)
    make_subset("test", start_index=1500, end_index=2500)
