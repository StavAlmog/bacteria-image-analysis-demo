"""
Download the public example images used by this project.

Dataset: "DeepBacs - Staphylococcus aureus widefield segmentation dataset"
         Pereira P.M. & Pinho M., ITQB NOVA, Portugal (2021)
         Zenodo, https://doi.org/10.5281/zenodo.5550933
         License: CC-BY 4.0 (free to reuse with credit)

The images show live S. aureus cells whose membranes are stained with a
fluorescent dye (Nile Red), so every cell appears as a glowing ring.
S. aureus naturally grows in clusters, which makes it a good example for
measuring aggregates. The dataset also contains an outline of every cell
drawn by hand, which we use only to double-check our results.

Run:  python download_data.py
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://zenodo.org/api/records/5550933/files/"
    "DeepBacs_Data_Segmentation_Staph_Aureus_dataset.zip/content"
)
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
ZIP_PATH = DATA_DIR / "saureus.zip"
UNPACK_DIR = DATA_DIR / "saureus_unpack"
IMAGES_DIR = DATA_DIR / "images"              # the folder our workflow reads
MASKS_DIR = DATA_DIR / "hand_drawn_masks"     # used only for checking results


def short(path):
    """Show a path relative to the project folder, e.g. data/images."""
    return path.relative_to(PROJECT_DIR)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Step 1: download the zip file (about 17 MB), unless we already have it.
    if ZIP_PATH.exists():
        print(f"Already downloaded: {short(ZIP_PATH)}")
    else:
        print("Downloading the dataset from Zenodo (about 17 MB)...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print(f"Saved to {short(ZIP_PATH)}")

    # Step 2: unpack it. We check every file name first so nothing can be
    # written outside the data folder.
    print("Unpacking...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for name in archive.namelist():
            target = (UNPACK_DIR / name).resolve()
            if not str(target).startswith(str(UNPACK_DIR.resolve())):
                raise ValueError(f"Unsafe file name inside zip: {name}")
        archive.extractall(UNPACK_DIR)

    # Step 3: collect the 12 full-size fluorescence images (and their
    # hand-drawn outlines) into two simple folders. The dataset also holds
    # bright-field images and small cut-out patches, which we do not need.
    IMAGES_DIR.mkdir(exist_ok=True)
    MASKS_DIR.mkdir(exist_ok=True)
    source = UNPACK_DIR / "fluorescence_dataset"
    image_folders = [source / "train" / "full_images" / "fluorescence",
                     source / "test" / "fluorescence"]
    count = 0
    for folder in image_folders:
        for image_file in sorted(folder.glob("*.tif")):
            shutil.copy(image_file, IMAGES_DIR / image_file.name)
            shutil.copy(folder.parent / "masks" / image_file.name,
                        MASKS_DIR / image_file.name)
            count += 1

    print(f"Done. {count} images are in {short(IMAGES_DIR)}")
    print(f"Their hand-drawn outlines are in {short(MASKS_DIR)}")


if __name__ == "__main__":
    main()
