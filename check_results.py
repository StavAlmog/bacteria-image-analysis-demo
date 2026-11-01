"""
Check the results of analyze_images.py in three simple ways.

1. Look at a random sample.
   Pick a few images at random and show, side by side:
   the original image | the aggregates our script found, outlined by size
   class | the cells a person outlined by hand. A person then looks at
   these and notes anything that looks wrong.

2. Compare against the hand-drawn outlines for every image:
   - what share of the hand-drawn cell pixels the script also found,
   - how many hand-drawn cells the script missed completely,
   - how many aggregates the script found that contain no hand-drawn
     cell (possible false detections, apart from cells cut off by the
     image edge, which the person did not outline),
   - how many aggregates each method gives.

3. Check the size classes.
   Count the hand-drawn cells inside each aggregate, to see whether a
   "single cell" really holds one cell, and how many a clump holds.

Saves:
  results/check/random_sample.png     the side-by-side pictures
  results/check/comparison.csv        the numbers, one row per image
  results/check/size_class_check.csv  hand-drawn cells per size class

Run:  python check_results.py            (run analyze_images.py first)
      python check_results.py --seed 7   (a different random sample)
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Patch
from PIL import Image
from skimage import color, measure, morphology

from analyze_images import OUTLINE_COLOURS, SIZE_CLASSES, load_image

IMAGES_DIR = Path("data/images")
HAND_DRAWN_DIR = Path("data/hand_drawn_masks")
RESULTS_DIR = Path("results")
CHECK_DIR = RESULTS_DIR / "check"
SAMPLE_SIZE = 4
TOUCH_GAP_PX = 2   # hand-drawn cells this close together count as one clump


def compare_one_image(ours, hand):
    """Compare our aggregates with the hand-drawn cells for one image."""
    ours_mask = ours > 0
    hand_mask = hand > 0

    # Share of hand-drawn cell pixels that the script also marked as cell.
    pixels_found = (ours_mask & hand_mask).sum() / hand_mask.sum()

    # A hand-drawn cell is "missed" if its centre is not inside any of
    # our aggregates.
    cells = measure.regionprops(hand)
    missed = sum(1 for c in cells
                 if not ours_mask[tuple(np.round(c.centroid).astype(int))])

    # Our aggregates with no hand-drawn cell inside. The person did not
    # outline cells cut off by the image edge, so we count those apart:
    # only the ones fully inside the image are possible false finds.
    height, width = ours.shape
    empty_at_edge = empty_inside = 0
    for agg in measure.regionprops(ours):
        if hand_mask[agg.slice][agg.image].any():
            continue
        top, left, bottom, right = agg.bbox
        if top == 0 or left == 0 or bottom == height or right == width:
            empty_at_edge += 1
        else:
            empty_inside += 1

    # The person drew each cell separately, so group hand-drawn cells that
    # touch (or nearly touch) into clumps, to compare aggregate counts.
    hand_clumps = measure.label(
        morphology.binary_dilation(hand_mask, morphology.disk(TOUCH_GAP_PX)))

    return {
        "hand_pixels_found_pct": 100 * pixels_found,
        "hand_cells": len(cells),
        "hand_cells_missed": missed,
        "script_aggregates": int(ours.max()),
        "hand_clumps": int(hand_clumps.max()),
        "no_hand_cell_at_edge": empty_at_edge,
        "no_hand_cell_inside": empty_inside,
        "area_ratio_script_to_hand": ours_mask.sum() / hand_mask.sum(),
    }


def hand_cells_per_aggregate(ours, hand):
    """Count the hand-drawn cell centres inside each of our aggregates."""
    centres = [tuple(np.round(c.centroid).astype(int)) for c in measure.regionprops(hand)]
    return np.bincount([ours[c] for c in centres], minlength=ours.max() + 1)


def coloured(image, labels):
    """Return a picture with every cell filled in its own colour."""
    low, high = np.percentile(image, [1, 99.5])
    grey = np.clip((image - low) / (high - low), 0, 1)
    return color.label2rgb(labels, grey, alpha=0.35, bg_label=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="change this to look at a different random sample")
    args = parser.parse_args()
    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    # Check only the images that have results: a skipped image has none.
    image_paths = [p for p in sorted(IMAGES_DIR.glob("*.tif"))
                   if (RESULTS_DIR / "labels" / f"{p.stem}.png").exists()]
    if not image_paths:
        raise SystemExit("Results are missing. Run analyze_images.py first.")

    def load_pair(path):
        ours = np.array(Image.open(RESULTS_DIR / "labels" / f"{path.stem}.png")).astype(int)
        hand = tifffile.imread(HAND_DRAWN_DIR / path.name).astype(int)
        return ours, hand

    # --- Checks 2 and 3: compare every image with the hand-drawn outlines --
    aggregates = pd.read_csv(RESULTS_DIR / "aggregates.csv")
    rows = []
    for path in image_paths:
        ours, hand = load_pair(path)
        rows.append({"image": path.name, **compare_one_image(ours, hand)})

        counts = hand_cells_per_aggregate(ours, hand)
        in_image = aggregates["image"] == path.name
        aggregates.loc[in_image, "hand_cells"] = counts[aggregates.loc[in_image, "aggregate_id"]]

    comparison = pd.DataFrame(rows)
    comparison.round(2).to_csv(CHECK_DIR / "comparison.csv", index=False)

    total = comparison.sum(numeric_only=True)
    print(f"Hand-drawn cell pixels also found by the script: "
          f"{comparison['hand_pixels_found_pct'].min():.0f}% to "
          f"{comparison['hand_pixels_found_pct'].max():.0f}% per image")
    print(f"Hand-drawn cells missed completely: "
          f"{total.hand_cells_missed:.0f} of {total.hand_cells:.0f}")
    print(f"Aggregates found by the script: {total.script_aggregates:.0f}  "
          f"(hand-drawn clumps: {total.hand_clumps:.0f})")
    print(f"Script aggregates with no hand-drawn cell: "
          f"{total.no_hand_cell_at_edge:.0f} cut off by the image edge (not outlined by hand), "
          f"{total.no_hand_cell_inside:.0f} inside the image (possible false finds)")

    # Size classes: only whole aggregates have a class.
    whole = aggregates[aggregates["size_class"] != "cut_off"]
    size_check = (whole.groupby("size_class")["hand_cells"]
                  .agg(aggregates="count", median_hand_cells="median",
                       fewest_hand_cells="min", most_hand_cells="max")
                  .reindex(list(SIZE_CLASSES))
                  .reset_index())
    size_check.to_csv(CHECK_DIR / "size_class_check.csv", index=False)
    singles = whole[whole["size_class"] == "single_cell"]
    print(f"Aggregates called single cells: {len(singles)}, of which "
          f"{(singles['hand_cells'] == 1).mean():.0%} hold exactly one hand-drawn cell")
    print(size_check.to_string(index=False))

    # --- Check 1: pictures of a random sample, for a person to review -----
    random.seed(args.seed)
    sample = sorted(random.sample(image_paths, min(SAMPLE_SIZE, len(image_paths))))
    print("Random sample to review: " + ", ".join(p.stem for p in sample))

    fig, axes = plt.subplots(len(sample), 3, figsize=(15, 5 * len(sample)), squeeze=False)
    for row, path in zip(axes, sample):
        image = load_image(path)
        _, hand = load_pair(path)
        outlines = np.array(Image.open(RESULTS_DIR / "outlines" / f"{path.stem}.png"))
        info = comparison.set_index("image").loc[path.name]

        row[0].imshow(image, cmap="gray")
        row[0].set_title(f"{path.stem}\noriginal")
        row[1].imshow(outlines, interpolation="nearest")
        row[1].set_title(f"Script: {int(info.script_aggregates)} aggregates")
        row[2].imshow(coloured(image, hand))
        row[2].set_title(f"By hand: {int(info.hand_cells)} cells "
                         f"in {int(info.hand_clumps)} clumps")
        for ax in row:
            ax.axis("off")

    # Colour key for the script's outlines.
    names = {**SIZE_CLASSES, "cut_off": "Cut off by the image edge"}
    key = [Patch(color=OUTLINE_COLOURS[name], label=label) for name, label in names.items()]
    fig.legend(handles=key, loc="upper center", ncol=len(key), frameon=False, fontsize=12)

    plt.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    plt.savefig(CHECK_DIR / "random_sample.png", dpi=100)
    print(f"Saved pictures to {CHECK_DIR / 'random_sample.png'}")


if __name__ == "__main__":
    main()
