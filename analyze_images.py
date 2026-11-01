"""
Measure how much of each image is covered by bacteria, and how the
bacteria group into clumps (aggregates), for every image in a folder.

For each image, the script:
  1. decides which pixels are cells and which are background,
  2. groups touching cell pixels into aggregates (a single cell or a clump),
  3. measures each aggregate and sorts it by size: single cell, small clump
     or large clump,
  4. saves a copy of the image with every aggregate outlined.

An image that cannot be read is skipped and noted, so one bad file does
not stop an unattended run.

At the end it writes:
  results/aggregates.csv      one row per aggregate
  results/image_summary.csv   one row per image
  results/clump_sizes.png     chart of the clump sizes in each image
  results/run_log.txt         when it ran, the settings, anything skipped

Run:  python analyze_images.py
      python analyze_images.py --input my_folder --output my_results
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from PIL import Image
from skimage import filters, measure, morphology, segmentation

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PIXEL_SIZE_UM = 0.08       # one pixel is 80 nanometres (from the dataset notes)
SMOOTHING = 1              # how much to blur before finding cells (in pixels)
MIN_AREA_PX = 30           # anything smaller than this is treated as noise
MAX_HOLE_PX = 100          # holes up to this size are cell centres (one
                           # whole cell is about 145 pixels)

# Aggregate size classes. These cut-offs were chosen for this dataset by
# comparing aggregate sizes with the hand-drawn cells: a typical single cell
# measures about 1.4 µm² and a pair about 2.8 µm². Change them for other cells.
SINGLE_CELL_MAX_UM2 = 2.0  # up to this size: a single cell
SMALL_CLUMP_MAX_UM2 = 8.0  # up to this size: a small clump (about 2 to 5 cells)
                           # above this size: a large clump

# The size classes, with the name shown on the chart.
SIZE_CLASSES = {
    "single_cell": "Single cells",
    "small_clump": "Small clumps",
    "large_clump": "Large clumps",
}

# One colour per size class: one set for the pale chart background, and steps
# of the same colours that show up on the dark microscope images.
CHART_COLOURS = {"single_cell": "#2a78d6", "small_clump": "#eb6834", "large_clump": "#1baf7a"}
OUTLINE_COLOURS = {"single_cell": "#3987e5", "small_clump": "#d95926", "large_clump": "#199e70",
                   "cut_off": "#9a9a94"}

# Chart background and text colours.
CHART_SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def load_image(path):
    """Open an image and return it as a 2D grid of brightness values."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        image = np.array(Image.open(path))
    if image.ndim == 3:            # colour file: keep the first channel
        image = image[..., 0]
    return image.astype(float)


def find_cell_pixels(image):
    """Return a True/False grid: True where there are cells."""
    # Blur very slightly to smooth out camera noise.
    smooth = filters.gaussian(image, sigma=SMOOTHING, preserve_range=True)

    # Pick a brightness cut-off automatically (Otsu's method) and keep
    # every pixel brighter than it. This catches the glowing membranes.
    cutoff = filters.threshold_otsu(smooth)
    is_cell = smooth > cutoff

    # The stain lights up the membrane, so a cell with a dim centre can come
    # out as a ring. Fill small holes so the whole cell counts. (This also
    # fills a few tiny gaps between tightly packed cells.) Only fill holes
    # smaller than a cell: a bigger hole is real background enclosed by a
    # ring of cells, and must not be counted as cell.
    is_cell = morphology.remove_small_holes(is_cell, area_threshold=MAX_HOLE_PX)

    # Drop tiny specks of noise.
    is_cell = morphology.remove_small_objects(is_cell, min_size=MIN_AREA_PX)
    return is_cell


def size_class(area_um2, touches_edge):
    """Sort an aggregate into a size class."""
    if touches_edge:
        return "cut_off"           # only partly visible, so its size is unknown
    if area_um2 <= SINGLE_CELL_MAX_UM2:
        return "single_cell"
    if area_um2 <= SMALL_CLUMP_MAX_UM2:
        return "small_clump"
    return "large_clump"


def measure_aggregates(is_cell, image, image_name):
    """Give each aggregate a number, measure it, and return one row per aggregate."""
    # Every separate group of touching cell pixels gets its own number.
    aggregates = measure.label(is_cell)
    height, width = is_cell.shape

    rows = []
    for agg in measure.regionprops(aggregates, intensity_image=image):
        top, left, bottom, right = agg.bbox
        # Aggregates cut off by the image edge are only partly measured.
        touches_edge = top == 0 or left == 0 or bottom == height or right == width
        area_um2 = agg.area * PIXEL_SIZE_UM ** 2
        rows.append({
            "image": image_name,
            "aggregate_id": agg.label,
            "area_px": int(agg.area),
            "area_um2": area_um2,
            "mean_brightness": agg.intensity_mean,
            "touches_edge": touches_edge,
            "size_class": size_class(area_um2, touches_edge),
        })
    return aggregates, rows


def summarise_image(image_name, is_cell, rows):
    """Return one summary row for an image."""
    # Sizes use only the aggregates fully inside the image.
    whole = [row for row in rows if not row["touches_edge"]]
    areas = [row["area_um2"] for row in whole]

    summary = {
        "image": image_name,
        "cell_pixels": int(is_cell.sum()),
        "cell_area_um2": is_cell.sum() * PIXEL_SIZE_UM ** 2,
        "percent_covered": 100 * is_cell.mean(),
        "aggregate_count": len(rows),
        "median_aggregate_um2": np.median(areas) if areas else 0,
        "largest_aggregate_um2": max(areas, default=0),
    }

    # Share of the cell area in each size class.
    for name in SIZE_CLASSES:
        class_area = sum(row["area_um2"] for row in whole if row["size_class"] == name)
        summary[f"pct_area_{name}"] = 100 * class_area / sum(areas) if areas else 0

    # How much of the cell area is left out of those shares because it is cut
    # off by the image edge. Large clumps are the most likely to be cut off,
    # so a high number here means the large-clump share is probably too low.
    all_area = sum(row["area_um2"] for row in rows)
    summary["pct_cell_area_cut_off"] = 100 * (1 - sum(areas) / all_area) if all_area else 0
    return summary


def save_outline_image(image, aggregates, rows, path):
    """Save the image with every aggregate outlined, coloured by size class."""
    # Stretch brightness so dim cells are visible, then make it colour.
    low, high = np.percentile(image, [1, 99.5])
    grey = np.clip((image - low) / max(high - low, 1), 0, 1)
    rgb = np.dstack([grey, grey, grey])

    for name, colour in OUTLINE_COLOURS.items():
        ids = [row["aggregate_id"] for row in rows if row["size_class"] == name]
        outline = segmentation.find_boundaries(np.isin(aggregates, ids), mode="outer")
        rgb[outline] = to_rgb(colour)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def short_names(file_names):
    """Drop the part of the file names that all the images share."""
    stems = [Path(name).stem for name in file_names]
    shared = os.path.commonprefix(stems)
    return [stem[len(shared):] or "(no number)" for stem in stems]


def save_size_chart(summary, path):
    """Save a chart: for each image, the share of cell area in each size class."""
    # Most aggregated image at the top (rows are drawn from the bottom up).
    data = summary.sort_values("pct_area_large_clump")
    rows = np.arange(len(data))

    fig = Figure(figsize=(9, 1.5 + 0.36 * len(data)), facecolor=CHART_SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(CHART_SURFACE)

    # One stacked bar per image; each segment starts where the last one ended.
    start = np.zeros(len(data))
    for name, label in SIZE_CLASSES.items():
        share = data[f"pct_area_{name}"].to_numpy()
        ax.barh(rows, share, left=start, height=0.66, color=CHART_COLOURS[name],
                edgecolor=CHART_SURFACE, linewidth=1, label=label)
        start += share

    # Label only the number the chart is about: the share in large clumps.
    for row, share in zip(rows, data["pct_area_large_clump"]):
        ax.text(101, row, f"{share:.0f}%", va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_yticks(rows, short_names(data["image"]), fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"],
                  fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    fig.suptitle("Share of cell area by clump size, in each image", x=0.01, y=0.985,
                 ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(0.01, 0.985 - 0.33 / fig.get_figheight(),
             "Whole aggregates only. Number at the right: share in large clumps.",
             fontsize=9, color=TEXT_SECONDARY)
    fig.legend(loc="upper left", bbox_to_anchor=(0.0, 0.985 - 0.42 / fig.get_figheight()),
               ncol=3, frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, handlelength=1,
               handleheight=1)
    fig.tight_layout(rect=(0, 0, 0.97, 1 - 0.72 / fig.get_figheight()))
    fig.savefig(path, dpi=130, facecolor=CHART_SURFACE)


def save_run_log(path, input_dir, image_count, summary_rows, skipped, aggregate_count):
    """Save a short record of this run, so results can be traced later."""
    no_cells = [row["image"] for row in summary_rows if row["cell_pixels"] == 0]
    lines = [
        f"Run finished: {datetime.now():%Y-%m-%d %H:%M}",
        f"Input folder: {input_dir}",
        f"Images found: {image_count}",
        f"Images analysed: {len(summary_rows)}",
        f"Images skipped: {len(skipped)}",
        f"Images with no cells found: {len(no_cells)}",
        f"Aggregates found: {aggregate_count}",
        "",
        "Settings:",
        f"  PIXEL_SIZE_UM = {PIXEL_SIZE_UM}",
        f"  SMOOTHING = {SMOOTHING}",
        f"  MIN_AREA_PX = {MIN_AREA_PX}",
        f"  MAX_HOLE_PX = {MAX_HOLE_PX}",
        f"  SINGLE_CELL_MAX_UM2 = {SINGLE_CELL_MAX_UM2}",
        f"  SMALL_CLUMP_MAX_UM2 = {SMALL_CLUMP_MAX_UM2}",
    ]
    if skipped:
        lines += ["", "Skipped images:"]
        lines += [f"  {name}: {reason}" for name, reason in skipped]
    if no_cells:
        lines += ["", "Images with no cells found (worth a look):"]
        lines += [f"  {name}" for name in no_cells]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyse_one_image(path, outline_dir, label_dir):
    """Run every step on one image and return its aggregate rows and summary row."""
    image = load_image(path)
    is_cell = find_cell_pixels(image)
    aggregates, rows = measure_aggregates(is_cell, image, path.name)
    summary_row = summarise_image(path.name, is_cell, rows)

    # Save the outlined picture for people to look at, and the aggregate
    # numbering so check_results.py can compare it later.
    save_outline_image(image, aggregates, rows, outline_dir / f"{path.stem}.png")
    Image.fromarray(aggregates.astype(np.uint16)).save(label_dir / f"{path.stem}.png")
    return rows, summary_row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--input", default="data/images",
                        help="folder of images to analyse")
    parser.add_argument("--output", default="results",
                        help="folder to save results in")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    outline_dir = output_dir / "outlines"
    label_dir = output_dir / "labels"
    outline_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir()
                         if p.suffix.lower() in {".png", ".tif", ".tiff"})
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")
    print(f"Analysing {len(image_paths)} images from {input_dir}...")

    all_aggregates = []
    summary_rows = []
    skipped = []
    for path in image_paths:
        try:
            rows, summary_row = analyse_one_image(path, outline_dir, label_dir)
        except Exception as error:
            # One bad file must not stop an unattended run: note it, remove
            # any older results for it, and carry on with the next image.
            print(f"  Skipped {path.name}: {error}")
            skipped.append((path.name, str(error)))
            (outline_dir / f"{path.stem}.png").unlink(missing_ok=True)
            (label_dir / f"{path.stem}.png").unlink(missing_ok=True)
            continue
        all_aggregates += rows
        summary_rows.append(summary_row)

    save_run_log(output_dir / "run_log.txt", input_dir, len(image_paths),
                 summary_rows, skipped, len(all_aggregates))
    if not summary_rows:
        raise SystemExit(f"No image could be analysed. See {output_dir / 'run_log.txt'}")

    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(all_aggregates).round(3).to_csv(output_dir / "aggregates.csv", index=False)
    summary.round(3).to_csv(output_dir / "image_summary.csv", index=False)
    save_size_chart(summary, output_dir / "clump_sizes.png")

    print(f"Found {len(all_aggregates)} aggregates in {len(summary_rows)} images.")
    if skipped:
        print(f"Skipped {len(skipped)} images. See {output_dir / 'run_log.txt'}")
    print(f"Tables, chart and outlined images saved in {output_dir}/")


if __name__ == "__main__":
    main()
