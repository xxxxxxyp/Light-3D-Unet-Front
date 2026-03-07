"""
Export expanded 3D bounding boxes from saved probability maps.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


PROB_MAP_SUFFIX = "_prob.nii.gz"


def process_single_case(prob_map_path, threshold, expansion_voxels):
    """Extract expanded bounding boxes for one probability map."""
    prob_map = nib.load(str(prob_map_path)).get_fdata()
    prob_map = np.asarray(prob_map)

    if prob_map.ndim == 4 and prob_map.shape[0] == 1:
        prob_map = prob_map[0]
    if prob_map.ndim != 3:
        raise ValueError(f"Expected a 3D probability map, got shape {prob_map.shape} from {prob_map_path}.")

    binary_pred = (prob_map >= threshold).astype(bool)

    struct_elem = np.ones((3, 3, 3), dtype=bool)
    binary_pred = ndimage.binary_closing(binary_pred, structure=struct_elem)

    labeled_pred, _ = ndimage.label(binary_pred)
    pred_slices = ndimage.find_objects(labeled_pred)

    bboxes = []
    for slices in pred_slices:
        if slices is None:
            continue

        z_slice, y_slice, x_slice = slices
        z_min = max(0, z_slice.start - expansion_voxels)
        z_max = min(prob_map.shape[0], z_slice.stop + expansion_voxels)
        y_min = max(0, y_slice.start - expansion_voxels)
        y_max = min(prob_map.shape[1], y_slice.stop + expansion_voxels)
        x_min = max(0, x_slice.start - expansion_voxels)
        x_max = min(prob_map.shape[2], x_slice.stop + expansion_voxels)

        bboxes.append([int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)])

    return bboxes


def export_bboxes(prob_maps_dir, output_json, threshold=0.100, expansion_voxels=3):
    """Export bounding boxes for all probability maps in a directory."""
    prob_maps_dir = Path(prob_maps_dir)
    output_json = Path(output_json)

    if not prob_maps_dir.is_dir():
        raise FileNotFoundError(f"Probability map directory does not exist: {prob_maps_dir}")

    prob_map_paths = sorted(prob_maps_dir.glob(f"*{PROB_MAP_SUFFIX}"))
    if not prob_map_paths:
        raise RuntimeError(f"No probability maps matching *{PROB_MAP_SUFFIX} found in {prob_maps_dir}")

    data = {}
    for prob_map_path in tqdm(prob_map_paths, desc="Exporting BBoxes"):
        case_id = prob_map_path.name[: -len(PROB_MAP_SUFFIX)]
        data[case_id] = process_single_case(prob_map_path, threshold, expansion_voxels)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Export expanded 3D bounding boxes from probability maps.")
    parser.add_argument("--prob_maps_dir", type=str, required=True, help="Directory containing probability maps.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the exported JSON file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.100,
        help="Threshold used to binarize probability maps.",
    )
    parser.add_argument(
        "--expansion_voxels",
        type=int,
        default=3,
        help="Number of voxels used to expand each bounding box.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_bboxes(
        prob_maps_dir=args.prob_maps_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        expansion_voxels=args.expansion_voxels,
    )


if __name__ == "__main__":
    main()
