"""
Prepare MedSAM2 NPZ data from processed NIfTI volumes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import nibabel as nib
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from light_unet.utils import find_case_files


def normalize_case_prompts(case_prompts: Optional[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize one case's prompt annotations to the slice-wise TP/FP structure."""
    normalized: Dict[str, List[Dict[str, Any]]] = {"TP": [], "FP": []}
    if case_prompts is None:
        return normalized
    if not isinstance(case_prompts, Mapping):
        raise TypeError("case_prompts must be a mapping with TP/FP prompt lists")

    for key in ("TP", "FP"):
        for entry in case_prompts.get(key, []):
            if not isinstance(entry, Mapping):
                continue
            if "z" not in entry or "box_2d" not in entry:
                continue
            box_2d = [int(value) for value in entry["box_2d"]]
            if len(box_2d) != 4:
                continue
            normalized[key].append({"z": int(entry["z"]), "box_2d": box_2d})
    return normalized


def serialize_prompt_entries(case_prompts: Optional[Any]) -> Dict[str, np.ndarray]:
    """Serialize TP/FP prompt entries as [z, y_min, x_min, y_max, x_max] arrays."""
    normalized = normalize_case_prompts(case_prompts)
    serialized: Dict[str, np.ndarray] = {}

    for key in ("TP", "FP"):
        rows = [
            [int(entry["z"]), *[int(value) for value in entry["box_2d"]]]
            for entry in normalized[key]
        ]
        serialized[f"{key.lower()}_boxes"] = np.asarray(rows, dtype=np.int16).reshape((-1, 5)) if rows else np.empty((0, 5), dtype=np.int16)

    return serialized


def load_prompts_index(prompts_json: Optional[str | Path]) -> Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Load the exported prompt JSON once for dataset preparation."""
    if not prompts_json:
        return None
    with Path(prompts_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(case_id): normalize_case_prompts(case_prompts) for case_id, case_prompts in payload.items()}


def transpose_to_dhw(array, source_path):
    """Transpose a 3D NIfTI array from (X, Y, Z)/(H, W, D) to (D, H, W)."""
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {array.shape} from {source_path}.")
    return np.transpose(array, (2, 0, 1))


def convert_to_npz(img_path, label_path, out_path, clip_min, clip_max, case_prompts=None):
    """Convert one image/label pair into compressed MedSAM2 NPZ format."""
    if clip_max <= clip_min:
        raise ValueError(f"clip_max must be greater than clip_min, got {clip_min} and {clip_max}.")

    img_data = transpose_to_dhw(nib.load(str(img_path)).get_fdata(), img_path)
    label_data = transpose_to_dhw(nib.load(str(label_path)).get_fdata(), label_path)

    img_data = np.clip(img_data, clip_min, clip_max)
    img_data = (img_data - clip_min) / (clip_max - clip_min)
    img_data = (img_data * 255.0).astype(np.uint8)

    label_data = (label_data > 0).astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"imgs": img_data, "gts": label_data}
    if case_prompts is not None:
        payload.update(serialize_prompt_entries(case_prompts))
    np.savez_compressed(out_path, **payload)


def read_case_ids(split_file):
    """Read non-empty case IDs from a split file."""
    with Path(split_file).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_case_files(data_dir, case_id):
    """Resolve the single PET image and label file for one case."""
    image_files = find_case_files(data_dir, case_id, file_type="image")
    label_files = find_case_files(data_dir, case_id, file_type="label")

    if len(image_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one image for case {case_id} in {data_dir}, found {len(image_files)}."
        )
    if len(label_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one label for case {case_id} in {data_dir}, found {len(label_files)}."
        )

    return image_files[0], label_files[0]


def prepare_split(data_dir, split_file, output_dir, clip_min, clip_max, prompts_by_case=None):
    """Convert every case listed in one split file."""
    case_ids = read_case_ids(split_file)
    for case_id in tqdm(case_ids, desc=f"Preparing {Path(output_dir).name} split"):
        img_path, label_path = resolve_case_files(data_dir, case_id)
        convert_to_npz(
            img_path=img_path,
            label_path=label_path,
            out_path=Path(output_dir) / f"{case_id}.npz",
            clip_min=clip_min,
            clip_max=clip_max,
            case_prompts=prompts_by_case.get(case_id) if prompts_by_case is not None else None,
        )


def prepare_dataset(data_dir, splits_dir, output_dir, clip_min, clip_max, prompts_json=None):
    """Prepare train and validation MedSAM2 NPZ datasets."""
    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    prompts_by_case = load_prompts_index(prompts_json)

    prepare_split(
        data_dir=data_dir,
        split_file=splits_dir / "train_list.txt",
        output_dir=output_dir / "train",
        clip_min=clip_min,
        clip_max=clip_max,
        prompts_by_case=prompts_by_case,
    )
    prepare_split(
        data_dir=data_dir,
        split_file=splits_dir / "val_list.txt",
        output_dir=output_dir / "val",
        clip_min=clip_min,
        clip_max=clip_max,
        prompts_by_case=prompts_by_case,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert processed PET NIfTI data into MedSAM2 NPZ format.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Processed data directory containing images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing train_list.txt and val_list.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/medsam2_npz",
        help="Root directory for generated MedSAM2 NPZ files.",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=0.0,
        help="Minimum absolute SUV value used for clipping before normalization.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=15.0,
        help="Maximum absolute SUV value used for clipping before normalization.",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="Optional prompt JSON file exported by scripts/export_bboxes.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_dataset(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        prompts_json=args.prompts_json,
    )


if __name__ == "__main__":
    main()
