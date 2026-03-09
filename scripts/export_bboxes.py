"""
Export slice-wise 2D prompt boxes from prediction masks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
DEFAULT_BODY_MASK_RATIO_THRESHOLD = 0.1
PREDICTION_SUFFIXES = (
    "_pred.nii.gz",
    "_prob.nii.gz",
    "_pred.nii",
    "_prob.nii",
)


def _load_3d_volume(volume_path: Path) -> np.ndarray:
    """Load a NIfTI volume and squeeze an optional leading singleton channel."""
    volume = np.asarray(nib.load(str(volume_path)).get_fdata())
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape} from {volume_path}.")
    return volume


def _binarize_prediction(prediction: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probability map or binary mask into a boolean prediction volume."""
    prediction = np.asarray(prediction)
    unique_values = np.unique(prediction)
    if prediction.dtype == bool or np.all(np.isin(unique_values, [0, 1])):
        return prediction.astype(bool)
    return prediction >= threshold


def _case_id_from_prediction_path(pred_path: Path) -> str:
    """Extract case id from known prediction file suffixes."""
    for suffix in PREDICTION_SUFFIXES:
        if pred_path.name.endswith(suffix):
            return pred_path.name[: -len(suffix)]
    return pred_path.name.split(".nii", 1)[0]


def _resolve_case_volume(case_dir: Path, case_id: str) -> Path:
    """Resolve one case-specific NIfTI file from a directory."""
    candidates = sorted(case_dir.glob(f"{case_id}.nii.gz"))
    candidates.extend(sorted(case_dir.glob(f"{case_id}.nii")))
    if not candidates:
        candidates = sorted(case_dir.glob(f"{case_id}_*.nii.gz"))
        candidates.extend(sorted(case_dir.glob(f"{case_id}_*.nii")))
    if not candidates:
        raise FileNotFoundError(f"Could not resolve volume for case {case_id} in {case_dir}")
    return candidates[0]


def _find_prediction_paths(pred_dir: Path) -> List[Path]:
    """Find prediction volumes while supporting both *_pred and *_prob naming."""
    pred_paths: Dict[str, Path] = {}
    for suffix in PREDICTION_SUFFIXES:
        for pred_path in pred_dir.glob(f"*{suffix}"):
            pred_paths.setdefault(pred_path.name, pred_path)
    return sorted(pred_paths.values())


def _compute_2d_box(mask_2d: np.ndarray, expansion_voxels: int) -> List[int]:
    """Compute an inclusive 2D bounding box [y_min, x_min, y_max, x_max]."""
    coords = np.argwhere(mask_2d)
    if coords.size == 0:
        raise ValueError("Cannot compute a 2D box from an empty mask.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, int(y_min) - expansion_voxels)
    x_min = max(0, int(x_min) - expansion_voxels)
    y_max = min(mask_2d.shape[0] - 1, int(y_max) + expansion_voxels)
    x_max = min(mask_2d.shape[1] - 1, int(x_max) + expansion_voxels)
    return [y_min, x_min, y_max, x_max]


def _normalize_case_prompts(case_prompts: MutableMapping[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Sort prompts per class by z to make JSON output deterministic."""
    return {
        "TP": sorted(case_prompts.get("TP", []), key=lambda item: (int(item["z"]), list(item["box_2d"]))),
        "FP": sorted(case_prompts.get("FP", []), key=lambda item: (int(item["z"]), list(item["box_2d"]))),
    }


def _process_case_arrays(
    prediction: np.ndarray,
    gt: np.ndarray,
    body_mask: np.ndarray,
    threshold: float,
    expansion_voxels: int,
    body_mask_ratio_threshold: float,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, int]:
    """
    生成单个病例的逐层 2D Prompt。

    关键逻辑说明：
    1. 先在 3D 预测图上做连通域分析，确保 TP/FP 的定义基于完整 3D 团块，而不是单层切片。
    2. 每个 3D 团块再按 Z 轴拆成多个 2D 切片框，供后端 2D Lite-MedSAM 使用。
    3. 每一层都会额外检查 body mask 覆盖比例；如果该切片大部分落在体外，则直接丢弃，
       这样可以过滤体外伪影，同时保留真正位于体内的切片框。
    """
    prediction_binary = _binarize_prediction(prediction, threshold)
    gt_binary = np.asarray(gt) > 0
    body_mask_binary = np.asarray(body_mask) > 0

    if prediction_binary.shape != gt_binary.shape or prediction_binary.shape != body_mask_binary.shape:
        raise ValueError(
            "Prediction, GT, and body mask must have identical shapes, got "
            f"{prediction_binary.shape}, {gt_binary.shape}, {body_mask_binary.shape}."
        )

    labeled_prediction, num_components = ndimage.label(prediction_binary)
    case_prompts: Dict[str, List[Dict[str, Any]]] = {"TP": [], "FP": []}
    filtered_fp_slices = 0
    filtered_total_slices = 0

    for component_id in range(1, num_components + 1):
        component_mask = labeled_prediction == component_id
        if not np.any(component_mask):
            continue

        component_kind = "TP" if np.any(component_mask & gt_binary) else "FP"
        slice_indices = np.where(np.any(component_mask, axis=(1, 2)))[0]

        for z_index in slice_indices:
            slice_mask = component_mask[z_index]
            slice_voxels = int(slice_mask.sum())
            if slice_voxels == 0:
                continue

            # 中文说明：
            # 当前切片中只有极少数像素位于体膜内时，通常代表体外噪声或重建伪影。
            # 这里用“切片内预测像素与 body mask 的重叠比例”做筛选，低于阈值则丢弃该层框。
            inside_body_ratio = float(np.count_nonzero(slice_mask & body_mask_binary[z_index])) / float(slice_voxels)
            if inside_body_ratio < body_mask_ratio_threshold:
                filtered_total_slices += 1
                if component_kind == "FP":
                    filtered_fp_slices += 1
                continue

            case_prompts[component_kind].append(
                {
                    "z": int(z_index),
                    "box_2d": _compute_2d_box(slice_mask, expansion_voxels),
                }
            )

    return _normalize_case_prompts(case_prompts), filtered_fp_slices, filtered_total_slices


def process_single_case(
    pred_path: str | Path,
    gt_path: str | Path,
    body_mask_path: str | Path,
    threshold: float = 0.1,
    expansion_voxels: int = 3,
    body_mask_ratio_threshold: float = DEFAULT_BODY_MASK_RATIO_THRESHOLD,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process one case and return slice-wise TP/FP prompt boxes."""
    prompts, _, _ = _process_case_arrays(
        prediction=_load_3d_volume(Path(pred_path)),
        gt=_load_3d_volume(Path(gt_path)),
        body_mask=_load_3d_volume(Path(body_mask_path)),
        threshold=threshold,
        expansion_voxels=expansion_voxels,
        body_mask_ratio_threshold=body_mask_ratio_threshold,
    )
    return prompts


def export_bboxes(
    pred_dir: str | Path,
    gt_dir: str | Path,
    body_mask_dir: str | Path,
    output_json: str | Path,
    threshold: float = 0.1,
    expansion_voxels: int = 3,
    body_mask_ratio_threshold: float = DEFAULT_BODY_MASK_RATIO_THRESHOLD,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Export slice-wise 2D TP/FP prompt boxes for every prediction volume."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    body_mask_dir = Path(body_mask_dir)
    output_json = Path(output_json)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")
    if not body_mask_dir.is_dir():
        raise FileNotFoundError(f"Body-mask directory does not exist: {body_mask_dir}")

    pred_paths = _find_prediction_paths(pred_dir)
    if not pred_paths:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    exported: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    total_filtered_fp_slices = 0

    for pred_path in tqdm(pred_paths, desc="Exporting slice-wise prompts"):
        case_id = _case_id_from_prediction_path(pred_path)
        gt_path = _resolve_case_volume(gt_dir, case_id)
        body_mask_path = _resolve_case_volume(body_mask_dir, case_id)

        prompts, filtered_fp_slices, filtered_total_slices = _process_case_arrays(
            prediction=_load_3d_volume(pred_path),
            gt=_load_3d_volume(gt_path),
            body_mask=_load_3d_volume(body_mask_path),
            threshold=threshold,
            expansion_voxels=expansion_voxels,
            body_mask_ratio_threshold=body_mask_ratio_threshold,
        )
        exported[case_id] = prompts
        total_filtered_fp_slices += filtered_fp_slices

        LOGGER.info(
            "Case %s exported %d TP slices and %d FP slices; body-mask filtered %d FP slices (%d total slices).",
            case_id,
            len(prompts["TP"]),
            len(prompts["FP"]),
            filtered_fp_slices,
            filtered_total_slices,
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(exported, handle, separators=(",", ":"), ensure_ascii=False)

    LOGGER.info("Body-mask filtered FP slices in total: %d", total_filtered_fp_slices)
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export slice-wise 2D TP/FP prompt boxes.")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="data/processed/pred",
        help="Directory containing prediction volumes such as *_pred.nii.gz or *_prob.nii.gz.",
    )
    parser.add_argument(
        "--prob_maps_dir",
        dest="pred_dir",
        type=str,
        help="Backward-compatible alias for --pred_dir.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="data/processed/labels",
        help="Directory containing ground-truth masks.",
    )
    parser.add_argument(
        "--body_mask_dir",
        type=str,
        default="data/processed/body_masks",
        help="Directory containing body-mask volumes.",
    )
    parser.add_argument("--output_json", type=str, required=True, help="Path to the exported JSON file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold used to binarize non-binary prediction maps.",
    )
    parser.add_argument(
        "--expansion_voxels",
        type=int,
        default=3,
        help="Number of pixels used to expand each 2D bounding box edge.",
    )
    parser.add_argument(
        "--body_mask_ratio_threshold",
        type=float,
        default=DEFAULT_BODY_MASK_RATIO_THRESHOLD,
        help="Minimum ratio of predicted pixels that must fall inside the body mask to keep a slice box.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    export_bboxes(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        body_mask_dir=args.body_mask_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        expansion_voxels=args.expansion_voxels,
        body_mask_ratio_threshold=args.body_mask_ratio_threshold,
    )


if __name__ == "__main__":
    main()
