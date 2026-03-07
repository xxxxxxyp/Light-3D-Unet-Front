"""
Evaluation script for saved probability maps.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from light_unet.core.config import ConfigManager


SUMMARY_HEADER = (
    f"{'Threshold':<10} {'Recall':>8} {'BBox_Rec':>10} {'Precision':>10} "
    f"{'F1':>10} {'DSC':>10} {'FP/case':>10}"
)


def format_summary_row(threshold, metrics):
    """Format one threshold row for the evaluation summary."""
    recall = metrics.get("lesion_wise_recall", metrics.get("recall", 0.0))
    bbox_recall = metrics.get("bbox_recall", 0.0)
    precision = metrics.get("lesion_wise_precision", metrics.get("precision", 0.0))
    f1 = metrics.get("lesion_wise_f1", 0.0)
    dsc = metrics.get("voxel_wise_dsc_micro", metrics.get("dsc", 0.0))
    fp_per_case = metrics.get("fp_per_case", 0.0)

    return (
        f"{threshold:<10.3f} {recall:>8.4f} {bbox_recall:>10.4f} {precision:>10.4f} "
        f"{f1:>10.4f} {dsc:>10.4f} {fp_per_case:>10.4f}"
    )


def format_default_threshold_report(default_threshold, metrics):
    """Format the detailed metrics report for the default threshold."""
    recall = metrics.get("lesion_wise_recall", metrics.get("recall", 0.0))
    bbox_recall = metrics.get("bbox_recall", 0.0)
    precision = metrics.get("lesion_wise_precision", metrics.get("precision", 0.0))
    f1 = metrics.get("lesion_wise_f1", 0.0)
    dsc = metrics.get("voxel_wise_dsc_micro", metrics.get("dsc", 0.0))
    fp_per_case = metrics.get("fp_per_case", 0.0)

    return "\n".join(
        [
            "Metrics at default threshold",
            f"  Threshold: {default_threshold:.3f}",
            f"  Recall: {recall:.4f}",
            f"  BBox Recall: {bbox_recall:.4f}",
            f"  Precision: {precision:.4f}",
            f"  F1: {f1:.4f}",
            f"  DSC: {dsc:.4f}",
            f"  FP/case: {fp_per_case:.4f}",
        ]
    )


def print_evaluation_summary(metrics_by_threshold, default_threshold):
    """Print the evaluation summary table and default-threshold details."""
    print("\nEVALUATION SUMMARY")
    print(SUMMARY_HEADER)

    for threshold, metrics in metrics_by_threshold:
        print(format_summary_row(threshold, metrics))

    default_metrics = None
    for threshold, metrics in metrics_by_threshold:
        if abs(threshold - default_threshold) < 1e-8:
            default_metrics = metrics
            break

    if default_metrics is None and metrics_by_threshold:
        default_metrics = metrics_by_threshold[0][1]

    if default_metrics is not None:
        print()
        print(format_default_threshold_report(default_threshold, default_metrics))


def load_evaluation_inputs(prob_maps_dir, data_dir, split_file):
    """Load probability maps, labels, and spacing values for evaluation."""
    import nibabel as nib

    from light_unet.models.metrics import DEFAULT_SPACING
    from light_unet.utils import find_case_files

    predictions = []
    labels = []
    spacings = []

    with open(split_file, "r", encoding="utf-8") as handle:
        case_ids = [line.strip() for line in handle if line.strip()]

    for case_id in case_ids:
        prob_map_path = Path(prob_maps_dir) / f"{case_id}_prob.nii.gz"
        label_files = find_case_files(data_dir, case_id, file_type="label")

        if not prob_map_path.exists() or not label_files:
            print(f"Warning: skipping {case_id} because probability map or label is missing.")
            continue

        prob_map_nii = nib.load(str(prob_map_path))
        label_nii = nib.load(str(label_files[0]))

        predictions.append(prob_map_nii.get_fdata())
        labels.append(label_nii.get_fdata())
        spacings.append(tuple(float(s) for s in label_nii.header.get_zooms()[:3]) or DEFAULT_SPACING)

    return predictions, labels, spacings


def evaluate_thresholds(predictions, labels, thresholds, spacings, expansion_voxels):
    """Evaluate all configured thresholds."""
    from light_unet.models.metrics import calculate_metrics

    results = []
    for threshold in thresholds:
        metrics = calculate_metrics(
            predictions,
            labels,
            threshold=threshold,
            spacing=spacings,
            expansion_voxels=expansion_voxels,
        )
        results.append((threshold, metrics))
    return results


def write_metrics_csv(output_path, metrics_by_threshold):
    """Persist evaluation metrics for downstream review."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "threshold",
                "recall",
                "bbox_recall",
                "precision",
                "f1",
                "dsc",
                "fp_per_case",
            ]
        )
        for threshold, metrics in metrics_by_threshold:
            writer.writerow(
                [
                    f"{threshold:.3f}",
                    f"{metrics.get('lesion_wise_recall', metrics.get('recall', 0.0)):.4f}",
                    f"{metrics.get('bbox_recall', 0.0):.4f}",
                    f"{metrics.get('lesion_wise_precision', metrics.get('precision', 0.0)):.4f}",
                    f"{metrics.get('lesion_wise_f1', 0.0):.4f}",
                    f"{metrics.get('voxel_wise_dsc_micro', metrics.get('dsc', 0.0)):.4f}",
                    f"{metrics.get('fp_per_case', 0.0):.4f}",
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved probability maps.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--prob_maps_dir", type=str, required=True, help="Directory with saved probability maps")
    parser.add_argument("--data_dir", type=str, required=True, help="Processed data directory")
    parser.add_argument("--split_file", type=str, required=True, help="Validation split file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for metrics CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ConfigManager.load(args.config)

    default_threshold = config["validation"]["default_threshold"]
    thresholds = config["validation"].get("threshold_sensitivity_range", [default_threshold])
    expansion_voxels = config.get("data", {}).get("bbox_expansion_voxels", 3)

    predictions, labels, spacings = load_evaluation_inputs(
        prob_maps_dir=args.prob_maps_dir,
        data_dir=args.data_dir,
        split_file=args.split_file,
    )

    if not predictions:
        raise RuntimeError("No evaluation cases could be loaded.")

    metrics_by_threshold = evaluate_thresholds(
        predictions=predictions,
        labels=labels,
        thresholds=thresholds,
        spacings=spacings,
        expansion_voxels=expansion_voxels,
    )

    metrics_csv = Path(args.output_dir) / "metrics.csv"
    write_metrics_csv(metrics_csv, metrics_by_threshold)
    print_evaluation_summary(metrics_by_threshold, default_threshold)


if __name__ == "__main__":
    main()
