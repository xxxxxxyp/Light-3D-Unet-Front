"""
Evaluation Script for Lightweight 3D U-Net
Evaluates model on validation set and generates metrics report
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_unet.metrics import calculate_lesion_metrics, calculate_dsc
from light_unet.utils import find_case_files


def evaluate_case(case_id, prob_maps_dir, data_dir, thresholds, spacing=(4.0, 4.0, 4.0)):
    """
    Evaluate a single case at multiple thresholds
    
    Args:
        case_id: Case identifier
        prob_maps_dir: Directory containing probability maps
        data_dir: Directory containing ground truth labels
        thresholds: List of thresholds to evaluate
        spacing: Voxel spacing
    
    Returns:
        results: Dictionary of results per threshold
    """
    # Load probability map
    prob_path = Path(prob_maps_dir) / f"{case_id}_prob.nii.gz"
    if not prob_path.exists():
        return None
    
    prob_nii = nib.load(prob_path)
    prob_map = prob_nii.get_fdata()
    
    # Load ground truth label from flat structure
    data_dir = Path(data_dir)
    
    # Find label file for this case
    label_files = find_case_files(data_dir, case_id, file_type="label")
    
    if len(label_files) == 0:
        return None
    
    label_nii = nib.load(label_files[0])
    label = label_nii.get_fdata()
    
    # Evaluate at each threshold
    results = {}
    
    for threshold in thresholds:
        # Binarize prediction
        pred_binary = (prob_map >= threshold).astype(np.float32)
        
        # Calculate voxel-wise DSC
        dsc = calculate_dsc(pred_binary, label)
        
        # Calculate lesion-wise metrics
        lesion_metrics = calculate_lesion_metrics(
            prob_map,
            label,
            threshold=threshold,
            min_size_voxels=0,
            iou_threshold=0.1,
            distance_threshold_mm=10.0,
            spacing=spacing
        )
        
        results[threshold] = {
            "dsc": dsc,
            "recall": lesion_metrics["recall"],
            "precision": lesion_metrics["precision"],
            "f1": lesion_metrics["f1"],
            "tp": lesion_metrics["tp"],
            "fp": lesion_metrics["fp"],
            "fn": lesion_metrics["fn"]
        }
    
    return results


def evaluate_split(split_file, prob_maps_dir, data_dir, config):
    """
    Evaluate all cases in a split
    
    Args:
        split_file: Path to split list file
        prob_maps_dir: Directory containing probability maps
        data_dir: Directory containing processed data
        config: Configuration dictionary
    
    Returns:
        summary: Summary statistics
        per_case_results: Results per case
    """
    # Load case list
    with open(split_file, "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    
    # Get thresholds
    thresholds = config["validation"]["threshold_sensitivity_range"]
    default_threshold = config["validation"]["default_threshold"]
    if default_threshold not in thresholds:
        thresholds = sorted(thresholds + [default_threshold])
    
    print(f"Evaluating {len(case_ids)} cases at {len(thresholds)} thresholds...")
    
    # Evaluate each case
    all_results = {}
    spacing = config["data"]["spacing"]["target"]
    
    for case_id in tqdm(case_ids, desc="Evaluating"):
        results = evaluate_case(case_id, prob_maps_dir, data_dir, thresholds, spacing=spacing)
        if results is not None:
            all_results[case_id] = results
    
    # Aggregate results
    summary = {}
    
    for threshold in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        dsc_values = []
        
        for case_id, results in all_results.items():
            if threshold in results:
                total_tp += results[threshold]["tp"]
                total_fp += results[threshold]["fp"]
                total_fn += results[threshold]["fn"]
                dsc_values.append(results[threshold]["dsc"])
        
        # Calculate aggregate metrics
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_per_case = total_fp / len(all_results) if len(all_results) > 0 else 0.0
        mean_dsc = np.mean(dsc_values) if len(dsc_values) > 0 else 0.0
        
        summary[threshold] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "dsc": mean_dsc,
            "fp_per_case": fp_per_case,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "num_cases": len(all_results)
        }
    
    return summary, all_results


def print_summary(summary, default_threshold):
    """Print evaluation summary"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Print results for each threshold
    thresholds = sorted(summary.keys())
    
    print(f"\n{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'DSC':>10} {'FP/case':>10}")
    print("-" * 70)
    
    for threshold in thresholds:
        metrics = summary[threshold]
        marker = " *" if threshold == default_threshold else ""
        print(f"{threshold:>10.2f} {metrics['recall']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['dsc']:>10.4f} {metrics['fp_per_case']:>10.2f}{marker}")
    
    print("\n* = default threshold")
    
    # Find best threshold
    best_recall_threshold = max(thresholds, key=lambda t: summary[t]['recall'])
    best_f1_threshold = max(thresholds, key=lambda t: summary[t]['f1'])
    
    print(f"\nBest Recall: {summary[best_recall_threshold]['recall']:.4f} at threshold {best_recall_threshold:.2f}")
    print(f"Best F1: {summary[best_f1_threshold]['f1']:.4f} at threshold {best_f1_threshold:.2f}")
    
    # Print default threshold metrics
    default_metrics = summary[default_threshold]
    print(f"\nMetrics at default threshold ({default_threshold:.2f}):")
    print(f"  Lesion-wise Recall: {default_metrics['recall']:.4f}")
    print(f"  Lesion-wise Precision: {default_metrics['precision']:.4f}")
    print(f"  Lesion-wise F1: {default_metrics['f1']:.4f}")
    print(f"  Voxel-wise DSC: {default_metrics['dsc']:.4f}")
    print(f"  FP per case: {default_metrics['fp_per_case']:.2f}")
    print(f"  TP: {default_metrics['tp']}, FP: {default_metrics['fp']}, FN: {default_metrics['fn']}")


def save_results(summary, per_case_results, output_dir):
    """Save evaluation results to CSV and JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary as CSV
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "threshold"
    summary_csv = output_dir / "metrics.csv"
    summary_df.to_csv(summary_csv)
    print(f"\nSummary saved to {summary_csv}")
    
    # Save detailed results as JSON
    results_json = output_dir / "detailed_results.json"
    with open(results_json, "w") as f:
        json.dump({
            "summary": summary,
            "per_case": per_case_results
        }, f, indent=2)
    print(f"Detailed results saved to {results_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lightweight 3D U-Net")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--prob_maps_dir", type=str, default="inference/prob_maps",
                        help="Directory containing probability maps")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--split_file", type=str, default="data/splits/val_list.txt",
                        help="Path to split file to evaluate")
    parser.add_argument("--output_dir", type=str, default="inference",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Run evaluation
    summary, per_case_results = evaluate_split(
        args.split_file,
        args.prob_maps_dir,
        args.data_dir,
        config
    )
    
    # Print summary
    print_summary(summary, config["validation"]["default_threshold"])
    
    # Save results
    save_results(summary, per_case_results, args.output_dir)


if __name__ == "__main__":
    main()
