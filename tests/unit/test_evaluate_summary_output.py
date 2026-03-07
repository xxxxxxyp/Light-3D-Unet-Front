"""
Tests for evaluation summary formatting.
"""

import contextlib
import importlib.util
import io
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVALUATE_PATH = os.path.join(REPO_ROOT, "scripts", "evaluate.py")


def load_evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate", EVALUATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluate module from {EVALUATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_row_includes_bbox_recall():
    evaluate = load_evaluate_module()
    metrics = {
        "lesion_wise_recall": 0.95,
        "bbox_recall": 0.952,
        "lesion_wise_precision": 0.8,
        "lesion_wise_f1": 0.8696,
        "voxel_wise_dsc_micro": 0.7777,
        "fp_per_case": 1.25,
    }

    row = evaluate.format_summary_row(0.3, metrics)

    assert "0.300" in row
    assert "0.9500" in row
    assert "0.9520" in row
    assert row.index("0.9500") < row.index("0.9520")


def test_printed_summary_and_default_report_include_bbox_recall():
    evaluate = load_evaluate_module()
    metrics = {
        "lesion_wise_recall": 0.81,
        "bbox_recall": 0.952,
        "lesion_wise_precision": 0.72,
        "lesion_wise_f1": 0.7624,
        "voxel_wise_dsc_micro": 0.7012,
        "fp_per_case": 0.5,
    }

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluate.print_evaluation_summary([(0.3, metrics)], default_threshold=0.3)

    rendered = output.getvalue()

    assert "Threshold" in rendered
    assert "Recall" in rendered
    assert "BBox_Rec" in rendered
    assert "Metrics at default threshold" in rendered
    assert "BBox Recall: 0.9520" in rendered


if __name__ == "__main__":
    test_summary_row_includes_bbox_recall()
    test_printed_summary_and_default_report_include_bbox_recall()
    print("Evaluation summary formatting tests passed! ✓")
