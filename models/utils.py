"""
Utility functions for file operations
"""

from pathlib import Path
from typing import List


def find_case_files(base_dir: Path, case_id: str, file_type: str = "image") -> List[Path]:
    """
    Find image or label files for a specific case
    
    Args:
        base_dir: Base directory containing images/ or labels/ subdirectory
        case_id: Case identifier (e.g., "0001")
        file_type: Type of file to find ("image" or "label")
    
    Returns:
        List of matching file paths
    """
    base_dir = Path(base_dir)
    
    if file_type == "image":
        # Images have pattern: case_id_*.nii or case_id_*.nii.gz (e.g., 0001_0000.nii.gz)
        subdir = base_dir / "images"
        patterns = [f"{case_id}_*.nii.gz", f"{case_id}_*.nii"]
    elif file_type == "label":
        # Labels have pattern: case_id.nii or case_id.nii.gz (e.g., 0001.nii.gz)
        subdir = base_dir / "labels"
        patterns = [f"{case_id}.nii.gz", f"{case_id}.nii"]
    else:
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'image' or 'label'")
    
    files = []
    if subdir.exists():
        for pattern in patterns:
            files.extend(subdir.glob(pattern))
    
    return files
