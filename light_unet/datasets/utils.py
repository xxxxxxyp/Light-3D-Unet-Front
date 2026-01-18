"""
Utility functions for dataset operations
"""

import warnings
from .constants import DEFAULT_FL_PREFIX_MAX, DEFAULT_DLBCL_PREFIX_MIN, DEFAULT_DLBCL_PREFIX_MAX

def filter_cases_by_domain(case_ids, domain_config):
    """Filter case IDs by domain based on case ID prefix."""
    if domain_config is None or domain_config.get('domain') is None:
        return case_ids
    
    domain = domain_config.get('domain', '').lower()
    fl_prefix_max = domain_config.get('fl_prefix_max', DEFAULT_FL_PREFIX_MAX)
    dlbcl_prefix_min = domain_config.get('dlbcl_prefix_min', DEFAULT_DLBCL_PREFIX_MIN)
    dlbcl_prefix_max = domain_config.get('dlbcl_prefix_max', DEFAULT_DLBCL_PREFIX_MAX)
    
    filtered = []
    for case_id in case_ids:
        try:
            prefix = int(case_id[:4])
            if domain == 'fl':
                if prefix <= fl_prefix_max:
                    filtered.append(case_id)
            elif domain == 'dlbcl':
                if dlbcl_prefix_min <= prefix <= dlbcl_prefix_max:
                    filtered.append(case_id)
            else:
                filtered.append(case_id)
        except (ValueError, IndexError):
            warnings.warn(f"Case ID {case_id} doesn't match expected format, skipping filter")
            filtered.append(case_id)
    
    return filtered

def create_missing_body_mask_error(missing_count, total_count, missing_cases, context=""):
    """Create a standardized error message for missing body masks."""
    case_list = ", ".join([f"'{c}'" for c in missing_cases[:5]])
    if len(missing_cases) > 5:
        case_list += "..."
    context_str = f" for {context}" if context else ""
    return FileNotFoundError(
        f"Body mask is required{context_str} but missing for {missing_count}/{total_count} cases: [{case_list}]. "
        f"Please ensure body masks are generated for all cases or disable body mask enforcement."
    )