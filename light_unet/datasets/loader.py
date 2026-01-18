"""
Factory methods for creating data loaders
"""

from torch.utils.data import DataLoader
from .case_dataset import CaseDataset
from .patch_dataset import PatchDataset, MixedPatchDataset

def _create_train_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)

def _get_common_params(config, is_train):
    return {
        'patch_size': config["data"]["patch_size"],
        'augmentation': config["augmentation"] if is_train else None,
        'seed': config["experiment"]["seed"],
        'body_mask_config': config.get("data", {}).get("body_mask", {})
    }

def _create_step_based_loaders(data_dir, split_file, config, params):
    domain_config = config.get("data", {}).get("domains", {})
    batch_size = config["training"]["batch_size"]
    lesion_ratio = config["training"]["class_balanced_sampling"]["lesion_patch_ratio"]
    
    # FL Setup
    fl_domain_config = {'domain': 'fl', **domain_config}
    fl_dataset = PatchDataset(
        data_dir, split_file, params['patch_size'], lesion_ratio, 
        params['augmentation'], params['seed'], fl_domain_config, params['body_mask_config']
    )
    fl_loader = _create_train_loader(fl_dataset, batch_size)
    
    # DLBCL Setup
    dlbcl_domain_config = {'domain': 'dlbcl', **domain_config}
    dlbcl_dataset = PatchDataset(
        data_dir, split_file, params['patch_size'], lesion_ratio, 
        params['augmentation'], params['seed'] + 1, dlbcl_domain_config, params['body_mask_config']
    )
    dlbcl_loader = _create_train_loader(dlbcl_dataset, batch_size)
    
    return {
        'mode': 'fl_epoch_plus_dlbcl',
        'fl_loader': fl_loader, 'dlbcl_loader': dlbcl_loader,
        'fl_dataset': fl_dataset, 'dlbcl_dataset': dlbcl_dataset
    }

def _create_probabilistic_loader(data_dir, split_file, config, params):
    domain_config = config.get("data", {}).get("domains", {})
    mixed_config = config.get("training", {}).get("mixed_domains", {})
    fl_ratio = mixed_config.get("fl_ratio", 0.5)
    batch_size = config["training"]["batch_size"]
    
    dataset = MixedPatchDataset(
        data_dir, split_file, params['patch_size'], 
        config["training"]["class_balanced_sampling"]["lesion_patch_ratio"],
        params['augmentation'], params['seed'], domain_config, fl_ratio, params['body_mask_config']
    )
    
    return {
        'mode': 'probabilistic',
        'train_loader': _create_train_loader(dataset, batch_size),
        'train_dataset': dataset
    }

def _create_standard_loader(data_dir, split_file, config, params):
    batch_size = config["training"]["batch_size"]
    dataset = PatchDataset(
        data_dir, split_file, params['patch_size'],
        config["training"]["class_balanced_sampling"]["lesion_patch_ratio"],
        params['augmentation'], params['seed'], None, params['body_mask_config']
    )
    return {
        'mode': 'standard',
        'train_loader': _create_train_loader(dataset, batch_size)
    }

def _create_val_loader(data_dir, split_file, config):
    mixed_config = config.get("training", {}).get("mixed_domains", {})
    use_mixed = mixed_config.get("enabled", False)
    body_mask_config = config.get("data", {}).get("body_mask", {})
    apply_to_validation = body_mask_config.get("apply_to_validation", False) and body_mask_config.get("enabled", False)
    
    domain_config = None
    if use_mixed:
        base_domains = config.get("data", {}).get("domains", {})
        domain_config = {'domain': 'fl', **base_domains}
        
    dataset = CaseDataset(
        data_dir, split_file, domain_config, 
        return_body_mask=apply_to_validation, 
        body_mask_required=apply_to_validation
    )
    
    return {
        'mode': 'validation',
        'val_loader': DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    }

def get_data_loader(data_dir, split_file, config, is_train=True):
    """Main factory entry point"""
    if not is_train:
        return _create_val_loader(data_dir, split_file, config)

    params = _get_common_params(config, is_train)
    mixed_config = config.get("training", {}).get("mixed_domains", {})
    
    if mixed_config.get("enabled", False):
        mode = mixed_config.get("mode", "probabilistic")
        if mode == "fl_epoch_plus_dlbcl":
            return _create_step_based_loaders(data_dir, split_file, config, params)
        else:
            return _create_probabilistic_loader(data_dir, split_file, config, params)
    else:
        return _create_standard_loader(data_dir, split_file, config, params)