"""
Training Script for Lightweight 3D U-Net
Implements complete training loop with validation, checkpointing, and logging
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet3d import Lightweight3DUNet
from models.losses import get_loss_function
from models.dataset import get_data_loader
from models.metrics import calculate_metrics, DEFAULT_SPACING
from models.utils import sliding_window_inference_3d


class Trainer:
    """Trainer class for 3D U-Net"""
    EPS = 1e-8
    
    def __init__(self, config_path):
        """Initialize trainer with configuration"""
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set random seeds
        seed = self.config["experiment"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = Lightweight3DUNet(
            in_channels=1,
            out_channels=self.config["model"]["output_channels"],
            start_channels=self.config["model"]["start_channels"],
            encoder_channels=self.config["model"]["encoder_channels"],
            use_depthwise_separable=self.config["model"]["use_depthwise_separable"],
            use_grouped=self.config["model"]["use_grouped_conv"],
            groups=self.config["model"]["groups"],
            dropout_p=self.config["model"]["dropout_p"] if self.config["model"]["use_dropout"] else 0.0
        ).to(self.device)
        
        params = self.model.count_parameters()
        print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
        
        # Create loss function
        self.criterion = get_loss_function(self.config["loss"])
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Create learning rate scheduler
        scheduler_config = self.config["training"]["scheduler"]
        if scheduler_config["name"] == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config["T_max"],
                eta_min=scheduler_config.get("eta_min", 1e-6)
            )
        elif scheduler_config["name"] == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get("mode", "max"),
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                min_lr=scheduler_config.get("min_lr", 1e-6)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
        
        # Setup data loaders
        data_dir = self.config.get("data_dir", "data/processed")
        splits_dir = self.config.get("splits_dir", "data/splits")
        
        train_result = get_data_loader(
            data_dir=data_dir,
            split_file=f"{splits_dir}/train_list.txt",
            config=self.config,
            is_train=True
        )
        
        # Determine training mode
        mixed_config = self.config.get("training", {}).get("mixed_domains", {})
        use_mixed = mixed_config.get("enabled", False)
        mixed_mode = mixed_config.get("mode", "probabilistic")
        
        # Handle different training modes
        if isinstance(train_result, dict) and 'fl_loader' in train_result:
            # New step-based mode: separate FL and DLBCL loaders
            self.fl_loader = train_result['fl_loader']
            self.dlbcl_loader = train_result['dlbcl_loader']
            self.train_loader = None
            self.train_dataset = None
            self.use_step_based_mixed = True
            self.use_mixed_training = False
        elif isinstance(train_result, tuple):
            # Old probabilistic mode: single loader with MixedPatchDataset
            self.train_loader, self.train_dataset = train_result
            self.fl_loader = None
            self.dlbcl_loader = None
            self.use_step_based_mixed = False
            self.use_mixed_training = True
        else:
            # Standard mode: single loader
            self.train_loader = train_result
            self.train_dataset = None
            self.fl_loader = None
            self.dlbcl_loader = None
            self.use_step_based_mixed = False
            self.use_mixed_training = False
        
        self.val_loader = get_data_loader(
            data_dir=data_dir,
            split_file=f"{splits_dir}/val_list.txt",
            config=self.config,
            is_train=False
        )
        
        # Log mixed training status
        if self.use_step_based_mixed:
            dlbcl_steps_ratio = mixed_config.get("dlbcl_steps_ratio", 0.0)
            dlbcl_steps_override = mixed_config.get("dlbcl_steps", None)
            fl_batches = len(self.fl_loader)
            
            if dlbcl_steps_override is not None:
                dlbcl_steps = dlbcl_steps_override
            else:
                dlbcl_steps = round(fl_batches * dlbcl_steps_ratio)
            
            print(f"\n*** Step-Based Mixed Domain Training Enabled ***")
            print(f"  Mode: fl_epoch_plus_dlbcl")
            print(f"  FL batches per epoch: {fl_batches}")
            print(f"  DLBCL steps per epoch: {dlbcl_steps}")
            print(f"  DLBCL steps ratio: {dlbcl_steps_ratio:.2f}")
            print(f"  Total steps per epoch: {fl_batches + dlbcl_steps}")
            print(f"  Validation: FL-only")
            print(f"  Val cases: {len(self.val_loader.dataset)} FL cases")
        elif self.use_mixed_training:
            print(f"\n*** Mixed Domain Training Enabled (Probabilistic Mode) ***")
            print(f"  FL ratio: {mixed_config.get('fl_ratio', 0.5):.2%}")
            print(f"  Validation: FL-only")
            print(f"  Val cases: {len(self.val_loader.dataset)} FL cases")
        
        # Setup logging
        log_dir = Path(self.config["output"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_dir = Path(self.config["output"]["tensorboard_dir"])
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        
        # Setup checkpoint directory
        checkpoint_dir = Path(self.config["output"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # Training state
        self.start_epoch = 0
        self.best_metric = 0.0
        self.best_recall = 0.0
        self.best_dsc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_recall": [],
            "val_precision": [],
            "val_dsc": [],
            "val_fp_per_case": [],
            "val_best_threshold": [],
            "learning_rate": []
        }

    def _is_better_metric(self, recall, dsc, best_recall, best_dsc, tie_threshold):
        tie_margin = tie_threshold + self.EPS
        if recall > best_recall + self.EPS:
            return True, True
        if abs(recall - best_recall) <= tie_margin and dsc > best_dsc + self.EPS:
            return True, False
        return False, False

    def _get_target_spacing(self):
        return tuple(self.config.get("data", {}).get("spacing", {}).get("target", DEFAULT_SPACING))

    def _resolve_spacing_value(self, spacings, index, target_spacing):
        if spacings is None:
            spacing_value = target_spacing
        elif isinstance(spacings, (torch.Tensor, list, tuple)):
            if len(spacings) <= index:
                spacing_value = target_spacing
            else:
                spacing_value = spacings[index]
        else:
            spacing_value = spacings
        if isinstance(spacing_value, torch.Tensor):
            spacing_value = spacing_value.tolist()
        return tuple(float(s) for s in spacing_value)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Step-based mixed training: FL full pass + DLBCL steps
        if self.use_step_based_mixed:
            return self._train_epoch_step_based(epoch)
        
        # Old probabilistic mixed training or standard training
        # Reset sample counts for old mixed training mode
        if self.use_mixed_training and self.train_dataset is not None:
            self.train_dataset.reset_sample_counts()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            images = images.float()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        
        # Log domain statistics for old mixed training
        if self.use_mixed_training and self.train_dataset is not None:
            counts = self.train_dataset.get_sample_counts()
            fl_samples = counts['fl_samples']
            dlbcl_samples = counts['dlbcl_samples']
            total_samples = counts['total_samples']
            
            if total_samples > 0:
                fl_ratio = fl_samples / total_samples
                dlbcl_ratio = dlbcl_samples / total_samples
                
                print(f"\n  Domain Statistics:")
                print(f"    FL samples: {fl_samples} ({fl_ratio:.2%})")
                print(f"    DLBCL samples: {dlbcl_samples} ({dlbcl_ratio:.2%})")
                print(f"    Total samples: {total_samples}")
                
                # Log to tensorboard
                self.writer.add_scalar("Domain/fl_samples", fl_samples, epoch)
                self.writer.add_scalar("Domain/dlbcl_samples", dlbcl_samples, epoch)
                self.writer.add_scalar("Domain/fl_ratio", fl_ratio, epoch)
                self.writer.add_scalar("Domain/dlbcl_ratio", dlbcl_ratio, epoch)
        
        return avg_loss
    
    def _train_epoch_step_based(self, epoch):
        """
        Train one epoch with step-based mixed training:
        1. Full pass through all FL batches
        2. Additional DLBCL steps based on ratio
        """
        mixed_config = self.config.get("training", {}).get("mixed_domains", {})
        dlbcl_steps_ratio = mixed_config.get("dlbcl_steps_ratio", 0.0)
        dlbcl_steps_override = mixed_config.get("dlbcl_steps", None)
        
        # Calculate DLBCL steps
        fl_batches = len(self.fl_loader)
        if dlbcl_steps_override is not None:
            dlbcl_steps = dlbcl_steps_override
        else:
            dlbcl_steps = round(fl_batches * dlbcl_steps_ratio)
        
        # Track losses and steps
        fl_total_loss = 0.0
        fl_steps = 0
        dlbcl_total_loss = 0.0
        dlbcl_steps_done = 0
        
        # Base global step for this epoch
        # Note: dlbcl_steps is fixed at epoch start for consistent global_step calculation
        base_global_step = epoch * (fl_batches + dlbcl_steps)
        
        # Stage 1: Full pass through FL data
        print(f"  Stage 1: FL training ({fl_batches} batches)")
        pbar = tqdm(self.fl_loader, desc=f"Epoch {epoch+1} [FL]", position=0)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            images = images.float()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            fl_total_loss += loss.item()
            fl_steps += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard with monotonic global step
            # Both train_step (unified) and fl_step (domain-specific) for flexible monitoring
            global_step = base_global_step + batch_idx
            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
            self.writer.add_scalar("Loss/fl_step", loss.item(), global_step)
        
        # Stage 2: DLBCL steps
        if dlbcl_steps > 0:
            print(f"  Stage 2: DLBCL training ({dlbcl_steps} steps)")
            
            # Create an iterator that can be cycled if needed
            dlbcl_iter = iter(self.dlbcl_loader)
            
            pbar = tqdm(range(dlbcl_steps), desc=f"Epoch {epoch+1} [DLBCL]", position=0)
            for step_idx in pbar:
                try:
                    images, labels = next(dlbcl_iter)
                except StopIteration:
                    # Cycle the DLBCL loader if we run out of data
                    dlbcl_iter = iter(self.dlbcl_loader)
                    images, labels = next(dlbcl_iter)
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = images.float()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                dlbcl_total_loss += loss.item()
                dlbcl_steps_done += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Log to tensorboard with monotonic global step
                # Both train_step (unified) and dlbcl_step (domain-specific) for flexible monitoring
                global_step = base_global_step + fl_steps + step_idx
                self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
                self.writer.add_scalar("Loss/dlbcl_step", loss.item(), global_step)
        
        # Calculate average losses
        fl_avg_loss = fl_total_loss / fl_steps if fl_steps > 0 else 0.0
        dlbcl_avg_loss = dlbcl_total_loss / dlbcl_steps_done if dlbcl_steps_done > 0 else 0.0
        
        # Combined loss weighted by number of steps
        total_steps = fl_steps + dlbcl_steps_done
        if total_steps > 0:
            combined_loss = (fl_total_loss + dlbcl_total_loss) / total_steps
        else:
            combined_loss = 0.0
        
        # Calculate effective ratios
        fl_ratio = fl_steps / total_steps if total_steps > 0 else 0.0
        dlbcl_ratio = dlbcl_steps_done / total_steps if total_steps > 0 else 0.0
        
        # Log domain statistics
        print(f"\n  Domain Statistics:")
        print(f"    FL steps: {fl_steps} ({fl_ratio:.2%}), avg loss: {fl_avg_loss:.4f}")
        print(f"    DLBCL steps: {dlbcl_steps_done} ({dlbcl_ratio:.2%}), avg loss: {dlbcl_avg_loss:.4f}")
        print(f"    Total steps: {total_steps}, combined loss: {combined_loss:.4f}")
        
        # Log to tensorboard
        self.writer.add_scalar("Domain/fl_steps", fl_steps, epoch)
        self.writer.add_scalar("Domain/dlbcl_steps", dlbcl_steps_done, epoch)
        self.writer.add_scalar("Domain/fl_ratio", fl_ratio, epoch)
        self.writer.add_scalar("Domain/dlbcl_ratio", dlbcl_ratio, epoch)
        self.writer.add_scalar("Loss/fl_avg", fl_avg_loss, epoch)
        self.writer.add_scalar("Loss/dlbcl_avg", dlbcl_avg_loss, epoch)
        self.writer.add_scalar("Loss/combined", combined_loss, epoch)
        
        return combined_loss
    
    def validate(self, epoch):
        """
        Validate model using sliding window inference on full cases.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average validation loss (0.0 for case-level validation)
            best_metrics: Dictionary with best metrics after threshold sweep
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_spacings = []
        target_spacing = self._get_target_spacing()
        default_threshold = self.config["validation"]["default_threshold"]
        patch_size = tuple(self.config["data"]["patch_size"])
        
        # Validation uses CaseDataset which returns full volumes
        # No gradients needed during validation
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in pbar:
                # CaseDataset returns: (image, label, case_id, spacing)
                if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                    images, labels, case_ids, spacings = batch[0], batch[1], batch[2], batch[3]
                else:
                    # Fallback for unexpected format
                    images, labels = batch[0], batch[1]
                    spacings = None
                
                # Process each case (batch_size=1 for CaseDataset)
                batch_size = images.shape[0]
                for b in range(batch_size):
                    # Get single case data - safely handle different tensor shapes
                    # CaseDataset returns [B, C, D, H, W] where C=1
                    if images.ndim == 5:
                        image = images[b, 0].cpu().numpy()  # Remove batch and channel: [D, H, W]
                        label = labels[b, 0].cpu().numpy()  # Remove batch and channel: [D, H, W]
                    elif images.ndim == 4:
                        # In case batch dimension is already removed
                        image = images[0].cpu().numpy() if b == 0 else images[b].cpu().numpy()
                        label = labels[0].cpu().numpy() if b == 0 else labels[b].cpu().numpy()
                    else:
                        raise ValueError(f"Unexpected image shape: {images.shape}")
                    
                    # Perform sliding window inference
                    prob_map = sliding_window_inference_3d(
                        image=image,
                        model=self.model,
                        patch_size=patch_size,
                        overlap=0.5,
                        device=self.device,
                        use_gaussian=True
                    )
                    
                    # Store predictions and labels
                    all_predictions.append(prob_map)
                    all_labels.append(label)
                    
                    # Resolve spacing for this case
                    if spacings is not None:
                        spacing_value = self._resolve_spacing_value(spacings, b, target_spacing)
                    else:
                        spacing_value = target_spacing
                    all_spacings.append(spacing_value)
                    
                    # Update progress bar
                    pbar.set_postfix({"cases": f"{len(all_predictions)}"})
        
        if len(all_predictions) == 0:
            warnings.warn(
                "No validation cases found. Ensure the validation dataset is populated and data paths are correct. Returning zero metrics.",
                UserWarning,
                stacklevel=2
            )
            empty_metrics = {
                "lesion_wise_recall": 0.0,
                "lesion_wise_precision": 0.0,
                "lesion_wise_f1": 0.0,
                "voxel_wise_dsc_micro": 0.0,
                "voxel_wise_dsc_macro": 0.0,
                "fp_per_case": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "best_threshold": default_threshold,
                "best_recall": 0.0,
                "best_dsc_macro": 0.0,
                # Backward compatibility
                "dsc": 0.0,
                "recall": 0.0,
                "precision": 0.0
            }
            return 0.0, empty_metrics
        
        # Perform threshold sweep to find best threshold
        thresholds = self.config["validation"].get("threshold_sensitivity_range", [default_threshold])
        if not thresholds:
            warnings.warn("Validation threshold list is empty; falling back to default threshold.", UserWarning, stacklevel=2)
            thresholds = [default_threshold]
        
        tie_threshold = self.config["metrics"]["model_selection"].get("tie_threshold", 0.0)
        
        # Evaluate at first threshold
        best_threshold = thresholds[0]
        best_metrics = calculate_metrics(
            all_predictions,
            all_labels,
            threshold=best_threshold,
            spacing=all_spacings if all_spacings else target_spacing
        )
        best_recall = best_metrics["lesion_wise_recall"]
        best_dsc_macro = best_metrics["voxel_wise_dsc_macro"]
        
        # Sweep through remaining thresholds
        for threshold in thresholds[1:]:
            metrics = calculate_metrics(
                all_predictions,
                all_labels,
                threshold=threshold,
                spacing=all_spacings if all_spacings else target_spacing
            )
            
            # Use macro DSC for tie-breaking (as per requirements)
            is_better, _ = self._is_better_metric(
                metrics["lesion_wise_recall"],
                metrics["voxel_wise_dsc_macro"],
                best_recall,
                best_dsc_macro,
                tie_threshold
            )
            
            if is_better:
                best_recall = metrics["lesion_wise_recall"]
                best_dsc_macro = metrics["voxel_wise_dsc_macro"]
                best_threshold = threshold
                best_metrics = metrics
        
        # Add best threshold and metrics to return dict
        best_metrics["best_threshold"] = best_threshold
        best_metrics["best_recall"] = best_recall
        best_metrics["best_dsc_macro"] = best_dsc_macro
        
        print(f"Validation sweep - best recall: {best_recall:.4f} at threshold {best_threshold:.2f}, DSC macro: {best_dsc_macro:.4f}")
        
        # Return 0.0 for loss since we don't compute it during case-level validation
        return 0.0, best_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "best_recall": self.best_recall,
            "best_dsc": self.best_dsc,
            "best_epoch": self.best_epoch,
            "config": self.config,
            "history": self.history
        }
        
        # Save regular checkpoint
        if self.config["output"]["save_checkpoints"]:
            if (epoch + 1) % self.config["output"]["save_every_n_epochs"] == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
                # Keep only last N checkpoints
                self._cleanup_checkpoints()
        
        # Save best model
        if is_best:
            best_model_path = Path(self.config["output"]["best_model_path"])
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model: {best_model_path}")
    
    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        keep_n = self.config["output"].get("keep_last_n_checkpoints", 5)
        
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()
    
    def train(self):
        """Main training loop"""
        epochs = self.config["training"]["epochs"]
        warmup_epochs = self.config["training"].get("warmup_epochs", 0) if self.config["training"].get("use_warmup", False) else 0
        
        early_stopping_config = self.config["training"]["early_stopping"]
        early_stopping_patience = early_stopping_config.get("patience", 20)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Warmup epochs: {warmup_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(self.start_epoch, epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            if (epoch + 1) % self.config["validation"].get("validate_every_n_epochs", 1) == 0:
                val_loss, val_metrics = self.validate(epoch)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Extract metrics using new clearer keys, with fallback to old keys
                current_recall = val_metrics.get("best_recall", val_metrics.get("lesion_wise_recall", val_metrics.get("recall", 0.0)))
                current_dsc_macro = val_metrics.get("best_dsc_macro", val_metrics.get("voxel_wise_dsc_macro", 0.0))
                current_dsc_micro = val_metrics.get("voxel_wise_dsc_micro", val_metrics.get("dsc", 0.0))
                current_precision = val_metrics.get("lesion_wise_precision", val_metrics.get("precision", 0.0))
                current_threshold = val_metrics.get("best_threshold", self.config["validation"]["default_threshold"])
                current_fp_per_case = val_metrics.get("fp_per_case", 0.0)
                
                # Update history with new metric names
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_recall"].append(current_recall)
                self.history["val_precision"].append(current_precision)
                self.history["val_dsc"].append(current_dsc_macro)  # Use macro DSC for tracking
                self.history["val_fp_per_case"].append(current_fp_per_case)
                self.history["val_best_threshold"].append(current_threshold)
                self.history["learning_rate"].append(current_lr)
                
                # Log to tensorboard with clearer names
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/lesion_wise_recall", current_recall, epoch)
                self.writer.add_scalar("Metrics/lesion_wise_precision", current_precision, epoch)
                self.writer.add_scalar("Metrics/voxel_wise_dsc_macro", current_dsc_macro, epoch)
                self.writer.add_scalar("Metrics/voxel_wise_dsc_micro", current_dsc_micro, epoch)
                self.writer.add_scalar("Metrics/fp_per_case", current_fp_per_case, epoch)
                self.writer.add_scalar("Metrics/best_threshold", current_threshold, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                
                # Print metrics
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Lesion-wise Recall: {current_recall:.4f} (best threshold: {current_threshold:.2f})")
                print(f"  Val Lesion-wise Precision: {current_precision:.4f}")
                print(f"  Val Voxel-wise DSC (macro): {current_dsc_macro:.4f}")
                print(f"  Val Voxel-wise DSC (micro): {current_dsc_micro:.4f}")
                print(f"  Val FP per case: {current_fp_per_case:.2f}")
                print(f"  Learning Rate: {current_lr:.6f}")
                
                # Check for improvement (use macro DSC for tie-breaking as per requirements)
                current_metric = current_recall  # Primary metric
                is_best = False
                tie_threshold = self.config["metrics"]["model_selection"].get("tie_threshold", 0.0)
                
                better_metric, recall_improved = self._is_better_metric(
                    current_metric,
                    current_dsc_macro,
                    self.best_recall,
                    self.best_dsc,
                    tie_threshold
                )

                if better_metric:
                    improvement = (current_metric - self.best_recall) if recall_improved else (current_dsc_macro - self.best_dsc)
                    self.best_recall = current_metric
                    self.best_dsc = current_dsc_macro
                    self.best_metric = self.best_recall
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    is_best = True
                    if recall_improved:
                        print(f"  *** New best {self.config['metrics']['primary']}: {self.best_recall:.4f} (↑{improvement:.4f}) ***")
                    else:
                        print(f"  *** Tie-broken improvement with DSC {self.best_dsc:.4f} (↑{improvement:.4f}) ***")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  No improvement for {self.epochs_without_improvement} epochs (best recall: {self.best_recall:.4f} at epoch {self.best_epoch+1})")
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Update learning rate scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()
                
                # Early stopping
                if early_stopping_config.get("enabled", True) and self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best {self.config['metrics']['primary']}: {self.best_metric:.4f} at epoch {self.best_epoch+1}")
                    break
            else:
                # Update learning rate scheduler even without validation
                if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
        
        # Save final training history
        history_path = Path(self.config["output"]["log_dir"]) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best {self.config['metrics']['primary']}: {self.best_metric:.4f} at epoch {self.best_epoch+1}")
        print(f"Training history saved to {history_path}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Lightweight 3D U-Net")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to processed data directory (overrides config)")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Path to splits directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load config and override if specified
    with open(args.config, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.data_dir:
        config["data_dir"] = args.data_dir
    else:
        config["data_dir"] = config.get("data_dir", "data/processed")
    
    if args.splits_dir:
        config["splits_dir"] = args.splits_dir
    else:
        config["splits_dir"] = config.get("splits_dir", "data/splits")
    
    # Save updated config
    with open(args.config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
