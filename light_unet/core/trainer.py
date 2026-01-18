"""
Core Trainer Module
Refactored to use ConfigManager and accept flexible config input.
"""

import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from light_unet.datasets.loader import get_data_loader
from models.unet3d import Lightweight3DUNet
from models.losses import get_loss_function
from models.metrics import calculate_metrics, DEFAULT_SPACING
from light_unet.utils import sliding_window_inference_3d

from light_unet.core.config import ConfigManager  # [NEW]

class Trainer:
    """Trainer class for 3D U-Net"""
    EPS = 1e-8
    
    def __init__(self, config_or_path):
        """
        Initialize trainer with configuration
        
        Args:
            config_or_path: Path to config file (str) or config dictionary
        """
        # [CHANGE] Load configuration using ConfigManager or use provided dict
        if isinstance(config_or_path, (str, Path)):
            self.config = ConfigManager.load(str(config_or_path))
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            raise TypeError(f"config_or_path must be str, Path or dict, got {type(config_or_path)}")
        
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
        
        # Extract loaders
        if not isinstance(train_result, dict):
            raise TypeError(f"get_data_loader must return a dict, got {type(train_result)}")
        
        mode = train_result.get('mode')
        if mode == 'fl_epoch_plus_dlbcl':
            self.fl_loader = train_result['fl_loader']
            self.dlbcl_loader = train_result['dlbcl_loader']
            self.train_loader = None
            self.train_dataset = None
            self.use_step_based_mixed = True
            self.use_mixed_training = False
        elif mode == 'probabilistic':
            self.train_loader = train_result['train_loader']
            self.train_dataset = train_result['train_dataset']
            self.fl_loader = None
            self.dlbcl_loader = None
            self.use_step_based_mixed = False
            self.use_mixed_training = True
        elif mode == 'standard':
            self.train_loader = train_result['train_loader']
            self.train_dataset = None
            self.fl_loader = None
            self.dlbcl_loader = None
            self.use_step_based_mixed = False
            self.use_mixed_training = False
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        
        val_result = get_data_loader(
            data_dir=data_dir,
            split_file=f"{splits_dir}/val_list.txt",
            config=self.config,
            is_train=False
        )
        
        self.val_loader = val_result['val_loader']
        
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
        
        if self.use_step_based_mixed:
            return self._train_epoch_step_based(epoch)
        
        if self.use_mixed_training and self.train_dataset is not None:
            self.train_dataset.reset_sample_counts()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            images = images.float()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        
        if self.use_mixed_training and self.train_dataset is not None:
            counts = self.train_dataset.get_sample_counts()
            fl_samples = counts['fl_samples']
            dlbcl_samples = counts['dlbcl_samples']
            total_samples = counts['total_samples']
            
            if total_samples > 0:
                fl_ratio = fl_samples / total_samples
                dlbcl_ratio = dlbcl_samples / total_samples
                self.writer.add_scalar("Domain/fl_samples", fl_samples, epoch)
                self.writer.add_scalar("Domain/dlbcl_samples", dlbcl_samples, epoch)
                self.writer.add_scalar("Domain/fl_ratio", fl_ratio, epoch)
                self.writer.add_scalar("Domain/dlbcl_ratio", dlbcl_ratio, epoch)
        
        return avg_loss
    
    def _train_epoch_step_based(self, epoch):
        mixed_config = self.config.get("training", {}).get("mixed_domains", {})
        dlbcl_steps_ratio = mixed_config.get("dlbcl_steps_ratio", 0.0)
        dlbcl_steps_override = mixed_config.get("dlbcl_steps", None)
        
        fl_batches = len(self.fl_loader)
        if dlbcl_steps_override is not None:
            dlbcl_steps = dlbcl_steps_override
        else:
            dlbcl_steps = round(fl_batches * dlbcl_steps_ratio)
        
        fl_total_loss = 0.0
        fl_steps = 0
        dlbcl_total_loss = 0.0
        dlbcl_steps_done = 0
        
        base_global_step = epoch * (fl_batches + dlbcl_steps)
        
        # Stage 1: FL
        pbar = tqdm(self.fl_loader, desc=f"Epoch {epoch+1} [FL]", position=0)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            images = images.float()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            fl_total_loss += loss.item()
            fl_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            global_step = base_global_step + batch_idx
            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
            self.writer.add_scalar("Loss/fl_step", loss.item(), global_step)
        
        # Stage 2: DLBCL
        if dlbcl_steps > 0:
            dlbcl_iter = iter(self.dlbcl_loader)
            pbar = tqdm(range(dlbcl_steps), desc=f"Epoch {epoch+1} [DLBCL]", position=0)
            for step_idx in pbar:
                try:
                    images, labels = next(dlbcl_iter)
                except StopIteration:
                    dlbcl_iter = iter(self.dlbcl_loader)
                    images, labels = next(dlbcl_iter)
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = images.float()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                dlbcl_total_loss += loss.item()
                dlbcl_steps_done += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                global_step = base_global_step + fl_steps + step_idx
                self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
                self.writer.add_scalar("Loss/dlbcl_step", loss.item(), global_step)
        
        fl_avg_loss = fl_total_loss / fl_steps if fl_steps > 0 else 0.0
        dlbcl_avg_loss = dlbcl_total_loss / dlbcl_steps_done if dlbcl_steps_done > 0 else 0.0
        
        total_steps = fl_steps + dlbcl_steps_done
        combined_loss = (fl_total_loss + dlbcl_total_loss) / total_steps if total_steps > 0 else 0.0
        
        fl_ratio = fl_steps / total_steps if total_steps > 0 else 0.0
        dlbcl_ratio = dlbcl_steps_done / total_steps if total_steps > 0 else 0.0
        
        self.writer.add_scalar("Domain/fl_steps", fl_steps, epoch)
        self.writer.add_scalar("Domain/dlbcl_steps", dlbcl_steps_done, epoch)
        self.writer.add_scalar("Domain/fl_ratio", fl_ratio, epoch)
        self.writer.add_scalar("Domain/dlbcl_ratio", dlbcl_ratio, epoch)
        self.writer.add_scalar("Loss/fl_avg", fl_avg_loss, epoch)
        self.writer.add_scalar("Loss/dlbcl_avg", dlbcl_avg_loss, epoch)
        self.writer.add_scalar("Loss/combined", combined_loss, epoch)
        
        return combined_loss
    
    def validate(self, epoch):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_spacings = []
        target_spacing = self._get_target_spacing()
        default_threshold = self.config["validation"]["default_threshold"]
        patch_size = tuple(self.config["data"]["patch_size"])
        
        body_mask_config = self.config.get("data", {}).get("body_mask", {})
        apply_body_mask = body_mask_config.get("apply_to_validation", False) and body_mask_config.get("enabled", False)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 5:
                        images, labels, case_ids, spacings, body_masks = batch[0], batch[1], batch[2], batch[3], batch[4]
                    elif len(batch) >= 4:
                        images, labels, case_ids, spacings = batch[0], batch[1], batch[2], batch[3]
                        body_masks = None
                    else:
                        images, labels = batch[0], batch[1]
                        spacings = None
                        body_masks = None
                else:
                    images, labels = batch[0], batch[1]
                    spacings = None
                    body_masks = None
                
                batch_size = images.shape[0]
                for b in range(batch_size):
                    if images.ndim == 5:
                        image = images[b, 0].cpu().numpy()
                        label = labels[b, 0].cpu().numpy()
                        body_mask = body_masks[b, 0].cpu().numpy() if body_masks is not None else None
                    elif images.ndim == 4:
                        image = images[0].cpu().numpy() if b == 0 else images[b].cpu().numpy()
                        label = labels[0].cpu().numpy() if b == 0 else labels[b].cpu().numpy()
                        body_mask = body_masks[0].cpu().numpy() if (body_masks is not None and b == 0) else (body_masks[b].cpu().numpy() if body_masks is not None else None)
                    else:
                        raise ValueError(f"Unexpected image shape: {images.shape}")
                    
                    prob_map = sliding_window_inference_3d(
                        image=image,
                        model=self.model,
                        patch_size=patch_size,
                        overlap=0.5,
                        device=self.device,
                        use_gaussian=True
                    )
                    
                    if apply_body_mask and body_mask is not None:
                        prob_map = prob_map * body_mask
                    
                    all_predictions.append(prob_map)
                    all_labels.append(label)
                    
                    if spacings is not None:
                        spacing_value = self._resolve_spacing_value(spacings, b, target_spacing)
                    else:
                        spacing_value = target_spacing
                    all_spacings.append(spacing_value)
                    
                    pbar.set_postfix({"cases": f"{len(all_predictions)}"})
        
        if len(all_predictions) == 0:
            return 0.0, {
                "lesion_wise_recall": 0.0, "lesion_wise_precision": 0.0,
                "voxel_wise_dsc_macro": 0.0, "voxel_wise_dsc_micro": 0.0,
                "fp_per_case": 0.0, "best_threshold": default_threshold,
                "best_recall": 0.0, "best_dsc_macro": 0.0
            }
        
        thresholds = self.config["validation"].get("threshold_sensitivity_range", [default_threshold])
        tie_threshold = self.config["metrics"]["model_selection"].get("tie_threshold", 0.0)
        
        best_threshold = thresholds[0]
        best_metrics = calculate_metrics(all_predictions, all_labels, threshold=best_threshold, spacing=all_spacings)
        best_recall = best_metrics["lesion_wise_recall"]
        best_dsc_macro = best_metrics["voxel_wise_dsc_macro"]
        
        for threshold in thresholds[1:]:
            metrics = calculate_metrics(all_predictions, all_labels, threshold=threshold, spacing=all_spacings)
            is_better, _ = self._is_better_metric(metrics["lesion_wise_recall"], metrics["voxel_wise_dsc_macro"], best_recall, best_dsc_macro, tie_threshold)
            
            if is_better:
                best_recall = metrics["lesion_wise_recall"]
                best_dsc_macro = metrics["voxel_wise_dsc_macro"]
                best_threshold = threshold
                best_metrics = metrics
        
        best_metrics["best_threshold"] = best_threshold
        best_metrics["best_recall"] = best_recall
        best_metrics["best_dsc_macro"] = best_dsc_macro
        
        return 0.0, best_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
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
        
        if self.config["output"]["save_checkpoints"]:
            if (epoch + 1) % self.config["output"]["save_every_n_epochs"] == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
                torch.save(checkpoint, checkpoint_path)
                self._cleanup_checkpoints()
        
        if is_best:
            best_model_path = Path(self.config["output"]["best_model_path"])
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_model_path)
    
    def _cleanup_checkpoints(self):
        keep_n = self.config["output"].get("keep_last_n_checkpoints", 5)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()
    
    def train(self):
        epochs = self.config["training"]["epochs"]
        warmup_epochs = self.config["training"].get("warmup_epochs", 0) if self.config["training"].get("use_warmup", False) else 0
        early_stopping_config = self.config["training"]["early_stopping"]
        early_stopping_patience = early_stopping_config.get("patience", 20)
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch(epoch)
            
            if (epoch + 1) % self.config["validation"].get("validate_every_n_epochs", 1) == 0:
                val_loss, val_metrics = self.validate(epoch)
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                current_recall = val_metrics.get("best_recall")
                current_dsc_macro = val_metrics.get("best_dsc_macro")
                
                # Update history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_recall"].append(current_recall)
                self.history["val_dsc"].append(current_dsc_macro)
                self.history["learning_rate"].append(current_lr)
                
                # TensorBoard
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Metrics/lesion_wise_recall", current_recall, epoch)
                self.writer.add_scalar("Metrics/voxel_wise_dsc_macro", current_dsc_macro, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                
                # Print
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Recall: {current_recall:.4f}, DSC: {current_dsc_macro:.4f}")
                
                # Model Selection
                tie_threshold = self.config["metrics"]["model_selection"].get("tie_threshold", 0.0)
                better_metric, recall_improved = self._is_better_metric(current_recall, current_dsc_macro, self.best_recall, self.best_dsc, tie_threshold)
                
                is_best = False
                if better_metric:
                    self.best_recall = current_recall
                    self.best_dsc = current_dsc_macro
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    is_best = True
                    print(f"  *** New best model! ***")
                else:
                    self.epochs_without_improvement += 1
                
                self.save_checkpoint(epoch, is_best=is_best)
                
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_recall)
                else:
                    self.scheduler.step()
                
                if early_stopping_config.get("enabled", True) and self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered.")
                    break
            else:
                if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
        
        self.writer.close()