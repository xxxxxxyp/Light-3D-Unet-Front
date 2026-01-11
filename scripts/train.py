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
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet3d import Lightweight3DUNet
from models.losses import get_loss_function
from models.dataset import get_data_loader
from models.metrics import calculate_metrics


class Trainer:
    """Trainer class for 3D U-Net"""
    
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
        
        self.train_loader = get_data_loader(
            data_dir=data_dir,
            split_file=f"{splits_dir}/train_list.txt",
            config=self.config,
            is_train=True
        )
        
        self.val_loader = get_data_loader(
            data_dir=data_dir,
            split_file=f"{splits_dir}/val_list.txt",
            config=self.config,
            is_train=False
        )
        
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
            "learning_rate": []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
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
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate metrics
        metrics = calculate_metrics(
            all_predictions,
            all_labels,
            threshold=self.config["validation"]["default_threshold"]
        )
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
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
                
                # Update history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_recall"].append(val_metrics["recall"])
                self.history["val_precision"].append(val_metrics["precision"])
                self.history["val_dsc"].append(val_metrics["dsc"])
                self.history["val_fp_per_case"].append(val_metrics.get("fp_per_case", 0.0))
                self.history["learning_rate"].append(current_lr)
                
                # Log to tensorboard
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Metrics/recall", val_metrics["recall"], epoch)
                self.writer.add_scalar("Metrics/precision", val_metrics["precision"], epoch)
                self.writer.add_scalar("Metrics/dsc", val_metrics["dsc"], epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                
                # Print metrics
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Recall: {val_metrics['recall']:.4f}")
                print(f"  Val Precision: {val_metrics['precision']:.4f}")
                print(f"  Val DSC: {val_metrics['dsc']:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")
                
                # Check for improvement
                current_metric = val_metrics["recall"]  # Primary metric
                is_best = False
                
                if current_metric > self.best_metric:
                    improvement = current_metric - self.best_metric
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    is_best = True
                    print(f"  *** New best {self.config['metrics']['primary']}: {self.best_metric:.4f} (↑{improvement:.4f}) ***")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  No improvement for {self.epochs_without_improvement} epochs (best: {self.best_metric:.4f} at epoch {self.best_epoch+1})")
                
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
