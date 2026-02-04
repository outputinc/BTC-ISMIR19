#!/usr/bin/env python
"""
Finetune BTC model on COCO Chorales submix dataset.

This script loads a pretrained BTC model and finetunes it on the
submix dataset created by create_finetuning_dataset.py.
"""

import os
import json
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from btc_model import BTC_model
from utils.hparams import HParams


def load_config(config_path: str) -> HParams:
    """Load config with proper yaml loader."""
    with open(config_path, 'r') as f:
        return HParams(**yaml.load(f, Loader=yaml.FullLoader))


class FinetuneDataset(Dataset):
    """Dataset for loading finetuning examples."""

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir) / split
        self.files = sorted(self.data_dir.glob('*.pt'))
        print(f"Loaded {len(self.files)} examples from {split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        # Apply log transform (same as AudioDataset)
        feature = np.log(np.abs(data['feature']) + 1e-6)
        chord = np.array(data['chord'], dtype=np.int64)
        return {
            'feature': feature,  # [144, T]
            'chord': chord,      # [T]
        }


TIMESTEP = 108  # BTC model expects exactly 108 frames per sequence


def collate_fn(batch):
    """Collate function that pads to TIMESTEP frames (108)."""
    batch_size = len(batch)

    features = []
    chords = []
    lengths = []

    for b in batch:
        feat = b['feature']  # [144, T]
        chord = b['chord']   # [T]
        seq_len = feat.shape[1]
        lengths.append(min(seq_len, TIMESTEP))

        # Pad or truncate to TIMESTEP
        if seq_len < TIMESTEP:
            feat = np.pad(feat, ((0, 0), (0, TIMESTEP - seq_len)), mode='constant')
            chord = np.pad(chord, (0, TIMESTEP - seq_len), mode='constant', constant_values=24)  # 24 = 'N' (no chord)
        elif seq_len > TIMESTEP:
            feat = feat[:, :TIMESTEP]
            chord = chord[:TIMESTEP]

        features.append(feat)
        chords.append(chord)

    # Stack into tensors
    features = torch.tensor(np.stack(features), dtype=torch.float32)  # [B, 144, TIMESTEP]
    chords = torch.tensor(np.stack(chords), dtype=torch.int64)        # [B, TIMESTEP]
    lengths = torch.tensor(lengths, dtype=torch.int64)

    return features, chords, lengths


def compute_accuracy(predictions, labels, lengths):
    """Compute accuracy only on valid (non-padded) positions."""
    correct = 0
    total = 0
    for i, length in enumerate(lengths):
        pred = predictions[i, :length]
        label = labels[i, :length]
        correct += (pred == label).sum().item()
        total += length.item()
    return correct / total if total > 0 else 0.0


def train_epoch(model, dataloader, optimizer, mean, std, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for features, chords, lengths in pbar:
        features = features.to(device)  # [B, 144, T]
        chords = chords.to(device)      # [B, T]

        # Normalize
        features = (features - mean) / std

        # Permute to [B, T, 144] for model
        features = features.permute(0, 2, 1)

        # Forward pass
        optimizer.zero_grad()
        batch_size, seq_len, _ = features.shape
        flat_chords = chords.view(-1)  # [B*T]

        prediction, loss, _, _ = model(features, flat_chords)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        prediction = prediction.view(batch_size, seq_len)
        acc = compute_accuracy(prediction, chords, lengths)

        total_loss += loss.item()
        total_correct += acc * sum(lengths).item()
        total_samples += sum(lengths).item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    return total_loss / num_batches, total_correct / total_samples


def validate(model, dataloader, mean, std, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for features, chords, lengths in pbar:
            features = features.to(device)
            chords = chords.to(device)

            # Normalize
            features = (features - mean) / std

            # Permute to [B, T, 144]
            features = features.permute(0, 2, 1)

            batch_size, seq_len, _ = features.shape
            flat_chords = chords.view(-1)

            prediction, loss, _, _ = model(features, flat_chords)

            prediction = prediction.view(batch_size, seq_len)
            acc = compute_accuracy(prediction, chords, lengths)

            total_loss += loss.item()
            total_correct += acc * sum(lengths).item()
            total_samples += sum(lengths).item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    return total_loss / num_batches, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description="Finetune BTC on COCO Chorales submixes")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/datasets/btc_finetuning"),
        help="Path to finetuning dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./test/btc_model.pt",
        help="Path to pretrained BTC model"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="run_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_models",
        help="Directory to save finetuned models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (lower for finetuning)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="btc-finetuning",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config_path)
    print("Config loaded")

    # Load pretrained model
    model = BTC_model(config=config.model).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    mean = checkpoint['mean']
    std = checkpoint['std']
    print(f"Pretrained model loaded from {args.model_path}")
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")

    # Create datasets and dataloaders
    train_dataset = FinetuneDataset(args.data_dir, split='train')
    valid_dataset = FinetuneDataset(args.data_dir, split='valid')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Optimizer with lower learning rate for finetuning
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        # Load dataset manifest if available for additional config info
        manifest_path = Path(args.data_dir) / "manifest.json"
        dataset_config = {}
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                dataset_config = manifest.get("config", {})
                dataset_config["train_examples"] = manifest.get("splits", {}).get("train", {}).get("num_examples", len(train_dataset))
                dataset_config["valid_examples"] = manifest.get("splits", {}).get("valid", {}).get("num_examples", len(valid_dataset))

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "model_path": args.model_path,
                "data_dir": args.data_dir,
                "train_examples": len(train_dataset),
                "valid_examples": len(valid_dataset),
                **dataset_config
            }
        )
        print("Weights & Biases initialized")
    elif not WANDB_AVAILABLE:
        print("Note: wandb not installed, logging disabled")

    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    print(f"\nStarting finetuning for {args.epochs} epochs...")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, mean, std, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, valid_loader, mean, std, device)
        print(f"Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}")

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "valid/loss": val_loss,
                "valid/accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = output_dir / "btc_finetuned_best.pt"
            torch.save({
                'model': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config.model,
            }, save_path)
            print(f"New best model saved to {save_path} (val_acc: {val_acc:.4f})")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_path = output_dir / f"btc_finetuned_epoch{epoch}.pt"
            torch.save({
                'model': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config.model,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

    # Save final model
    save_path = output_dir / "btc_finetuned_final.pt"
    torch.save({
        'model': model.state_dict(),
        'mean': mean,
        'std': std,
        'epoch': args.epochs,
        'val_acc': val_acc,
        'config': config.model,
    }, save_path)
    print(f"\nFinal model saved to {save_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    print(f"\nFinetuning complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Finish wandb logging
    if use_wandb:
        # Log final summary metrics
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["final_train_acc"] = history['train_acc'][-1]
        wandb.summary["final_val_loss"] = history['val_loss'][-1]

        # Save best model as artifact
        artifact = wandb.Artifact(
            name="btc-finetuned-model",
            type="model",
            description=f"BTC model finetuned on submix data, best val_acc={best_val_acc:.4f}"
        )
        artifact.add_file(str(output_dir / "btc_finetuned_best.pt"))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("Weights & Biases logging complete")


if __name__ == "__main__":
    main()
