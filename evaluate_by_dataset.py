#!/usr/bin/env python
"""Evaluate a finetuned BTC checkpoint on COCO vs Slakh validation sets separately."""

import os
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from btc_model import BTC_model
from utils.hparams import HParams
import yaml


def load_config(config_path: str) -> HParams:
    with open(config_path, 'r') as f:
        return HParams(**yaml.load(f, Loader=yaml.FullLoader))


class FinetuningDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str, mean: float, std: float):
        self.data_dir = Path(data_dir) / split
        self.mean = mean
        self.std = std
        self.files = sorted(list(self.data_dir.glob("*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        feature = data['feature']  # [144, T]
        chord = data['chord']  # [T]

        # Apply log and normalize
        feature = np.log(np.abs(feature) + 1e-6)
        feature = (feature - self.mean) / self.std
        feature = feature.T  # [T, 144]

        # Pad/truncate to 108 frames
        timestep = 108
        if feature.shape[0] < timestep:
            pad = np.zeros((timestep - feature.shape[0], feature.shape[1]))
            feature = np.vstack([feature, pad])
            chord = chord + [0] * (timestep - len(chord))
        else:
            feature = feature[:timestep]
            chord = chord[:timestep]

        # Get dataset type from filename
        filename = self.files[idx].name
        is_slakh = filename.startswith("Track")

        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(chord, dtype=torch.long),
            is_slakh
        )


def evaluate(model, dataloader, device, dataset_filter=None):
    """Evaluate model, optionally filtering by dataset type."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, chords, is_slakh in tqdm(dataloader, desc="Evaluating"):
            # Filter batches if requested
            if dataset_filter is not None:
                mask = is_slakh if dataset_filter == "slakh" else ~is_slakh
                if not mask.any():
                    continue
                features = features[mask]
                chords = chords[mask]

            features = features.to(device)
            chords = chords.to(device)

            # Forward pass
            self_attn_output, _ = model.self_attn_layers(features)
            prediction, _ = model.output_layer(self_attn_output)

            # Calculate accuracy
            predicted = prediction.view(-1)
            target = chords.view(-1)
            correct += (predicted == target).sum().item()
            total += target.numel()

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="finetuned_models/btc_finetuned_best.pt")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/datasets/btc_finetuning"))
    parser.add_argument("--config_path", type=str, default="run_config.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    config = load_config(args.config_path)
    model = BTC_model(config=config.model).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    mean = checkpoint['mean']
    std = checkpoint['std']
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load validation dataset
    val_dataset = FinetuningDataset(args.data_dir, "valid", mean, std)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Count examples by dataset
    n_coco = sum(1 for f in val_dataset.files if not f.name.startswith("Track"))
    n_slakh = sum(1 for f in val_dataset.files if f.name.startswith("Track"))
    print(f"\nValidation set: {len(val_dataset)} total ({n_coco} COCO, {n_slakh} Slakh)")

    # Evaluate on all data
    print("\n=== Evaluating on ALL validation data ===")
    acc_all = evaluate(model, val_loader, device, dataset_filter=None)
    print(f"All validation accuracy: {acc_all:.4f} ({acc_all*100:.2f}%)")

    # Evaluate on COCO only
    print("\n=== Evaluating on COCO only ===")
    acc_coco = evaluate(model, val_loader, device, dataset_filter="coco")
    print(f"COCO validation accuracy: {acc_coco:.4f} ({acc_coco*100:.2f}%)")

    # Evaluate on Slakh only
    print("\n=== Evaluating on Slakh only ===")
    acc_slakh = evaluate(model, val_loader, device, dataset_filter="slakh")
    print(f"Slakh validation accuracy: {acc_slakh:.4f} ({acc_slakh*100:.2f}%)")

    print("\n=== Summary ===")
    print(f"{'Dataset':<15} {'Examples':<10} {'Accuracy':<10}")
    print(f"{'-'*35}")
    print(f"{'COCO':<15} {n_coco:<10} {acc_coco*100:.2f}%")
    print(f"{'Slakh':<15} {n_slakh:<10} {acc_slakh*100:.2f}%")
    print(f"{'Combined':<15} {len(val_dataset):<10} {acc_all*100:.2f}%")


if __name__ == "__main__":
    main()
