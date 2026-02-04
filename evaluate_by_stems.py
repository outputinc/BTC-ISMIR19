#!/usr/bin/env python
"""Evaluate a finetuned BTC checkpoint broken down by number of stems."""

import os
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from btc_model import BTC_model
from utils.hparams import HParams
import yaml


def load_config(config_path: str) -> HParams:
    with open(config_path, 'r') as f:
        return HParams(**yaml.load(f, Loader=yaml.FullLoader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="finetuned_models/btc_finetuned_best.pt")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/datasets/btc_finetuning"))
    parser.add_argument("--config_path", type=str, default="run_config.yaml")
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

    # Get validation files
    val_dir = Path(args.data_dir) / "valid"
    val_files = sorted(list(val_dir.glob("*.pt")))
    print(f"Validation files: {len(val_files)}")

    # Track results by (dataset, n_stems)
    results = defaultdict(lambda: {"correct": 0, "total": 0, "count": 0})

    model.eval()
    with torch.no_grad():
        for filepath in tqdm(val_files, desc="Evaluating"):
            data = torch.load(filepath, weights_only=False)
            feature = data['feature']  # [144, T]
            chord = data['chord']  # [T]
            metadata = data.get('metadata', {})

            # Get dataset and stem count
            dataset = metadata.get('dataset', 'coco' if not filepath.name.startswith("Track") else 'slakh')
            stems = metadata.get('stems', [])
            n_stems = len(stems)

            # Preprocess
            feature = np.log(np.abs(feature) + 1e-6)
            feature = (feature - mean) / std
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

            # Forward pass
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            chord_tensor = torch.tensor(chord, dtype=torch.long).to(device)

            self_attn_output, _ = model.self_attn_layers(feature_tensor)
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()

            # Calculate accuracy
            correct = (prediction == chord_tensor).sum().item()
            total = len(chord)

            # Store by dataset and stem count
            key = (dataset, n_stems)
            results[key]["correct"] += correct
            results[key]["total"] += total
            results[key]["count"] += 1

    # Print results
    print("\n" + "=" * 60)
    print("Results by Dataset and Number of Stems")
    print("=" * 60)

    for dataset in ["coco", "slakh"]:
        print(f"\n### {dataset.upper()} ###")
        print(f"{'Stems':<8} {'Examples':<10} {'Accuracy':<10}")
        print("-" * 30)

        dataset_correct = 0
        dataset_total = 0
        dataset_count = 0

        for n_stems in sorted(set(k[1] for k in results.keys() if k[0] == dataset)):
            key = (dataset, n_stems)
            r = results[key]
            acc = r["correct"] / r["total"] if r["total"] > 0 else 0
            print(f"{n_stems:<8} {r['count']:<10} {acc*100:.2f}%")
            dataset_correct += r["correct"]
            dataset_total += r["total"]
            dataset_count += r["count"]

        overall_acc = dataset_correct / dataset_total if dataset_total > 0 else 0
        print("-" * 30)
        print(f"{'Total':<8} {dataset_count:<10} {overall_acc*100:.2f}%")

    # Combined summary table
    print("\n" + "=" * 60)
    print("Combined Summary Table")
    print("=" * 60)
    print(f"{'Dataset':<10} {'Stems':<8} {'Examples':<10} {'Accuracy':<10}")
    print("-" * 40)

    for key in sorted(results.keys()):
        dataset, n_stems = key
        r = results[key]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        print(f"{dataset:<10} {n_stems:<8} {r['count']:<10} {acc*100:.2f}%")


if __name__ == "__main__":
    main()
