#!/usr/bin/env python
"""
Gradio demo for comparing harmonic retrieval between base and finetuned BTC embeddings.

Shows side-by-side comparison of top 10 most similar stems using:
- Column 1: Base BTC model embeddings
- Column 2: Finetuned BTC model embeddings (COCO+Slakh silence-filtered, 71.92% val acc)

Finetuned checkpoint: finetuned_models/btc_finetuned_best.pt
"""

import gradio as gr
import torch
import numpy as np
import librosa
import soundfile as sf
import random
from pathlib import Path
import tempfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model info
FINETUNED_CHECKPOINT = "finetuned_models/btc_finetuned_best.pt"
FINETUNED_INFO = "COCO+Slakh (silence-filtered), 71.92% val acc"

# Dataset configurations - unified embedding file with both datasets
EMBEDDING_FILE = "comparison_embeddings_combined.pt"

# Legacy separate files (fallback)
LEGACY_DATASETS = {
    "coco_chorales": "comparison_embeddings_coco_chorales.pt",
    "slakh": "comparison_embeddings_slakh.pt",
}

# Cache for loaded data
loaded_data = None

def load_embeddings():
    """Load and cache embeddings."""
    global loaded_data
    if loaded_data is not None:
        return loaded_data

    # Try combined file first, then fall back to legacy
    if os.path.exists(EMBEDDING_FILE):
        filepath = EMBEDDING_FILE
        print(f"Loading combined embeddings from {filepath}...")
    else:
        # Fall back to legacy files - merge them
        print("Combined embedding file not found, trying legacy files...")
        all_base = []
        all_finetuned = []
        all_metadata = []

        for name, fp in LEGACY_DATASETS.items():
            if os.path.exists(fp):
                print(f"Loading {name} from {fp}...")
                data = torch.load(fp, weights_only=False)
                all_base.append(data['base_embeddings'])
                all_finetuned.append(data['finetuned_embeddings'])
                for m in data['metadata']:
                    m['dataset'] = name
                    all_metadata.append(m)

        if not all_base:
            raise FileNotFoundError("No embedding files found!")

        loaded_data = {
            'base_embeddings': torch.cat(all_base, dim=0),
            'finetuned_embeddings': torch.cat(all_finetuned, dim=0),
            'metadata': all_metadata,
        }
        # Normalize
        loaded_data['base_normalized'] = loaded_data['base_embeddings'] / (loaded_data['base_embeddings'].norm(dim=2, keepdim=True) + 1e-8)
        loaded_data['finetuned_normalized'] = loaded_data['finetuned_embeddings'] / (loaded_data['finetuned_embeddings'].norm(dim=2, keepdim=True) + 1e-8)
        loaded_data['num_stems'] = len(all_metadata)

        print(f"Merged {loaded_data['num_stems']} stems total")
        return loaded_data

    data = torch.load(filepath, weights_only=False)

    base_embeddings = data['base_embeddings']
    finetuned_embeddings = data['finetuned_embeddings']
    metadata = data['metadata']

    # Normalize embeddings for cosine similarity
    base_normalized = base_embeddings / (base_embeddings.norm(dim=2, keepdim=True) + 1e-8)
    finetuned_normalized = finetuned_embeddings / (finetuned_embeddings.norm(dim=2, keepdim=True) + 1e-8)

    num_stems = len(metadata)

    # Count by dataset
    coco_count = sum(1 for m in metadata if m.get('dataset') == 'coco' or 'coco' in m.get('stem_path', '').lower())
    slakh_count = sum(1 for m in metadata if m.get('dataset') == 'slakh' or 'slakh' in m.get('stem_path', '').lower())

    print(f"Loaded {num_stems} stems ({coco_count} COCO, {slakh_count} Slakh)")
    print(f"Base embeddings shape: {base_embeddings.shape}")
    print(f"Finetuned embeddings shape: {finetuned_embeddings.shape}")

    loaded_data = {
        'base_normalized': base_normalized,
        'finetuned_normalized': finetuned_normalized,
        'base_embeddings': base_embeddings,
        'finetuned_embeddings': finetuned_embeddings,
        'metadata': metadata,
        'num_stems': num_stems,
    }

    return loaded_data

# Preload embeddings
print(f"Finetuned model: {FINETUNED_INFO}")
try:
    load_embeddings()
except FileNotFoundError as e:
    print(f"Warning: {e}")

DURATION = 5.0
SAMPLE_RATE = 22050


def load_audio_segment(audio_path, duration=DURATION, sr=SAMPLE_RATE):
    """Load first N seconds of audio."""
    wav, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    target_len = int(duration * sr)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    return wav


def save_audio_temp(wav, sr=SAMPLE_RATE):
    """Save audio to a temporary file."""
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    sf.write(path, wav, sr)
    return path


def find_top_k_similar(query_idx, normalized_embeddings, metadata, k=10, exclude_same_track=True):
    """Find top-k most similar stems using cosine similarity."""
    query_emb = normalized_embeddings[query_idx]  # [54, 24]
    query_track = metadata[query_idx]['track_name']

    # Compute mean cosine similarity for all stems
    # Element-wise multiply and sum along feature dim, then mean along time
    similarities = (normalized_embeddings * query_emb.unsqueeze(0)).sum(dim=2).mean(dim=1)  # [N]

    # Sort by similarity (descending)
    sorted_indices = torch.argsort(similarities, descending=True)

    results = []
    for idx in sorted_indices.tolist():
        if idx == query_idx:
            continue
        if exclude_same_track and metadata[idx]['track_name'] == query_track:
            continue

        results.append({
            'index': idx,
            'similarity': similarities[idx].item(),
            'track_name': metadata[idx]['track_name'],
            'stem_name': metadata[idx]['stem_name'],
            'stem_path': metadata[idx]['stem_path'],
        })
        if len(results) >= k:
            break

    return results


def mix_audio(wav1, wav2, ratio=0.5):
    """Mix two audio signals."""
    return wav1 * ratio + wav2 * (1 - ratio)


def get_dataset_type(meta):
    """Determine dataset type from metadata."""
    if meta.get('dataset'):
        return meta['dataset']
    stem_path = meta.get('stem_path', '').lower()
    if 'coco' in stem_path:
        return 'coco'
    elif 'slakh' in stem_path:
        return 'slakh'
    return 'unknown'


def compare_retrieval(source_dataset):
    """Main function called when user clicks the comparison button."""
    # Load embeddings
    try:
        data = load_embeddings()
    except FileNotFoundError as e:
        # Return error message in the UI
        error_msg = f"**Error:** {e}"
        empty_outputs = [None, error_msg, ""]
        for _ in range(10):
            empty_outputs.extend([None, "", None, ""])
        return tuple(empty_outputs)

    base_normalized = data['base_normalized']
    finetuned_normalized = data['finetuned_normalized']
    metadata = data['metadata']
    num_stems = data['num_stems']

    # Filter indices by source dataset if specified
    if source_dataset == "both":
        valid_indices = list(range(num_stems))
    else:
        valid_indices = [i for i, m in enumerate(metadata) if get_dataset_type(m) == source_dataset]

    if not valid_indices:
        error_msg = f"**Error:** No stems found for dataset '{source_dataset}'"
        empty_outputs = [None, error_msg, ""]
        for _ in range(10):
            empty_outputs.extend([None, "", None, ""])
        return tuple(empty_outputs)

    # Select random stem from filtered set
    query_idx = random.choice(valid_indices)
    query_meta = metadata[query_idx]

    # Load query audio
    query_wav = load_audio_segment(query_meta['stem_path'])
    query_audio_path = save_audio_temp(query_wav)

    # Find top 10 matches for both models
    base_matches = find_top_k_similar(query_idx, base_normalized, metadata, k=10)
    finetuned_matches = find_top_k_similar(query_idx, finetuned_normalized, metadata, k=10)

    # Format query info
    query_dataset = get_dataset_type(query_meta)
    query_info = f"**Query:** {query_meta['track_name']} / {query_meta['stem_name']} ({query_dataset.upper()})"

    # Calculate overlap between top 10 results
    base_stems = set((m['track_name'], m['stem_name']) for m in base_matches)
    finetuned_stems = set((m['track_name'], m['stem_name']) for m in finetuned_matches)
    overlap = len(base_stems & finetuned_stems)
    overlap_info = f"**Overlap in top 10:** {overlap}/10 stems appear in both lists"

    # Create mixed audio for all base matches
    base_audios = []
    base_infos = []
    for i, match in enumerate(base_matches):
        match_wav = load_audio_segment(match['stem_path'])
        mixed_wav = mix_audio(query_wav, match_wav, ratio=0.5)
        mixed_path = save_audio_temp(mixed_wav)
        base_audios.append(mixed_path)
        base_infos.append(f"**#{i+1}** sim={match['similarity']:.3f}\n{match['stem_name']}")

    # Create mixed audio for all finetuned matches
    finetuned_audios = []
    finetuned_infos = []
    for i, match in enumerate(finetuned_matches):
        match_wav = load_audio_segment(match['stem_path'])
        mixed_wav = mix_audio(query_wav, match_wav, ratio=0.5)
        mixed_path = save_audio_temp(mixed_wav)
        finetuned_audios.append(mixed_path)
        finetuned_infos.append(f"**#{i+1}** sim={match['similarity']:.3f}\n{match['stem_name']}")

    # Pad if fewer than 10 matches
    while len(base_audios) < 10:
        base_audios.append(None)
        base_infos.append("")
    while len(finetuned_audios) < 10:
        finetuned_audios.append(None)
        finetuned_infos.append("")

    # Build output tuple
    outputs = [query_audio_path, query_info, overlap_info]
    for i in range(10):
        outputs.extend([base_audios[i], base_infos[i], finetuned_audios[i], finetuned_infos[i]])

    return tuple(outputs)


# Build Gradio interface
with gr.Blocks(title="BTC Harmonic Retrieval Comparison") as demo:
    gr.Markdown(f"""
    # Harmonic Retrieval Comparison: Base vs Finetuned BTC

    This demo compares harmonic retrieval between the **base BTC model** and the **finetuned BTC model**.

    **Finetuned Model:** {FINETUNED_INFO}

    Click **Compare Retrieval** to:
    1. Select a random query stem from the chosen dataset
    2. Find the top 10 most similar stems using cosine similarity on embeddings
    3. Compare results side-by-side: base model (left) vs finetuned model (right)

    Each audio player plays a 50/50 mix of the query stem + matched stem.
    """)

    with gr.Row():
        source_dropdown = gr.Dropdown(
            choices=["coco", "slakh", "both"],
            value="both",
            label="Source Dataset",
            info="Select which dataset to draw the query stem from"
        )
        compare_btn = gr.Button("Compare Retrieval", variant="primary", size="lg")

    gr.Markdown("## Query Stem")
    with gr.Row():
        query_audio = gr.Audio(label="Query Audio (5 sec)", type="filepath")
        with gr.Column():
            query_info = gr.Markdown()
            overlap_info = gr.Markdown()

    gr.Markdown("## Top 10 Similar Stems (Mixed with Query)")
    with gr.Row():
        gr.Markdown("### Base Model", elem_id="base-header")
        gr.Markdown("### Finetuned Model", elem_id="finetuned-header")

    # Create 10 rows of audio players, each with base (left) and finetuned (right)
    base_audios = []
    base_infos = []
    finetuned_audios = []
    finetuned_infos = []

    for i in range(10):
        with gr.Row():
            with gr.Column():
                base_audio = gr.Audio(label=f"Base #{i+1}", type="filepath")
                base_info = gr.Markdown()
                base_audios.append(base_audio)
                base_infos.append(base_info)
            with gr.Column():
                finetuned_audio = gr.Audio(label=f"Finetuned #{i+1}", type="filepath")
                finetuned_info = gr.Markdown()
                finetuned_audios.append(finetuned_audio)
                finetuned_infos.append(finetuned_info)

    # Build outputs list
    outputs = [query_audio, query_info, overlap_info]
    for i in range(10):
        outputs.extend([base_audios[i], base_infos[i], finetuned_audios[i], finetuned_infos[i]])

    compare_btn.click(
        fn=compare_retrieval,
        inputs=[source_dropdown],
        outputs=outputs
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
