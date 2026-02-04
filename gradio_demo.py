#!/usr/bin/env python
"""
Gradio demo for BTC model stem similarity search.

Allows users to select a random stem and find the top 10 most similar stems
based on BTC output layer embeddings with temporal similarity analysis.
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

# Load embeddings and metadata
print("Loading embeddings and metadata...")
embeddings_data = torch.load('stem_embeddings.pt', weights_only=False)
embeddings = embeddings_data['embeddings']  # [N, 54, 24]
metadata = torch.load('stem_metadata.pt', weights_only=False)

print(f"Loaded {len(metadata)} stems from {len(set(m['track_name'] for m in metadata))} tracks")
print(f"Embeddings shape: {embeddings.shape}")

# Normalize embeddings along the feature dimension for cosine similarity
# Shape: [N, 54, 24] -> normalize along dim=2
normalized_embeddings = embeddings / (embeddings.norm(dim=2, keepdim=True) + 1e-8)

DURATION = 5.0
SAMPLE_RATE = 22050
NUM_TIMESTEPS = embeddings.shape[1]  # 54


def load_audio_segment(audio_path, duration=DURATION, sr=SAMPLE_RATE):
    """Load first N seconds of audio."""
    wav, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    # Ensure consistent length
    target_len = int(duration * sr)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    return wav


def mix_audio(wav1, wav2, ratio=0.5):
    """Mix two audio signals."""
    return wav1 * ratio + wav2 * (1 - ratio)


def compute_temporal_similarity(query_idx, target_idx):
    """
    Compute cosine similarity at each timestep between query and target.

    Returns:
        similarities: numpy array of shape [54] with cosine sim at each timestep
        mean_sim: mean similarity across all timesteps
    """
    query_emb = normalized_embeddings[query_idx]  # [54, 24]
    target_emb = normalized_embeddings[target_idx]  # [54, 24]

    # Compute cosine similarity at each timestep
    # Element-wise multiply and sum along feature dimension
    similarities = (query_emb * target_emb).sum(dim=1)  # [54]

    mean_sim = similarities.mean().item()
    return similarities.numpy(), mean_sim


def find_similar_stems(query_idx, top_k=5, bottom_k=5, exclude_same_track=True):
    """Find the top-k most similar and bottom-k least similar stems by mean temporal similarity."""
    query_emb = normalized_embeddings[query_idx]  # [54, 24]

    # Compute mean cosine similarity for all stems
    # query_emb: [54, 24], normalized_embeddings: [N, 54, 24]
    # Element-wise multiply and sum along feature dim, then mean along time
    similarities = (normalized_embeddings * query_emb.unsqueeze(0)).sum(dim=2).mean(dim=1)  # [N]

    # Get query track name
    query_track = metadata[query_idx]['track_name']

    # Sort by similarity (descending for top, ascending for bottom)
    sorted_indices_desc = torch.argsort(similarities, descending=True)
    sorted_indices_asc = torch.argsort(similarities, descending=False)

    def collect_results(sorted_indices, k):
        results = []
        for idx in sorted_indices.tolist():
            if idx == query_idx:
                continue
            if exclude_same_track and metadata[idx]['track_name'] == query_track:
                continue

            # Get temporal similarity curve
            temporal_sims, mean_sim = compute_temporal_similarity(query_idx, idx)

            results.append({
                'index': idx,
                'mean_similarity': mean_sim,
                'temporal_similarities': temporal_sims,
                'track_name': metadata[idx]['track_name'],
                'stem_name': metadata[idx]['stem_name'],
                'stem_path': metadata[idx]['stem_path'],
            })
            if len(results) >= k:
                break
        return results

    top_results = collect_results(sorted_indices_desc, top_k)
    bottom_results = collect_results(sorted_indices_asc, bottom_k)

    return top_results, bottom_results


def save_audio_temp(wav, sr=SAMPLE_RATE):
    """Save audio to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    sf.write(path, wav, sr)
    return path


def create_similarity_plot(temporal_sims, mean_sim, title=""):
    """Create a plot of cosine similarity vs timestep."""
    fig, ax = plt.subplots(figsize=(4, 2.5))

    timesteps = np.arange(len(temporal_sims))
    time_seconds = timesteps * (DURATION / NUM_TIMESTEPS)

    ax.plot(time_seconds, temporal_sims, 'b-', linewidth=1.5)
    ax.axhline(y=mean_sim, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_sim:.3f}')
    ax.fill_between(time_seconds, temporal_sims, alpha=0.3)

    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Cosine Sim', fontsize=9)
    ax.set_xlim(0, DURATION)
    ax.set_ylim(-1, 1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return path


def choose_random():
    """Main function called when user clicks 'Choose Random'."""
    # Select random stem
    query_idx = random.randint(0, len(metadata) - 1)
    query_meta = metadata[query_idx]

    # Load query audio
    query_wav = load_audio_segment(query_meta['stem_path'])
    query_audio_path = save_audio_temp(query_wav)

    # Find similar and dissimilar stems
    top_matches, bottom_matches = find_similar_stems(query_idx, top_k=5, bottom_k=5)

    # Prepare outputs
    query_info = f"**Source:** {query_meta['track_name']} / {query_meta['stem_name']}"

    # Create mixed audio and plots for top matches (most similar)
    top_audios = []
    top_infos = []
    top_plots = []
    for i, match in enumerate(top_matches):
        match_wav = load_audio_segment(match['stem_path'])
        mixed_wav = mix_audio(query_wav, match_wav, ratio=0.5)
        mixed_path = save_audio_temp(mixed_wav)
        top_audios.append(mixed_path)
        top_infos.append(
            f"**#{i+1}** (mean sim={match['mean_similarity']:.3f}): "
            f"{match['track_name']} / {match['stem_name']}"
        )
        plot_path = create_similarity_plot(
            match['temporal_similarities'],
            match['mean_similarity']
        )
        top_plots.append(plot_path)

    # Create mixed audio and plots for bottom matches (least similar)
    bottom_audios = []
    bottom_infos = []
    bottom_plots = []
    for i, match in enumerate(bottom_matches):
        match_wav = load_audio_segment(match['stem_path'])
        mixed_wav = mix_audio(query_wav, match_wav, ratio=0.5)
        mixed_path = save_audio_temp(mixed_wav)
        bottom_audios.append(mixed_path)
        bottom_infos.append(
            f"**#{i+1}** (mean sim={match['mean_similarity']:.3f}): "
            f"{match['track_name']} / {match['stem_name']}"
        )
        plot_path = create_similarity_plot(
            match['temporal_similarities'],
            match['mean_similarity']
        )
        bottom_plots.append(plot_path)

    # Pad if fewer than 5 matches
    while len(top_audios) < 5:
        top_audios.append(None)
        top_infos.append("")
        top_plots.append(None)
    while len(bottom_audios) < 5:
        bottom_audios.append(None)
        bottom_infos.append("")
        bottom_plots.append(None)

    return (
        query_audio_path,
        query_info,
        top_audios[0], top_infos[0], top_plots[0],
        top_audios[1], top_infos[1], top_plots[1],
        top_audios[2], top_infos[2], top_plots[2],
        top_audios[3], top_infos[3], top_plots[3],
        top_audios[4], top_infos[4], top_plots[4],
        bottom_audios[0], bottom_infos[0], bottom_plots[0],
        bottom_audios[1], bottom_infos[1], bottom_plots[1],
        bottom_audios[2], bottom_infos[2], bottom_plots[2],
        bottom_audios[3], bottom_infos[3], bottom_plots[3],
        bottom_audios[4], bottom_infos[4], bottom_plots[4],
    )


# Build Gradio interface
with gr.Blocks(title="BTC Stem Similarity Search") as demo:
    gr.Markdown("""
    # BTC Stem Similarity Search

    This demo uses hidden representations from the BTC (Bidirectional Transformer for Chords)
    model to find similar and dissimilar musical stems.

    Click **Choose Random** to:
    1. Select a random stem from the dataset
    2. Find the 5 most similar and 5 least similar stems (from different tracks)
    3. Listen to the source and mixed audio (source + match at 50/50)
    4. View cosine similarity over time for each match

    The similarity is based on the 24-dimensional output layer (chord logits) of the BTC model,
    computed at each timestep and sorted by mean similarity.
    """)

    with gr.Row():
        random_btn = gr.Button("Choose Random", variant="primary", size="lg")

    gr.Markdown("## Source Stem")
    with gr.Row():
        source_audio = gr.Audio(label="Source Audio (5 sec)", type="filepath")
        source_info = gr.Markdown()

    gr.Markdown("## Top 5 Most Similar (Mixed: 50% Source + 50% Match)")

    # Create 5 top match rows with plots
    top_audios = []
    top_infos = []
    top_plots = []
    for i in range(5):
        with gr.Row():
            audio = gr.Audio(label=f"Best #{i+1}", type="filepath")
            info = gr.Markdown()
            plot = gr.Image(label="Similarity over Time")
            top_audios.append(audio)
            top_infos.append(info)
            top_plots.append(plot)

    gr.Markdown("## Bottom 5 Least Similar (Mixed: 50% Source + 50% Match)")

    # Create 5 bottom match rows with plots
    bottom_audios = []
    bottom_infos = []
    bottom_plots = []
    for i in range(5):
        with gr.Row():
            audio = gr.Audio(label=f"Worst #{i+1}", type="filepath")
            info = gr.Markdown()
            plot = gr.Image(label="Similarity over Time")
            bottom_audios.append(audio)
            bottom_infos.append(info)
            bottom_plots.append(plot)

    # Connect button to function
    outputs = [source_audio, source_info]
    for i in range(5):
        outputs.extend([top_audios[i], top_infos[i], top_plots[i]])
    for i in range(5):
        outputs.extend([bottom_audios[i], bottom_infos[i], bottom_plots[i]])

    random_btn.click(fn=choose_random, inputs=[], outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
