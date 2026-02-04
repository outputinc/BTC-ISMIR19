# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC-ISMIR19 is the official implementation of "A Bi-Directional Transformer for Musical Chord Recognition" (ISMIR 2019). It performs automatic chord recognition from audio files using a bi-directional transformer architecture.

## Common Commands

### Inference (Chord Recognition)
```bash
python test.py --audio_dir ./test --save_dir ./test --voca False
```
- `--voca False`: 25 chord classes (major/minor)
- `--voca True`: 170 chord classes (large vocabulary)
- Outputs: LAB files (time-stamped chord annotations) and MIDI files

### Training
```bash
python train.py --index 0 --kfold 0 --model btc --voca False
```
- `--index`: Dataset index (0=Isophonics, 1=UsPop2002, 2=Robbie Williams)
- `--kfold`: Cross-validation fold (0-4)
- `--model`: Model type (btc, cnn, crnn)

### Fine-tuning
```bash
python finetune_btc.py --data_dir path/to/finetuning_data
```

### Interactive Demo
```bash
python gradio_demo.py
```

## Architecture

### Core Model (btc_model.py)
The BTC model uses stacked bi-directional self-attention layers:
- 8 attention layers with 4 heads each
- Fixed input shape: (batch, 108 timesteps, 144 features)
- Processes audio forward and backward simultaneously

### Data Pipeline
1. Audio -> CQT features (144 bins via librosa)
2. Chunking into 108-frame windows
3. Global normalization (mean/std from training set)
4. Model inference -> chord predictions
5. Post-processing -> LAB/MIDI output

### Key Modules
- `utils/transformer_modules.py`: Multi-head attention, positional encoding, layer normalization
- `utils/preprocess.py`: CQT extraction, chord label processing
- `utils/mir_eval_modules.py`: Evaluation metrics (root, thirds, triads, majmin, mirex)
- `audio_dataset.py`: PyTorch Dataset with k-fold support

## Configuration

All hyperparameters in `run_config.yaml`:
- Audio: 22050 Hz sample rate, 10s segments, 5s skip interval
- Features: 144 CQT bins, 24 bins/octave, 2048 hop length
- Model: 8 layers, 4 heads, 128 hidden size, 25 or 170 output classes
- Training: 0.0001 learning rate, batch size 128, 100 max epochs

## Model Constants
- Timestep: 108 frames (fixed, padding applied for variable-length audio)
- Feature size: 144 bins
- Checkpoints saved to `model/` directory
