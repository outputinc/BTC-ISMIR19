# Fine-tuning Experiments

This document tracks fine-tuning experiments for the BTC (Bi-Directional Transformer for Chord Recognition) model on stem separation datasets.

## Datasets

### COCO Chorales
- Synthesized Bach chorales with clean, well-separated stems
- 4 stems per track (soprano, alto, tenor, bass voices)
- All stems are tonal and have consistent audio throughout

### Slakh2100
- Real-world multi-instrument arrangements (MIDI rendered to audio)
- Variable number of stems per track (typically 5-15)
- Challenges:
  - Many stems are silent at the start (instruments enter later in the song)
  - Contains non-tonal instruments (drums, percussion)
  - More complex harmonic content than COCO

### MUSDB18
- Professional multi-track recordings for music source separation
- 150 full-length tracks (100 train, 50 test), ~3-7 minutes each
- 4 stems per track: bass, drums, other, vocals
- High-quality studio recordings with real-world complexity
- Location: `~/datasets/musdb18`
- Tonality: drums is non-tonal; bass, other, vocals are tonal

## Experiment 1: COCO Only (Baseline)

**Date:** 2024-02

**Configuration:**
- Training data: COCO Chorales only
- Train examples: 121,359
- Validation examples: 40,311
- Learning rate: 1e-5
- Epochs: 10

**Results:**
| Metric | Value |
|--------|-------|
| Best Val Accuracy | **77.07%** |

**Observations:**
- Strong performance on synthesized Bach chorales
- Clean, consistent stems make chord recognition easier
- Serves as upper bound for COCO-style data

---

## Experiment 2: COCO + Slakh (Submix-level Silence Filtering)

**Date:** 2024-02

**Configuration:**
- Training data: COCO + Slakh2100
- Silence filtering: RMS threshold on combined submix (not individual stems)
- Train examples: 183,619 (121,359 COCO + 62,260 Slakh)
- Validation examples: 58,629 (40,311 COCO + 18,318 Slakh)
- Learning rate: 1e-5
- Epochs: 10

**Results:**
| Metric | Value |
|--------|-------|
| Best Val Accuracy | **71.92%** |

**Per-Dataset Breakdown:**
| Dataset | Examples | Accuracy |
|---------|----------|----------|
| COCO | 40,311 | 75.66% |
| Slakh | 18,318 | 57.37% |
| Combined | 58,629 | 70.71% |

**Per-Stem Breakdown:**
| Dataset | Stems | Examples | Accuracy |
|---------|-------|----------|----------|
| COCO | 1 | 13,397 | 62.32% |
| COCO | 2 | 13,369 | 78.71% |
| COCO | 3 | 13,545 | 85.82% |
| Slakh | 1 | 6,982 | 47.88% |
| Slakh | 2 | 5,948 | 60.04% |
| Slakh | 3 | 5,388 | 66.93% |

**Observations:**
- Adding Slakh data decreased overall accuracy (77.07% → 71.92%)
- Slakh is significantly harder than COCO (57.37% vs 75.66%)
- More stems = higher accuracy (consistent pattern)
- Problem identified: Submix-level filtering still allowed silent individual stems

---

## Experiment 3: COCO + Slakh (Stem-level Silence Filtering)

**Date:** 2024-02

**Configuration:**
- Training data: COCO + Slakh2100
- Silence filtering: RMS threshold (0.03) on **individual stems** before creating submixes
- Train examples: 210,807 (121,359 COCO + 89,448 Slakh)
- Validation examples: 58,629 (40,311 COCO + 18,318 Slakh)
- Learning rate: 1e-5
- Epochs: 10

**Results:**
| Metric | Value |
|--------|-------|
| Best Val Accuracy | **71.36%** |

**Per-Dataset Breakdown:**
| Dataset | Examples | Accuracy |
|---------|----------|----------|
| COCO | 40,311 | 75.40% |
| Slakh | 18,318 | **60.39%** |
| Combined | 58,629 | 70.71% |

**Per-Stem Breakdown:**
| Dataset | Stems | Examples | Accuracy |
|---------|-------|----------|----------|
| COCO | 1 | 13,397 | 62.01% |
| COCO | 2 | 13,369 | 78.83% |
| COCO | 3 | 13,545 | 85.27% |
| Slakh | 1 | 6,982 | 51.19% |
| Slakh | 2 | 5,948 | 62.87% |
| Slakh | 3 | 5,388 | 69.57% |

**Observations:**
- Stem-level filtering increased Slakh training data: 62,260 → 89,448 (+44%)
- Slakh accuracy improved: 57.37% → **60.39%** (+3.0%)
- 1-stem Slakh improved most: 47.88% → 51.19% (+3.3%)
- COCO performance remained stable (~75%)
- Overall val accuracy slightly lower due to more Slakh examples in training mix

---

## Key Findings

### 1. Dataset Difficulty
COCO Chorales is significantly easier than Slakh2100:
- COCO: ~75% accuracy (synthesized, clean stems)
- Slakh: ~60% accuracy (real-world, complex arrangements)

### 2. Stem Count Correlation
More stems consistently leads to higher accuracy:
- 1 stem → 2 stems: +15-17% accuracy
- 2 stems → 3 stems: +6-7% accuracy

This makes sense: more harmonic information = easier chord recognition.

### 3. Silence Filtering Matters
Filtering at the stem level (not submix level) is important for Slakh:
- Slakh instruments often don't play at the start of songs
- Submix filtering can pass silent individual stems
- Stem-level filtering ensures each stem contributes meaningful audio

### 4. Non-tonal Instrument Filtering
For demo/inference, filtering out drums and percussion improves quality:
- `is_drum: true` stems should be excluded
- Non-tonal instrument classes (Drums, Percussion, Sound Effects) don't contribute to chord recognition

---

## Model Checkpoints

| Experiment | Checkpoint Path | Val Acc |
|------------|-----------------|---------|
| COCO only | `finetuned_models/btc_finetuned_coco_only.pt` | 77.07% |
| COCO + Slakh (submix filter) | `finetuned_models/btc_finetuned_submix_filter.pt` | 71.92% |
| COCO + Slakh (stem filter) | `finetuned_models/btc_finetuned_best.pt` | 71.36% |

---

## Future Experiments


### Potential improvements to try:
1. **More Multi-Track Datasets**: Add MUSDB, MOISESDB
2. **Stem Separate Original Datasets**: Lower quality audio, higher quality chord labels.
3. **Data augmentation**: Pitch shifting, time stretching on stems
4. **Match Inference Distribution**: Time stretch to 120bpm and use 8/16 seconds (4/8 bars)

### Metrics to track:
- Per-instrument accuracy on Slakh
- Confusion matrix for chord types (major/minor)
- Performance on specific chord transitions
