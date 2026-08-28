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

## Experiment 4: Large Vocabulary (170 chords)

**Date:** 2026-02-05

**Motivation:** Test Option A from `PROJECT_STATUS.md` — whether finer-grained chord
labels (7ths, sus, aug, dim) force the model to learn actual harmony rather than
segment identity.

**Configuration:**
- Vocabulary: 170 classes (`--voca True`), teacher = `btc_model_large_voca.pt`
- Two datasets built with `create_finetuning_dataset.py --voca True`
- Learning rate: 1e-5, batch size 32, epochs 20 (identical to Experiments 1-3)

**Results:**
| Run | Dataset | Train / Valid | Output dir | Best Val Acc |
|-----|---------|---------------|------------|--------------|
| 4a | `btc_finetuning_large_voca` (slakh) | 89,448 / 18,318 | `finetuned_models/` | **53.78%** (ep 16) |
| 4b | `btc_finetuning_large_voca_all` (coco+slakh+musdb) | 216,957 / 62,049 | `finetuned_models_all/` | **66.26%** (ep 20) |

**Nominal comparison against matched small-vocabulary runs:**
| Dataset | Small voca (25) | Large voca (170) | Delta |
|---------|-----------------|------------------|-------|
| slakh only, ~89k | 71.36% | 53.78% | -17.6 pts |
| all, ~217k | 69.89% (ep 13, interrupted) | 66.26% (ep 20) | -3.6 pts |

**These deltas are not interpretable.** See "What Val Accuracy Actually Measures"
below — the two columns are agreement rates against two *different* teacher models
(`btc_model.pt` vs `btc_model_large_voca.pt`), each labeling its own dataset. There
is no shared yardstick. A drop is also expected on difficulty grounds alone: 170-way
vs 25-way classification, where `C:maj7` and `C:maj` are distinct labels in one
vocabulary and identical in the other.

**Status: trained but never evaluated.** Neither run has a wandb record (wandb
logging was not enabled for either), and neither was ever run through
`evaluate_by_dataset.py` or `evaluate_by_stems.py`. No conclusion was reached about
whether large vocabulary addresses the pseudo-label problem. Option A in
`PROJECT_STATUS.md` should still be considered untested.

**Note:** both eval scripts and `gradio_comparison_demo.py` currently fail to load
these checkpoints — they construct `BTC_model` from the default 25-class config and
hit a size mismatch on `output_layer.output_projection.weight` ([170, 128] vs
[25, 128]). The checkpoints now record `large_voca` and `num_chords`, so the fix is
to read those and set `config.feature['large_voca']` / `config.model['num_chords']`
before constructing the model.

---

## What Val Accuracy Actually Measures

Worth stating explicitly, because it is easy to misread every number in this file.

The training objective is **frame-wise cross-entropy** (`F.nll_loss` in
`SoftmaxOutputLayer.loss`, `utils/transformer_modules.py:86-89`). It is *not*
contrastive — no embedding distances, no positive/negative pairs. `compute_accuracy`
(`finetune_btc.py:104-113`) is top-1 agreement per frame, masked to non-padded frames.

The labels are **pseudo-labels**: `create_finetuning_dataset.py:459` gives every
submix of a chunk the same `chord_chunk`, sliced from the pretrained BTC's prediction
on the **full mix**. No ground truth is involved at any point.

So val accuracy answers: *given a subset of stems, how often does the model reproduce
what the teacher said about the full mix?* It is a teacher-agreement rate, not chord
recognition accuracy. Comparing it across experiments is only meaningful when the
teacher and vocabulary are held fixed.

This also explains the shape of the results. Because all submixes of a chunk share one
label vector, cross-entropy imposes a "same chunk -> same output" constraint, which the
model can satisfy by learning segment identity rather than harmony. That is the same
failure mode documented in `PROJECT_STATUS.md`, arrived at from the loss function
rather than from the accuracy numbers.

**Known inconsistency:** the loss includes padded frames (`labels.view(-1)` covers all
108 positions) while accuracy excludes them via `lengths`. At 10s chunks this is
`int(10*22050/2048)` = 107 real frames out of 108, so ~1% of the training signal is the
model learning to predict `N` on padding.

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
