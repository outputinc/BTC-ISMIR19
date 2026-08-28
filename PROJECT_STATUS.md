# BTC Chord Recognition - Project Status

**Last Updated:** August 2026
**Status:** Shelved - documented for a clean restart

This direction is not being continued in its current form. The infrastructure and
findings are recorded here and in `EXPERIMENTS.md`; any pickup should retrain from
scratch rather than resume from the checkpoints described below. See "Restart Notes"
at the end of this file.

---

## Current Blockers

### 1. Audio Dataset Acquisition

**MoisesDB** (240 tracks, 12 genres, multi-stem):
- Download requires manual browser interaction at https://music.ai/research/
- Website uses JavaScript modals for license acceptance (CC BY-NC-SA 4.0)
- Automated download via Playwright unsuccessful - headless browser blocked
- Script `download_moisesdb.py` created but requires manual download step
- Expected size: ~15GB compressed

**Original BTC Training Datasets:**
- Isophonics (Beatles, Queen, etc.) - copyrighted audio not redistributable
- UsPop2002 - academic access only
- RWC Popular - requires license agreement
- These datasets have ground-truth chord annotations but audio is not freely available

### 2. Chord Annotation Quality

The finetuning approach uses **pseudo-labels** from BTC inference on mix audio:
1. Run BTC on full mix → get chord predictions per frame
2. Create random submixes from stems
3. Train model to predict same chords for submixes

**Problem:** Model learns to predict *which audio segment it's listening to* rather than the actual chords. Evidence:
- High accuracy (~77%) on COCO Chorales where segments are distinct
- Model predictions are consistent within a segment regardless of which stems are included
- This is essentially learning a segment embedding, not chord recognition

**Root Cause:** Pseudo-labels from BTC on the mix are the same for all submixes of that chunk, so the model learns segment identity rather than harmonic content.

---

## Potential Solutions for Annotation Quality

### Option A: Large Vocabulary (170 chords vs 25) - ATTEMPTED, NOT EVALUATED
- Current: 25 classes (12 major + 12 minor + N)
- Large vocab: 170 classes (includes 7ths, sus, aug, dim, etc.)
- More granular labels might force model to learn actual harmony
- Risk: May just learn finer-grained segment identity
- **Two runs completed 2026-02-05** (Experiment 4 in `EXPERIMENTS.md`), reaching
  53.78% and 66.26% teacher-agreement. Neither was ever evaluated, and the numbers
  are not comparable to the 25-class runs (different teacher, different vocabulary).
  **The hypothesis remains untested.**

### Option B: External Chord Prediction (Kord or similar)
- Use a different chord recognition model to generate labels
- Kord, Chordino, or other ACR systems
- Would provide independent labels, breaking segment-identity correlation
- Challenge: Need reliable ACR that works on stems/submixes

### Option C: Contrastive Learning Approach
- Don't use chord labels at all during finetuning
- Train embeddings where different submixes of same segment are similar
- Then fine-tune classification head on small labeled dataset
- More robust to pseudo-label noise

### Option D: Stem-Separate Original Datasets
- Take Isophonics/McGill Billboard audio (if obtainable)
- Run source separation (Demucs, Spleeter)
- Use ground-truth chord labels with separated stems
- Lower audio quality but higher annotation quality

---

## What Was Accomplished

### Scripts Created
| Script | Purpose |
|--------|---------|
| `create_finetuning_dataset.py` | Process COCO/Slakh/MUSDB into submix training data |
| `finetune_btc.py` | Finetuning loop with wandb logging |
| `prepare_moisesdb.py` | Process MoisesDB for finetuning (untested - no data) |
| `download_moisesdb.py` | Attempted automated download (requires manual step) |
| `evaluate_by_dataset.py` | Per-dataset accuracy breakdown |
| `evaluate_by_stems.py` | Per-stem-count accuracy analysis |
| `gradio_demo.py` | Interactive demo for testing |

### Datasets Processed
| Dataset | Tracks | Stems/Track | Status |
|---------|--------|-------------|--------|
| COCO Chorales | 1,200+ | 4 (SATB voices) | ✅ Processed |
| Slakh2100 | 2,100 | 5-15 (variable) | ✅ Processed |
| MUSDB18 | 150 | 4 (bass/drums/other/vocals) | ✅ Processed |
| MoisesDB | 240 | Variable | ❌ Blocked on download |

### Experiment Results Summary
| Configuration | Val Accuracy | Notes |
|---------------|--------------|-------|
| COCO only | 77.07% | Clean synthetic data |
| COCO + Slakh | 71.36% | Stem-level silence filtering |
| COCO + Slakh + MUSDB | ~70% | Added real-world complexity |
| Slakh, large voca (170) | 53.78% | Never evaluated; not comparable to rows above |
| All, large voca (170) | 66.26% | Never evaluated; not comparable to rows above |

**These are teacher-agreement rates, not chord accuracy.** Labels are the pretrained
BTC's own predictions on the full mix; there is no ground truth in this pipeline. The
25-class and 170-class rows were scored against different teacher models and cannot be
compared. See "What Val Accuracy Actually Measures" in `EXPERIMENTS.md`.

### Key Finding: Stem Count Correlation
More stems = higher accuracy (consistent across all experiments):
- 1 stem: ~50-62%
- 2 stems: ~62-79%
- 3 stems: ~67-86%

This suggests the model relies on harmonic density rather than learning robust single-instrument chord recognition.

---

## Repository State

### Tracked
`CLAUDE.md`, `EXPERIMENTS.md`, `PROJECT_STATUS.md`, and all pipeline scripts are
committed to `outputinc/BTC-ISMIR19`.

### Untracked (gitignored)
```
finetuned_models*/           # Checkpoint files
wandb/                       # Training logs
*.pt                         # Embedding files
*.png                        # Analysis visualizations
```
Checkpoints from Experiment 4 were archived to
`btc_finetuned_20260828.tar.gz` (both runs plus their `training_history.json`).

### Key Configuration
- Model: BTC (8 attention layers, 4 heads, 128 hidden)
- Input: CQT features (144 bins, 108 timesteps)
- Output: 25 chord classes (major/minor + N), or 170 with `--voca True`
- Training: Adam optimizer, lr=1e-5, batch_size=32

---

## Restart Notes

Read before picking this up again.

### Do not resume from the existing checkpoints
They were trained against pseudo-labels with the objective described below. Whatever
they learned is entangled with that setup. Start training over.

### Known issues in the current code
1. **Eval scripts cannot load large-voca checkpoints.** `evaluate_by_dataset.py`,
   `evaluate_by_stems.py`, and `gradio_comparison_demo.py` build `BTC_model` from the
   default 25-class config and fail with a size mismatch on
   `output_layer.output_projection.weight`. The checkpoints record `large_voca` and
   `num_chords` - read those before constructing the model.
2. **`test.py` cannot load a finetuned checkpoint at all** - `model_file` is hardcoded
   to the pretrained paths. It needs a `--checkpoint` argument to serve as an
   end-to-end audio -> LAB/MIDI demo for finetuned models.
3. **Loss counts padded frames, accuracy does not.** `labels.view(-1)` in the loss
   covers all 108 positions while `compute_accuracy` masks via `lengths`. At 10s
   chunks that is 107 real frames of 108, so ~1% of the training signal is predicting
   `N` on padding.
4. **`gradio_comparison_demo.py` has a stale header** claiming "71.92% val acc" for a
   checkpoint that is actually a 53.78% large-voca model.

### The core problem is unchanged
The objective is frame-wise cross-entropy against pseudo-labels: predict the full-mix
chord classes given only a subset of stems. Every submix of a chunk shares one label
vector, so the objective can be satisfied by learning segment identity rather than
harmony. No amount of retraining on this setup fixes that. A restart should change the
labels (ground truth, or an independent ACR) or change the objective (contrastive,
Option C) - not just rerun with different hyperparameters.

### If evaluating anyway
Map large-voca predictions down to majmin and score with `utils/mir_eval_modules.py`
(`root`, `thirds`, `majmin`, `mirex`) so vocabularies share a yardstick. Nothing in
the repo currently wires that to a finetuned checkpoint.

---

## Related Resources

- **BTC Paper:** [ISMIR 2019](https://archives.ismir.net/ismir2019/paper/000019.pdf)
- **MoisesDB Paper:** [arXiv:2307.15913](https://arxiv.org/abs/2307.15913)
- **Isophonics Annotations:** http://isophonics.net/datasets
- **McGill Billboard:** https://ddmal.music.mcgill.ca/research/billboard

---

## Contact / Handoff Notes

This work is shelved as of August 2026. The core challenge is that **finetuning on pseudo-labels teaches segment identity, not chord recognition**. Any continuation should focus on:

1. Getting better chord annotations (ground truth or better ACR)
2. Or changing the training objective (contrastive learning)
3. Or obtaining the original BTC training data with real annotations

The infrastructure (data pipelines, training loops, evaluation) is solid. The bottleneck is data quality, not code.
