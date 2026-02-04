# BTC Chord Recognition - Project Status

**Last Updated:** February 2026
**Status:** Paused - Blocked on data acquisition and annotation quality

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

### Option A: Large Vocabulary (170 chords vs 25)
- Current: 25 classes (12 major + 12 minor + N)
- Large vocab: 170 classes (includes 7ths, sus, aug, dim, etc.)
- More granular labels might force model to learn actual harmony
- Risk: May just learn finer-grained segment identity

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

### Key Finding: Stem Count Correlation
More stems = higher accuracy (consistent across all experiments):
- 1 stem: ~50-62%
- 2 stems: ~62-79%
- 3 stems: ~67-86%

This suggests the model relies on harmonic density rather than learning robust single-instrument chord recognition.

---

## Repository State

### Untracked Files (not committed)
```
CLAUDE.md                    # Claude Code instructions
EXPERIMENTS.md               # Experiment tracking
PROJECT_STATUS.md            # This file
finetuned_models/            # Checkpoint files
wandb/                       # Training logs
*.pt                         # Embedding files
*.png                        # Analysis visualizations
```

### Key Configuration
- Model: BTC (8 attention layers, 4 heads, 128 hidden)
- Input: CQT features (144 bins, 108 timesteps)
- Output: 25 chord classes (major/minor + N)
- Training: Adam optimizer, lr=1e-5, batch_size=32

---

## Recommended Next Steps

### Short Term (if resuming)
1. **Manual MoisesDB download** - Visit music.ai/research in browser
2. **Test prepare_moisesdb.py** - Verify pipeline works with real data
3. **Evaluate on held-out test sets** - Beyond validation accuracy

### Medium Term (addressing annotation quality)
1. **Implement contrastive pretraining** - Learn stem-invariant embeddings
2. **Integrate external ACR** - Kord or Chordino for independent labels
3. **Obtain licensed datasets** - Contact Isophonics maintainers

### Long Term (if pursuing this direction)
1. **Collect ground-truth annotations** - Manual labeling of stem datasets
2. **Hybrid architecture** - Separate stem encoder + chord decoder
3. **Multi-task learning** - Joint stem classification + chord recognition

---

## Related Resources

- **BTC Paper:** [ISMIR 2019](https://archives.ismir.net/ismir2019/paper/000019.pdf)
- **MoisesDB Paper:** [arXiv:2307.15913](https://arxiv.org/abs/2307.15913)
- **Isophonics Annotations:** http://isophonics.net/datasets
- **McGill Billboard:** https://ddmal.music.mcgill.ca/research/billboard

---

## Contact / Handoff Notes

The core challenge is that **finetuning on pseudo-labels teaches segment identity, not chord recognition**. Any continuation should focus on:

1. Getting better chord annotations (ground truth or better ACR)
2. Or changing the training objective (contrastive learning)
3. Or obtaining the original BTC training data with real annotations

The infrastructure (data pipelines, training loops, evaluation) is solid. The bottleneck is data quality, not code.
