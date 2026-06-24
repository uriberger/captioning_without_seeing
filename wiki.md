# Captioning Without Seeing — Project Wiki

## Overview

A research project exploring whether a language model can produce high-quality image captions **without ever seeing the image**, by strategically querying an oracle that can see it.

The core question: how much of the captioning task is visual perception, and how much is reasoning and language?

---

## Setup

### Roles

- **Blind model**: The model under evaluation (or training). It cannot see the image. Its goal is to produce a caption.
- **Oracle**: A strong VLM (e.g., GPT-4V, LLaVA) that can see the image. It answers the blind model's questions in free-form text.

### Protocol

1. The blind model is told it must produce a caption of an image it cannot see.
2. It may send up to **N queries** to the oracle. Each query must be a single question (compound questions like "How many dogs and how many cats are there?" are disallowed).
3. After exhausting its queries (or choosing to stop), the blind model produces a final caption.

### The N Parameter

N is the central experimental variable:

- **N = 0**: Pure prior-based captioning — a hallucination baseline. Useful as a lower bound.
- **N = small (1–5)**: The model must be highly strategic about what to ask.
- **N = large**: Approaches having full image information; useful as an upper bound.

The **N-vs-quality curve** is expected to be a central empirical result of the project.

---

## Research Questions

1. **How does caption quality scale with N?** Is there a point of diminishing returns?
2. **Do VLMs outperform LLMs as the blind model?** Two competing hypotheses:
   - *VLMs may be better*: richer learned associations between visual concepts and language may lead to more informative questions.
   - *LLMs may be better*: stronger general reasoning may lead to better integration of oracle answers into a coherent caption.
3. **What questioning strategies emerge?** Does the blind model learn to ask about objects first, then attributes, then relations? Does strategy differ between VLMs and LLMs?
4. **Does performance vary by cultural context?** (Relevant if/when switching to CrossModal3600 — see Dataset section.)

---

## Evaluation

### Metrics

**Polos** is the primary metric — a learned, CLIP-based reference-free captioning metric (CVPR 2024). It is well-suited for this project because:
- It does not require many reference captions per image (unlike CIDEr/BLEU).
- It handles culturally diverse content better than n-gram metrics.
- It measures alignment with the actual image, not just lexical overlap with references.

**CLIPScore** is used as a secondary metric. It measures cosine similarity between image and caption embeddings in CLIP space (ViT-B/16), with no reference captions needed. It provides a simpler, interpretable baseline for image-text alignment.

Standard n-gram metrics (BLEU, METEOR, CIDEr) may be used as additional secondary metrics for comparability with prior work.

### Running Evaluation

```bash
source venv/bin/activate
python evaluate.py
```

This discovers all directories under `outputs/`, computes Polos and CLIPScore for each, prints a summary table, and writes per-sample scores to `eval_scores.jsonl` inside each output directory.

### Environment Architecture

Polos has deep dependency conflicts with the main project environment (it requires `pytorch-lightning==1.3.8` and `fairseq==0.12.2`, both of which are incompatible with modern torch/torchmetrics). The solution is two separate virtual environments:

- **`venv/`** — main environment for running experiments and computing CLIPScore.
- **`venv_polos/`** — isolated environment for Polos only. Called as a subprocess by `evaluate.py` via `score_polos.py`.

### Environment Setup from Scratch

#### Prerequisites

- Python 3.9 (the project path is `/cm/local/apps/python3/bin/python3`)
- The `Polos` repo must be cloned at the project root: `git clone https://github.com/YuigaWada/Polos.git`

#### 1. Main environment (`venv/`)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "torchmetrics[multimodal]"
```

`requirements.txt` covers torch, transformers, datasets, Pillow, qwen-vl-utils, and accelerate. The `torchmetrics[multimodal]` line adds CLIPScore support.

#### 2. Polos environment (`venv_polos/`)

This environment requires careful pinning due to cascading compatibility issues across pytorch-lightning, fairseq, and their transitive dependencies.

```bash
python3 -m venv venv_polos
source venv_polos/bin/activate

# PyTorch — CPU build is sufficient (Polos inference is fast)
pip install "torch==2.1.2" "torchvision==0.16.2" \
    --index-url https://download.pytorch.org/whl/cpu

# Old pytorch-lightning stack with exact compatible versions
pip install \
    "pytorch-lightning==1.3.8" \
    "PyYAML==5.3.1" \
    "pyDeprecate==0.3.0" \
    "torchmetrics==0.5.1" \
    "tensorboard" \
    "numpy<2" \
    "tqdm==4.65.0"

# fairseq and its transitive dependencies
pip install \
    "omegaconf==2.0.6" \
    "hydra-core==1.0.7" \
    "antlr4-python3-runtime==4.8" \
    "sacrebleu<2.0" \
    "portalocker" \
    "bitarray"
pip install --no-deps "fairseq==0.12.2"

# Remaining Polos dependencies
pip install \
    "transformers==4.30.0" \
    "pytorch-nlp==0.5.0" \
    "click" \
    "pandas" \
    "scipy" \
    "sentencepiece" \
    "ftfy"

# Install Polos itself (no deps — all handled above)
pip install --no-deps -e Polos/
```

#### 3. Patches required in `venv_polos/`

Three files must be patched after installation. These patches are already applied to the current environment; only needed when rebuilding from scratch.

**Patch A — `venv_polos/lib/python3.9/site-packages/pytorch_lightning/metrics/__init__.py`**

Replace the entire file with an empty stub. The `metrics` shim in pytorch-lightning 1.3.8 imports old torchmetrics APIs that are broken with numpy 2 and modern tqdm. Polos only uses `LightningModule.load_from_checkpoint`, not these metric classes.

```python
# Stubbed out — original imports are incompatible with modern torchmetrics/numpy.
# Polos only needs LightningModule.load_from_checkpoint, not these metric classes.
```

**Patch B — `venv_polos/lib/python3.9/site-packages/torchmetrics/functional/text/bert.py`**

Add `import tqdm.auto` after `import tqdm` (around line 31). Without this, the `tqdm.auto.tqdm` type annotation on the `_get_progress_bar` function causes an `AttributeError` at import time.

```python
if _TQDM_AVAILABLE:
    import tqdm
    import tqdm.auto    # ← add this line
```

**Patch C — `Polos/polos/models/model_base.py`**

In `ModelBase.__init__` (around line 154), change `self.hparams = ...` to `self._hparams = ...`. In PyTorch 2.x, `hparams` is a read-only property on `nn.Module` and cannot be set via direct attribute assignment.

```python
# Before:
self.hparams = Namespace(**hparams)
# ...
self.hparams = hparams

# After:
self._hparams = Namespace(**hparams)
# ...
self._hparams = hparams
```

**Patch D — `Polos/polos/models/__init__.py`**

Add `strict=False` to both `load_from_checkpoint` calls. The saved Polos checkpoint predates a `position_ids` buffer added in newer transformers versions; without `strict=False`, loading fails with a missing-key error.

```python
model = str2model[...].load_from_checkpoint(
    checkpoint, hparams=hparams, strict=False    # ← add strict=False
)
```

---

## Dataset

### Current: COCO Captions

**COCO Captions** (2014) is the standard benchmark dataset for image captioning. Each of the 5,000 validation images has 5 human-written reference captions, making it well-suited for evaluation with standard metrics.

- Loaded via `Multimodal-Fatima/COCO_captions_{split}` on HuggingFace, which provides all 5 reference captions grouped per image.
- Streaming mode is used — no full download required.

### Future Candidate: CrossModal3600

**CrossModal3600** remains a longer-term candidate for its cultural diversity angle (36 locales). It was removed from HuggingFace; the original source is the Google Research Datasets repo. Reasons it is still worth pursuing:
- Performance variation by locale is an interesting analysis axis.
- Content diversity makes the N=0 baseline weaker, making oracle queries more clearly necessary.
- Pairs well with Polos (reference-free evaluation handles cross-cultural content better).

---

## Query Constraints

To prevent the blind model from extracting too much information per query:
- Each query must be a **single, atomic question**.
- Compound questions (e.g., "What color is the car and how many people are there?") are disallowed.
- Enforcement approach: **TBD** — options include prompt-level instruction only, or an additional validation step that detects and rejects compound questions.

---

## Future Directions

### Training the Blind Model

A potential second phase: train the blind model to be a better questioner. Possible approach:
- **Reinforcement learning**: reward signal derived from final caption quality (e.g., Polos score).
- The model learns a policy for generating maximally informative questions given its current state of knowledge about the image.

This direction is under consideration but not yet planned.

---

## Open Questions

- Which specific models to use as the blind model (LLM candidates: GPT-4, LLaMA; VLM candidates: GPT-4V, LLaVA, etc.)?
- Which model to use as the oracle?
- Final dataset choice.
- Whether to enforce single-question constraint via prompting alone or with a validator.
- Whether to pursue the RL training direction.
