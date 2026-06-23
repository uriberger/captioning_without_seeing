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

### Metric: Polos

The primary metric is **Polos**, a CLIP-based reference-free captioning metric. It is well-suited for this project because:
- It does not require many reference captions per image (unlike CIDEr/BLEU).
- It handles culturally diverse content better than n-gram metrics.
- It measures alignment with the actual image, not just lexical overlap with references.

Standard n-gram metrics (BLEU, METEOR, CIDEr) may be used as secondary metrics for comparability with prior work.

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
