"""
Evaluate all experiment outputs using Polos and CLIPScore.

For each output directory under outputs/, loads results.jsonl, fetches the
corresponding COCO images, computes per-sample Polos and CLIPScore, and
prints a summary table grouped by (blind_model, n_queries).

Polos runs in a subprocess using venv_polos (which has the compatible
pytorch-lightning 1.3 / fairseq environment). CLIPScore runs in the main
process using the standard venv.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torchmetrics.multimodal import CLIPScore

from dataset import iter_coco_captions


OUTPUTS_DIR = "outputs"
POLOS_PYTHON = str(Path(__file__).parent / "venv_polos" / "bin" / "python")
SCORE_POLOS_SCRIPT = str(Path(__file__).parent / "score_polos.py")


def _load_experiment(out_dir: str) -> tuple[dict, list[dict]]:
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    records = []
    with open(os.path.join(out_dir, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return cfg, records


def _discover_experiments(outputs_dir: str) -> list[str]:
    if not os.path.isdir(outputs_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(outputs_dir)):
        candidate = os.path.join(outputs_dir, name)
        if (
            os.path.isdir(candidate)
            and os.path.isfile(os.path.join(candidate, "config.json"))
            and os.path.isfile(os.path.join(candidate, "results.jsonl"))
        ):
            dirs.append(candidate)
    return dirs


def _build_image_lookup(image_ids: set, split: str) -> dict:
    """Stream COCO and return {image_id: PIL.Image} for all requested ids."""
    lookup = {}
    for sample in iter_coco_captions(split, max_samples=None):
        if sample["image_id"] in image_ids:
            lookup[sample["image_id"]] = sample["image"]
        if len(lookup) == len(image_ids):
            break
    return lookup


def _compute_clipscore(records: list, images: dict, device: str) -> list:
    import torchvision.transforms.functional as TF
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    metric.eval()
    scores = []
    with torch.no_grad():
        for rec in records:
            img = images[rec["image_id"]]
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
            img_uint8 = (img_tensor * 255).to(torch.uint8)
            score = metric(img_uint8, [rec["caption"]])
            scores.append(score.item())
    return scores


def _compute_polos(records: list, images: dict) -> list:
    """Run score_polos.py in venv_polos via subprocess. Images are saved to
    a temp dir and passed as file paths."""
    with tempfile.TemporaryDirectory() as tmp:
        # Save images to temp dir
        input_records = []
        for i, rec in enumerate(records):
            img_path = os.path.join(tmp, f"{i}.png")
            images[rec["image_id"]].save(img_path)
            input_records.append({
                "mt": rec["caption"],
                "refs": rec["reference_captions"],
                "img_path": img_path,
            })

        input_json = os.path.join(tmp, "input.json")
        with open(input_json, "w") as f:
            json.dump(input_records, f)

        result = subprocess.run(
            [POLOS_PYTHON, SCORE_POLOS_SCRIPT, input_json],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            print("Polos subprocess error:", file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            raise RuntimeError("Polos scoring failed")

        return json.loads(result.stdout.strip())


def _load_cached_scores(exp_dir: str, expected_count: int) -> tuple[list, list] | None:
    """Return (clip_scores, polos_scores) from eval_scores.jsonl if complete, else None."""
    scores_path = os.path.join(exp_dir, "eval_scores.jsonl")
    if not os.path.isfile(scores_path):
        return None
    rows = []
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != expected_count:
        return None
    clip_scores = [r["clip_score"] for r in rows]
    polos_scores = [r["polos"] for r in rows]
    return clip_scores, polos_scores


def main():
    exp_dirs = _discover_experiments(OUTPUTS_DIR)
    if not exp_dirs:
        print(f"No experiment directories found under '{OUTPUTS_DIR}/'.")
        return

    print(f"Found {len(exp_dirs)} experiment(s).\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect all records; determine which experiments need scoring
    all_experiments = []
    needs_scoring_image_ids = set()
    for exp_dir in exp_dirs:
        cfg, records = _load_experiment(exp_dir)
        cached = _load_cached_scores(exp_dir, len(records))
        all_experiments.append((exp_dir, cfg, records, cached))
        if cached is None:
            for rec in records:
                needs_scoring_image_ids.add(rec["image_id"])

    n_cached = sum(1 for _, _, _, c in all_experiments if c is not None)
    n_fresh = len(all_experiments) - n_cached
    print(f"Cached: {n_cached}  |  To evaluate: {n_fresh}\n")

    images = {}
    if needs_scoring_image_ids:
        split = all_experiments[0][1].get("split", "validation")
        print(f"Loading {len(needs_scoring_image_ids)} images from COCO {split} split...")
        images = _build_image_lookup(needs_scoring_image_ids, split)
        print(f"Loaded {len(images)} images.\n")

    results_summary = []
    for exp_dir, cfg, records, cached in all_experiments:
        blind = cfg.get("blind_model", "?").split("/")[-1]
        oracle = cfg.get("oracle_model", "?").split("/")[-1]
        n = cfg.get("n_queries", "?")
        n_samples = len(records)
        scores_path = os.path.join(exp_dir, "eval_scores.jsonl")

        if cached is not None:
            clip_scores, polos_scores = cached
            print(f"Cached:    {os.path.basename(exp_dir)}  ({n_samples} samples)")
        else:
            print(f"Evaluating: {os.path.basename(exp_dir)}  ({n_samples} samples)")

            clip_scores = _compute_clipscore(records, images, device)
            print(f"  CLIPScore done.")

            polos_scores = _compute_polos(records, images)
            print(f"  Polos done.")

            with open(scores_path, "w") as f:
                for rec, cs, ps in zip(records, clip_scores, polos_scores):
                    f.write(json.dumps({
                        "image_id": rec["image_id"],
                        "clip_score": cs,
                        "polos": ps,
                    }) + "\n")
            print(f"  Per-sample scores → {scores_path}")

        avg_clip = sum(clip_scores) / len(clip_scores)
        avg_polos = sum(polos_scores) / len(polos_scores)
        print(f"  CLIPScore: {avg_clip:.4f}  |  Polos: {avg_polos:.4f}\n")

        results_summary.append({
            "dir": os.path.basename(exp_dir),
            "blind_model": blind,
            "oracle_model": oracle,
            "n_queries": n,
            "n_samples": n_samples,
            "clip_score": avg_clip,
            "polos": avg_polos,
        })

    # Summary table
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Blind model':<22} {'Oracle':<22} {'N':>4} {'Samples':>8} {'CLIPScore':>10} {'Polos':>8}")
    print("-" * 72)
    for r in sorted(results_summary, key=lambda x: (x["blind_model"], x["n_queries"])):
        print(
            f"{r['blind_model']:<22} {r['oracle_model']:<22} {r['n_queries']:>4} "
            f"{r['n_samples']:>8} {r['clip_score']:>10.4f} {r['polos']:>8.4f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
