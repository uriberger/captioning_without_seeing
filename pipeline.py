import json
import os
from datetime import datetime
from typing import Optional

from config import Config
from dataset import iter_coco_captions
from oracle import Oracle
from blind_model import BlindModel


def _find_resumable_dir(cfg: Config) -> Optional[str]:
    """Return an existing output dir whose config matches blind_model, oracle_model, and n_queries."""
    if not os.path.isdir(cfg.output_dir):
        return None
    for name in sorted(os.listdir(cfg.output_dir)):
        candidate = os.path.join(cfg.output_dir, name)
        config_path = os.path.join(candidate, "config.json")
        results_path = os.path.join(candidate, "results.jsonl")
        if not os.path.isfile(config_path) or not os.path.isfile(results_path):
            continue
        with open(config_path) as f:
            saved = json.load(f)
        if (
            saved.get("blind_model") == cfg.blind_model
            and saved.get("oracle_model") == cfg.oracle_model
            and saved.get("n_queries") == cfg.n_queries
        ):
            return candidate
    return None


def _load_completed_ids(results_path: str) -> set[str]:
    completed = set()
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                completed.add(json.loads(line)["image_id"])
    return completed


def _make_output_dir(cfg: Config) -> str:
    blind_short = cfg.blind_model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{blind_short}_N{cfg.n_queries}"
    path = os.path.join(cfg.output_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def run(cfg: Config) -> str:
    """
    Run the full experiment. Returns the output directory path.
    Resumes a previous partial run if one exists with matching blind_model, oracle_model, and n_queries.
    """
    existing = _find_resumable_dir(cfg)
    if existing:
        out_dir = existing
        results_path = os.path.join(out_dir, "results.jsonl")
        completed_ids = _load_completed_ids(results_path)
        print(f"Resuming run: {out_dir} ({len(completed_ids)} samples already done)")
        file_mode = "a"
    else:
        out_dir = _make_output_dir(cfg)
        cfg.save(os.path.join(out_dir, "config.json"))
        results_path = os.path.join(out_dir, "results.jsonl")
        completed_ids = set()
        print(f"Output dir: {out_dir}")
        file_mode = "w"

    print(f"Loading oracle: {cfg.oracle_model}")
    oracle = Oracle(
        cfg.oracle_model,
        thinking=cfg.oracle_thinking,
        max_new_tokens=cfg.max_new_tokens_answer,
        max_answer_tokens=cfg.max_answer_tokens,
    )

    print(f"Loading blind model: {cfg.blind_model}")
    blind = BlindModel(
        cfg.blind_model,
        n_queries=cfg.n_queries,
        thinking=cfg.blind_thinking,
        max_new_tokens_question=cfg.max_new_tokens_question,
        max_new_tokens_caption=cfg.max_new_tokens_caption,
    )

    with open(results_path, file_mode) as f:
        for i, sample in enumerate(iter_coco_captions(cfg.split, cfg.max_samples)):
            if sample["image_id"] in completed_ids:
                continue
            print(f"[{i}] image_id={sample['image_id']}")
            record = _process_sample(sample, oracle, blind, cfg.n_queries)
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Done. Results saved to {results_path}")
    return out_dir


def _process_sample(sample: dict, oracle: Oracle, blind: BlindModel, n_queries: int) -> dict:
    image = sample["image"]
    transcript = []

    # Conversation history passed to the blind model (no images, text only)
    history = []

    if n_queries == 0:
        caption_out = blind.generate_caption_zero_shot()
    else:
        for round_idx in range(n_queries):
            # Blind model asks a question
            question_out = blind.ask_question(history, round_idx)
            question_text = question_out["text"]

            # Oracle answers
            answer_out = oracle.answer(image, question_text)
            answer_text = answer_out["text"]

            transcript.append({
                "round": round_idx + 1,
                "question": question_text,
                "question_full": question_out["full"],
                "answer": answer_text,
                "answer_full": answer_out["full"],
            })

            # Extend blind model history
            history.append({"role": "assistant", "content": question_out["full"]})
            history.append({"role": "user", "content": f"Oracle's answer: {answer_text}"})

        caption_out = blind.generate_caption(history)

    return {
        "image_id": sample["image_id"],
        "reference_captions": sample["captions"],
        "n_queries": n_queries,
        "blind_model": blind.model_name,
        "oracle_model": oracle.model_name,
        "transcript": transcript,
        "caption": caption_out["text"],
        "caption_full": caption_out["full"],
    }
