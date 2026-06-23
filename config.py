from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Config:
    # Models
    oracle_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    blind_model: str = "Qwen/Qwen3-8B"
    oracle_thinking: bool = True
    blind_thinking: bool = False

    # Experiment
    n_queries: int = 5

    # Dataset
    dataset_name: str = "Multimodal-Fatima/COCO_captions_validation"
    split: str = "validation"
    max_samples: Optional[int] = None

    # Output
    output_dir: str = "outputs"

    # Generation limits (oracle answer tokens larger to accommodate thinking chain)
    max_new_tokens_question: int = 1024
    max_new_tokens_answer: int = 2048
    max_new_tokens_caption: int = 512

    # Max tokens in the oracle's visible answer (prompt-enforced; M in the wiki)
    max_answer_tokens: int = 10

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
