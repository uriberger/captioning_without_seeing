import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _build_system_prompt(n_queries: int) -> str:
    return (
        f"You are tasked with writing a caption for an image you cannot see. "
        f"You may ask an oracle, who can see the image, exactly {n_queries} question(s). "
        f"Each question must be a single, atomic question — do not combine multiple questions "
        f"into one (e.g. 'What color is the car and how many people are there?' is not allowed). "
        f"After all questions are answered, you will write the final image caption."
    )


class BlindModel:
    def __init__(
        self,
        model_name: str,
        n_queries: int,
        thinking: bool = False,
        max_new_tokens_question: int = 1024,
        max_new_tokens_caption: int = 512,
    ):
        self.model_name = model_name
        self.n_queries = n_queries
        self.thinking = thinking
        self.max_new_tokens_question = max_new_tokens_question
        self.max_new_tokens_caption = max_new_tokens_caption

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self._system_prompt = _build_system_prompt(n_queries)

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(self, messages: list, max_new_tokens: int) -> dict:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        return {
            "full": full_text,
            "text": _strip_thinking(full_text),
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask_question(self, history: list, round_index: int) -> dict:
        """
        Generate the question for round `round_index` (0-based).
        `history` is the conversation so far (list of {"role", "content"} dicts).
        Returns {"text": ..., "full": ...}.
        """
        messages = [{"role": "system", "content": self._system_prompt}] + history
        messages.append({
            "role": "user",
            "content": f"Ask question {round_index + 1}/{self.n_queries}:",
        })
        return self._generate(messages, self.max_new_tokens_question)

    def generate_caption(self, history: list) -> dict:
        """
        Generate the final caption after all N query rounds.
        Returns {"text": ..., "full": ...}.
        """
        messages = [{"role": "system", "content": self._system_prompt}] + history
        messages.append({
            "role": "user",
            "content": "You have used all your questions. Now write the final image caption:",
        })
        return self._generate(messages, self.max_new_tokens_caption)

    def generate_caption_zero_shot(self) -> dict:
        """Caption with no oracle queries (N=0 baseline)."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": "Write an image caption. You have no information about the image:"},
        ]
        return self._generate(messages, self.max_new_tokens_caption)
