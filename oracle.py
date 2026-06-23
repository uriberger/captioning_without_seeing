import re
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


SYSTEM_PROMPT_TEMPLATE = (
    "You are a vision assistant. Answer the question about the provided image "
    "accurately and concisely. Answer only what was explicitly asked — do not "
    "volunteer any additional information. Your answer must be at most {max_answer_tokens} tokens."
)


class Oracle:
    def __init__(self, model_name: str, thinking: bool = True, max_new_tokens: int = 2048, max_answer_tokens: int = 10):
        self.model_name = model_name
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self.max_answer_tokens = max_answer_tokens

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def answer(self, image: Image.Image, question: str) -> dict:
        """
        Returns dict with:
          "text":  the answer with thinking stripped (used as blind model input)
          "full":  the raw model output including <think> block (saved in transcript)
        """
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_answer_tokens=self.max_answer_tokens)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        # qwen_vl_utils extracts image/video tensors from the message content
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        full_text = self.processor.decode(generated[0], skip_special_tokens=True)

        return {
            "full": full_text,
            "text": _strip_thinking(full_text),
        }
