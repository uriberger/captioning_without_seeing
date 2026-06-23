from typing import Iterator, Optional
from datasets import load_dataset
from PIL import Image


def iter_crossmodal3600(
    locale: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[dict]:
    """
    Yields dicts with keys:
      image_id    : str
      locale      : str
      image       : PIL.Image
      captions    : list[str]  (reference captions for evaluation)
    """
    if locale is not None:
        configs = [locale]
    else:
        # CrossModal3600 has one config per language/locale
        configs = _all_locales()

    count = 0
    for loc in configs:
        ds = load_dataset("google/crossmodal-3600", loc, split="test", trust_remote_code=True)
        for sample in ds:
            yield {
                "image_id": str(sample.get("image/key", sample.get("id", count))),
                "locale": loc,
                "image": sample["image"].convert("RGB"),
                "captions": _extract_captions(sample, loc),
            }
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def _extract_captions(sample: dict, locale: str) -> list:
    # Field naming in CrossModal3600 HuggingFace version
    for key in (f"caption/{locale}", "caption/en", "captions", "caption"):
        if key in sample:
            val = sample[key]
            if isinstance(val, list):
                return val
            return [val]
    return []


def _all_locales() -> list:
    return [
        "ar", "bn", "cs", "da", "de", "el", "en", "es", "fa", "fi",
        "fil", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko",
        "mi", "nl", "no", "pl", "pt", "quz", "ro", "ru", "sv", "sw",
        "te", "th", "tr", "uk", "vi", "zh",
    ]
