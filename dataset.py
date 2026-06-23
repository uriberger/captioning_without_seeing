from typing import Iterator, Optional
from datasets import load_dataset


# Multimodal-Fatima repos have all 5 reference captions grouped per image.
_SPLIT_TO_REPO = {
    "validation": "Multimodal-Fatima/COCO_captions_validation",
    "train": "Multimodal-Fatima/COCO_captions_train",
    "test": "Multimodal-Fatima/COCO_captions_test",
}


def iter_coco_captions(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> Iterator[dict]:
    """
    Yields dicts with keys:
      image_id  : str
      image     : PIL.Image
      captions  : list[str]  (5 reference captions for evaluation)
    """
    repo = _SPLIT_TO_REPO[split]
    ds = load_dataset(repo, split=split, streaming=True)
    for count, sample in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            return
        yield {
            "image_id": str(sample["cocoid"]),
            "image": sample["image"].convert("RGB"),
            "captions": sample["sentences_raw"],
        }
