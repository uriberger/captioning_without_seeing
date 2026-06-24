"""
Standalone Polos scoring script. Run via venv_polos Python interpreter.

Reads a JSON file containing a list of {mt, refs, img_path} dicts,
downloads/loads the Polos checkpoint, scores them, and prints JSON to stdout.

Usage:
    venv_polos/bin/python score_polos.py <input_json>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Polos"))

from PIL import Image
from polos.models import download_model, load_checkpoint


def main():
    if len(sys.argv) != 2:
        print("Usage: score_polos.py <input_json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        records = json.load(f)

    checkpoint = download_model("polos")
    model = load_checkpoint(checkpoint)
    model.eval()

    samples = []
    for rec in records:
        img = Image.open(rec["img_path"]).convert("RGB")
        samples.append({
            "mt": rec["mt"],
            "refs": rec["refs"],
            "img": img,
        })

    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        model.to("cuda")

    _, scores = model.predict(samples, cuda=cuda, show_progress=False)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
