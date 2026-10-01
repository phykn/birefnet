"""Save native-size label IDs, an overlay, and a prediction contact sheet."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._preview import colorize, overlay, save_labels, save_preview
from src.data.image import read_image, read_mask
from src.predict.inference import predict_logits, probabilities
from src.predict.model import load as load_model, load_run_config
from src.prepare.spec import PreprocessSpec


def run(model, image_path: Path, output: Path, tiles=(1,), mask_path=None, class_id=None,
        ignore_index=None) -> dict:
    if class_id is not None and not 0 <= class_id < model.num_classes:
        raise ValueError("class_id must be in [0, num_classes)")
    meta = getattr(model, "loaded_meta", None)
    spec = PreprocessSpec.from_meta(meta)
    if ignore_index is None:
        ignore_index = (meta or {}).get("ignore_index", 255)
    image = read_image(str(image_path))
    panels = [("Input", image)]
    if mask_path is not None:
        target = read_mask(str(mask_path))
        if target.shape != image.shape[:2]:
            raise ValueError("Image and mask dimensions differ")
        panels.append(("Ground truth", colorize(target, model.num_classes, target != ignore_index)))
    logits = predict_logits(model, image, size=spec.size, is_sem=spec.is_sem, tiles=tiles)
    labels = logits.argmax(axis=0).astype(np.uint8)
    colors = colorize(labels, model.num_classes)
    blended = overlay(image, colors)
    panels.extend([("Prediction", colors), ("Overlay", blended)])
    output.mkdir(parents=True, exist_ok=True)
    save_labels(labels, model.num_classes, output / "labels.png")
    Image.fromarray(blended).save(output / "overlay.png")
    save_preview(panels, output / "preview.png")
    if class_id is not None:
        probs = probabilities(logits)
        Image.fromarray(np.rint(probs[class_id] * 255).astype(np.uint8)).save(output / f"probability_{class_id}.png")
    report = {"image": str(image_path), "shape": list(labels.shape), "num_classes": model.num_classes,
              "preprocess": spec.to_meta(), "tiles": list(tiles),
              "class_pixels": np.bincount(labels.ravel(), minlength=model.num_classes).tolist()}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="Optional ground-truth mask for comparison")
    parser.add_argument("--config", type=Path, help="Defaults to the checkpoint's run config, then model defaults")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tiles", type=int, nargs="+", default=[1])
    parser.add_argument("--class-id", type=int, help="Also save this class's probability PNG")
    parser.add_argument("--output", type=Path, default=Path("run/checks/predict"))
    args = parser.parse_args()
    cfg = load_run_config(args.weight, args.config)
    model = load_model(cfg, args.weight, torch.device(args.device))
    report = run(model, args.image, args.output, args.tiles, args.mask, args.class_id)
    print(f"Class pixel counts: {report['class_pixels']}")
    print(f"Labels, overlay, preview and summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
