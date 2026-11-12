import importlib

import numpy as np
from PIL import Image

generate = importlib.import_module("scripts.00_synthetic").generate


def read(path):
    with Image.open(path) as image:
        return np.asarray(image)


def test_synthetic_samples(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    generate(first, seed=3, train=4, test=1, size=256)
    generate(second, seed=3, train=4, test=1, size=256)
    files = sorted(p.relative_to(first) for p in first.rglob("*.png"))
    assert len(files) == 4 * 5
    assert all((first / f).read_bytes() == (second / f).read_bytes() for f in files)

    label_wrong = mask_wrong = 0
    for split, count in (("train", 4), ("test", 1)):
        root = first / split
        names = {
            n: sorted(p.name for p in (root / n).iterdir())
            for n in ("image", "mask", "label", "truth")
        }
        assert all(len(v) == count and v == names["image"] for v in names.values())
        for name in names["image"]:
            image, mask, label, truth = (
                read(root / n / name) for n in ("image", "mask", "label", "truth")
            )
            assert image.shape == mask.shape == label.shape == truth.shape == (256, 256)
            assert not (truth == 255).any()
            painted = label != 255
            assert painted.mean() >= 0.005
            assert (mask[painted] == label[painted]).all()
            label_wrong += np.count_nonzero(label[painted] != truth[painted])
            mask_wrong += np.count_nonzero(mask[~painted] != truth[~painted])
    assert label_wrong > 0
    assert mask_wrong > 0
