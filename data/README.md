# Synthetic sample data

Four generated image/mask pairs for pipeline checks, **not real observations**.
Each `image/sample_XX.png` (grayscale, 256x256) pairs with the same filename in
`mask/` (PNG palette mode P). Pixel values remain integer class IDs:

| ID | Meaning | Palette color |
|---|---|---|
| 0 | Background | Dark navy |
| 1 | Filled shapes | Gray |
| 2 | Branching lines | Coral red |
| 3 | Interior holes | Cyan |

These meanings describe the synthetic fixtures only. Define class IDs for your own dataset.

Use Pillow without RGB conversion to read label IDs. `preview.png` shows each
image beside its colored mask. The deterministic generator is
`scripts/make_sample_data.py`; it refuses to overwrite existing pairs.
These samples verify loading/training mechanics only, not segmentation accuracy.
