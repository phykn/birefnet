# Synthetic sample data

Four generated SEM-like fixtures, **not real battery measurements**.
Each `image/sample_XX.png` (grayscale, 256x256) pairs with the same filename in
`mask/` (PNG palette mode P). Pixel values remain integer class IDs:

| ID | Meaning | Palette color |
|---|---|---|
| 0 | Inter-particle pore | Dark navy |
| 1 | Solid particle | Gray |
| 2 | Internal crack | Coral red |
| 3 | Internal pore | Cyan |

Use Pillow without RGB conversion to read label IDs. `preview.png` shows each
image beside its colored mask. The deterministic generator is
`scripts/make_sample_data.py`; it refuses to overwrite existing pairs.
These samples verify loading/training mechanics only, not segmentation accuracy.
