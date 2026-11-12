# NEU-Seg steel surface defects

The example dataset for this project. It contains hot-rolled steel strip images with three defect classes, taken from the [DHW-Master/NEU_Seg](https://github.com/DHW-Master/NEU_Seg) copy of NEU-Seg.

The source repository states no license, so the image and mask folders are git-ignored. Only this README is committed.

## Layout

```text
data/
├── train/image/   3630 JPEG, 200 × 200 (grayscale content, RGB files)
├── train/mask/    3630 palette PNG
├── test/image/     840 JPEG, 200 × 200
└── test/mask/      840 palette PNG
```

- Images and masks pair by filename stem, for example `000201.jpg` ↔ `000201.png`.
- The default [`config.yaml`](../config.yaml) trains on `data/train/` and holds out `data.valid_ratio` (10%) of it for validation.
- `data/test/` is never used in training. Score it with `scripts/04_evaluate.py`.

## Classes

Masks are palette (mode `P`) PNGs, and each pixel value is a class ID.

| ID | Class | Palette color | Share of training pixels |
| --- | --- | --- | --- |
| 0 | Background | Black | 87.9% |
| 1 | Inclusion | White | 2.1% |
| 2 | Patches | Blue | 7.4% |
| 3 | Scratches | Red | 2.6% |

Read masks with Pillow and do not convert them to RGB, so the class IDs stay intact.

## Rebuild

1. Clone [DHW-Master/NEU_Seg](https://github.com/DHW-Master/NEU_Seg).
2. Move `images/training` to `train/image/` and `annotations/training` to `train/mask/`.
3. Move `images/test` to `test/image/` and `annotations/test` to `test/mask/`.
4. Save each mask as a palette PNG with the colors in the table above.
