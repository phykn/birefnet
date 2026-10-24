# NEU-Seg steel surface defects

Hot-rolled steel strip images with three defect classes, from the
[DHW-Master/NEU_Seg](https://github.com/DHW-Master/NEU_Seg) copy of NEU-Seg.
The source repository states no license, so the image and mask folders are
git-ignored; only this README is committed.

```
data/
├── train/image/  3630 x 200x200 JPEG (grayscale content, RGB files)
├── train/mask/   3630 palette PNGs
├── test/image/    840 x 200x200 JPEG
└── test/mask/     840 palette PNGs
```

Images and masks pair by filename stem (`000201.jpg` ↔ `000201.png`).
`config.yaml` trains on `data/train/` and splits off `valid_ratio` for
validation; `data/test/` is held out.

Masks are palette (mode `P`) PNGs whose pixel values are class IDs:

| ID | Meaning | Palette color |
|---|---|---|
| 0 | Background | Black |
| 1 | Inclusion | White |
| 2 | Patches | Blue |
| 3 | Scratches | Red |

Read masks with Pillow without RGB conversion to keep the class IDs.
Background covers about 88% of training pixels.

To rebuild: clone the source repository, move `images/training`,
`annotations/training`, `images/test` and `annotations/test` to the folders
above, and save each mask as a palette PNG with the colors in the table.
