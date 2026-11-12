<div align="center">

# BiRefNet for SEM Segmentation

**Fine-tune BiRefNet to segment multiple classes in scanning electron microscope images.**

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.10%2B-EE4C2C?logo=pytorch&logoColor=white)
![Devices](https://img.shields.io/badge/device-CUDA%20%7C%20Apple%20MPS%20%7C%20CPU-555555)

</div>

## Highlights

- Pretrained BiRefNet with a frozen Swin-L backbone. Only the squeeze module and decoder train.
- Grayscale input and SEM-style augmentations: gamma, uneven illumination, blur, noise.
- Learns from sparse brush labels and imperfect AI masks together.
- Trains at 1024 × 1024 and predicts large images with overlapping tiles.

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/phykn/birefnet.git
cd birefnet
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the [BiRefNet v1 checkpoint](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth) to `weight/`. Put image and mask pairs in `data/train/image/` and `data/train/mask/`. They pair by filename stem. Masks hold class IDs, and `255` marks ignored pixels. See [the example dataset](data/README.md).

```bash
.venv/bin/python scripts/01_dataset.py   # preview data
.venv/bin/python scripts/02_augment.py   # preview augmentations
.venv/bin/python run_train.py
```

## Training

Defaults live in [`config.yaml`](config.yaml). Override only what you need:

```bash
.venv/bin/python run_train.py --config my_overrides.yaml
.venv/bin/python run_train.py --resume run/RUN_ID/weights/last.train.pth
.venv/bin/tensorboard --logdir run
```

Each run saves its config, data split, logs, and checkpoints to `run/RUN_ID/`. Use `best_miou.pth` for prediction and `last.train.pth` to resume.

## Predict and evaluate

```bash
.venv/bin/python scripts/03_predict.py --weight run/RUN_ID/weights/best_miou.pth \
  --image data/test/image/000001.jpg --tiles 1 3
.venv/bin/python scripts/04_evaluate.py --weight run/RUN_ID/weights/best_miou.pth \
  --images data/test/image --masks data/test/mask
```

`--tiles 1 3` averages the whole image with a 3 × 3 tile grid. Evaluation reports mIoU, per-class IoU, and pixel accuracy.

## Brush labels and AI masks

If people painted a few strokes and an AI filled in the rest, add a label folder. Painted pixels are trusted, and the AI mask is a weaker target.

```text
DATA/image/0001.png
DATA/mask/0001.png    # AI mask
DATA/label/0001.png   # brush strokes, 255 elsewhere
```

Set `data.label_dir` to that folder. `scripts/00_synthetic.py` makes synthetic data with known ground truth for tuning `loss.label_weight` and `loss.mask_weight`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## References

1. P. Zheng, D. Gao, D.-P. Fan, L. Liu, J. Laaksonen, W. Ouyang, and N. Sebe, "Bilateral Reference for High-Resolution Dichotomous Image Segmentation," *CAAI Artificial Intelligence Research*, 2024. [[code]](https://github.com/ZhengPeng7/BiRefNet)
2. Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in *Proc. IEEE/CVF ICCV*, 2021. [[code]](https://github.com/microsoft/Swin-Transformer)
3. H. Dong, K. Song, Y. He, J. Xu, Y. Yan, and Q. Meng, "PGA-Net: Pyramid Feature Fusion and Global Context Attention Network for Automated Surface Defect Detection," *IEEE Trans. Industrial Informatics*, 2020. (NEU-Seg dataset) [[data]](https://github.com/DHW-Master/NEU_Seg)
