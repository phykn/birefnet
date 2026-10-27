# BiRefNet Multiclass Segmentation

Fine-tunes [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) for multiclass
semantic segmentation of SEM images. Training imitates operator-dependent
brightness, contrast, focus, and noise so the model learns to ignore them.

- Frozen Swin-L backbone; only the squeeze module and decoder train
- Optional Depth Anything V2 depth fed into every decoder level
- SEM augmentation: zoom, gamma, uneven light, blur, grain, hidden edges
- Soft targets from label blur and an EMA teacher
- Runs on CUDA, Apple GPU (MPS), or CPU

## Setup

Python 3.10+ and PyTorch 2.10+.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the pretrained models into `weight/`:

- [BiRefNet-general-epoch_244.pth](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth)
- Depth Anything V2-Small (skip if `model.depth: false`):

  ```bash
  .venv/bin/hf download depth-anything/Depth-Anything-V2-Small-hf --local-dir weight/Depth-Anything-V2-Small-hf
  ```

## Data

```
data/train/image/0001.jpg
data/train/mask/0001.png
```

Masks are single-channel or palette images holding class IDs `0` to
`model.classes - 1`; `255` is ignored. Every image needs a mask with the same
filename stem. 16-bit images are stretched to 8 bits. The default setup uses the
[NEU-Seg dataset](data/README.md).

## Train

```bash
python run_train.py                     # uses config.yaml
python run_train.py --config my.yaml    # overrides only the keys in my.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

All settings live in [`config.yaml`](config.yaml). Each run writes to
`run/<run-id>/`: its config, the train/valid split, TensorBoard logs, and
`weights/best_miou.pth` (EMA weights with the best validation mIoU).

GPU training runs in mixed precision. If memory runs short, set
`train.grad_checkpoint: true`: the decoder recomputes its activations during
backward, using about a quarter of the memory for roughly 20% more time per
step. The freed memory can go to a larger `loader.batch_size` with a smaller
`train.accum`.

## Predict

```bash
python scripts/03_predict.py --weight run/<run-id>/weights/best_miou.pth --image path/to/image.jpg
```

Add `--mask` to show the ground truth or `--tiles 1 3` for tiled prediction.
Training and prediction run on CUDA if present, otherwise on an Apple GPU,
otherwise on the CPU. `scripts/01_dataset.py` and `scripts/02_augment.py` show
raw and augmented training samples.

## Test

```bash
python -m pytest -q
```
