# BiRefNet for SEM Segmentation

Fine-tune BiRefNet to label multiple classes in scanning electron microscopy (SEM) images. The pipeline uses grayscale preprocessing and SEM-oriented augmentations for contrast, uneven illumination, blur, and noise. It freezes the Swin-L backbone and trains the squeeze module and decoder.

Train, preview predictions, and evaluate on CUDA, Apple Silicon (MPS), or CPU. The default training crop is 1024 × 1024. Prediction supports overlapping tiles for large images.

## Quick start

Use Python 3.10+ and uv. The requirements include PyTorch 2.10+. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/phykn/birefnet.git
cd birefnet
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the base checkpoint and place it at `weight/BiRefNet-general-epoch_244.pth`. [BiRefNet v1 checkpoint](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth)

Put your image and mask pairs in the following directories. Use matching filename stems and single-channel or palette masks with class IDs from `0` to `model.classes - 1`. Pixels with value `255` are ignored. The loader scales each 16-bit image to 8-bit using its own intensity range. [NEU-Seg setup and label IDs](data/README.md)

```text
data/train/image/0001.jpg
data/train/mask/0001.png
```

```bash
.venv/bin/python scripts/01_dataset.py   # preview image and mask pairs
.venv/bin/python scripts/02_augment.py   # preview training augmentations
.venv/bin/python run_train.py
```

## Configure and resume training

Edit `config.yaml`, or pass `--config path/to/overrides.yaml` with just the settings you want to override. `train.crop` accepts multiples of 32 from 64 to 1024. Prediction uses a 1024 × 1024 model input and restores the result to the original image size. [Default settings](config.yaml)

Relative config, dataset, base-weight, and training-run paths resolve from the project root, even when launched from another directory.

Each run saves its resolved config, training and validation split, TensorBoard logs, and checkpoints under `run/RUN_ID/`. Replace `RUN_ID` with your run directory name in the commands below.

```bash
.venv/bin/tensorboard --logdir run
.venv/bin/python run_train.py --resume run/RUN_ID/weights/last.train.pth
```

`best_miou.pth` holds the EMA weights with the best validation mIoU. Use `last.train.pth` to resume training. Use the earlier implementation to load a v2 checkpoint. [Code at `750d7d9`](https://github.com/phykn/birefnet/tree/750d7d93456d25efeda6b7728b94f0ee1b442829)

## Predict and evaluate

```bash
.venv/bin/python scripts/03_predict.py \
  --weight run/RUN_ID/weights/best_miou.pth \
  --image data/test/image/0001.jpg --tiles 1 3

.venv/bin/python scripts/04_evaluate.py \
  --weight run/RUN_ID/weights/best_miou.pth \
  --images data/test/image --masks data/test/mask
```

Use a filename from your dataset in the prediction command. The preview shows the predicted class map and overlay. Add `--mask PATH` to compare a ground-truth mask or `--class-id 1` to see a class probability map. Evaluation reports mean IoU, per-class IoU, and pixel accuracy. Both commands accept `--device cpu` to select the CPU explicitly.

## Train with brush labels and AI masks

Use this mode when people painted only small parts of each image and an AI filled in a full mask from those strokes. Training trusts the painted pixels strongly and the AI mask weakly, and treats both as possibly wrong. Every image needs all three files, matched by filename stem. Both `mask/` and `label/` hold class IDs; `255` in `label/` means unpainted.

```text
DATA/image/0001.png
DATA/mask/0001.png    # AI mask, every pixel filled
DATA/label/0001.png   # brush strokes, 255 elsewhere
```

| Config key | Default | Meaning |
| --- | --- | --- |
| `data.label_dir` | `""` | Folder of brush labels. Empty turns this mode off. |
| `loss.label_weight` | `1.0` | Weight of cross-entropy on painted pixels, averaged over painted pixels only |
| `loss.mask_weight` | `3.0` | Weight of cross-entropy and Dice on the AI mask, away from painted pixels |
| `loss.label_smooth` | `0.05` | Label smoothing on painted pixels, so one wrong stroke cannot force full confidence |

The defaults scored best against the true answer on synthetic data: `mask_weight: 3.0` beat 0.3 and 1.0. The label term is averaged over painted pixels only, so each painted pixel still counts for far more than each mask pixel. If your AI masks are less accurate than the synthetic ones, raise the error rates in `scripts/00_synthetic.py` and compare weights again.

Validation then reports `label_miou` on painted pixels and `mask_miou` against the AI mask; `best_miou.pth` is chosen by `label_miou`. Painted pixels usually sit where the AI mask is already right, so `label_miou` checks that trusted pixels are kept, not how well mask errors were fixed.

### Choose settings on synthetic data

Synthetic samples come with the true answer in `truth/`, so they can show which settings fix mask errors. The error rates and sizes are constants at the top of `scripts/00_synthetic.py`; adjust them to resemble your real AI masks and strokes.

```bash
.venv/bin/python scripts/00_synthetic.py   # data/synthetic/{train,test}/{image,mask,label,truth}
.venv/bin/python scripts/02_augment.py --config experiments/syn_trial.yaml
.venv/bin/python run_train.py --config experiments/syn_trial.yaml
```

`experiments/syn_trial.yaml` is a local 20-step config with `model.classes: 5`, the three `data/synthetic/train` folders, and `train.warmup: 0`. Score a run three ways by changing `--masks`:

```bash
for target in truth label mask; do
  .venv/bin/python scripts/04_evaluate.py \
    --weight run/RUN_ID/weights/best_miou.pth \
    --images data/synthetic/test/image --masks data/synthetic/test/$target
done
```

### Train on your own data

Create two override configs with your folders and class count. Train both with the same number of steps, then compare the best `valid/label_miou` in TensorBoard. The labeled run should not score lower than the mask-only run.

```yaml
# labeled.yaml
model:
  classes: 5
data:
  image_dir: DATA/image
  mask_dir: DATA/mask
  label_dir: DATA/label
loss:
  mask_weight: 3.0
```

```yaml
# mask_only.yaml: same data, learns from the AI mask alone
model:
  classes: 5
data:
  image_dir: DATA/image
  mask_dir: DATA/mask
  label_dir: DATA/label
loss:
  label_weight: 0.0
  mask_weight: 1.0
  teacher: 0.0
```

## Optional training features

The default recipe uses cross-entropy, Dice loss, and EMA weights. The following features are off by default. To try one, set its key to the example value in your config.

| Feature | Config key | Example value |
| --- | --- | --- |
| Relative depth input | `model.depth` | `true` |
| Boundary-band loss | `loss.boundary` | `0.5` |
| Edge-guidance loss | `loss.aux` | `1.0` |
| Label blur | `loss.label_blur` | `1.0` |
| EMA teacher | `loss.teacher` | `0.3` |

For depth input, place the model files in the directory specified by `model.depth_weight`. The default is `weight/Depth-Anything-V2-Small-hf`. [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests skip hardware and model checks when the required GPU or checkpoints are unavailable.

## Code layout

`src/run.py` assembles training from the schema in `src/config.py`. Data pairing, splitting, augmentation, and model-input fitting live in `src/data/`. `src/model/` preserves the pretrained architecture and checkpoint keys. `src/train/` owns losses, EMA, optimizer steps, and training checkpoints. Both validation and the evaluation CLI use `src/evaluate.py`, which calls the whole-image and tiled inference in `src/predict.py`.
