# BiRefNet Multiclass Segmentation

Fine-tunes [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) for multiclass
semantic segmentation (four classes by default). It is built for SEM images,
where brightness, contrast, focus, and noise change with the operator: training
imitates those changes so the model learns to ignore them. Development uses the
public [NEU-Seg steel defect dataset](data/README.md).

## Setup

Run every command from the repository root. Python 3.10 or newer is required.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Run scripts with `.venv/bin/python`, or activate the environment first.
Keep PyTorch at 2.10 or newer: older versions have known checkpoint
deserialization vulnerabilities even with `weights_only=True`. Only load
checkpoints from trusted sources. The Werkzeug floor in `requirements.txt`
protects the TensorBoard dependency on Windows.

Download two pretrained models into `weight/`:

1. [BiRefNet weights](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth)
   to `weight/BiRefNet-general-epoch_244.pth`.
2. Depth Anything V2-Small. Skip this when `model.depth` is `false`.

   ```bash
   .venv/bin/hf download depth-anything/Depth-Anything-V2-Small-hf --local-dir weight/Depth-Anything-V2-Small-hf
   ```

Put training images in `data/train/image/` and masks in `data/train/mask/`;
an image and its mask share a filename stem (`0001.jpg` ↔ `0001.png`), and a
file without a partner stops training with an error. Images deeper than 8 bits,
such as 16-bit TIFFs, are stretched to 0–255 between their own darkest and
brightest pixel. Masks
are single-channel or palette images whose pixel values are class IDs `0` to
`model.classes - 1`. RGB masks are rejected. The value `255` means "ignore":
use it for pixels outside the region of interest, and no loss or metric counts
them.

## Training

```bash
python run_train.py
python run_train.py --config my.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

`--config` loads a file over `config.yaml`, so it only needs the keys it
changes. Each run gets its own folder:

```
run/<run-id>/
├── config.yaml      settings used by this run
├── train.csv        image/mask filenames in the training split
├── valid.csv        image/mask filenames in the validation split
├── logs/            TensorBoard logs (tensorboard --logdir run)
└── weights/
    ├── best_miou.pth    EMA weights with the best validation mIoU
    ├── last.pth         latest weights
    ├── last_ema.pth     latest EMA weights
    └── last.train.pth   latest weights plus optimizer state, for --resume
```

`--resume` reads `config.yaml`, `train.csv`, and `valid.csv` from the run
folder, so keep them next to `weights/`; it cannot be combined with `--config`.
The dataset folders must still hold exactly the saved file pairs. The run
continues from the saved step, but the random augmentation sequence starts over.

The split is drawn per image. Keep crops or related images from one source
together in the same split, or validation will see near-copies of training data.

Training uses CUDA, then an Apple GPU (MPS), then the CPU, whichever is
available first. On MPS the decoder's deformable convolutions use a
`grid_sample` version, because torchvision's has no MPS backward; both give the
same results, so training runs fully on the GPU.

## Configuration

Every setting lives in `config.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `model.weight` | `weight/BiRefNet-general-epoch_244.pth` | Pretrained BiRefNet weights |
| `model.depth` | `true` | Feed Depth Anything depth into the decoder ([Depth](#depth)) |
| `model.depth_weight` | `weight/Depth-Anything-V2-Small-hf` | Local Depth Anything folder |
| `model.classes` | `4` | Number of classes |
| `data.image_dir`, `data.mask_dir` | `data/train/image`, `data/train/mask` | Training images and masks |
| `data.valid_ratio` | `0.1` | Share of pairs held out for validation |
| `data.sem` | `true` | SEM input conversion ([Input](#input)); `false` for ordinary RGB |
| `augment.zoom` | `[0.5, 2.0]` | Magnification range |
| `augment.hide_edges` | `0.3` | Share of samples with hidden edges |
| `augment.brightness` | `0.2` | Brightness shift, up to ±0.2 of full range |
| `augment.contrast` | `0.4` | Contrast gain between 0.6 and 1.4 |
| `augment.gamma` | `[0.7, 1.5]` | Gamma range |
| `augment.light` | `0.2` | Uneven lighting, up to ±20% across the image |
| `augment.blur` | `1.0` | Largest blur sigma, in pixels |
| `augment.noise` | `0.05` | Largest grain standard deviation, as a share of 0–255 |
| `loader.*` | | PyTorch `DataLoader` options of the same name |
| `train.seed` | `0` | Fixes the split, augmentation, and initialization |
| `train.steps` | `5000` | Optimizer steps |
| `train.val_every`, `train.save_every` | `250`, `500` | Validation and save intervals, in steps |
| `train.lr`, `train.min_lr` | `1e-4`, `1e-6` | Peak and final learning rate |
| `train.warmup` | `200` | Steps rising from `min_lr` to `lr`; `0` starts at `lr` |
| `train.weight_decay` | `0.01` | AdamW weight decay |
| `train.grad_clip` | `1.0` | Gradient norm limit |
| `train.accum` | `4` | Batches per optimizer step |
| `train.ema` | `0.995` | EMA decay |
| `loss.ce`, `loss.dice`, `loss.boundary`, `loss.aux` | `1`, `1`, `0.5`, `1` | Loss weights ([Losses](#losses)) |
| `loss.boundary_width` | `3` | Band around class edges for the boundary loss, in pixels |
| `loss.label_blur` | `1.0` | Mask edge blur, in pixels ([Losses](#losses)) |
| `loss.teacher`, `loss.teacher_start` | `0.3`, `1000` | EMA teacher share and start step ([Losses](#losses)) |

`data.ignore_index` (default `255`) changes the ignore value. The `loader` keys
are `batch_size`, `num_workers`, `prefetch_factor`, `persistent_workers`,
`pin_memory`, and `drop_last`. Booleans are YAML `true`/`false` without quotes.
Pinning applies only on CUDA, and the two worker options only when
`num_workers > 0`.

## How training works

Only the squeeze module and the decoder train. The Swin backbone, its
BatchNorm statistics, and Depth Anything stay frozen.

### Input

Every image is fitted into a 1024×1024 input, keeping its aspect ratio and
padding the rest; the padding is ignored by every loss. The size is fixed by the
pretrained weights and cannot be changed.

With `data.sem: true`, the image is converted to grayscale, clipped to
mean ± 3 standard deviations, rescaled to 0–255, and repeated in all three
channels. The statistics come from the whole image, including pixels the mask
ignores, and never from the mask itself, so prediction converts images the same
way; tiled prediction converts each tile on its own.

### Augmentation

Each training sample is changed in this order, all before the SEM conversion
above, so the conversion sees what a real shot would give it:

1. **Zoom**: a factor drawn log-uniformly from `augment.zoom`, relative to
   fitting the whole image. Above 1, a random square window of side
   `long side / zoom` is cut out and enlarged; below 1, the whole image shrinks
   and the rest is padding. Window edges cut through the image are marked, and
   the boundary loss skips them.
2. **Rotate and flip**: a random multiple of 90° and an optional mirror.
3. **Brightness and contrast**, then **gamma** (log-uniform).
4. **Uneven light**: a linear ramp in a random direction brightens one side and
   darkens the other.
5. **Blur** with a sigma between 0 and `augment.blur`, like a slightly
   unfocused shot.
6. **Grain**: Gaussian noise with a standard deviation between 0 and
   `augment.noise`, like a fast scan.
7. **Hidden edges** on `augment.hide_edges` of the samples: each side loses up to
   25% of the image, filled with black, white, mid-gray, or a random gray, while
   at least half of the labeled pixels stay visible. Hidden pixels get the
   ignore label. They stand in for an uncropped info bar or a dark rim, so they
   also enter the SEM conversion statistics.

Validation and prediction use no augmentation.

### Losses

The loss adds cross-entropy, Dice, a boundary cross-entropy on a band
`loss.boundary_width` pixels wide around class edges, and BiRefNet's auxiliary
edge-guidance loss. Side outputs at coarser decoder levels share the
cross-entropy and Dice terms.

Masks are not trusted completely: hand-drawn edges are often a pixel or two
off, and zooming moves them too. Two settings soften the targets:

- `loss.label_blur: 1.0` blurs each class mask with a 1-pixel Gaussian, so
  targets mix neighboring classes only near class edges. Ignored pixels and
  padding never blur into the targets.
- `loss.teacher: 0.3` mixes in the EMA model's prediction: targets become 70%
  mask and 30% teacher, so where the averaged model keeps disagreeing with a
  mask, the target leans its way. The teacher reuses the frozen backbone and
  depth features, so it costs one extra decoder pass. The share stays 0 for the
  first `loss.teacher_start` steps, while the average is still poor, then rises
  linearly to its full value by twice that step.

Setting both to `0` trains on the masks exactly as drawn.

### Depth

With `model.depth: true`, a frozen Depth Anything V2-Small runs inside BiRefNet
on the exact input tensor, after every augmentation, so training and prediction
compute depth the same way and nothing is precomputed. Each image's depth is
scaled to 0–1 over its own pixels (2nd–98th percentile); padding stays 0. A
small branch adds depth next to the image patches at every decoder level. Its
last layers start at zero, so training begins at the pretrained decoder and uses
depth only as far as it helps.

### Validation and checkpoints

Every `train.val_every` steps, the EMA weights predict each validation image
whole, at its own size, exactly as prediction does, and are scored by mIoU.
The best score is saved as `best_miou.pth`. The EMA weights are updated after
every optimizer step and also serve as the teacher above.

Checkpoints store only the trained part (squeeze module and decoder), about a
tenth of the full model. Loading one first builds BiRefNet from `model.weight`
and Depth Anything from `model.depth_weight`, then puts the trained part on top,
so keep those files unchanged for as long as you need the run. Checkpoints from
earlier versions of this code cannot be loaded.

## Checking data and predicting

```bash
# Random original image/mask pairs
python scripts/01_dataset.py

# Random training samples after augmentation
python scripts/02_argument.py

# Input, predicted class map, overlay, and optional ground truth
python scripts/03_predict.py --weight run/<run-id>/weights/best_miou.pth --image data/test/image/000001.jpg
```

The scripts only open a matplotlib window; they write no files.
`03_predict.py --help` lists its options: `--mask` adds the ground truth,
`--tiles 1 3` averages predictions over the whole image and a 3×3 tile grid,
and `--class-id 3` adds that class's probability map. Prediction reuses the
checkpoint's input conversion and ignore value, and the run's `config.yaml`
unless `--config` names another file.

`--device cuda` (or `cuda:1`) predicts on a GPU, with BF16 where the GPU
supports it natively and FP16 otherwise. The default is the CPU.

## Code layout

| Folder | Holds |
|---|---|
| `src/data/` | Image/mask pairing, train/valid splits and their CSV files, augmentation, the training dataset |
| `src/prepare/` | Input conversion and the resize/padding geometry shared by training and prediction |
| `src/model/` | The network, Depth Anything wrapper, and checkpoint format |
| `src/train/` | Losses, trainer, EMA, learning rate schedule, validation, training checkpoints |
| `src/predict/` | Loading a trained model, whole-image and tiled prediction |
| `src/build/` | Builds the dataset, model, loss, and trainer from the config |

`src/run.py` creates or resumes a run and starts training; `run_train.py` reads
its command-line arguments. Run the tests with:

```bash
python -m pytest -q
```
