# BiRefNet Multiclass Segmentation

Fine-tune **BiRefNet for multiclass semantic segmentation** on your own image datasets.
Supports four classes by default, tiled inference, and a FastAPI server.

## Quick start

Run commands from the repository root.

```bash
pip install -r requirements.txt
```

1. Download the [pretrained weights](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth) to `weight/BiRefNet-general-epoch_244.pth`.
2. Place images and masks with matching filename stems in `data/image/` and `data/mask/`.
3. Set data paths and training options in `config/train.yaml`. Set the class count in `config/model.yaml`.

Masks must be **single-channel or palette images containing class IDs 0-3**. RGB masks are unsupported; `255` is ignored.
The included [four synthetic pairs](data/README.md) are pipeline fixtures. Use your own annotated data for training and evaluation.

## Training

```bash
python run_train.py --config config/train.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

The backbone is always frozen; only the squeeze module and decoder are trained.
BatchNorm statistics remain frozen. CUDA is used when available.
`train.warmup_steps: 0` starts directly at `max_lr`; a positive value ramps
from `min_lr` before cosine decay.
The segmentation objective uses cross-entropy, Dice overlap loss, and boundary
cross-entropy, plus the model's auxiliary guidance loss. Dice is fixed; IoU
remains an evaluation metric, not an additional training loss.

Configuration keeps dataset paths, `is_sem`, augmentation, batch/worker
counts, training, and loss options available for experiments. Backbone
channel widths use the model's architecture defaults; ignored labels default to
`255`. Loader pinning follows CUDA availability, persistent workers follow
`num_workers > 0`, and prefetch defaults to 2. These defaults no longer need to
be repeated in YAML. Existing explicit overrides in saved runs are still read.

Results are saved under `run/<run-id>/`. `best_miou.pth` is selected by validation
mIoU on native-size predictions,
`last.pth` is the latest model, and `last.train.pth` includes the state needed to resume.
`last_ema.pth` contains a separate full model with EMA weights (`train.ema_decay`, default 0.99),
updated only after successful optimizer steps. EMA is used for saving only;
validation and best-model selection use the live model. Training checkpoints
preserve EMA state; older checkpoints without EMA initialize it from restored
model weights. Frozen parameters are reused and persistent model buffers are tracked.
To resume, keep `config.yaml`, `train.csv`, and `valid.csv` beside the run's
`weights/` directory. The current dataset must match the saved file pairs;
split membership and order are restored before the model is allocated.
`--resume` uses the saved config and cannot be combined with `--config`.
Random sampling and augmentation sequences are not restored.
Keep related images and crops from the same source in the same split. Automatic splitting operates on individual images.

### SEM inputs and augmentation

Training, validation, and inference inputs are fixed at 1024x1024 to match the
configured pretrained weights. Input size has no config or CLI override.
Checkpoints declaring another input size are rejected.

The default training config uses `data.is_sem: true`: grayscale,
1–99 percentile normalization, and two successive z-score/CLAHE passes.
Training crops and rotates/flips the grayscale image, applies brightness/contrast
jitter, then derives all three channels from that same augmented image.
Each training sample has one augmented image. Brightness is sampled uniformly
from -0.20 to +0.20, and contrast gain from 0.60 to 1.40.
`augment.brightness` and `augment.contrast` set these limits.
Validation has no random jitter.
ImageNet normalization follows feature generation and edge masking.

`data.crop_prob: 1.0` crops every training sample. A square side length is drawn
uniformly from `data.min_crop_size` (default 256) through 1024, inclusive; each
axis is clipped to the available source pixels for smaller images. The minimum
must be an integer from 1 to 1024. Lowering `crop_prob` also allows whole images.
After cropping, the image is resized with its aspect ratio preserved until the
long side reaches 1024, then padded to 1024x1024. Masks and crop-edge flags use
the same geometry with nearest-neighbor resizing; padding labels are ignored.
Crop centers are selected randomly, independently of mask labels.
`augment.masking_prob: 1.0` attempts edge masking on every training sample
(`0.0` disables it). Each side hides at most 25% of the image, retaining at
least 50% of the pre-masking valid ROI pixels. Sampling retries at most 32
times, then leaves the sample unchanged if no valid candidate is found.
Hidden areas receive a black, white, mid-gray, or random grayscale fill. Hidden labels use `data.ignore_index` (default `255`).

Encode ROI-exterior pixels as `255` in the input class-ID mask; a separate ROI
file is not loaded. Feature statistics use the whole crop, without GT-mask
statistics, so the same conversion is available at inference. Validation and
inference disable random augmentation and retain aspect-preserving resize and
padding. Tiled SEM inference derives features independently for each tile.
Set `data.is_sem: false` for ordinary RGB input. Checkpoints record the boolean
`is_sem` setting, which validation and inference reuse automatically. Legacy
`rgb` and `sem_features` checkpoint metadata are accepted; the removed
`gray_repeat` and `gray_features` representations cannot map to this boolean
and require the earlier code to run their checkpoints.

Split original images before online crops. Externally generated crops from one
source still need to be assigned to the same split. Before selecting a trained
SEM model, compare mIoU/Dice and porosity changes under brightness, contrast,
and edge-background perturbations on the same retained ROI. The included
synthetic fixtures establish pipeline behavior, not SEM robustness.

## Data and model checks

```bash
# Inspect loader inputs, augmentations, masks, and valid regions
python scripts/check_data.py --batches 2

# Check gradient connectivity and finite values for trainable parameters
python scripts/check_gradients.py

# Save native-size class IDs, an overlay, and a comparison image
python scripts/predict.py --weight run/<run-id>/weights/best_miou.pth --image data/image/sample_01.png
```

Outputs are saved under `run/checks/`. Use each script's `--help` for options.
For prediction, add `--mask` to display ground truth, `--tiles 1 3` for tiled inference,
or `--class-id 3` to save a class probability PNG.
Prediction reuses the saved preprocessing settings and ignored-label index.
Palette indices in `labels.png` are class IDs.

## API

```bash
python run_api.py --host 127.0.0.1 --port 8000 --weight run/<run-id>/weights/best_miou.pth
```

Open `http://127.0.0.1:8000/docs` for the API documentation. Send `base64_str` to `POST /predict` to receive a base64-encoded class ID PNG.
Use `--device cuda` for GPU inference. Both the API and prediction script default
to the checkpoint's run config when available, then model defaults. `--config`
selects an explicit configuration.
The API reads the checkpoint's preprocessing settings by default and uses the
same Pillow RGB decoding as the training loader, including for 16-bit PNGs.

## Development

`src/data/` owns image/mask pairs, split membership, CSV serialization, and
training augmentation. Splits and datasets keep each image and mask together
as a pair; CSVs store their filenames so dataset directories can be relocated.
`src/prepare/`
owns shared channel conversion and resize/padding geometry; `Fit` applies the
same geometry to images, masks, and native-size restoration. `src/model/`
owns the network and model checkpoint format, while `src/train/` owns losses,
optimization, EMA, validation, and atomic training checkpoint writes.
`src/predict/` loads inference models and their run configuration, restores
predictions, and blends tiles; `src/build/` assembles
these components from configuration, including the loss shared by training
and gradient inspection. `src/run.py` owns training run creation, saved config
and split restoration, and the training launch; `run_train.py` handles CLI
arguments. `src/config.py` reads and migrates config values. `backend/` contains
the HTTP API and `scripts/` contains inspection tools.

```bash
python -m pytest -q
```

Based on [BiRefNet](https://github.com/ZhengPeng7/BiRefNet). Current checkpoints store the full model and are incompatible with earlier LoRA and binary segmentation checkpoints.
