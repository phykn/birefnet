# BiRefNet multiclass fine-tuning

Fine-tune pretrained BiRefNet for battery SEM semantic segmentation. LoRA is removed.
The default has four mutually exclusive pixel classes and trains the decoder plus
squeeze module while freezing the Swin-L backbone and BatchNorm statistics.

## Project layout

- `src/`: reusable model, preprocessing, training, and inference logic.
- `backend/`: FastAPI app, HTTP routes, request/response schemas, and image codecs.
- `run_train.py` / `run_api.py`: training and API entrypoints.
- `notebooks/`: examples that use the core logic directly.

The backend imports `src`; the core logic does not depend on the backend.
`src/build/` assembles configured components. `src/model/checkpoint.py` owns the
shared model checkpoint format and pretrained head conversion; `src/train/checkpoint.py`
adds optimizer, scheduler, scaler, EMA, and training progress for resume.

## Data and pretrained weights

Four synthetic 256x256 sample pairs are included; see `data/README.md` and
`data/preview.png`. They are pipeline fixtures, not real SEM measurements.

Put matching filename stems under `data/image` and `data/mask`.
Masks must be single-channel integer or palette images with IDs `0, 1, 2, 3`,
not RGB visualization masks or 0/255 binary masks. The default `255` is ignored.
For example: 0 inter-particle pore, 1 solid particle, 2 internal crack,
3 internal pore. Class names are your annotation convention, not inferred by code.
All four classes, including class 0, contribute to the loss and macro metrics.

The official initial checkpoint is
[BiRefNet-general-epoch_244.pth](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth),
from [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
Place it at `weight/BiRefNet-general-epoch_244.pth` (885,082,437 bytes).
SHA256: `11341a6a1c12646627e8d28da025bfec8aad027929d377cbe8fd4759636cc77c`.
The weight directory is git-ignored; copy the file separately to the training computer.

Initial loading preserves all compatible pretrained tensors and initializes only
the four segmentation heads (final + three auxiliary heads) whose output changes
from one to four channels. Missing keys and unrelated shape mismatches are errors.
Binary GDT attention remains one channel; its guidance now uses detached boundaries
across all softmax classes, rather than treating a selected class as foreground.

## Train

```bash
pip install -r requirements.txt
python run_train.py --config config/train.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

Run from the project root. Training selects CUDA if available, otherwise CPU.
Class count is `birefnet.num_classes` in `config/model.yaml`; data paths, resolution,
ignore index, optimizer and fine-tuning options are in `config/train.yaml`.

- `train.mode: decoder` (default): frozen backbone, train squeeze + decoder.
- `train.mode: partial`: also train the final `train.backbone_stages` backbone stages.
- `train.mode: full`: train all model parameters.
- `train.backbone_lr_scale: 0.1`: lower learning rate for an unfrozen backbone.
- `train.freeze_bn: true`: keep BatchNorm statistics fixed for small batches.
- `teacher.enabled: false`: omit EMA memory and extra prediction by default.

The loss is masked multiclass Cross-Entropy + Dice, with final-output boundary
weighting, multiscale supervision, and binary GDT auxiliary supervision. Geometry
is shared between the weak/strong views. Indexed masks use nearest-neighbor resizing.
GPU training uses mixed precision and optional backbone gradient checkpointing.
The configured batch contains two augmentation views during training; 1024-pixel
training on 24GB has not been measured. Reduce batch/resolution if needed, preserving
thin cracks, or use gradient accumulation when increasing the effective batch.

Only train/validation sets are used; binary threshold calibration is removed.
Membership is saved in CSV files. Automatic splitting is by input image, not specimen:
keep related images from the same specimen and crops from the same original in one
split. For grouped evaluation, prepare the run's CSV membership consistently.
Report per-class IoU/Dice/recall, not only the mean, especially for rare cracks.

## Checkpoints and prediction

Runs are stored under `run/<run-id>`. `weights/best_miou.pth` is selected by native-size
validation mIoU, `last.pth` is a standalone full model, and `last.train.pth` also saves
optimizer/scheduler/scaler/optional EMA state for resume. Saved preprocessing is reused
for inference. Resume/inference do not require the original pretrained file.
Resume restores the saved configuration and split membership; it does not restore RNG
or loader position exactly. Old LoRA overlays and binary runs cannot be resumed.
Resume rejects a preprocessing size or input mode that differs from the checkpoint.

```bash
python run_api.py --host 127.0.0.1 --port 8000 --weight run/<run-id>/weights/best_miou.pth
```

The API defaults to CPU; use `--device cuda` on a GPU machine. For a nondefault class
count, pass `--config run/<run-id>/config.yaml`. `POST /predict` accepts `base64_str`,
optional `id`, `tiles` (e.g. `[1, 3]`), and `overlap`. All class logits are blended
before argmax. Default `output_mode: labels` returns an indexed PNG with original
class IDs, plus `num_classes` metadata. Decode palette PNG indices with Pillow:
`np.asarray(Image.open(...))`; RGB conversion produces a visualization instead.
`output_mode: probability` requires `class_id` and returns that class probability
scaled to uint8 0..255. Binary `threshold` is no longer accepted.

Python `predict_logits` returns float32 C,H,W. Python `predict(...,
output_mode="probability")` without a class ID returns all softmax probabilities.
See `notebooks/01_predict.ipynb` for a CPU example using a trained full checkpoint.
Training and API decoding both use stored image pixels without applying EXIF rotation
or mirroring, so output labels share the stored image's coordinates.

## Verification

```bash
python -m pytest -q
```

Tests cover indexed masks, loss/backprop, fine-tuning parameter selection, pretrained
head replacement, resume, multiclass tiling and API output. They run on CPU with small
fixtures. Real SEM accuracy and full-resolution GPU memory still require training
and evaluation on the target computer. `scripts/check_upstream.py` is an optional
binary architecture compatibility check against an explicitly supplied upstream
checkout; it does not validate multiclass task accuracy.
