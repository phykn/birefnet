# BiRefNet SEM Segmentation

Fine-tune **BiRefNet for multiclass semantic segmentation** of battery SEM images.
Supports four classes by default, tiled inference, and a FastAPI server.

## Quick start

Run commands from the repository root.

```bash
pip install -r requirements.txt
```

1. Download the [pretrained weights](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth) to `weight/BiRefNet-general-epoch_244.pth`.
2. Place images and masks with matching filename stems in `data/image/` and `data/mask/`.
3. Set data paths, resolution, and training options in `config/train.yaml`. Set the class count in `config/model.yaml`.

Masks must be **single-channel or palette images containing class IDs 0-3**. RGB masks are unsupported; `255` is ignored.
The included [four synthetic pairs](data/README.md) are pipeline fixtures. Use separate SEM data for real training and evaluation.

## Training

```bash
python run_train.py --config config/train.yaml
python run_train.py --resume run/<run-id>/weights/last.train.pth
```

The default `decoder` mode freezes the backbone and trains the squeeze module and decoder.
`train.mode: partial` also trains the final backbone stages; `full` trains the entire model. CUDA is used when available.

Results are saved under `run/<run-id>/`. `best_miou.pth` is selected by validation mIoU,
`last.pth` is the latest model, and `last.train.pth` includes the state needed to resume.
Keep images and crops from the same specimen in the same split. Automatic splitting operates on individual images.

## Data and model checks

```bash
# Inspect loader inputs, augmentations, masks, and valid regions
python scripts/check_data.py --size 256 --batches 2

# Check gradient connectivity and finite values for trainable parameters
python scripts/check_gradients.py --size 64

# Save native-size class IDs, an overlay, and a comparison image
python scripts/predict.py --weight run/<run-id>/weights/best_miou.pth --image data/image/sample_01.png
```

Outputs are saved under `run/checks/`. Use each script's `--help` for options.
For prediction, add `--mask` to display ground truth, `--tiles 1 3` for tiled inference,
or `--class-id 3` to save a class probability PNG.
Prediction reuses the saved preprocessing settings. Palette indices in `labels.png` are class IDs.

## API

```bash
python run_api.py --host 127.0.0.1 --port 8000 --weight run/<run-id>/weights/best_miou.pth
```

Open `http://127.0.0.1:8000/docs` for the API documentation. Send `base64_str` to `POST /predict` to receive a base64-encoded class ID PNG.
Use `--device cuda` for GPU inference and `--config run/<run-id>/config.yaml` for a nondefault class configuration.

## Development

`src/` contains model, data, training, and inference code; `backend/` contains the HTTP API; `scripts/` contains inspection tools.

```bash
python -m pytest -q
```

Based on [BiRefNet](https://github.com/ZhengPeng7/BiRefNet). Current checkpoints store the full model and are incompatible with earlier LoRA and binary segmentation checkpoints.
