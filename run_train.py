import argparse

import torch
from omegaconf import OmegaConf

from src.build.data import build as build_data
from src.build.model import build as build_model
from src.config import load_run
from src.data.split import load as load_splits
from src.data.split import save as save_splits
from src.build.trainer import build as build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, checkpoint, run_dir = load_run(args.resume, args.config)
    saved_splits = load_splits(run_dir) if run_dir is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, load_pretrained=checkpoint is None).to(device)
    train_loader, valid_loader, splits = build_data(
        cfg,
        saved_splits,
    )
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=run_dir,
    )
    if checkpoint is not None:
        trainer.load_resume(str(checkpoint))

    n_train = len(splits["train_image"])
    n_valid = len(splits["valid_image"])
    total, trainable = model.stats["total"], model.stats["trainable"]
    print(f"\n[Dataset] train={n_train}, valid={n_valid}")
    print(
        f"[BiRefNet] total={total:,}  trainable={trainable:,}  "
        f"ratio={trainable / total:.2%}"
    )

    if run_dir is None:
        OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
        save_splits(splits, trainer.save_dir)

    trainer.train(
        steps=cfg.train.steps,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
    )
    print(f"\nTraining finished. Weights saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
