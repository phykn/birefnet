import argparse

import torch
import uvicorn

from src.build.model import build_predictor
from src.config import load_config
from backend.app import build_app, read_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu", help="cpu or cuda on the training machine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_config(args.config)
    model = build_predictor(cfg, args.weight, device)
    app = build_app(
        model=model,
        device=device,
        preprocess=read_preprocess(model),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
