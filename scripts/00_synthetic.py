import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve
from src.data.synthetic import generate


def main():
    parser = argparse.ArgumentParser(
        description="Write synthetic images with a known truth, a flawed mask and brush labels"
    )
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = generate(resolve(args.out), args.seed)
    print(f"images: {result['images']}")
    for cls, ratio in enumerate(result["label_ratio"]):
        print(f"class_{cls}_label_ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
