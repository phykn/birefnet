import argparse

from src.run import train


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--config")
    args = parser.parse_args()
    train(args.resume, args.config)


if __name__ == "__main__":
    main()
