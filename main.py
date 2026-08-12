"""Inspect the shared model registry."""

from __future__ import annotations

import argparse

from models import available_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the canonical names accepted by the model registry.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.list:
        for name in available_models():
            print(name)
        return
    print("Use --list to show registered model names.")


if __name__ == "__main__":
    main()
