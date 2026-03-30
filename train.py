#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_CODES = ("cru", "crv", "cwu", "cwv", "vru", "vrv", "vwu", "vwv")
MODEL_NAMES = (
    "baseline",
    "equivariant_edges",
    "multiscale_relu",
    "multiscale_relu_pe",
    "multiscale_tanh",
    "multiscale_tanh_pe",
    "multiscale_tanh_pe_32",
    "multiscale_tanh_pe_4mp",
    "multiscale_tanh_pe_64",
    "multiscale_tanh_pe_8mp",
    "multiscale_topk",
    "spherical_coords_sin",
    "spherical_coords_tanh",
    "stress",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a single training job from models/.")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--dataset", required=True, choices=DATASET_CODES)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the underlying model script.",
    )
    args, extra = parser.parse_known_args()

    model_script = Path(__file__).resolve().parent / "models" / f"{args.model}.py"
    cmd = [args.python, str(model_script), args.dataset, *extra]
    return subprocess.call(cmd, cwd=model_script.parent.parent)


if __name__ == "__main__":
    raise SystemExit(main())
