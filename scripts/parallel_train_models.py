#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
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
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch model training jobs in parallel."
    )
    parser.add_argument("--models", nargs="+", default=["baseline"], choices=MODEL_NAMES)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CODES), choices=DATASET_CODES)
    parser.add_argument("--workers", type=int, default=1, help="Maximum concurrent training jobs.")
    parser.add_argument(
        "--devices",
        nargs="*",
        default=[],
        help="Optional CUDA device ids to assign round-robin via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = deque((model_name, dataset) for model_name in args.models for dataset in args.datasets)
    running: list[tuple[subprocess.Popen[str], str]] = []
    device_cycle = deque(args.devices)

    while queue or running:
        while queue and len(running) < args.workers:
            model_name, dataset = queue.popleft()
            cmd = [args.python, "train.py", "--model", model_name, "--dataset", dataset]
            env = os.environ.copy()
            if device_cycle:
                device = device_cycle[0]
                device_cycle.rotate(-1)
                env["CUDA_VISIBLE_DEVICES"] = device
                label = f"{model_name}:{dataset} [gpu {device}]"
            else:
                label = f"{model_name}:{dataset}"

            print(f"[launch] {label}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
            running.append((proc, label))

        next_running: list[tuple[subprocess.Popen[str], str]] = []
        for proc, label in running:
            return_code = proc.poll()
            if return_code is None:
                next_running.append((proc, label))
                continue
            if return_code != 0:
                for other_proc, _ in next_running:
                    other_proc.terminate()
                raise subprocess.CalledProcessError(return_code, proc.args)
            print(f"[done] {label}", flush=True)
        running = next_running
        if running:
            time.sleep(0.5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
