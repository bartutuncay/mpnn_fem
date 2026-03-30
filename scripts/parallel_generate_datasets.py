#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

dataset_ids = ("cru", "crv", "cwu", "cwv", "vru", "vrv", "vwu", "vwv")
data_dir = Path(__file__).resolve().parents[1]
sim_dir = data_dir/"datasets_src"/"training_datasets"
processor = data_dir/"datasets_src"/"save_train_data.py"
normalizer = data_dir/"datasets_src"/"calc_norm_stats.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ground truth simulations; convert into graphs, compute normalization stats."
    )
    parser.add_argument("--datasets", nargs="+", default=list(dataset_ids), choices=dataset_ids)
    parser.add_argument("--start", type=int, default=0, help="starting index")
    parser.add_argument("--stop", type=int, required=True, help="last index")
    parser.add_argument(
        "--split-index",
        type=int,
        default=80,
        help="Simulations below are written to training, rest to testing.",
    )
    parser.add_argument("--raw-workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--process-workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--python", default=sys.executable, help="python")
    parser.add_argument("--skip-raw", action="store_true", help="skip ground truth generation")
    parser.add_argument("--skip-process", action="store_true", help="skip processing")
    parser.add_argument("--skip-norm", action="store_true", help="skip norm.stat. calculation")
    return parser.parse_args()


def iter_indices(start: int, stop: int) -> Iterable[int]:
    if stop <= start:
        raise ValueError("--stop must be greater than --start")
    return range(start, stop)

def run_parallel(commands: list[list[str]], max_workers: int, cwd: Path) -> None:
    pending = list(commands)
    running: list[tuple[subprocess.Popen[str], list[str]]] = []

    def launch(cmd: list[str]) -> tuple[subprocess.Popen[str], list[str]]:
        proc = subprocess.Popen(cmd, cwd=cwd)
        print(f"[launch] {' '.join(cmd)}", flush=True)
        return proc, cmd

    while pending or running:
        while pending and len(running) < max_workers:
            cmd = pending.pop(0)
            running.append(launch(cmd))

        next_running: list[tuple[subprocess.Popen[str], list[str]]] = []
        for proc, original_cmd in running:
            return_code = proc.poll()
            if return_code is None:
                next_running.append((proc, original_cmd))
                continue
            if return_code != 0:
                for other_proc, _ in next_running:
                    other_proc.terminate()
                raise subprocess.CalledProcessError(return_code, original_cmd)
            print(f"[done] {' '.join(original_cmd)}", flush=True)
        running = next_running
        if running:
            time.sleep(0.2)


def main() -> int:
    args = parse_args()
    indices = list(iter_indices(args.start, args.stop))

    if not args.skip_raw:
        raw_commands = []
        for dataset in args.datasets:
            raw_script = sim_dir / f"{dataset}.py"
            for idx in indices:
                raw_commands.append([args.python, str(raw_script), str(idx)])
        run_parallel(raw_commands, max_workers=args.raw_workers, cwd=data_dir)

    if not args.skip_process:
        process_commands = []
        for dataset in args.datasets:
            for idx in indices:
                split = "train" if idx < args.split_index else "test"
                process_commands.append(
                    [args.python, str(processor), str(idx), dataset, split]
                )
        run_parallel(process_commands, max_workers=args.process_workers, cwd=data_dir)

    if not args.skip_norm:
        norm_commands = [[args.python, str(normalizer), dataset] for dataset in args.datasets]
        run_parallel(norm_commands, max_workers=min(len(norm_commands), args.process_workers), cwd=data_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
