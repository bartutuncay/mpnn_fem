# Message Passing Neural Networks for Structural Engineering: Bridging MPNNs and FEM

This repository contains:

- Ground truth FEM data generation scripts in `datasets_src/training_datasets`
- Script to convert data into graphs `datasets_src/save_train_data.py`
- Dataloaders in `forward_src`
- Model training scripts in `models`
- Evaluation scripts in `eval`

## Dataset subset codes

In total, 8 different combinations are possible. Slab, panel and tube datasets are included as experimental datasets and are not validated.
- Boundary conditions: `c` or `v`: `cantilever` or `var_bc`
- Mesh geometry: `r` or `w`: `regular` or `warped`
- Load condition: `u` or `v`: `uniform` or `non_uniform`

## Installation

Create an environment and install the Python dependencies from [requirements.txt](requirements.txt).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

*Python 3.12 with CUDA preferred.*
Note: `torch`, `torch-scatter`, and `torch-geometric` should be installed directly from PyTorch and PyTorch Geometric servers with matching versions and matching CUDA, otherwise version mismatches are likely to occur.

## Project layout

- [datasets_src/training_datasets](datasets_src/training_datasets): deterministic FEM ground truth data generators, one script per subset
- [datasets_src/save_train_data.py](datasets_src/save_train_data.py): converts raw simulation files to data graphs. Raw simulation data is not required after these are generated.
- [datasets_src/calc_norm_stats.py](datasets_src/calc_norm_stats.py): computes normalization statistics for processed *training* sets.
- [forward_src/dataloader.py](forward_src/dataloader.py): dataloader (also stress and spherical coordinates variables available)
- [models](models): model training scripts
- [train.py](/Users/bartu/gnn/mpnn_fem/train.py): initiates single model training run
- [scripts/parallel_generate_datasets.py](/Users/bartu/gnn/mpnn_fem/scripts/parallel_generate_datasets.py): dataset generation batch operation (use if hardware can handle parallelization)

## Data pipeline

1. Generate raw FEM simulations into `datasets/support/geom/loads/sim_*.pt`.
2. Convert each raw simulation into per-timestep graph samples under `datasets/support/geom/loads/{train|test}/`.
3. Compute normalization statistics into `datasets/support/geom/loads/norm/train_norm_stats.pt`.

The conversion step keeps the last timestep and writes one processed graph file per simulation.

## Usage

Generate one raw dataset family manually:

```bash
python3 datasets_src/training_datasets/cru.py 0
```

Convert one simulation into processed training graphs:

```bash
python3 datasets_src/save_train_data.py 0 cru train
```

Compute normalization statistics:

```bash
python3 datasets_src/calc_norm_stats.py cru
```

Train one model:

```bash
python3 train.py --model baseline --dataset cru
```

## Parallel scripts

Generate multiple datasets in parallel:

```bash
python3 scripts/parallel_generate_datasets.py --datasets cru crv vrv --start 0 --stop 100 --split-index 80 --raw-workers 4 --process-workers 4
```


## Notes

- Most training scripts run on `CUDA` by default.
- Model checkpoints and loss CSVs are written under `training/*/`.
- Each model has to be trained individually since a combined trainer is not implemented yet.