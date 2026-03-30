## Calculate dataset statistics and write results in dictionary

import torch
from forward_src.dataloader_stress import make_loader
from typing import Dict, List, Tuple, Union
import argparse

parser = argparse.ArgumentParser(description="dataset path")
parser.add_argument("dataset", type=str, help="define dataset: use 3-letter abbreviation")
args = parser.parse_args()
sim_dataset = args.dataset
support = 'cantilever' if sim_dataset[0] == 'c' else 'var_bc'
geom = 'regular' if sim_dataset[1] == 'r' else 'warped'
loads = 'uniform' if sim_dataset[2] == 'u' else 'non_uniform'
data_dir = f'{support}/{geom}/{loads}'

class RunningMeanStd:
    def __init__(self, feat_dim: int, eps: float = 1e-8, device="cpu"):
        self.eps = eps
        self.device = device
        self.count = torch.tensor(0.0, device=device)
        self.mean = torch.zeros(feat_dim, device=device)
        self.M2 = torch.zeros(feat_dim, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor,update_partial:bool):
        x = x.to(self.device).float()
        if x.numel() == 0:
            return
        if update_partial == True:
            x = x[:,3:6]

        n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if self.count.item() == 0:
            self.mean = batch_mean
            self.M2 = batch_var * n
            self.count = torch.tensor(float(n), device=self.device)
            return

        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * (n / total)
        self.M2 = self.M2 + batch_var * n + (delta ** 2) * (self.count * n / total)
        self.count = total

    def finalize(self):
        var = self.M2 / torch.clamp(self.count, min=1.0)
        std = torch.sqrt(var + self.eps)
        return self.mean, std


FeatShape = Union[int, Tuple[int, ...]]

@torch.no_grad()
def compute_norm_stats_from_loader(
    loader,
    keys_and_shapes: Dict[str, FeatShape],
    max_batches: int | None = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, torch.Tensor]]:

    def _shape_tuple(s: FeatShape) -> Tuple[int, ...]:
        return (s,) if isinstance(s, int) else tuple(s)

    def _prod(t: Tuple[int, ...]) -> int:
        p = 1
        for v in t:
            p *= int(v)
        return p

    rms = {k: RunningMeanStd(_prod(_shape_tuple(sh)), device=device)
           for k, sh in keys_and_shapes.items()}

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        for key, feat_shape in keys_and_shapes.items():
            x = batch[key]
            if x is None:
                continue

            feat_shape_t = _shape_tuple(feat_shape)
            feat_ndim = len(feat_shape_t)
            feat_dim = _prod(feat_shape_t)

            # Flatten all feature dimensions into one:
            x2 = x.reshape(-1, feat_dim)

            if key == "x":
                rms[key].update(x2, True)
            else:
                rms[key].update(x2, False)

    stats = {}
    for k in keys_and_shapes:
        mean, std = rms[k].finalize()
        stats[k] = {"mean": mean.cpu(), "std": std.cpu()}
    return stats

train_loader = make_loader(f'datasets/{data_dir}/train', batch_size=1, shuffle=True, num_workers=4)

keys_and_dims = {
    "x": 46,
    "fext": 3,
    "edge_attr": 3,
    "y_u": 3,
    "y_fint": 3,
    "y_str": (3,3),
}

stats = compute_norm_stats_from_loader(train_loader, keys_and_dims)
torch.save(stats, f"datasets/{data_dir}/norm/train_norm_stats.pt")
