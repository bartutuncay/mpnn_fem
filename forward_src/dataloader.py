from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

#Data(edge_index=[2, 24804], edge_attr=[24804, 3],
#  pos=[4900, 3], num_nodes=4900, x=[4900, 46], fext=[4900, 3],
#  y_u=[4900, 3], y_fint=[4900, 3], coarse_features=[4900, 1],
#  edge_index_coarse=[2, 1002], edge_attr_coarse=[1002, 3], edge_index_diff=[2, 9800],
#  edge_attr_diff=[9800, 3], edge_index_aggr=[2, 9800], edge_attr_aggr=[9800, 3])

@dataclass
class DataBatch:
    edge_index: Optional[torch.Tensor] = None           
    edge_attr: Optional[torch.Tensor] = None           
    pos: Optional[torch.Tensor] = None    
    num_nodes: Optional[torch.Tensor] = None  
    x: Optional[torch.Tensor] = None           
    fext: Optional[torch.Tensor] = None       
    y_u: Optional[torch.Tensor] = None       
    y_fint: Optional[torch.Tensor] = None       
    coarse_features: Optional[torch.Tensor] = None       
    edge_index_coarse: Optional[torch.Tensor] = None       
    edge_attr_coarse: Optional[torch.Tensor] = None       
    edge_index_diff: Optional[torch.Tensor] = None       
    edge_attr_diff: Optional[torch.Tensor] = None       
    edge_index_aggr: Optional[torch.Tensor] = None       
    edge_attr_aggr: Optional[torch.Tensor] = None       
    batch: Optional[torch.Tensor] = None       

    # Allow dict-style access: batch['img'] etc.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    # Match your training loop: batch = batch.to(device)
    def to(self, device: Union[str, torch.device]) -> "DataBatch":
        for field_name in self.__dataclass_fields__:
            v = getattr(self, field_name)
            if torch.is_tensor(v):
                setattr(self, field_name, v.to(device))
        return self

class PtDictFolderDataset(Dataset):
    """
    """
    def __init__(self, root_dir: Union[str, Path], pattern: str = "*.pt"):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched {pattern} under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        sample = torch.load(path, map_location="cpu")

        # Add filename if useful for debugging
        sample["_path"] = str(path)
        return sample
    
def _require_all(samples, key: str):
    vals = [s.get(key, None) for s in samples]
    if any(v is None for v in vals):
        missing = [i for i, v in enumerate(vals) if v is None]
        raise KeyError(f"Some samples missing '{key}': indices {missing}")
    return vals

def _concat_node_tensor(
    samples: Sequence[Dict[str, Any]],
    key: str,
    num_nodes: List[int],
    allow_missing: bool = True,
) -> Optional[torch.Tensor]:
    vals = [s.get(key, None) for s in samples]
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        if allow_missing:
            # If partially missing, that's usually a dataset bug; fail loudly.
            missing = [i for i, v in enumerate(vals) if v is None]
            raise KeyError(f"Some samples missing '{key}' but others have it: indices {missing}")
        else:
            raise KeyError(f"Missing required key '{key}' in some samples.")
    # Validate first dim matches num_nodes
    for i, v in enumerate(vals):
        if not torch.is_tensor(v):
            raise TypeError(f"'{key}' in sample {i} is not a tensor. Got {type(v)}")
        if v.shape[0] != num_nodes[i]:
            raise ValueError(
                f"'{key}' in sample {i} has shape[0]={v.shape[0]} but expected num_nodes={num_nodes[i]}"
            )
    return torch.cat(vals, dim=0)

def _collate_edges(
    samples: Sequence[Dict[str, Any]],
    edge_key: str,
    attr_key: str,
    num_nodes: List[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    edge_list = [s.get(edge_key, None) for s in samples]
    if all(e is None for e in edge_list):
        return None, None
    if any(e is None for e in edge_list):
        missing = [i for i, e in enumerate(edge_list) if e is None]
        raise KeyError(f"Some samples missing '{edge_key}': indices {missing}")

    attr_list = [s.get(attr_key, None) for s in samples]
    has_any_attr = any(a is not None for a in attr_list)
    has_all_attr = all(a is not None for a in attr_list)
    if has_any_attr and not has_all_attr:
        missing = [i for i, a in enumerate(attr_list) if a is None]
        raise KeyError(f"Some samples missing '{attr_key}' but others have it: indices {missing}")

    edge_index_cat: List[torch.Tensor] = []
    edge_attr_cat: List[torch.Tensor] = []

    node_offset = 0
    for i, (ei, n) in enumerate(zip(edge_list, num_nodes)):
        if not torch.is_tensor(ei):
            raise TypeError(f"'{edge_key}' in sample {i} is not a tensor. Got {type(ei)}")
        if ei.dim() != 2 or ei.shape[0] != 2:
            raise ValueError(f"'{edge_key}' in sample {i} must have shape [2, E]. Got {tuple(ei.shape)}")

        ei_shift = ei.clone() + node_offset
        edge_index_cat.append(ei_shift)

        if has_all_attr:
            a = attr_list[i]
            if not torch.is_tensor(a):
                raise TypeError(f"'{attr_key}' in sample {i} is not a tensor. Got {type(a)}")
            edge_attr_cat.append(a)

        node_offset += n

    edge_index = torch.cat(edge_index_cat, dim=1)
    edge_attr = torch.cat(edge_attr_cat, dim=0) if has_all_attr else None
    return edge_index, edge_attr

def collate_pt_dicts(samples: Sequence[Dict[str, Any]]) -> DataBatch:
    # Required: y_u determines num_nodes
    y_us = _require_all(samples, "y_u")
    for i, y in enumerate(y_us):
        if not torch.is_tensor(y):
            raise TypeError(f"'y_u' in sample {i} is not a tensor. Got {type(y)}")
        if y.dim() < 1:
            raise ValueError(f"'y_u' in sample {i} must have at least 1 dimension. Got {y.dim()}")

    num_nodes_list: List[int] = [y.shape[0] for y in y_us]
    y_u = torch.cat(y_us, dim=0)

    # Batch vector
    batch_vec = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes_list)],dim=0)

    # Main edges
    edge_index, edge_attr = _collate_edges(
        samples, edge_key="edge_index", attr_key="edge_attr", num_nodes=num_nodes_list)

    # Optional node-level tensors
    x = _concat_node_tensor(samples, "x", num_nodes_list, allow_missing=True)
    fext = _concat_node_tensor(samples, "fext", num_nodes_list, allow_missing=True)
    y_fint = _concat_node_tensor(samples, "y_fint", num_nodes_list, allow_missing=True)
    coarse_features = _concat_node_tensor(samples, "coarse_features", num_nodes_list, allow_missing=True)

    # pos: optional; if present, assume per-node with matching N and concat
    pos_vals = [s.get("pos", None) for s in samples]
    if all(v is None for v in pos_vals):
        pos = None
    else:
        if any(v is None for v in pos_vals):
            missing = [i for i, v in enumerate(pos_vals) if v is None]
            raise KeyError(f"Some samples missing 'pos' but others have it: indices {missing}")
        for i, v in enumerate(pos_vals):
            if not torch.is_tensor(v):
                raise TypeError(f"'pos' in sample {i} is not a tensor. Got {type(v)}")
            if v.shape[0] != num_nodes_list[i]:
                raise ValueError(
                    f"'pos' in sample {i} has shape[0]={v.shape[0]} but expected num_nodes={num_nodes_list[i]}"
                )
        pos = torch.cat(pos_vals, dim=0)

    # Additional edge sets (coarse/diff/aggr)
    edge_index_coarse, edge_attr_coarse = _collate_edges(
        samples, edge_key="edge_index_coarse", attr_key="edge_attr_coarse", num_nodes=num_nodes_list)
    edge_index_diff, edge_attr_diff = _collate_edges(
        samples, edge_key="edge_index_diff", attr_key="edge_attr_diff", num_nodes=num_nodes_list)
    edge_index_aggr, edge_attr_aggr = _collate_edges(
        samples, edge_key="edge_index_aggr", attr_key="edge_attr_aggr", num_nodes=num_nodes_list)
    # Return num_nodes as tensor to match your dataclass annotation (or change the annotation instead)
    num_nodes = torch.tensor(num_nodes_list, dtype=torch.long)

    return DataBatch(
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        num_nodes=num_nodes,
        x=x,
        fext=fext,
        y_u=y_u,
        y_fint=y_fint,
        coarse_features=coarse_features,
        edge_index_coarse=edge_index_coarse,
        edge_attr_coarse=edge_attr_coarse,
        edge_index_diff=edge_index_diff,
        edge_attr_diff=edge_attr_diff,
        edge_index_aggr=edge_index_aggr,
        edge_attr_aggr=edge_attr_aggr,
        batch=batch_vec,
    )

def make_loader(
    root_dir: Union[str, Path],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    pattern: str = "*.pt",
) -> DataLoader:
    ds = PtDictFolderDataset(root_dir=root_dir, pattern=pattern)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_pt_dicts,
        persistent_workers=(num_workers > 0),
    )