## PyG Graph with Mesh Nodes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from comet_ml import start
from comet_ml.integration.pytorch import log_model
import glob
from torch_scatter import scatter_add
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
import pandas as pd
import torch.nn.functional as F
from torch.nn import ModuleList
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, GENConv, GCNConv
import torch.nn as nn
from torch_geometric.nn import pool
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree, Delaunay
from gnn_2026.datasets_src.dataloader import make_loader
import os
import time
import torch_geometric.typing as pyg_typing

torch.set_default_dtype(torch.float32)
device = torch.device('cpu')

from copy import deepcopy
from pathlib import Path

class GNN(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,latent_dim, out_channels):
        super().__init__()

        self.node_proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.layers = layers

        self.encoding_layers = ModuleList([
            GENConv(latent_dim * 2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])

        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='leaky_relu',plain_last=True)
    
    def forward(self,x,edge_index,edge_attr, batch):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_attr=edge_attr)
            x = x + m

        x = self.inv_proj(x)
        return x

model = GNN(in_channels=46,edge_in=3,layers=12,latent_dim=128,out_channels=3).to(device)
alias = 'baseline_cantilever_regular_uniform'
ckpt_path = Path(f"../../../scratch/btuncay/gnn/ablation_models/1_simple_dataset/{alias}/weights_8400.pt")
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

class StandardScaler:
    def __init__(self, node_stats_dict: dict, device):
        self.device = device
        # move stats once
        self.m_x = node_stats_dict['x']["mean"].to(device)
        self.s_x = node_stats_dict['x']["std"].to(device)
        self.m_u = node_stats_dict['y_u']["mean"].to(device)
        self.s_u = node_stats_dict['y_u']["std"].to(device)
        self.m_f = node_stats_dict['y_fint']["mean"].to(device)
        self.s_f = node_stats_dict['y_fint']["std"].to(device)

    def norm_inputs_targets(self, batch):
        x = batch.x[:, :6].float()
        x_pe = batch.x[:, 6:].float()
        x_force = (x[:, 3:6] - self.m_x) / self.s_x
        x_norm = torch.cat([x[:, :3], x_force, x_pe], dim=1)

        y_u = ((batch.y_u.float() - self.m_u) / self.s_u)
        # y_f = ((batch.y_fint.float() - self.m_f) / self.s_f)

        return x_norm, y_u

    def inv_u(self, u_norm):
        return u_norm * self.s_u + self.m_u

def dirichlet_loss(x, edge_index):
    row, col = edge_index  # [E], [E]
    diff = x[row] - x[col] # [E, C]
    sq = (diff * diff).sum(dim=-1)
    w = torch.ones_like(sq)
    E = 0.5 * (w * sq).sum()
    denom = (w.sum().clamp_min(1.0))
    
    return E / denom

#train_set, val_set = split_dataset(dataset_val, val_ratio=0.1)
train_loader = make_loader('../../../scratch/btuncay/gnn/ablation_datasets/cantilever/regular/uniform/val', batch_size=1, shuffle=True, num_workers=4)
norm_stats = torch.load(f"../../../scratch/btuncay/gnn/ablation_datasets/cantilever/regular/uniform/norm/train_norm_stats.pt",weights_only=False)
scaler = StandardScaler(norm_stats,device)

c = 0
out_samples = []
with torch.no_grad():
    for data in train_loader:
        t1 = time.time()
        d = deepcopy(data)
        d = d.to(device)
        x = d.x
        edge_index = d.edge_index
        edge_attr = d.edge_attr
        #edge_index = data.edge_index_dict.values()
        #edge_attr = data.edge_attr_dict.values()
        batch = d.to(device)
        x_in, y_u = scaler.norm_inputs_targets(batch)

        pred = model(
            x_in,
            batch.edge_index,
            batch.edge_attr,
            batch.batch)

        # Detach and move predictions to CPU for saving

        d.y_u = scaler.inv_u(y_u)
        d.pred_u = scaler.inv_u(pred)
        # Move the data structure back to CPU before attaching CPU tensors

        # Attach predictions into the dictionary
        t2 = time.time()

        #METRICS
        #time
        dur = t2-t1
        d.dur = dur

        #MSE
        pos_diff = np.linalg.norm(y_u,axis=-1)
        pos_diff_pred = np.linalg.norm(pred,axis=-1)
        mask = pos_diff > 1e-6
        pos_delta = np.mean(np.abs(pos_diff_pred[mask] - pos_diff[mask]) / pos_diff[mask])
        d.mae = pos_delta

        #physicality: dirichlet loss + boundary
        bc = batch.x[:,:3]
        pred_mask = pred > 0.999
        pred_u_bc = pred[pred_mask].mean() #0 if bcs stay in place
        dir_loss = dirichlet_loss(pred,batch.edge_index)
        physical_score = 1-(pred_u_bc)
        d.phy = physical_score

        sample = {
            "x": d.x.cpu(),
            "edge_index": d.edge_index.cpu(),
            "edge_attr": d.edge_attr.cpu() if d.edge_attr is not None else None,
            "y_u": d.y_u.cpu(),
            "pred_u": d.pred_u.cpu(),
            "dur": float(dur),
            "mae": float(pos_delta),
            "phy": float(physical_score),
            "batch": d.batch.cpu() if hasattr(d, "batch") and d.batch is not None else None,
        }

        out_samples.append(sample)
        print(d.dur,d.mae)
        c+=1
        if c >= 100:
            break
# Save to a new file
save_path = f"../../../scratch/btuncay/gnn/ablation_models/1_simple_dataset/{alias}/preds_0328_{alias}.pt"
torch.save(out_samples, save_path)
print(f"Saved {len(out_samples)} samples with predictions to {save_path}")