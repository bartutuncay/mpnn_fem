# Evaluate Multi-Scale Model with TopK Pooling

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
from torch_geometric.nn import pool, TopKPooling
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree, Delaunay
from datasets_src.dataloader_stress import make_loader
import os
import time
import argparse
import torch_geometric.typing as pyg_typing

torch.set_default_dtype(torch.float32)
device = torch.device('cpu')

from copy import deepcopy
from pathlib import Path

parser = argparse.ArgumentParser(description="dataset path")
parser.add_argument("dataset", type=str, help="define dataset: use 3-letter abbreviation")
parser.add_argument("test_dataset", type=str, help="define testing dataset: use 3-letter abbreviation")
args = parser.parse_args()
sim_dataset = args.dataset
test_dataset = args.test_dataset
support = 'cantilever' if sim_dataset[0] == 'c' else 'var_bc'
geom = 'regular' if sim_dataset[1] == 'r' else 'warped'
loads = 'uniform' if sim_dataset[2] == 'u' else 'non_uniform'
data_dir = f'{support}/{geom}/{loads}'
support = 'cantilever' if test_dataset[0] == 'c' else 'var_bc'
geom = 'regular' if test_dataset[1] == 'r' else 'warped'
loads = 'uniform' if test_dataset[2] == 'u' else 'non_uniform'
test_data_dir = f'{support}/{geom}/{loads}'

class GEN_TopK(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,layers_coarse,latent_dim, out_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = layers
        self.layers_coarse = layers_coarse

        self.proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')

        self.fine_layers_1 = ModuleList([
            GENConv(latent_dim*2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        
        self.fine_layers_2 = ModuleList([
            GENConv(latent_dim, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        
        self.pool = TopKPooling(in_channels=latent_dim*2,ratio=0.2,min_score=None)
        
        self.coarse_layers = ModuleList([
            GENConv(latent_dim*2, latent_dim*2, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers_coarse)])

        self.unpool_proj = MLP(in_channels=latent_dim * 2,hidden_channels=latent_dim,
            out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        
        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='tanh',plain_last=True)
        self.inv_proj_f = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='tanh',plain_last=True)
    
    def forward(self,x,edge_index,edge_attr,batch):
        x = self.proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # fine pass (initial)
        for conv in self.fine_layers_1:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index=edge_index, edge_attr=edge_attr)
            x = x + m
        
        x_global = pool.global_mean_pool(x, batch)         # [B, latent]
        x_expanded = x_global[batch]                       # [N, latent]
        x2 = torch.cat([x, x_expanded], dim=1)    
        x_c, ei_c, ea_c, batch_c, perm, score = self.pool(
            x2, edge_index, edge_attr=edge_attr, batch=batch)

        for conv in self.coarse_layers:
            m_c = conv(x_c, edge_index=ei_c, edge_attr=ea_c)
            x_c = x_c + m_c

        x2_up = x2.new_zeros((x2.size(0), x_c.size(1)))      # [N_fine, 2*latent]
        x2_up[perm] = x_c
        x = x + self.unpool_proj(x2_up) 

        # fine pass (final)
        for conv in self.fine_layers_2:
            m = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = x + m
        x_f = self.inv_proj_f(x)
        x_u = self.inv_proj(x)
        return x_f, x_u


model = GEN_TopK(in_channels=10,edge_in=3,layers=4,layers_coarse=12,latent_dim=128,out_channels=3).to(device)
alias = f'multiscale_topk_{sim_dataset}'
ckpt_path = Path(f"training/{alias}/weights.pt")
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

class StandardScaler:
    def __init__(self, node_stats_dict: dict, device):
        self.device = device
        self.m_x = node_stats_dict['x']["mean"].to(device)
        self.s_x = node_stats_dict['x']["std"].to(device)
        self.m_u = node_stats_dict['y_u']["mean"].to(device)
        self.s_u = node_stats_dict['y_u']["std"].to(device)
        self.m_f = node_stats_dict['y_fint']["mean"].to(device)
        self.s_f = node_stats_dict['y_fint']["std"].to(device)

    def norm_inputs_targets(self, batch):
        x = batch.x[:, :6].float()
        x_force = (x[:, 3:6] - self.m_x) / self.s_x
        x_norm = torch.cat([x[:, :3], x_force], dim=1)

        y_u = ((batch.y_u.float() - self.m_u) / self.s_u)
        y_f = ((batch.y_fint.float() - self.m_f) / self.s_f)

        return x_norm, y_u, y_f

    def inv_u(self, u_norm):
        return u_norm * self.s_u + self.m_u
    
    def inv_f(self, f_norm):
        return f_norm * self.s_f + self.m_f

train_loader = make_loader(f'datasets/{test_data_dir}/test', batch_size=1, shuffle=True, num_workers=4)
norm_stats = torch.load(f"datasets/{test_data_dir}/norm/train_norm_stats.pt",weights_only=False)
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
        batch = d.to(device)
        x_in, y_u, y_fint = scaler.norm_inputs_targets(batch)
        x_in = torch.cat([batch.l_cen,batch.l_dist,x_in],dim=-1)

        pred_f, pred = model(
            x_in,
            batch.edge_index,
            batch.edge_attr,
            batch.batch)


        d.y_u = scaler.inv_u(y_u)
        d.pred_u = scaler.inv_u(pred)
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

# Save to a new file
os.makedirs(f'test/{alias}',exist_ok=True)
save_path = f"test/{alias}/preds_{alias}_{test_dataset}.pt"
torch.save(out_samples, save_path)
print(f"Saved {len(out_samples)} samples with predictions to {save_path}")