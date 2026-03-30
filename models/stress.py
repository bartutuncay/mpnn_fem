# Stress Field Prediction Model
## learn transformation from gradient of displacement field u'[3] ==> stress [3,3]

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
from torch_geometric.nn import pool
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree, Delaunay
from forward_src.dataloader_stress import make_loader
import os
import argparse
import torch_geometric.typing as pyg_typing

pyg_typing.WITH_INDEX_SORT = False

torch.set_default_dtype(torch.float32)
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="dataset path")
parser.add_argument("dataset", type=str, help="define dataset: use 3-letter abbreviation")
args = parser.parse_args()
sim_dataset = args.dataset
support = 'cantilever' if sim_dataset[0] == 'c' else 'var_bc'
geom = 'regular' if sim_dataset[1] == 'r' else 'warped'
loads = 'uniform' if sim_dataset[2] == 'u' else 'non_uniform'
data_dir = f'{support}/{geom}/{loads}'

## pipeline
# get all original edge lengths and node indices
# recalculate edge lengths with displacement field
# use edge length ratio as edge value
# predict node values (stresses)

# node attrs: u, BC, F
# edge attrs: L0, 

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
        self.m_s = node_stats_dict['y_str']['mean'].to(device)
        self.s_s = node_stats_dict['y_str']['std'].to(device)
        self.m_s = self.m_s.view(3, 3)
        self.s_s = self.s_s.view(3, 3)

    def norm_inputs_targets(self, batch):
        x = batch.x[:, :6].float()
        x_force = (x[:, 3:6] - self.m_x) / self.s_x
        x_norm = torch.cat([x[:, :3], x_force], dim=1)

        #y_u = ((batch.y_u.float() - self.m_u) / self.s_u)
        y_s = ((batch.y_str.float() - self.m_s) / self.s_s)

        return x_norm, y_s

    def inv_u(self, u_norm):
        return u_norm * self.s_u + self.m_u

class GNN(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,latent_dim, out_channels):
        super().__init__()

        self.node_proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        self.layers = layers

        self.encoding_layers = ModuleList([
            GENConv(latent_dim * 2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])

        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='tanh',plain_last=True)
    
    def forward(self,x,edge_attr_str,edge_index, batch):
        x = self.node_proj(x)
        edge_attr_str = self.edge_proj(edge_attr_str)
        
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_attr=edge_attr_str)
            x = x + m
        x = self.inv_proj(x)
        return x

def strain_proxies(u, pos, edge_index, edge_attr, eps=1e-8):
    src, dst = edge_index[0], edge_index[1]

    # initial and deformed edge vectors
    edge0 = edge_attr.float()
    pos_deformed = pos + u
    edge1 = (pos_deformed[dst] - pos_deformed[src]).float()

    # relative displacement across edge
    edge_def = (u[dst] - u[src]).float()

    # initial and deformed lengths
    edge_len0 = torch.linalg.norm(edge0, dim=1, keepdim=True)
    edge_len1 = torch.linalg.norm(edge1, dim=1, keepdim=True)
    edge_dir = edge0 / edge_len0.clamp_min(eps)

    # stretch ratio
    lam = edge_len1 / edge_len0.clamp_min(eps)

    # scalar axial strain
    eps_axial = (edge_len1 - edge_len0) / edge_len0.clamp_min(eps)

    # shear component of relative displacement
    axial_proj = (edge_def * edge_dir).sum(dim=1, keepdim=True) * edge_dir
    shear_vec = edge_def - axial_proj

    # scalar shear strain
    eps_shear = torch.linalg.norm(shear_vec, dim=1, keepdim=True) / edge_len0.clamp_min(eps)

    return eps_axial, eps_shear, lam, edge1

def compute_losses(batch, pred_eps, eps_str):
    eps = 1e-8
    gid = batch.batch
    num_graphs = int(gid.max()) + 1
    #bc = batch.x[:,:3]
    pred_eps = pred_eps.view(-1, 3, 3)
    e2 = (pred_eps - eps_str).pow(2).sum(dim=1)
    y2 = (eps_str).pow(2).sum(dim=1)

    e2_g = scatter_add(e2, gid, dim=0, dim_size=num_graphs)
    y2_g = scatter_add(y2, gid, dim=0, dim_size=num_graphs)
    Lg = e2_g / (y2_g + eps)
    L_u = Lg.mean()
    
    loss = L_u

    return loss, {'L_u': float(L_u.detach())}

## Training Loop
model = GNN(in_channels=6,edge_in=6,layers=12,latent_dim=128,out_channels=9).to(device)

alias = f"stress_prediction_{sim_dataset}"
print(alias)
os.makedirs(f"training/stress_prediction/{alias}",exist_ok=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=300)

def run_epoch(loader, train=True):
    model.train(train)
    total, last_loss_dict = 0.0, {}
    i=0
    for batch in loader:
        batch = batch.to(device)
        x_in, y_u = scaler.norm_inputs_targets(batch)
        eps_axial, eps_shear, lam, edge_attr_str = strain_proxies(batch.y_u, batch.pos, batch.edge_index, batch.edge_attr)
        edge_attr_stress = torch.cat([eps_axial, eps_shear, lam, edge_attr_str],dim=1)
        opt.zero_grad(set_to_none=True)

        pred = model(
            x_in,
            edge_attr_stress,
            batch.edge_index,
            batch.batch)

        loss, loss_dict = compute_losses(batch, pred, y_u)
        last_loss_dict = loss_dict

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.detach())
    steps = max(1, len(loader))
    return total / steps, last_loss_dict

EPOCHS = 10000
best_val = float("inf")
best_state = None
loss_records = []

#torch.cuda.empty_cache()
train_loader = make_loader(f'datasets/{data_dir}', batch_size=1, shuffle=True, num_workers=4)
norm_stats = torch.load(f"datasets/{data_dir}/norm/train_norm_stats.pt",weights_only=False)
scaler = StandardScaler(norm_stats,device)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_ld = run_epoch(train_loader, train=True)

    scheduler.step(train_loss)
    loss_records.append({"epoch": epoch, "train_loss": train_loss})

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | total loss: {train_loss:.6f} | losses: {train_ld}")
    if epoch == 1 or epoch % 100 == 0:
        torch.save(model.state_dict(), f"training/stress_prediction/{alias}/weights_{epoch}.pt")

print("Best val:", best_val)

pd.DataFrame(loss_records).to_csv(f"training/stress_prediction/{alias}/losses.csv", index=False)
