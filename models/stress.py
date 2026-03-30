## learn transformation from gradient of displacement field u'[3] ==> stress [3,3]

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
from gnn_2026.datasets_src.dataloader_stress import make_loader
import os
import time
import torch_geometric.typing as pyg_typing

pyg_typing.WITH_INDEX_SORT = False

torch.set_default_dtype(torch.float32)
device = torch.device('cuda')

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
        self.m_s = node_stats_dict['str']['mean'].to(device)
        self.s_s = node_stats_dict['str']['std'].to(device)

    def norm_inputs_targets(self, batch):
        x = batch.x[:, :6].float()
        x_pe = batch.x[:, 6:].float()
        x_force = (x[:, 3:6] - self.m_x) / self.s_x
        x_norm = torch.cat([x[:, :3], x_force, x_pe], dim=1)

        #y_u = ((batch.y_u.float() - self.m_u) / self.s_u)
        y_s = ((batch.str.float() - self.m_s) / self.s_s)

        return x_norm#, y_s

    def inv_u(self, u_norm):
        return u_norm * self.s_u + self.m_u

def split_dataset(dataset, val_ratio=0.1, shuffle=True):
    n = len(dataset)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    return train_set, val_set


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

def dirichlet_loss(x, edge_index):
    row, col = edge_index  # [E], [E]
    diff = x[row] - x[col] # [E, C]
    sq = (diff * diff).sum(dim=-1)
    w = torch.ones_like(sq)
    E = 0.5 * (w * sq).sum()
    denom = (w.sum().clamp_min(1.0))
    
    return E / denom

def strain_proxies(u, pos, edge_index, edge_attr):
    # get initial edge lengths/unit vector dir
    src, dst = edge_index[0], edge_index[1]
    #edge_norm = torch.linalg.norm(edge_aanglttr[dst]-edge_attr[src],dim=1)
    pos_deformed = pos + u
    edge_attr_str = (pos_deformed[dst] - pos_deformed[src]).float()
    edge_def = (u[dst] - u[src]).float()
    edge_dir = edge_attr / edge_attr.norm()

    # final/initial edge norm --> axial strain
    eps_axial = (edge_attr_str-edge_attr) / edge_attr.norm()
    
    # shear to axial strain ratio
    eps_shear = (edge_def-(edge_def*edge_dir)*edge_dir)/edge_attr.norm()

    # stretch ratio
    lam = torch.linalg.norm(edge_attr_str-edge_attr)/edge_dir

    return eps_axial, eps_shear, lam, edge_attr_str

def compute_losses(batch, pred_eps, eps_str):
    eps = 1e-8
    gid = batch.batch
    num_graphs = int(gid.max()) + 1
    bc = batch.x[:,:3]

    e2 = (pred_u - y_u).pow(2).sum(dim=1)
    y2 = (y_u).pow(2).sum(dim=1)

    e2_g = scatter_add(e2, gid, dim=0, dim_size=num_graphs)
    y2_g = scatter_add(y2, gid, dim=0, dim_size=num_graphs)
    Lg = e2_g / (y2_g + eps)
    L_u = Lg.mean()
    
    pred_u_bc = pred_u*bc
    y_u_bc = y_u*bc
    
    e_bc = pred_u_bc - y_u_bc
    L_u_bc = (e_bc.pow(2).sum(dim=1).mean()) / (y_u_bc.pow(2).sum(dim=1).mean() + eps)

    #fext   = batch.x[:,3:6]
    #pred_f = pred_f*(1-bc) + 0*bc
    #L_fint = F.mse_loss(pred_f, y_f)
    #L_eq   = F.mse_loss(((torch.ones_like(bc)-bc)*(pred_f - fext)).sum(),torch.zeros((),device=bc.device))
    

    return loss, {'L_u': float(L_u.detach()), 'L_u_bc':float(L_u_bc.detach())}

## Training Loop
model = GNN(in_channels=9,edge_in=12,layers=12,latent_dim=128,out_channels=9).to(device)

alias = "0307_stress_prediction"
print(alias)
os.makedirs(f"../../../scratch/btuncay/gnn/genconv_2026/{alias}",exist_ok=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=300)

def run_epoch(loader, train=True):
    model.train(train)
    total, last_loss_dict = 0.0, {}
    i=0
    for batch in loader:
        batch = batch.to(device)
        x_in, y_u = scaler.norm_inputs_targets(batch)
        edge_attr_str = strain_proxies(batch.y_u, batch.pos, batch.edge_index, batch.edge_attr)
        opt.zero_grad(set_to_none=True)

        pred = model(
            x_in,
            edge_attr_str,
            batch.edge_index,
            batch.batch)
        
        #dir_loss = dirichlet_loss(pred,batch.edge_index)
        loss, loss_dict = compute_losses(batch, pred, y_u)
        #loss=loss+(0.1*dir_loss)
        last_loss_dict = loss_dict  # keep something to return

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        #total += loss.item()
        total += float(loss.detach())
        #print('load: ',t2-t1,'s, forward pass: ',t3-t2,'s, backpropagation: ',t4-t3,'s')
        i+=1
        if i >= 100:
            break
    steps = i
    #steps = max(1, len(loader))
    return total / steps, last_loss_dict


# Comet - logging
experiment = start(api_key="7VD3oulgQdsrnNz60JDDhY86O",project_name="mpnn-fem",workspace="btuncay")

hyper_params = {'learning_rate': 1e-4,'steps': 10000,'batch_size':1}
experiment.log_parameters(hyper_params)
log_model(experiment,model=model,model_name=alias)

EPOCHS = 10000
best_val = float("inf")
best_state = None
loss_records = []

#torch.cuda.empty_cache()
train_loader = make_loader('../../../scratch/btuncay/gnn/torchfem_dataset/beam_reduced_cant_ts', batch_size=1, shuffle=True, num_workers=4)
norm_stats = torch.load(f"../../../scratch/btuncay/gnn/torchfem_dataset/beam_reduced_cant_ts/norm/train_norm_stats.pt",weights_only=False)
scaler = StandardScaler(norm_stats,device)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_ld = run_epoch(train_loader, train=True)

    scheduler.step(train_loss)
    loss_records.append({"epoch": epoch, "train_loss": train_loss})

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | total loss: {train_loss:.6f} | losses: {train_ld}")
    if epoch == 1 or epoch % 100 == 0:
        torch.save(model.state_dict(), f"../../../scratch/btuncay/gnn/genconv_2026/{alias}/weights_{epoch}.pt")

print("Best val:", best_val)

pd.DataFrame(loss_records).to_csv(f"../../../scratch/btuncay/gnn/genconv_2026/{alias}/losses.csv", index=False)