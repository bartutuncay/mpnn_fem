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
from torch_geometric.nn import MLP, GENConv
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
        x_force = (x[:, 3:6] - self.m_x) / self.s_x
        x_norm = torch.cat([x[:, :3], x_force], dim=1)

        y_u = ((batch.y_u.float() - self.m_u) / self.s_u)
        y_f = ((batch.y_fint.float() - self.m_f) / self.s_f)

        return x_norm, y_u, y_f

    def inv_u(self, u_norm):
        return u_norm * self.s_u + self.m_u
    
    def inv_f(self, f_norm):
        return f_norm * self.s_f + self.m_f


class GEN_Multiscale(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,layers_coarse,latent_dim, out_channels):
        super().__init__()
        self.proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        
        # coarse projection: takes in zeros, outputs latent*2, shares nodes with fine graph
        self.coarse_proj = MLP(in_channels=1,hidden_channels=latent_dim*2,out_channels=latent_dim*2,num_layers=2,act='tanh',norm='layer')
        
        self.coarse_edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')

        self.aggr_edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        self.diff_edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='tanh',norm='layer')
        
        self.layers = layers

        self.fine_layers_1 = ModuleList([
            GENConv(latent_dim*2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        
        self.fine_layers_2 = ModuleList([
            GENConv(latent_dim, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        
        self.coarse_layers = ModuleList([
            GENConv(latent_dim*2, latent_dim*2, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers_coarse)])
        
        self.aggr_conv = GENConv(latent_dim, latent_dim*2, norm='layer',msg_norm=True,edge_dim=latent_dim)
        self.diff_conv = GENConv(latent_dim*2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)

        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='tanh',plain_last=True)
        self.inv_proj_f = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='tanh',plain_last=True)
    
    def forward(self,x,edge_index,edge_attr,edge_index_coarse,edge_attr_coarse,edge_index_diff,edge_attr_diff,edge_index_aggr,edge_attr_aggr,batch):
        x = self.proj(x)
        #x_c = self.coarse_proj(x_c)
        edge_attr = self.edge_proj(edge_attr)
        edge_attr_coarse = self.coarse_edge_proj(edge_attr_coarse)
        edge_attr_diff = self.diff_edge_proj(edge_attr_diff)
        edge_attr_aggr = self.aggr_edge_proj(edge_attr_aggr)
        # fine pass (initial)
        for conv in self.fine_layers_1:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index=edge_index, edge_attr=edge_attr)
            x = x + m
        # fine --aggregate--> coarse
        # x: fine nodes, x_c: coarse nodes
        m_a = self.aggr_conv(x, edge_index=edge_index_aggr, edge_attr=edge_attr_aggr)
        x_c = m_a
        # coarse pass
        for conv in self.coarse_layers:
            m_c = conv(x_c, edge_index=edge_index_coarse, edge_attr=edge_attr_coarse)
            x_c = x_c + m_c
        # coarse --broadcast--> fine
        m_d = self.diff_conv(x_c, edge_index=edge_index_diff, edge_attr=edge_attr_diff)
        x = x + m_d
        # fine pass (final)
        for conv in self.fine_layers_2:
            m = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = x + m
        x_f = self.inv_proj_f(x)
        x_u = self.inv_proj(x)
        return x_f, x_u

def dirichlet_loss(x, edge_index):
    row, col = edge_index  # [E], [E]
    diff = x[row] - x[col] # [E, C]
    sq = (diff * diff).sum(dim=-1)
    w = torch.ones_like(sq)
    E = 0.5 * (w * sq).sum()
    denom = (w.sum().clamp_min(1.0))
    
    return E / denom

def compute_losses(batch, pred_u, pred_f, y_u, y_f):
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

    fext   = batch.x[:,3:6]
    pred_f = pred_f*(1-bc) + 0*bc
    L_fint = F.mse_loss(pred_f, y_f)
    L_eq   = F.mse_loss(((torch.ones_like(bc)-bc)*(pred_f - fext)).sum(),torch.zeros((),device=bc.device))

    dir_loss = dirichlet_loss(pred_u,batch.edge_index)
    
    loss = L_u + 2*L_u_bc + 0.2*L_fint + 2e-7*L_eq + 0.8*dir_loss

    return loss, {'L_u': float(L_u.detach()), 'L_fint':float(L_fint.detach()), 'L_eq':float(L_eq.detach()),
                  'L_dir':float(dir_loss.detach()), 'L_u_bc':float(L_u_bc.detach())}

## Training Loop
model = GEN_Multiscale(in_channels=10,edge_in=3,layers=4,layers_coarse=8,latent_dim=128,out_channels=3).to(device)

alias = "multiscale_tanh_pe_8mp_bcs_warped_non_uniform"
print(alias)
os.makedirs(f"../../../scratch/btuncay/gnn/ablation_models/3_loads_bcs/{alias}",exist_ok=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=300)

def run_epoch(loader, train=True):
    model.train(train)
    total, last_loss_dict = 0.0, {}
    i=0
    for batch in loader:
        t1 = time.time()
        batch = batch.to(device)
        x_in, y_u, y_fint = scaler.norm_inputs_targets(batch)
        t2 = time.time()
        opt.zero_grad(set_to_none=True)
        #print(batch)
        x_in = torch.cat([batch.l_cen,batch.l_dist,x_in],dim=-1)

        pred_f, pred_u = model(
            x_in,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_index_coarse,
            batch.edge_attr_coarse,
            batch.edge_index_diff,
            batch.edge_attr_diff,
            batch.edge_index_aggr,
            batch.edge_attr_aggr,
            batch.batch)
        t3 = time.time()
        
        #dir_loss = dirichlet_loss(pred,batch.edge_index)
        loss, loss_dict = compute_losses(batch, pred_u, pred_f, y_u, y_fint)
        #loss=loss+(0.1*dir_loss)
        last_loss_dict = loss_dict  # keep something to return

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        #total += loss.item()
        total += float(loss.detach())
        t4=time.time()
        #print('load: ',t2-t1,'s, forward pass: ',t3-t2,'s, backpropagation: ',t4-t3,'s')
        i+=1
        if i >= 100:
            break
    #steps = i
    steps = max(1, len(loader))
    return total / steps, last_loss_dict


# Comet - logging
experiment = start(api_key="7VD3oulgQdsrnNz60JDDhY86O",project_name="mpnn-fem",workspace="btuncay")

hyper_params = {'learning_rate': 5e-4,'steps': 10000,'batch_size':1}
experiment.log_parameters(hyper_params)
log_model(experiment,model=model,model_name=alias)

EPOCHS = 10000
best_val = float("inf")
best_state = None
loss_records = []

#torch.cuda.empty_cache()
train_loader = make_loader('../../../scratch/btuncay/gnn/ablation_datasets/var_bc/warped/non_uniform/train', batch_size=1, shuffle=True, num_workers=4)
norm_stats = torch.load(f"../../../scratch/btuncay/gnn/ablation_datasets/var_bc/warped/non_uniform/norm/train_norm_stats.pt",weights_only=False)
scaler = StandardScaler(norm_stats,device)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_ld = run_epoch(train_loader, train=True)

    scheduler.step(train_loss)
    loss_records.append({"epoch": epoch, "train_loss": train_loss})

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | total loss: {train_loss:.6f} | losses: {train_ld}")
    if epoch == 1 or epoch % 200 == 0:
        torch.save(model.state_dict(), f"../../../scratch/btuncay/gnn/ablation_models/3_loads_bcs/{alias}/weights_{epoch}.pt")

print("Best val:", best_val)

pd.DataFrame(loss_records).to_csv(f"../../../scratch/btuncay/gnn/ablation_models/3_loads_bcs/{alias}/losses.csv", index=False)

## NEW:
# generalization study
# physics loss
# variable cantilever fix position
# high weighted losses at supports
# distance to supports
# rotate shape along x/y/z

## TODO:
# save node-wise element stresses
# subgraph mean pooling
# edge attrs in spherical coordinates
# torsion/moment/shear problem
# holes/reduced section beams
# distance to loads --> path from node to load?
# circular activation function
# results description - what worked, what didn't, why
# attention based pooling
# sensitivity analysis
# EGNN

# equivariant positions: se3
# neural operator
# inverted edge weights