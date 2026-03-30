# Save training/test data
# Define index for batch operation, including training/testing split

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm
import itertools
import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
import pandas as pd
import torch.nn.functional as F
from torch.nn import ModuleList
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, GENConv, GCNConv
import torch.nn as nn
from collections import deque
from torch_geometric.nn import pool
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree, Delaunay
import os
import argparse
import torch_geometric.typing as pyg_typing
pyg_typing.WITH_INDEX_SORT = False

torch.set_default_dtype(torch.float32)
device = torch.device('cpu')

## Define parameters: mesh connectivity type,
mesh_conn = 'element'#, 'knn'

parser = argparse.ArgumentParser(description="index, dataset")
parser.add_argument("value", type=int, help="integer to process")
parser.add_argument("dataset", type=str, help="define dataset: use 3-letter abbreviation")
parser.add_argument("train", type=str, choices=("train", "test"), help="split to write")
args = parser.parse_args()
sim_idx = args.value
sim_dataset = args.dataset
sim_train = args.train
np.random.seed(sim_idx)

support = 'cantilever' if sim_dataset[0] == 'c' else 'var_bc'
geom = 'regular' if sim_dataset[1] == 'r' else 'warped'
loads = 'uniform' if sim_dataset[2] == 'u' else 'non_uniform'
data_dir = f'{support}/{geom}/{loads}'
train = sim_train

def mesh_edges_from_conn(conn: torch.Tensor)->torch.Tensor:
    conn = conn.long() # original connectivity matrix
    hex_edges = torch.tensor([
        [0,1], [1,2], [2,3], [3,0],     # bottom face
        [4,5], [5,6], [6,7], [7,4],     # top face
        [0,4], [1,5], [2,6], [3,7],     # vertical edges
    ])
    pairs = conn[:,hex_edges]           # [E, 12, 2]
    pairs = pairs.reshape(-1, 2)
    pairs = torch.unique(torch.sort(pairs, dim=1).values, dim=0).T
    edge_index = pairs
    return edge_index

def mesh_edges_nearest(nodes: torch.Tensor) -> torch.Tensor:
    # build mesh connections from nearest neighbors
    num_nodes = int(nodes.size(0))
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=nodes.device)

    coords = nodes.detach().cpu().numpy()
    k = min(8, num_nodes - 1)
    _, nn_idx = KDTree(coords).query(coords, k=k + 1, workers=-1)
    neighbors = nn_idx[:, 1:].reshape(-1)  # only consider neighbors
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), k)

    pairs = torch.from_numpy(np.stack([src, neighbors], axis=1))
    pairs = torch.sort(pairs, dim=1).values
    pairs = torch.unique(pairs, dim=0)
    return pairs.t().to(nodes.device)

# Add averaged element stresses to nodes
def get_nodal_stresses(conn:torch.Tensor,nodes:torch.Tensor,stress:torch.Tensor)->torch.Tensor:
    # data: [E,3,3]
    E, nen = conn.shape
    N = nodes.shape[0]
    conn = conn.long()
    # get mean of stresses from each node
    T = stress.shape[0]
    s = stress[:,:,None,:,:].expand(T,E,nen,3,3)
    idx = conn.reshape(-1)
    val = s.reshape(T,E*nen,3,3)
    node_sum = torch.zeros(T, N, 3, 3, device=stress.device, dtype=stress.dtype)
    node_sum.index_add_(1, idx, val)
    ones = torch.ones(idx.shape[0], device=stress.device, dtype=stress.dtype)
    node_count = torch.zeros(N, device=stress.device, dtype=stress.dtype)
    node_count.index_add_(0, idx, ones)
    node_count = node_count.clamp_min(1.0)
    node_stress = node_sum / node_count[None, :, None, None]
    return node_stress

# Convert edge attributes to spherical coordinates: (x,y,z) ==> (r,θ,ϕ)
def to_spherical_coords(edge_attrs:torch.Tensor)->torch.Tensor:
    r = torch.linalg.norm(edge_attrs,dim=1)
    theta = torch.acos((edge_attrs[:,2]/(r+1e-8)).clamp(-1.0,1.0))
    phi = torch.atan2(edge_attrs[:,1],edge_attrs[:,0])
    phi = (phi + 2*torch.pi) % (2*torch.pi)
    return torch.stack([r,theta,phi],dim=1)

## Rigid transform along one axis
perms = [(0, 1, 2),(0, 2, 1),
    (1, 0, 2),(1, 2, 0),
    (2, 0, 1),(2, 1, 0)]
axes = random.choice(perms)

def data_to_graph(path:str,device):
    data = Data()
    simdata = torch.load(path,weights_only=False)
    conn = simdata["elements"]
    conn = conn.cpu().numpy() if isinstance(conn, torch.Tensor) else np.asarray(conn)
    nodes = simdata['nodes']
    
    # mesh node properties: positions, forces, BC, dirichlet displacement
    data.pos = nodes[:,axes] # [N,3]
    bc = simdata['boundary']+1e-6 #boundary encoding tolerance to avoid zeros
    data.bc = bc[:,axes]

    f_ext = simdata['ext_forces'] #forces in timeseries format [T,N,3]
    data.f_ext = f_ext[:,:,axes]

    # target properties
    # mesh: displacement over time
    f_ts = simdata['forces'] #forces in timeseries format [T,N,3]
    data.f_ts = f_ts[:,:,axes]
    u_ts = simdata['u_history']
    data.u_ts = u_ts[:,:,axes]

    if mesh_conn == 'element':
        ei = mesh_edges_from_conn(simdata["elements"]) # [2,N]
    if mesh_conn == 'knn':
        ei = mesh_edges_nearest(nodes)
    data.y_str = get_nodal_stresses(simdata['elements'],nodes,simdata['stress_history'])
    src, dst = ei
    disp = (nodes[dst] - nodes[src]).float()
    data.edge_index = ei
    data.edge_attr = disp
    data.edge_attr_sph = to_spherical_coords(disp)
    data = data.to(device)
    return data

def generate_dataset(data_dir:str):
    device = torch.device('cpu')
    file = data_dir
    samples = []
    data = data_to_graph(file,device)
    samples.append(data)
    print(file)

    print(len(samples))
    return samples

dataset = generate_dataset(f"datasets/{data_dir}/sim_{sim_idx}.pt")

ea = dataset[0].edge_attr
ea_sph = dataset[0].edge_attr_sph
edge_index = dataset[0].edge_index
edge_index2,ea_sph = to_undirected(edge_index,edge_attr=ea_sph)
edge_index,ea = to_undirected(edge_index,edge_attr=ea)
dataset[0].edge_attr = ea
dataset[0].edge_attr_sph = ea_sph
dataset[0].edge_index = edge_index

# topological distance to boundary nodes 
@torch.no_grad()
def graph_distance(edge_index: torch.Tensor,sources: torch.Tensor,
                            num_nodes: int)->torch.Tensor:
    device = edge_index.device
    unreachable = 1e9
    dist = torch.full((num_nodes,), unreachable, device=device)
    
    if sources.numel() == 0:
        return dist

    # mask visited nodes
    visited = torch.zeros((num_nodes,), dtype=torch.bool, device=device)
    frontier = torch.zeros((num_nodes,), dtype=torch.bool, device=device)
    visited[sources] = True
    frontier[sources] = True
    dist[sources] = 0.0

    row, col = edge_index[0], edge_index[1]
    step = 0.0

    while frontier.any():
        step += 1.0

        # edges whose source is in frontier
        hit = frontier[row]     # [E] bool
        nbrs = col[hit]         # neighbors reached this layer

        if nbrs.numel() == 0:
            break

        # next frontier==unique neighbors not visited
        next_frontier = torch.zeros_like(frontier)
        next_frontier[nbrs] = True
        next_frontier &= ~visited

        if not next_frontier.any():
            break
        dist[next_frontier] = step
        visited |= next_frontier
        frontier = next_frontier
    return dist

@torch.no_grad()
def bc_closeness_3ch(data,gamma:float=1.0,eps:float=1e-8):
    edge_index = data.edge_index
    N = data.num_nodes

    # source sets per DOF
    bc = data.bc  # [N,3]: 1e-5 to 1
    src_x = torch.where(bc[:,0] > 1e-5)[0]
    src_y = torch.where(bc[:,1] > 1e-5)[0]
    src_z = torch.where(bc[:,2] > 1e-5)[0]

    dists = []
    for src in (src_x, src_y, src_z):
        d = graph_distance(edge_index, src, N)  # [N]
        # if no BC sources: --> all zeros
        if (d >= 1e8).all():
            s = torch.zeros((N,), device=edge_index.device)
        else:
            d_max = torch.max(d[d < 1e8])
            s = 1.0 - d / (d_max + eps)
            s = torch.exp(-0.02*d) #exponential decrease
            s = torch.clamp(s, 0.0, 1.0)
            if gamma != 1.0:
                s = s.pow(gamma)
        dists.append(s)

    return torch.stack(dists, dim=-1) # [N,3]

@torch.no_grad()
def load_closeness_3ch(data,load,gamma:float=1.0,eps:float = 1e-8):
    edge_index = data.edge_index
    N = data.num_nodes

    # source sets per DOF
    #load = data.fext  # [N,3]: float
    src_x = torch.where(load[:,0].abs() > 1e-2)[0]
    src_y = torch.where(load[:,1].abs() > 1e-2)[0]
    src_z = torch.where(load[:,2].abs() > 1e-2)[0]

    dists = []
    for src in (src_x, src_y, src_z):
        d = graph_distance(edge_index, src, N)  # [N]
        d_max = torch.max(d[d < 1e8])
        s = 1.0 - d / (d_max + eps)
        s = torch.exp(-0.02*d) #exponential decrease
        s = torch.clamp(s, 0.0, 1.0)
        if gamma != 1.0:
            s = s.pow(gamma)
        dists.append(s)

    return torch.stack(dists, dim=-1)  # [N,3]

def closeness_centrality(graph):
    # Calculate closeness centrality with breadth-first search
    ## Note: disable function if memory is limited!
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    device = edge_index.device

    row, col = edge_index

    # adjacency list
    adj = [[] for _ in range(num_nodes)]

    row = row.detach().cpu().tolist()
    col = col.detach().cpu().tolist()

    for u, v in zip(row, col):
        adj[u].append(v)

    closeness = torch.zeros(num_nodes, dtype=torch.float)

    for src in range(num_nodes):
        dist = [-1] * num_nodes
        dist[src] = 0

        q = deque([src])

        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)

        reachable_dists = [d for d in dist if d >= 0]
        n_reachable = len(reachable_dists) - 1
        sum_dist = sum(reachable_dists)

        if sum_dist > 0:
            c = n_reachable / sum_dist
            c *= n_reachable / (num_nodes - 1)
            closeness[src] = c
        else:
            closeness[src] = 0.0

    return closeness.to(device).view(-1, 1)

def build_timestep_features(data,dtype=torch.float32):
    datalist = []
    len_t = data.f_ext.shape[0]
    pe = T.AddLaplacianEigenvectorPE(k=24,attr_name='laplacian',is_undirected=True)
    pe_rw = T.AddRandomWalkPE(walk_length=16,attr_name='rwse')
    data_pe = pe(data)
    data_pe = pe_rw(data_pe)
    lap_enc = data_pe.laplacian.to(dtype)
    rwse_enc = torch.log(data_pe.rwse + 1e-6).to(dtype)
    #bc    = data.bc
    bc = bc_closeness_3ch(data)
    cen = closeness_centrality(data)

    static_kwargs = {}
    for k in ["pos", "edge_index", "edge_attr", "faces", "num_nodes", "edge_attr_sph"]:
        if hasattr(data_pe, k):
            static_kwargs[k] = getattr(data_pe, k)

    for t in range(len_t):
        f_ext_t = torch.as_tensor(data_pe.f_ext[t], dtype=dtype)
        x = torch.cat([bc, f_ext_t, lap_enc, rwse_enc], dim=-1)
        d = Data(**static_kwargs)
        load_dist = load_closeness_3ch(d,f_ext_t)
        d.x = x
        d.fext = f_ext_t
        d.y_u = data_pe.u_ts[t].to(dtype)
        d.y_fint = data_pe.f_ts[t].to(dtype)
        d.y_str = data_pe.y_str[t].to(dtype)
        d.l_dist = load_dist
        d.l_cen = cen
        datalist.append(d)

    # delete timeseries fields to save space
    del data.bc, data.u_ts, data.f_ext, data.f_int
    del data.f_ts
    return datalist

def coarsen(data):
    # sample points spatially
    pts = data.pos
    frac = 0.05                         # fraction of nodes to be preserved
    N = pts.shape[0]                    # number of total nodes
    M = max(1, int(np.round(frac * N))) # number of selected nodes
    rng = np.random.default_rng(42)
    selected_idx = np.empty(M, dtype=int)

    start = rng.integers(N)
    selected_idx[0] = start
    diff = (pts - pts[start]).cpu()
    min_dist2 = np.einsum("ij,ij->i", diff, diff)
    min_dist2[start] = 0.0 # node-node distances

    for i in range(1, M): # elimination
        idx = np.argmax(min_dist2) # select farthest node
        selected_idx[i] = idx
        diff = (pts - pts[idx]).cpu()
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[idx] = 0.0

    pts_significant = pts[selected_idx]
    tree = cKDTree(pts) # neighborhood search
    sig_idx_original = torch.from_numpy(selected_idx) # map back to original node ids

    # connect points via Delaunay triangulation
    tri = Delaunay(pts_significant)
    tets = tri.simplices
    tet_edges = np.array([[0, 1], [0, 2], [0, 3],
                          [1, 2], [1, 3], [2, 3]])
    edges = tets[:, tet_edges].reshape(-1, 2)

    # Alpha shape for connections (invalid edges deleted)
    # calculate % of given edge inside original shape by subsampling each coarse connection
    # -> for each sampled point in line get nearest neighbors & distances to neighbors
    edges = torch.tensor(edges,dtype=torch.int)
    edge_start = pts_significant[edges[:,0]]
    edge_end = pts_significant[edges[:,1]]
    num_points = 20 # number of subsampling points
    direction = edge_end - edge_start
    t = torch.linspace(0, 1, num_points, device=edge_start.device)  # distances along edge
    edge_sampled = edge_start.unsqueeze(-2) + t.unsqueeze(-1) * direction.unsqueeze(-2) # [E,N,3]
    # get distances from sample points to nearest node
    edge_dists, _ = tree.query(edge_sampled[:,:],k=2)
    edge_dists = edge_dists[:,1:num_points-1,1] # exclude self
    # calculate average distance
    query_dist,_ = tree.query(pts,k=2)
    query_dist = query_dist[:,1].mean(axis=0)
    # eliminate outliers
    dist_sigma = 1
    dist_threshold = 0.05
    bad_sample = (edge_dists > query_dist*dist_sigma)
    bad_frac = bad_sample.mean(axis=1)
    keep_line = (bad_frac <= dist_threshold)
    edges_filt = edges[keep_line]
    edges_undirected = np.sort(edges_filt, axis=1)
    edges_undirected = np.unique(edges_undirected, axis=0)

    # map coarse edge endpoints back to original node ids
    edges_original_idx = torch.as_tensor(sig_idx_original, dtype=torch.long)[
        torch.as_tensor(edges_undirected, dtype=torch.long)]  # shape [E, 2], indexes into data.pos
    
    # aggregate info
    tree_f = cKDTree(pts_significant)
    _, idx_nearest = tree_f.query(pts, k=1)
    groups = [data.x[idx_nearest == j] for j in range(len(pts_significant))]
    pooled = torch.stack([g.sum(dim=0) for g in groups]) #aggregated BCs 0-3, forces 3-5

    tree_s = cKDTree(pts[sig_idx_original]) #k-dim tree of significant points
    coarse = torch.zeros((len(pts),pooled.size(-1)+1)).to(device)
    ## aggregated node information is not utilized

    ## coarse to fine edge connections
    # features: BC, force
    # edge features: relative position to coarse node
    _, idx_broadcast = tree_s.query(pts,k=2)
    orig_idx_1 = sig_idx_original[idx_broadcast[:,0]]
    orig_idx_2 = sig_idx_original[idx_broadcast[:,1]]
    dst_1 = torch.arange(len(pts), device=orig_idx_1.device)
    dst_2 = torch.arange(len(pts), device=orig_idx_2.device)
    
    edges_1 = torch.stack([orig_idx_1, dst_1], dim=0)  # [2, N]
    edges_2 = torch.stack([orig_idx_2, dst_2], dim=0)  # [2, N]
    edges_broadcast = torch.cat([edges_1, edges_2], dim=1) 
    edges_aggregate = torch.cat([edges_2, edges_1], dim=1) 

    return edges_original_idx.T, coarse, edges_broadcast, edges_aggregate

dataset_p = [[g.to(device) for g in build_timestep_features(d.clone())] for d in dataset] # dataset_p: List[List[Data]]

for seq_proc in dataset_p:                      # seq_proc is List[Data]
    for step, data_proc in enumerate(seq_proc): # data_proc is per-timestep Data
        if step < 49:
            continue
        extra_ei, agg_feat, ei_diff, ei_aggr = coarsen(data_proc)
        src, dst = extra_ei
        extra_attr = (data_proc.pos[dst] - data_proc.pos[src]).float()
        extra_attr_sph = to_spherical_coords(extra_attr)
        src_d, dst_d = ei_diff
        diff_attr = (data_proc.pos[dst_d] - data_proc.pos[src_d]).float()
        diff_attr_sph = to_spherical_coords(diff_attr)
        src_a, dst_a = ei_aggr
        aggr_attr = (data_proc.pos[dst_a] - data_proc.pos[src_a]).float()
        aggr_attr_sph = to_spherical_coords(aggr_attr)
        ei_old = data_proc.edge_index
        ea_old = data_proc.edge_attr
        if ea_old.size(1) != extra_attr.size(1):
            extra_attr_full = torch.zeros((extra_attr.size(0), ea_old.size(1)),dtype=ea_old.dtype,device=ea_old.device)
            extra_attr_full[:, :extra_attr.size(1)] = extra_attr.to(ea_old.dtype)
        else:
            extra_attr_full = extra_attr.to(ea_old.dtype)

        data_proc.coarse_features   = torch.zeros((agg_feat.size(0), 1), device=ea_old.device, dtype=ea_old.dtype)
        data_proc.edge_index_coarse = extra_ei.to(ei_old.device)
        data_proc.edge_attr_coarse  = extra_attr_full
        data_proc.edge_attr_coarse_sph  = extra_attr_sph.to(ea_old.device, dtype=ea_old.dtype)

        data_proc.edge_index_diff   = ei_diff.to(ei_old.device)
        data_proc.edge_attr_diff    = diff_attr.to(ea_old.device, dtype=ea_old.dtype)
        data_proc.edge_attr_diff_sph = diff_attr_sph.to(ea_old.device, dtype=ea_old.dtype)

        data_proc.edge_index_aggr   = ei_aggr.to(ei_old.device)
        data_proc.edge_attr_aggr    = aggr_attr.to(ea_old.device, dtype=ea_old.dtype)
        data_proc.edge_attr_aggr_sph = aggr_attr_sph.to(ea_old.device, dtype=ea_old.dtype)
        torch.save(data_proc,f"datasets/{data_dir}/{train}/sim_{sim_idx}_{step}.pt")
