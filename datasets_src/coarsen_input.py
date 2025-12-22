# Load Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Delaunay
import time

torch.set_default_dtype(torch.float32)
dataset = torch.load("../../../scratch/btuncay/gnn/torchfem_dataset/beam_reduced/combined_50.pt",weights_only=False)

def build_laststep_features(data,dtype=torch.float32):
    ## Collapse to peak load timestep
    #print(data.keys)
    len = data['nodes'].f_ext.shape[0]
    t = int(len-1)
    # Nodes: x = [pos, bc, f_ext[-1], f_int[-1]]  (no leakage of u_ts into x)
    #pos   = data['nodes'].pos.to(dtype) #no need to include
    bc    = data['nodes'].bc
    f_ext = torch.Tensor(data['nodes'].f_ext[t]).to(dtype)
    data['nodes'].fext = torch.Tensor(data['nodes'].f_ext[t]).to(dtype)
    data['nodes'].x = torch.cat([bc, f_ext], dim=-1).to(dtype)


    # Target: nodes â†’ u_ts[-1] (3D)
    data['nodes'].y_u = data['nodes'].u_ts[t].to(dtype)
    data['nodes'].y_fint = data['nodes'].f_ts[t].to(dtype)

    # (Optional) free large tensors you won't use further to save RAM/VRAM
    del data['nodes'].bc, data['nodes'].u_ts, data['nodes'].f_ext, data['nodes'].f_int, data['elements'].material#, data['nodes'].pos
    del data['elements'].s_ts, data['elements'].d_ts, data['elements'].material, data['nodes'].f_ts, data['elements'].s_ts, data['elements']
    del data[('nodes','belongs_to','elements')], data[('elements','contributes','nodes')]

    #print(data.keys)

    return data

class HeteroStandardScaler:
    def __init__(self):
        self.node_stats = {}
        self.edge_stats = {}

    def fit(self, dataset):
        node_x = torch.cat([d['nodes'].x[:,3:].float() for d in dataset],dim=0)
        node_f = torch.cat([d['nodes'].y_fint.float() for d in dataset],dim=0)
        node_u = torch.cat([d['nodes'].y_u.float() for d in dataset],dim=0)
        edge_acc = {}
        # compute mean/std
        for data in dataset:

            for etype in data.edge_types:
                if "edge_attr" in data[etype]:
                    edge_acc.setdefault(etype, []).append(data[etype].edge_attr.float())
        
        self.node_stats['nodes_x'] = {
            "mean": node_x.mean(dim=0, keepdim=True),
            "std":  node_x.std(dim=0, keepdim=True) + 1e-8}
        self.node_stats['nodes_f'] = {
            "mean": node_f.mean(dim=0, keepdim=True),
            "std":  node_f.std(dim=0, keepdim=True) + 1e-8}
        self.node_stats['nodes_u'] = {
            #"mean": node_u.mean(dim=0, keepdim=True),
            "mean": torch.zeros_like(node_u.mean(dim=0, keepdim=True)),
            "std":  node_u.std(dim=0, keepdim=True) + 1e-8}

        for etype, mats in edge_acc.items():
            E = torch.cat(mats, dim=0)
            self.edge_stats[etype] = {
                "mean": E.mean(dim=0, keepdim=True),
                "std":  E.std(dim=0, keepdim=True) + 1e-8}


    def transform(self, data: HeteroData):
        # apply normalization
        x = data['nodes'].x[:,3:].float()
        m_x = self.node_stats['nodes_x']["mean"]
        s_x = self.node_stats['nodes_x']["std"]
        data['nodes'].x[:,3:] = (x - m_x) / s_x

        y_f = data['nodes'].y_fint.float()
        m_f = self.node_stats['nodes_f']["mean"]
        s_f = self.node_stats['nodes_f']["std"]
        data['nodes'].y_fint = (y_f - m_f) / s_f
        
        y_u = data['nodes'].y_u.float()
        m_u = self.node_stats['nodes_u']["mean"]
        s_u = self.node_stats['nodes_u']["std"]
        data['nodes'].y_u = (y_u - m_u) / s_u

        # edges
        for etype in data.edge_types:
            if etype in self.edge_stats and "edge_attr" in data[etype]:
                e = data[etype].edge_attr.float()
                m = self.edge_stats[etype]["mean"]
                s = self.edge_stats[etype]["std"]
                data[etype].edge_attr = (e - m) / s

        return data
    
    def inverse_transform(self, data: HeteroData):
        # Nodes
        m, s = self.node_stats["nodes_u"]["mean"], self.node_stats["nodes_u"]["std"]
        data = data * s + m

        return data

def make_undirected(data):
    e = data['nodes','adjacent','nodes']
    ei = torch.cat([e.edge_index, e.edge_index.flip(0)], dim=1)
    ea = torch.cat([e.edge_attr,  e.edge_attr], dim=0)
    ei, ea = coalesce(ei, ea, num_nodes=data['nodes'].num_nodes, reduce='first')
    data['nodes','adjacent','nodes'].edge_index = ei
    data['nodes','adjacent','nodes'].edge_attr  = ea
    if ('nodes','adjacent_rev','nodes') in data.edge_types:
        del data['nodes','adjacent_rev','nodes']
    return data

scaler = HeteroStandardScaler()
## DataLoader with train/val split
dataset_p = [build_laststep_features(d) for d in dataset]
dataset_p = [make_undirected(d) for d in dataset_p]
scaler.fit(dataset_p)
dataset_t = [scaler.transform(d) for d in dataset_p]

def split_dataset(dataset, val_ratio=0.1, shuffle=True):
    n = len(dataset)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    return train_set, val_set

# preprocess first
train_set, val_set = split_dataset(dataset_t, val_ratio=0.1)

def coarsen(data):
    # sample points spatially
    pts = data['nodes']['pos']
    frac = 0.05
    N = pts.shape[0] # number of points
    M = max(1, int(np.round(frac * N))) # number of selected points
    rng = np.random.default_rng(42)
    selected_idx = np.empty(M, dtype=int)

    start = rng.integers(N)
    selected_idx[0] = start
    diff = pts - pts[start]
    min_dist2 = np.einsum("ij,ij->i", diff, diff)
    min_dist2[start] = 0.0

    for i in range(1, M): # random elimination
        idx = np.argmax(min_dist2)
        selected_idx[i] = idx
        diff = pts - pts[idx]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[idx] = 0.0

    pts_filtered = pts[selected_idx]

    # neighborhood search
    k = 20
    tree = cKDTree(pts)
    query_pts = tree.query(pts_filtered,k=k+1)
    query_pts = query_pts[1]
    query_self = query_pts[:,0]
    query_pts = query_pts[:,1:]

    # normals estimation

    features = data['nodes']['x'][query_pts]
    #bc_norm = minmax_norm(features[:,:,:3])
    #fext_norm = minmax_norm(features[:,:,3:])
    bc_std = torch.std(features[:,:,:3],dim=1)
    fext_std = torch.std(features[:,:,3:],dim=1)
    std_sum = torch.mean(fext_std,dim=1) #standard deviations of BCs + forces (0-1 normalized)
    bc_std_sum = torch.mean(bc_std,dim=1)
    
    rel_pos = data['nodes']['pos'][query_pts]-(data['nodes']['pos'][query_self]).unsqueeze(1).repeat(1, k, 1)
    rel_pos = minmax_norm(rel_pos)
    relpos_std = torch.std(rel_pos,dim=1)
    pos_std_sum = torch.mean(relpos_std,dim=1) #standard deviations of relative positions
    #feature_norm = torch.stack([std_sum,pos_std_sum],dim=1) # geometry & fixity & external forces -> based on neighborhood
    #feature_norm = torch.sum(feature_norm,axis=1)

    # mask by feature and position variance
    sig_mask_bc = (bc_std_sum > torch.quantile(bc_std_sum, 0.95))
    mid_mask_bc = (bc_std_sum < torch.quantile(bc_std_sum, 0.05)) # if any, usually will return no points
    sig_idx_bc = torch.where(sig_mask_bc | mid_mask_bc)[0]

    sig_mask_feat = (std_sum > torch.quantile(std_sum, 0.95))
    mid_mask_feat = (std_sum < torch.quantile(std_sum, 0.15))
    sig_idx_in_feat = torch.where(sig_mask_feat | mid_mask_feat)[0]

    sig_mask_pos = (pos_std_sum > torch.quantile(pos_std_sum, 0.95))
    mid_mask_pos = (pos_std_sum < torch.quantile(pos_std_sum, 0.15))
    sig_idx_in_pos = torch.where(sig_mask_pos | mid_mask_pos)[0]          # positions in pts_filtered

    sig_idx_in_filtered = torch.unique(torch.cat([sig_idx_bc,sig_idx_in_feat,sig_idx_in_pos],dim=0))

    # map back to original node ids
    sig_idx_original = torch.as_tensor(selected_idx)[sig_idx_in_filtered]  # indices into data['nodes'].pos

    # use original-order points for the coarse set
    pts_significant = pts[sig_idx_original]

    # fixed BC => fixed also after aggregation
    fixity_mask = ((data['nodes']['x'][sig_idx_original, :3]).sum(dim=1) == 0)
    pts_free = pts_significant[fixity_mask]
    pts_fixed = pts_significant[~fixity_mask]

    # connect points via Delaunay tetrahedralization (local coarse ordering)
    tri = Delaunay(pts_significant)
    tets = tri.simplices
    tet_edges = np.array([[0, 1], [0, 2], [0, 3],
                          [1, 2], [1, 3], [2, 3]])
    edges = tets[:, tet_edges].reshape(-1, 2)
    edges_undirected = np.sort(edges, axis=1)
    edges_undirected = np.unique(edges_undirected, axis=0)

    ## Alpha shape for connections (invalid edges deleted)
    # get inside volume with original points
    # calculate % of given edge inside original shape
    # -> for each sampled point in line get nearest neighbors & distances of neighbors
    edges = torch.tensor(edges,dtype=torch.int)
    edge_start = pts_significant[edges[:,0]]
    edge_end = pts_significant[edges[:,1]]
    num_points = 20
    direction = edge_end - edge_start
    t = torch.linspace(0, 1, num_points, device=edge_start.device)  # distances along edge
    edge_sampled = edge_start.unsqueeze(-2) + t.unsqueeze(-1) * direction.unsqueeze(-2) # (N_edges, num_points, ndims)
    # get distances from sample points to nearest node
    edge_dists, _ = tree.query(edge_sampled[:,:],k=2)
    edge_dists = edge_dists[:,1:num_points-1,1] # exclude self
    #print(edge_dists.shape)
    print(edge_dists)
    # calculate average distance
    query_dist,_ = tree.query(pts,k=2)
    query_dist = query_dist[:,1].mean(axis=0)
    print('query dist:',query_dist)
    # eliminate outliers
    dist_sigma = 1
    dist_threshold = 0.05
    bad_sample = (edge_dists > query_dist*dist_sigma)
    bad_frac = bad_sample.mean(axis=1)
    keep_line = (bad_frac <= dist_threshold)
    edge_mask_lines   = keep_line
    print(edge_mask_lines)

    # edge_attr in coarse (original) coordinates
    i_local = edges_undirected[:, 0]
    j_local = edges_undirected[:, 1]
    edge_attr = pts_significant[j_local] - pts_significant[i_local]  # same as before

    # map coarse-local edge endpoints back to original node ids
    edges_original_idx = torch.as_tensor(sig_idx_original, dtype=torch.long)[
        torch.as_tensor(edges_undirected, dtype=torch.long)
    ]  # shape [E, 2], indexes into data['nodes'].pos

    # aggregate info (unchanged)
    tree_f = cKDTree(pts_free)
    dists, idx_nearest = tree_f.query(pts, k=1)
    groups = [data['nodes']['x'][idx_nearest == j] for j in range(len(pts_free))]
    pooled = torch.stack([g.sum(dim=0) for g in groups])
    pooled[:, :3] = 0

    # return edges in original indexing + attributes

    return edges_original_idx.T#, torch.as_tensor(edge_attr, dtype=torch.float32)

save_path = "../../../scratch/btuncay/gnn/torchfem_dataset/beam_reduced/combined_50_alpha_conn.pt"
new_edges_dict = {}

for i, data in enumerate(dataset_t):
    extra_ei = coarsen(data)  # extra_ei shape [2, E], extra_attr [E, 3]
    src, dst = extra_ei
    extra_attr = (data['nodes'].pos[dst] - data['nodes'].pos[src]).float()

    ei_old = data['nodes', 'adjacent', 'nodes'].edge_index
    ea_old = data['nodes', 'adjacent', 'nodes'].edge_attr
    data['nodes', 'adjacent', 'nodes'].edge_index = torch.cat([ei_old, extra_ei], dim=1)
    data['nodes', 'adjacent', 'nodes'].edge_attr  = torch.cat([ea_old, extra_attr], dim=0)

torch.save(dataset_t, save_path)