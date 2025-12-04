## PyG Graph with Mesh Nodes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import numpy as np
import torch
from scipy.spatial import KDTree
from torch_geometric.data import HeteroData
import pandas as pd

# Material encoding (1-h)
def load_material_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = ['steel'] #fixed
    vocab = {lbl: i for i, lbl in enumerate(labels)}
    return vocab

def one_hot(vocab: Dict[str, int], device=None, dtype=torch.float) -> torch.Tensor:
    vec = torch.ones(len(vocab), dtype=dtype, device=device) #elements only placeholder
    return vec

def stiffness_to_node_adj_edge_index(K: torch.Tensor, num_nodes: int, dof_per_node: int = 3) -> torch.Tensor: #mesh-mesh nodes
    K = K.coalesce()
    dof_rows, dof_cols = K.indices()
    node_rows = torch.div(dof_rows, dof_per_node, rounding_mode="floor")
    node_cols = torch.div(dof_cols, dof_per_node, rounding_mode="floor")
    ei = torch.stack([node_rows.long(), node_cols.long()], dim=0)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    ei = torch.unique(ei, dim=1)
    return ei

def incidence_edges_from_conn(conn: np.ndarray, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Nc, Nv = conn.shape
    c_idx = np.repeat(np.arange(Nc, dtype=np.int64), Nv)
    n_idx = conn.reshape(-1)
    edge_index = torch.from_numpy(np.vstack([c_idx, n_idx])).long()
    node_tensor = nodes.to(edge_index.device)
    conn_tensor = torch.from_numpy(conn).long().to(edge_index.device)
    #edge attribute -> mesh to element centroid in xyz
    elem_nodes = node_tensor[conn_tensor]
    centroids = elem_nodes.mean(dim=1, keepdim=True)
    rel_disp = (elem_nodes - centroids).reshape(-1, elem_nodes.size(-1))

    return edge_index, rel_disp

def mesh_edges_from_conn(conn: torch.Tensor) -> torch.Tensor:
    conn = conn.long()                             # [E, nverts]
    hex_edges = torch.tensor([
        [0,1], [1,2], [2,3], [3,0],     # bottom face
        [4,5], [5,6], [6,7], [7,4],     # top face
        [0,4], [1,5], [2,6], [3,7],     # vertical edges
    ])
    pairs = conn[:,hex_edges]          # [E, 12, 2]
    pairs = pairs.reshape(-1, 2)
    pairs = torch.unique(torch.sort(pairs, dim=1).values, dim=0).T
    edge_index = pairs
    return edge_index

def mesh_edges_nearest(nodes: torch.Tensor) -> torch.Tensor: ##TODO
    # build mesh connections from nearest neighbors

    return

# Connectivity
# mesh nodes <-> mesh nodes
# mesh nodes <-> element nodes

def data_to_graph(path:str,device):
    data = HeteroData()
    vocab = load_material_vocab()
    simdata = torch.load(path,weights_only=False)
    conn = simdata["elements"]
    conn = conn.cpu().numpy() if isinstance(conn, torch.Tensor) else np.asarray(conn)
    nodes = simdata['nodes']
    c2n_ei, c2n_w = incidence_edges_from_conn(conn,nodes)  # [2, E_cn], [E_cn, 1]
    tree = KDTree(nodes)
    
    # mesh node properties: positions, forces, BC, dirichlet displacement
    data['nodes'].pos = nodes #                                             [N,3]
    data['nodes'].bc = simdata['boundary'] #                                [N,3]
    data['nodes'].f_ext = simdata['ext_forces'] #forces in timeseries format     [T,N,3]
    #data['nodes'].dr = simdata['dirichlet_disp'] #dirichlet displacement    [N,3]

    # element node properties: material, stiffness matrix
    #data['elements'].stiffness = #stiffness is a learned feature

    # target properties
    # mesh: displacement over time
    # element: stress, damage state
    data['nodes'].f_ts = simdata['forces'] #forces in timeseries format     [T,N,3]
    data['nodes'].u_ts = simdata['u_history']
    data['elements'].s_ts = simdata['stress_history']
    data['elements'].d_ts = simdata['state']

    # edges: connectivity mesh-mesh, mesh-element
    # mesh-element: distance to element centroid
    #elem_mat = one_hot(vocab, device=device).unsqueeze(0).repeat(int(c2n_ei.shape[1]), 1)
    

    #data["nodes", "belongs_to", "elements"].edge_attr = torch.cat([-c2n_w,elem_mat],dim=-1)
    data['elements'].num_nodes = simdata['stress_history'].size(1)

    # mesh-mesh: distance
    #ei = stiffness_to_node_adj_edge_index(simdata["stiffness"], num_nodes=nodes.size(0), dof_per_node=nodes.size(1))
    ei = mesh_edges_from_conn(simdata["elements"]) # [2,N]
    src, dst = ei
    edge_mat = one_hot(vocab, device=device).unsqueeze(0).repeat(int(ei.shape[1]), 1)
    disp = (nodes[dst] - nodes[src]).float()
    data["nodes", "adjacent", "nodes"].edge_index = ei
    data["nodes", "adjacent_rev", "nodes"].edge_index = ei.flip(0)
    data["nodes", "adjacent", "nodes"].edge_attr = torch.cat([disp,edge_mat],dim=-1)
    data["nodes", "adjacent_rev", "nodes"].edge_attr = torch.cat([-disp,edge_mat],dim=-1)
    #data['elements'].edge_index = simdata['elements'] #mesh-element [E,8]
    #data['nodes'].edge_index = stiffness_to_node_adj_edge_index(simdata["stiffness"], num_nodes=nodes.size(0), dof_per_node=nodes.size(1)) #from stiffness matrix, only connectivity

    #print(data['nodes'].num_nodes)                              # N
    #print(data['elements'].num_nodes)                           # E
    #print(data.num_edges)                                       # 2*((E*num_vertices)+())
    #print(data['nodes','adjacent','nodes'].num_edges)           # 
    #print(data['nodes','adjacent_rev','nodes'].num_edges)       #
    #print(data['elements','contributes','nodes'].num_edges)     # E*num_vertices
    #print(data['nodes','belongs_to','elements'].num_edges)      # E*num_vertices

    #data = data.pin_memory()
    data = data.to(device)
    return data

#data_to_graph('../base/torchfem_dataset/processed/simulation_dump_3.pt','../base/torchfem_dataset/processed/mat.csv',device)

def generate_dataset(data_dir:str):
    device = torch.device('cpu')
    files = sorted(Path(data_dir).glob("sim*.pt"))
    samples = []
    for file in files:
        filename = int(str(file).split('sim_')[1].split('.pt')[0])
        if filename > 100:
            break
        data = data_to_graph(file,device)
        samples.append(data)
        print(file)
    print(len(files))
    return samples

dataset = generate_dataset('datasets/simple_beam')
torch.save(dataset, "datasets/simple_beam/combined.pt")