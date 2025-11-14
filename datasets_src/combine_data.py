## PyG Graph with Mesh & Element Nodes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import numpy as np
import torch
from torch_geometric.data import HeteroData
import pandas as pd

# Material encoding (1-h)
def load_material_vocab(materials_csv: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    # CSV: idx,label
    df = pd.read_csv(materials_csv, header=None, names=["sim_id", "label"])
    #labels = sorted(df["label"].astype(str).unique().tolist())
    labels = ['concrete','steel','aluminum','CFRP'] #0,1,2,3 fixed material labels for now
    vocab = {lbl: i for i, lbl in enumerate(labels)}
    sim_to_label = {int(r.sim_id): str(r.label) for _, r in df.iterrows()}
    return vocab, sim_to_label

def one_hot(label: str, vocab: Dict[str, int], device=None, dtype=torch.float) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=dtype, device=device)
    vec[vocab[label]] = 1.0
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
    pairs = conn[:, hex_edges]          # [E, 12, 2]
    pairs = pairs.reshape(-1, 2)
    pairs = torch.unique(torch.sort(pairs, dim=1).values, dim=0).T
    edge_index = pairs
    return edge_index

# Connectivity
# mesh nodes <-> mesh nodes
# mesh nodes <-> element nodes

def data_to_graph(idx, path:str,materials_csv:str,device):
    data = HeteroData()
    vocab, sim_to_label = load_material_vocab(materials_csv)
    simdata = torch.load(path)
    idx = idx
    conn = simdata["elements"]
    conn = conn.cpu().numpy() if isinstance(conn, torch.Tensor) else np.asarray(conn)
    nodes = simdata['nodes']
    label = sim_to_label[idx]
    c2n_ei, c2n_w = incidence_edges_from_conn(conn,nodes)  # [2, E_cn], [E_cn, 1]
    
    # mesh node properties: positions, forces, BC, dirichlet displacement
    data['nodes'].pos = nodes #                                             [N,3]
    data['nodes'].f_ts = simdata['forces'] #forces in timeseries format     [T,N,3]
    data['nodes'].bc = simdata['boundary'] #                                [N,3]
    data['nodes'].dr = simdata['dirichlet_disp'] #dirichlet displacement    [N,3]

    # element node properties: material, stiffness matrix
    data['elements'].material = one_hot(label, vocab, device=device).unsqueeze(0).repeat(int(conn.shape[0]), 1) # [E,len(materials)]
    #data['elements'].stiffness = #stiffness is a learned feature

    # target properties
    # mesh: displacement over time
    # element: stress, damage state
    data['nodes'].u_ts = simdata['u_history']
    data['elements'].s_ts = simdata['stress_history']
    data['elements'].d_ts = simdata['state']

    # edges: connectivity mesh-mesh, mesh-element
    # mesh-element: distance to element centroid
    data["elements", "contributes", "nodes"].edge_index = c2n_ei
    data["elements", "contributes", "nodes"].edge_attr = c2n_w
    data["nodes", "belongs_to", "elements"].edge_index = c2n_ei.flip(0)
    data["nodes", "belongs_to", "elements"].edge_attr = -c2n_w
    data['elements'].num_nodes = simdata['stress_history'].size(1)

    # mesh-mesh: distance
    ei = stiffness_to_node_adj_edge_index(simdata["stiffness"], num_nodes=nodes.size(0), dof_per_node=nodes.size(1))
    ei = mesh_edges_from_conn(simdata["stiffness"])
    data["nodes", "adjacent", "nodes"].edge_index = ei
    data["nodes", "adjacent_rev", "nodes"].edge_index = ei.flip(0)
    data["nodes", "adjacent", "nodes"].edge_attr = (nodes[ei[1]] - nodes[ei[0]]).float()

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

def generate_dataset(data_dir:str,materials_csv:str):
    device = torch.device('cpu')
    files = sorted(Path(data_dir).glob("simulation_dump*.pt"))
    samples = []
    for file in files:
        idx = int(str(file).split("simulation_dump_")[1].split(".pt")[0])
        data = data_to_graph(idx,file,materials_csv,device)
        samples.append(data)
        print(file)
    print(len(files))
    return samples

dataset = generate_dataset('../base/torchfem_dataset/panel_euler/panel_plasticity','../base/torchfem_dataset/processed/mat.csv')
torch.save(dataset, "../base/torchfem_dataset/processed/panel_combined.pt")