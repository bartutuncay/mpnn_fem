import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# --------------------------------------------
# Example MPNN block (replace with your own)
# Can be PNA, GAT, GENConv, MLP-MPNN, etc.
# --------------------------------------------
class MPNNBlock(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(msg)

    def update(self, aggr_out):
        return aggr_out


# --------------------------------------------
# Dual-Branch Multiscale Graph Network
# --------------------------------------------
class DualScaleGraphUNet(nn.Module):
    def __init__(self, 
                 fine_dim=64,
                 coarse_dim=128,
                 depth_fine=3,
                 depth_coarse=6):
        super().__init__()
        
        # Fine graph encoders
        self.fine_enc = nn.Linear(fine_dim, fine_dim)
        
        # Coarse graph encoders
        self.coarse_enc = nn.Linear(fine_dim, coarse_dim)
        
        # Fine branch MPNN layers
        self.fine_layers = nn.ModuleList([
            MPNNBlock(fine_dim, fine_dim) for _ in range(depth_fine)
        ])
        
        # Coarse branch MPNN layers
        self.coarse_layers = nn.ModuleList([
            MPNNBlock(coarse_dim, coarse_dim) for _ in range(depth_coarse)
        ])

        # Final decoder
        self.decoder = nn.Linear(fine_dim, fine_dim)
    
    # ---------------------------------------------------------
    # NOTE:
    # pool_f2c and interp_c2f must be provided by YOU.
    # They depend on the coarsening strategy (Graclus, METIS,
    # custom clustering, barycentric interpolation, etc.)
    # ---------------------------------------------------------
    def pool_f2c(self, x_f, assign_matrix):
        """
        x_f: (N_fine, F)
        assign_matrix: (N_fine, N_coarse) sparse/binary/soft
        Returns: pooled coarse features (N_coarse, F)
        """
        return assign_matrix.T @ x_f
    
    def interp_c2f(self, x_c, assign_matrix):
        """
        x_c: (N_coarse, F)
        Interpolates coarse → fine.
        """
        return assign_matrix @ x_c

    # ---------------------------------------------------------
    # Forward pass
    # fine_data: PyG Data object for fine graph
    # coarse_data: PyG Data object for coarse graph
    # assign: matrix that maps fine → coarse nodes (pooling)
    # ---------------------------------------------------------
    def forward(self, fine_data, coarse_data, assign):

        # -----------------------------------------------------
        # Encode features
        # -----------------------------------------------------
        h_f = self.fine_enc(fine_data.x)        # (N_f, D_f)
        h_c = self.coarse_enc(coarse_data.x)    # (N_c, D_c)

        # Initial pooling: fine → coarse
        h_c = h_c + self.pool_f2c(h_f, assign)

        # -----------------------------------------------------
        # Parallel message passing on both graphs
        # -----------------------------------------------------
        for f_layer, c_layer in zip(self.fine_layers, self.coarse_layers[:len(self.fine_layers)]):
            
            # Fine update
            h_f = h_f + f_layer(
                h_f, 
                fine_data.edge_index,
                fine_data.edge_attr
            )

            # Coarse update
            h_c = h_c + c_layer(
                h_c,
                coarse_data.edge_index,
                coarse_data.edge_attr
            )

            # Skip: coarse → fine interpolation
            h_f = h_f + self.interp_c2f(h_c, assign)

            # Optional skip: fine → coarse (for stability)
            h_c = h_c + self.pool_f2c(h_f, assign)

        # -----------------------------------------------------
        # If coarse branch deeper than fine branch
        # Continue coarse-only updates
        # -----------------------------------------------------
        for c_layer in self.coarse_layers[len(self.fine_layers):]:
            h_c = h_c + c_layer(
                h_c,
                coarse_data.edge_index,
                coarse_data.edge_attr
            )
            h_f = h_f + self.interp_c2f(h_c, assign)

        # -----------------------------------------------------
        # Decode back on fine graph
        # -----------------------------------------------------
        out = self.decoder(h_f)
        return out
