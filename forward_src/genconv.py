import torch
import torch.nn.functional as F
from torch.nn import ModuleList

from torch_geometric.nn import pool
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, GENConv, to_hetero, MessageNorm, Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#device = torch.device('mps')

class GNN(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,latent_dim, out_channels):
        super().__init__()
        #self.proj = nn.Linear(in_channels, latent_dim)
        #self.edge_proj = nn.Linear(edge_in, latent_dim)

        self.proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.layers = layers
        #self.msg_norm = MessageNorm(learn_scale=True)

        self.encoding_layers = ModuleList([
            GENConv(latent_dim * 2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        #self.dropout = torch.nn.Dropout(p=0.1)

        #self.conv1 = GENConv(latent_dim,latent_dim,edge_dim=latent_dim) #num_layers=layers
        #self.conv2 = GENConv(latent_dim,out_channels,edge_dim=latent_dim) #num_layers=layers
        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='leaky_relu',plain_last=True)
    

    def forward(self,x,edge_index,edge_attr, batch):
        #x = torch.relu(self.proj(x))
        #edge_attr = torch.relu(self.edge_proj(edge_attr))
        x = self.proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_attr=edge_attr)
            #m = self.msg_norm(x, m)
            x = x + m
            #x = self.dropout(x)
        #for _ in range(self.layers):
        #    x = torch.relu(self.conv1(x,edge_index,edge_attr))
        #x = self.conv2(x, edge_index, edge_attr)
        x = self.inv_proj(x)
        return x

class GNN_double_pool(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,latent_dim, out_channels,z_dim=16):
        super().__init__()
        #self.proj = nn.Linear(in_channels, latent_dim)
        #self.edge_proj = nn.Linear(edge_in, latent_dim)

        self.proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.layers = layers
        #self.msg_norm = MessageNorm(learn_scale=True)

        self.encoding_layers = ModuleList([
            GENConv(latent_dim * 2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        self.to_z = Linear(latent_dim, z_dim)

        #self.conv1 = GENConv(latent_dim,latent_dim,edge_dim=latent_dim) #num_layers=layers
        #self.conv2 = GENConv(latent_dim,out_channels,edge_dim=latent_dim) #num_layers=layers
        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='leaky_relu',plain_last=True)
    

    def forward(self,x,edge_index,edge_attr, batch):
        #x = torch.relu(self.proj(x))
        #edge_attr = torch.relu(self.edge_proj(edge_attr))
        x = self.proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_attr=edge_attr)
            x = x + m
        #x_pooled = pool.global_mean_pool(x,batch)
        #z = self.to_z(x_pooled)
        x = self.inv_proj(x)
        return x

class GNN_Multihead(torch.nn.Module):
    def __init__(self, in_channels,edge_in,layers,latent_dim, out_channels):
        super().__init__()
        #self.proj = nn.Linear(in_channels, latent_dim)
        #self.edge_proj = nn.Linear(edge_in, latent_dim)

        self.proj = MLP(in_channels=in_channels,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.edge_proj = MLP(in_channels=edge_in,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='leaky_relu',norm='layer')
        self.layers = layers
        #self.msg_norm = MessageNorm(learn_scale=True)

        self.encoding_layers = ModuleList([
            GENConv(latent_dim * 2, latent_dim, norm='layer',msg_norm=True,edge_dim=latent_dim)
            for _ in range(layers)])
        
        self.inv_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='leaky_relu',plain_last=True)
        self.inv_proj2 = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=out_channels,num_layers=2,act='leaky_relu',plain_last=True)
    

    def forward(self,x,edge_index,edge_attr, batch):
        x = self.proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_attr=edge_attr)
            x = x + m
        x = self.inv_proj(x)
        f = self.inv_proj2(x)
        return x,f

##TODO: 
# deeper model - test message passing steps vs. error
# mean pooling - in encoding step: magedgepooling

##later:
# element stresses simple MLP
# force-based calcs & sequential input