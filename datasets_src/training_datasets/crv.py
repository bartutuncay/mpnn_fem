## Cantilever supports - Regular mesh - Non-uniform load
# Recommended to run in parallel using a scheduler

import torch
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')
torch.set_default_device(device)

from torchfem import Solid
from torchfem.materials import IsotropicPlasticity3D
from torchfem.mesh import cube_hexa
from torchfem.sdfs import Box
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser(description="index")
parser.add_argument("value", type=int, help="integer to process")
args = parser.parse_args()
idx = args.value

E = 210e3
nu = 0.3
sigma_y = 250
k = 2e3

# Hardening function
def sigma_f(q):
    return sigma_y + k * q
# Derivative of the hardening function
def sigma_f_prime(q):
    return k

material = IsotropicPlasticity3D(E, nu, sigma_f, sigma_f_prime)

model = None
nodes = None
elements = None

torch.random.manual_seed(idx)
np.random.seed(idx)

Lx = np.random.uniform(30,70)       # flange
Ly = np.random.uniform(50,100)      # web
Lz = np.random.uniform(200,800)     # length
w = np.random.uniform(2,6)          # flange thickness
t = np.random.uniform(2,5)          # web thickness
support_loc = np.random.choice(['bottom','top'])
force_loc = np.random.choice(['right','left'])
# problem types:
load_type = np.random.choice(['bending','shear','torsion','axial','point','ramp'])

resolution = torch.randint(3,6,(1,))

## Initialize mesh
nodes, elements = cube_hexa(int(Lx/resolution), int((Ly+2*w)/resolution), int(Lz/resolution), Lx, Ly+(2*w), Lz)

# Create a solid object
model = Solid(nodes, elements, material)
center = torch.tensor([Lx/2,(Ly+(2*w))/2,Lz/2])

flange_top = Box(center=center+torch.tensor([0,Ly/2+w/2,0]),size=torch.tensor([Lx,w,Lz]))
flange_bottom = Box(center=center+torch.tensor([0,-Ly/2-w/2,0]),size=torch.tensor([Lx,w,Lz]))
web = Box(center=center+torch.tensor([0,0,0]),size=torch.tensor([t,Ly,Lz]))

body = flange_top|flange_bottom|web

# Apply distance function to filter nodes
sdf_vals = body.sdf(nodes)
inside = sdf_vals <= 2
mask = inside[elements].any(dim=1)
elements = elements[mask]
used = torch.unique(elements)
new_index = -torch.ones(len(nodes), dtype=torch.long)
new_index[used] = torch.arange(len(used))
elements = new_index[elements]
nodes = nodes[used]

# Define boundary nodes
eps = 2e-3  # tolerance for coordinates
xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
zmin, zmax = nodes[:, 2].min(), nodes[:, 2].max()
on_z_zero = torch.isclose(nodes[:, 2], zmin, atol=eps)
on_z_top = torch.isclose(nodes[:, 2], zmax, atol=eps)
if force_loc == 'right':
    on_z_edge = torch.isclose(nodes[:, 1], ymax, atol=eps)
    on_z_support = torch.isclose(nodes[:, 1], ymin, atol=eps)
else:
    on_z_edge = torch.isclose(nodes[:, 1], ymin, atol=eps)
    on_z_support = torch.isclose(nodes[:, 1], ymax, atol=eps)

model = Solid(nodes, elements, material)

# Set constraints
model.displacements[on_z_zero, :] = 0
model.constraints[on_z_zero, :] = True

n_steps = 100
peak_load = np.random.uniform(1e3,1e4)
force_dir = torch.randn(3)
if load_type == 'bending':
    force_dir /= force_dir.norm()
if load_type == 'axial':
    force_dir = torch.tensor([0,0,1])

force_vector = torch.zeros_like(nodes)

# Incremental loading
increments = torch.linspace(0.0, 1.0, int(n_steps/2))
force_vector[on_z_edge, 0] = -peak_load / on_z_edge.sum() * force_dir[0]
force_vector[on_z_edge, 1] = -peak_load / on_z_edge.sum() * force_dir[1]
force_vector[on_z_edge, 2] = -peak_load / on_z_edge.sum() * force_dir[2]

if load_type == 'shear':
    force_vector[on_z_edge, 0] = -peak_load / on_z_edge.sum() * 0
    force_vector[on_z_edge, 1] = -peak_load / on_z_edge.sum() * (nodes[on_z_edge,2]/torch.abs(zmax-zmin)-0.5)
    force_vector[on_z_edge, 2] = -peak_load / on_z_edge.sum() * 0

if load_type == 'ramp':
    force_vector[on_z_edge, 0] = -peak_load / on_z_edge.sum() * 0
    force_vector[on_z_edge, 1] = -peak_load / on_z_edge.sum() * (nodes[on_z_edge,2]/torch.abs(zmax-zmin))
    force_vector[on_z_edge, 2] = -peak_load / on_z_edge.sum() * 0

if load_type == 'torsion':
    force_vector[on_z_edge, 0] = -peak_load / on_z_edge.sum() * (nodes[on_z_edge,2]/torch.abs(zmax-zmin)-0.5)
    force_vector[on_z_edge, 1] = -peak_load / on_z_edge.sum() * (nodes[on_z_edge,2]/torch.abs(zmax-zmin)-0.5)
    force_vector[on_z_edge, 2] = -peak_load / on_z_edge.sum() * 0

if load_type == 'point':
    load_point = on_z_edge[torch.randint(0,on_z_edge.size(0),(1,))]
    peak_load /= 1e-1
    force_vector[load_point, 0] = -peak_load / on_z_edge.sum() * 0
    force_vector[load_point, 1] = -peak_load / on_z_edge.sum() * 0
    force_vector[load_point, 2] = -peak_load / on_z_edge.sum() * 0

if support_loc == 'top':
    model.displacements[on_z_top] = 0
    model.constraints[on_z_top] = True
    force_vector[on_z_top, 0] = 0
    force_vector[on_z_top, 1] = 0
    force_vector[on_z_top, 2] = 0
    model.constraints[on_z_zero,:] = False
if support_loc == 'bottom':
    model.displacements[on_z_zero] = 0
    model.constraints[on_z_zero] = True
    force_vector[on_z_zero, 0] = 0
    force_vector[on_z_zero, 1] = 0
    force_vector[on_z_zero, 2] = 0
    model.constraints[on_z_top,:] = False

model.forces = force_vector
scaled = force_vector.unsqueeze(0) * increments.view(-1, 1, 1)
scaled_np = scaled.detach().cpu().numpy()

t1 = time.time()
u, f, stress, F, state = model.solve(increments=increments,method="spsolve",return_intermediate=True,verbose=True) #rtol=1e-5
t2 = time.time()

os.makedirs('datasets/cantilever/regular/non_uniform',exist_ok=True)
torch.save(
    {
        "nodes": model.nodes,
        "stiffness": model.K,
        "u_history": u.detach().cpu(),
        "stress_history":stress.detach().cpu(),
        "deform_grad":F.detach().cpu(),
        "forces":f.detach().cpu(),
        "state":state.detach().cpu(),
        "boundary":model.constraints.detach().cpu(),
        "dirichlet_disp": model.displacements.detach().cpu(),
        "elements":model.elements.detach().cpu(),
        "ext_forces": scaled,
        "support_loc": support_loc,
        "force_loc": force_loc,
        "load_type": load_type,
        "load_type": load_type,
        "sim_time": t2-t1,
    },
    f"datasets/cantilever/regular/non_uniform/sim_{idx}.pt",
)

