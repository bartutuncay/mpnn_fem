## Unreinforced concrete slab

import torch
from torchfem import Solid
from torchfem.materials import IsotropicDamage3D
from torchfem.mesh import cube_hexa
from torchfem.sdfs import Cylinder,Box
from torchfem.io import export_mesh
import numpy as np
import argparse

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Process one integer.")
parser.add_argument("value", type=int, help="The integer to process")
args = parser.parse_args()
idx = args.value

###

# Damage parameters (Damage initiation strain and fracture energy)
eps_0 = 2.0e-4
G_f = 0.048


def d(kappa, cl):
    # Damage evolution law for exponential strain softening
    eps_f = G_f / (E * eps_0 * cl) + eps_0 / 2.0
    evolution = 1.0 - eps_0 / kappa * torch.exp(-(kappa - eps_0) / (eps_f - eps_0))
    evolution[kappa < eps_0] = 0.0

    return evolution


def d_prime(kappa, cl):
    # Derivative of the damage evolution law
    eps_f = G_f / (E * eps_0 * cl) + eps_0 / 2.0
    derivative = (
        eps_0
        * torch.exp(-(kappa - eps_0) / (eps_f - eps_0))
        * (1 / kappa**2 + 1 / (kappa * (eps_f - eps_0)))
    )
    derivative[kappa < eps_0] = 0.0

    return derivative
###
Lxrange = torch.linspace(60,140,5)
Lyrange = torch.linspace(15,45,5)
zrange = torch.linspace(50,120,5)
hole_r = torch.linspace(5,30,10)
torch.random.manual_seed(idx)
np.random.seed(idx)
ribcount = torch.randint(1,2,(1,))
loadtypes = ['uniform', 'point','incremental','incr_point']
loadtype = np.random.choice(loadtypes)
resolution = np.random.uniform(2,5)

# concrete
E = np.random.uniform(20e3,30e3)
nu = 0.2
material_concrete = IsotropicDamage3D(E,nu,d,d_prime,'rankine')

Lx = np.random.choice(Lxrange,1)[0]
Ly = np.random.choice(Lyrange,1)[0]
z = np.random.choice(zrange,1)[0]
rbrad = np.random.choice(hole_r,1)[0]
ribp = np.linspace(np.random.uniform(0.4*-Ly,0.2*-Ly),np.random.uniform(0.4*Ly,0.2*Ly),ribcount).tolist()
#ribp_l = np.linspace(np.random.uniform(0.4*-Lx,0.2*-Lx),np.random.uniform(0.4*Lx,0.2*Lx),ribcount).tolist()

## Reinforced concrete slab
nodes, elements = cube_hexa(int(Lx/resolution), int(Ly/resolution), int(z/resolution), Lx, Ly, z)

model = Solid(nodes, elements, material_concrete)

# Create a solid object
center = torch.tensor([Lx/2,Ly/2,z/2])
slab = Box(center,size=torch.tensor([Lx,Ly,z])) #center, size

for rebar in range(ribcount):
    hole = Cylinder(center+torch.tensor([np.random.uniform(-5,10),0,ribp[rebar]]),rbrad,Ly)
    slab = (slab-hole)


###
sdf_vals = slab.sdf(nodes)
inside = sdf_vals <= 0.5
mask = inside[elements].all(dim=1)
elements = elements[mask]

used = torch.unique(elements)
new_index = -torch.ones(len(nodes), dtype=torch.long)
new_index[used] = torch.arange(len(used))
elements = new_index[elements]
nodes = nodes[used]
model = Solid(nodes, elements, material_concrete)
###
#model.plot(
#    node_property={"SDF1": slab.sdf(nodes)})#,contour=("SDF2",[0]),
    #color="skyblue")

#model.plot(node_property={"SDF": body.sdf(nodes)},contour=("SDF", [0.0]), color="skyblue")

# Set constraints
DL = 0.1
model.displacements[nodes[:, 0] == 1.0, 0] = DL
eps = 1e-8  # tolerance for floating-point coords
z0 = torch.isclose(nodes[:, 2], torch.tensor(0.0), atol=eps)
xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
on_x_edge = torch.isclose(nodes[:, 0], xmin, atol=eps) | torch.isclose(nodes[:, 0], xmax, atol=eps)
on_y_edge = torch.isclose(nodes[:, 1], ymin, atol=eps) | torch.isclose(nodes[:, 1], ymax, atol=eps)
edge_mask = z0 & (on_x_edge | on_y_edge)
model.constraints[edge_mask, :] = True

n_steps = 200
peak_load = np.random.uniform(1e-2,1e-1)
theta = torch.linspace(0, torch.pi*4+np.random.uniform(0,torch.pi), n_steps)
increments = 0.5 * (1 - torch.cos(theta))            # starts at 0, peaks at 1, returns to 0
# or for a simple quarter-cycle: torch.sin(torch.linspace(0, math.pi/2, n_steps))
force_vector = torch.zeros_like(nodes)

if loadtype == 'uniform':
    # uniform load on plate
    side = torch.isclose(nodes[:, 1], torch.tensor(0.0, dtype=nodes.dtype), atol=1e-2)
    z_0 = torch.isclose(nodes[:, 2], torch.tensor(0.0, dtype=nodes.dtype), atol=1)
    side_mask = side & ~z_0
    force_vector[side_mask, 0] = peak_load
if loadtype == 'point':
    # point load
    #point = torch.Tensor([0])
    point_y = torch.isclose(nodes[:, 1], torch.tensor(0.0, dtype=nodes.dtype), atol=1e-2)
    point_x = torch.isclose(nodes[:,0],torch.tensor(np.random.uniform(0.1*Lx,0.9*Lx),dtype=nodes.dtype),atol = 1)
    point_z = torch.isclose(nodes[:,2],torch.tensor(np.random.uniform(0.8*z,z),dtype=nodes.dtype),atol = 5)
    point = point_x & point_y & point_z
    v = torch.randn(3) # random unit vector
    v /= v.norm()
    force_vector[point, 0] = peak_load * v[0]
    force_vector[point, 1] = peak_load * v[1]
    force_vector[point, 2] = peak_load * v[2]
    print(point.sum())
if loadtype == 'incremental':
    # Incremental loading
    increments = torch.cat((torch.linspace(0.0, 1.0, int(n_steps/2)), torch.linspace(1.0, 0.0, int(n_steps/2))))
    side = torch.isclose(nodes[:, 1], torch.tensor(0.0, dtype=nodes.dtype), atol=1e-2)
    z_0 = torch.isclose(nodes[:, 2], torch.tensor(0.0, dtype=nodes.dtype), atol=1)
    side_mask = side & ~z_0
    force_vector[side_mask, 0] = peak_load
if loadtype == 'incr_point':
    # Incremental loading
    increments = torch.cat((torch.linspace(0.0, 1.0, int(n_steps/2)), torch.linspace(1.0, 0.0, int(n_steps/2))))
    point_y = torch.isclose(nodes[:, 1], torch.tensor(0.0, dtype=nodes.dtype), atol=1e-2)
    point_x = torch.isclose(nodes[:,0],torch.tensor(np.random.uniform(0.1*Lx,0.9*Lx),dtype=nodes.dtype),atol = 1)
    point_z = torch.isclose(nodes[:,2],torch.tensor(np.random.uniform(0.8*z,z),dtype=nodes.dtype),atol = 5)
    point = point_x & point_y & point_z
    v = torch.randn(3) # random unit vector
    v /= v.norm()
    force_vector[point, 0] = peak_load * v[0]
    force_vector[point, 1] = peak_load * v[1]
    force_vector[point, 2] = peak_load * v[2]
    print(point.sum())

model.displacements = force_vector
print(loadtype)

#print(idx)
u, f, stress, F, state = model.solve(increments=increments,method="spsolve",return_intermediate=True)
print(f'completed idx:{idx}')
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
    },
    f"../../scratch/btuncay/gnn/torchfem_dataset/unreinforced_concrete/simulation_dump_{idx}.pt",
)