import torch
from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D,IsotropicPlasticity3D,OrthotropicElasticity3D
from torchfem.mesh import cube_hexa
from torchfem.sdfs import Sphere, Cylinder, Box, Gyroid
from torchfem.io import export_mesh
import numpy as np
import argparse

torch.set_default_dtype(torch.float64)


parser = argparse.ArgumentParser(description="Process one integer.")
parser.add_argument("value", type=int, help="The integer to process")
args = parser.parse_args()
idx = args.value

# Elastic material model
#material = IsotropicElasticity3D(E=1000.0, nu=0.3)


def sigma_f(q):
    return sigma_y + k * q
# Derivative of the hardening function
def sigma_f_prime(q):
    return k

model = None
nodes = None
elements = None
matlist = ['steel', 'aluminum', 'CFRP']
Lxrange = torch.linspace(25,45,5)
Lyrange = torch.linspace(40,90,5)
trange = torch.linspace(0.4,0.8,10)
ribwrange = torch.linspace(2,5,4)
ribhrange = torch.linspace(3,10,4)
torch.random.manual_seed(idx)
np.random.seed(idx)
ribcount = torch.randint(2,4,(1,))
ribdirs = torch.randint(0,1,(1,))
loadtypes = ['uniform', 'point','incremental','incr_point']
mat = np.random.choice(matlist)
loadtype = np.random.choice(loadtypes)
if mat == 'steel':
    E = 210e3
    nu = 0.3
    sigma_y = 250
    k = 2e3
    material = IsotropicElasticity3D(E, nu)
elif mat == 'aluminum':
    E = 70e3
    nu = 0.33
    sigma_y = 240
    k = 500
    material = IsotropicPlasticity3D(E, nu, sigma_f, sigma_f_prime)
else:
    E1 = 10e3
    E2, E3 = 130e3,10e3
    nu1 = 0.3
    nu2, nu3 = 0.3, 0.3
    sigma1 = 500
    sigma2, sigma3 = 1.6e3, 500
    material = OrthotropicElasticity3D(E1,E2,E3,nu1,nu2,nu3,sigma1,sigma2,sigma3)
resolution = torch.randint(1,4,(1,))
Lx = np.random.choice(Lxrange,1)[0]
Ly = np.random.choice(Lyrange,1)[0]
radius_inner = np.random.choice(trange,1)[0]
ribw = np.random.choice(ribwrange,1)[0]
ribh = np.random.choice(ribhrange,1)[0]
ribp = np.linspace(np.random.uniform(0.4*-Ly,0.2*-Ly),np.random.uniform(0.4*Ly,0.2*Ly),ribcount).tolist()
ribp_l = np.linspace(np.random.uniform(0.4*-Lx,0.2*-Lx),np.random.uniform(0.4*Lx,0.2*Lx),ribcount).tolist()

## Hollow tube
nodes, elements = cube_hexa(int(Lx/resolution), int(Ly/resolution), max(5,int(Lx/resolution)), Lx, Ly, Lx)
#print(len(elements))
# Create a solid object
model = Solid(nodes, elements, material)
center = torch.tensor([Lx/2,Ly/2,Lx/2])
body = Cylinder(center,Lx/2,Ly) #cylinder: center, radius, height
inner_cyl = Cylinder(center,Lx*radius_inner/2,Ly)
body = body-inner_cyl

###
sdf_vals = body.sdf(nodes)
inside = sdf_vals <= 0.2
mask = inside[elements].all(dim=1)
elements = elements[mask]
used = torch.unique(elements)
new_index = -torch.ones(len(nodes), dtype=torch.long)
new_index[used] = torch.arange(len(used))
elements = new_index[elements]
nodes = nodes[used]
model = Solid(nodes, elements, material)
###

#model.plot(node_property={"SDF": body.sdf(nodes)},contour=("SDF", [0.0]), color="skyblue")

# Set constraints
DL = 0.1
model.displacements[nodes[:, 0] == 1.0, 0] = DL
eps = 1e-8  # tolerance for floating-point coords
xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
#on_x_edge = torch.isclose(nodes[:, 0], xmin, atol=eps) | torch.isclose(nodes[:, 0], xmax, atol=eps)
on_y_edge = torch.isclose(nodes[:, 1], ymin, atol=eps)
edge_mask = on_y_edge
model.constraints[edge_mask, :] = True

n_steps = 200
peak_load = np.random.uniform(2e3,5e4)
theta = torch.linspace(0, torch.pi*4+np.random.uniform(0,torch.pi), n_steps)
increments = 0.5 * (1 - torch.cos(theta))            # starts at 0, peaks at 1, returns to 0
# or for a simple quarter-cycle: torch.sin(torch.linspace(0, math.pi/2, n_steps))
force_vector = torch.zeros_like(nodes)

print(mat)
print(loadtype)
if loadtype == 'uniform':
    # uniform load on plate
    bottom = torch.isclose(nodes[:, 1], ymax, atol=1e-2)
    v = torch.randn(3) # random unit vector
    v /= v.norm()
    force_vector[bottom, 0] = peak_load / bottom.sum() * v[0]
    force_vector[bottom, 1] = peak_load / bottom.sum() * v[1]
    force_vector[bottom, 2] = peak_load / bottom.sum() * v[2]
if loadtype == 'point':
    # edge load
    point_x = torch.isclose(nodes[:, 0], nodes[:,0].min(), atol=1)
    point = point_x
    v = torch.randn(3) # random unit vector
    v /= v.norm()
    force_vector[point, 0] = peak_load / point.sum() * v[0]
    force_vector[point, 1] = peak_load / point.sum() * v[1]
    force_vector[point, 2] = peak_load / point.sum() * v[2]
    print(point.sum())
if loadtype == 'incremental':
    # Incremental loading
    increments = torch.cat((torch.linspace(0.0, 1.0, int(n_steps/2)), torch.linspace(1.0, 0.0, int(n_steps/2))))
    bottom = torch.isclose(nodes[:, 1], ymax, atol=1e-2)
    v = torch.randn(3) # random unit vector
    v /= v.norm()
    force_vector[bottom, 0] = peak_load / bottom.sum() * v[0]
    force_vector[bottom, 1] = peak_load / bottom.sum() * v[1]
    force_vector[bottom, 2] = peak_load / bottom.sum() * v[2]
if loadtype == 'incr_point':
    # Incremental loading
    increments = torch.cat((torch.linspace(0.0, 1.0, int(n_steps/2)), torch.linspace(1.0, 0.0, int(n_steps/2))))
    point_z = torch.isclose(nodes[:, 2], nodes[:,2].min(), atol=1)
    point = point_z
    v = torch.randn(3) # random unit vector
    v /= v.norm()
    force_vector[point, 0] = peak_load / point.sum() * v[0]
    force_vector[point, 1] = peak_load / point.sum() * v[1]
    force_vector[point, 2] = peak_load / point.sum() * v[2]
    print(point.sum())


model.forces = force_vector

u, f, stress, F, state = model.solve(increments=increments,method="spsolve",return_intermediate=True)
print(f'Completed idx: {idx}')

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
    f"../../scratch/btuncay/gnn/torchfem_dataset/tube/simulation_dump_{idx}.pt",
)
    
#nodes = []
#elements = []

