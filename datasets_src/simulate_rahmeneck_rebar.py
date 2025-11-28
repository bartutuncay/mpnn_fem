import torch
from torchfem import Solid
from torchfem.materials import IsotropicPlasticity3D
from torchfem.mesh import cube_hexa
from torchfem.sdfs import Sphere, Cylinder, Box, Gyroid, Shell, Torus
from torchfem.io import export_mesh
import numpy as np

torch.set_default_dtype(torch.float64)

E = 210e3
nu = 0.3
sigma_y = 250
k = 2e3

def sigma_f(q):
    return sigma_y + k * q
# Derivative of the hardening function
def sigma_f_prime(q):
    return k

material = IsotropicPlasticity3D(E, nu, sigma_f, sigma_f_prime)

model = None
nodes = None
elements = None

leg1 = 1730
leg2 = 1800
diameter = 20
corner_radius = 150

resolution = 3

type = 'l' #'l'

## Hollow tube
nodes, elements = cube_hexa(int(leg1/resolution), int(leg2/resolution), int(diameter/resolution), leg1, leg2, diameter)
#print(len(elements))
# Create a solid object
model = Solid(nodes, elements, material)
leg1_cyl = Cylinder(torch.tensor([diameter/2,leg2/2+diameter/2,diameter/2]),diameter/2,leg2-2*corner_radius)
leg2_cyl = Cylinder(torch.tensor([leg1/2+diameter/2+corner_radius/2,diameter/2,diameter/2]),diameter/2,leg1-corner_radius).rotate(
    torch.tensor([0.0, 0.0, 1.0]), torch.tensor(torch.pi / 2.0))
leg3_cyl = Cylinder(torch.tensor([leg1/2+diameter/2+corner_radius/2,leg2-diameter/2,diameter/2]),diameter/2,leg1-corner_radius).rotate(
    torch.tensor([0, 0, 1.]), torch.tensor(torch.pi / 2.0))
torus1= Torus(torch.tensor([corner_radius+diameter/2,corner_radius+diameter/2,diameter/2]),corner_radius,diameter/2).rotate(
    torch.tensor([1., 0, 0]), torch.tensor(torch.pi / 2.0))
mask1 = Box(torch.tensor([corner_radius/2,corner_radius/2,diameter/2]),torch.tensor([corner_radius+diameter,corner_radius+diameter,corner_radius]))
torus2=Torus(torch.tensor([corner_radius+diameter/2,leg2-corner_radius-diameter/2,diameter/2]),corner_radius,diameter/2).rotate(
    torch.tensor([1., 0, 0]), torch.tensor(torch.pi / 2.0))
mask2 = Box(torch.tensor([corner_radius/2,leg2-corner_radius/2,diameter/2]),torch.tensor([corner_radius+diameter,corner_radius+diameter,corner_radius]))

torus1 = torus1&mask1
torus2 = torus2&mask2

if type == 'l':
    mask = Box(torch.tensor([leg1/2,leg2,diameter/2]),torch.tensor([leg1,leg2,diameter]))
if type == 'c':
    mask = Box(torch.tensor([leg1/2,leg2/2,diameter/2]),torch.tensor([leg1,leg2,diameter]))

body = (leg1_cyl|leg2_cyl|leg3_cyl|torus1|torus2)&mask


###
sdf_vals = body.sdf(nodes)
inside = sdf_vals <= 0
mask = inside[elements].all(dim=1)
elements = elements[mask]
used = torch.unique(elements)
new_index = -torch.ones(len(nodes), dtype=torch.long)
new_index[used] = torch.arange(len(used))
elements = new_index[elements]
nodes = nodes[used]
model = Solid(nodes, elements, material)
###

# Set constraints
#DL = 0.1
#model.displacements[nodes[:, 0] == 0, 0] = DL
eps = 1e-3  # tolerance for floating-point coords
xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
#on_x_edge = torch.isclose(nodes[:, 0], xmin, atol=eps) | torch.isclose(nodes[:, 0], xmax, atol=eps)
if type == 'l':
    loc_mask = torch.isclose(nodes[:, 1], ymin, atol=1e-1)
if type == 'c':
    on_y_edge = torch.isclose(nodes[:, 1], ymin, atol=eps)
    edge_mask = on_y_edge
    loc_mask = edge_mask & (nodes[:,0] >= 1000)
model.constraints[loc_mask, :] = True
model.constraints[:, 2] = True

n_steps = 200
peak_load = 1e3
# or for a simple quarter-cycle: torch.sin(torch.linspace(0, math.pi/2, n_steps))
force_vector = torch.zeros_like(nodes)

# Incremental loading
increments = torch.cat((torch.linspace(0.0, 1.0, int(n_steps/2)), torch.linspace(1.0, 0.0, int(n_steps/2))))
bottom = torch.isclose(nodes[:, 1], ymax, atol=eps)
force_mask = bottom & (nodes[:, 0] >= 1300)
force_vector[force_mask, 0] = 0
force_vector[force_mask, 1] = -peak_load / force_mask.sum()
force_vector[force_mask, 2] = 0

#print(force_mask.sum(),loc_mask.sum())
#print(nodes[loc_mask])
#print(nodes[force_mask])

model.forces = force_vector
scaled = np.array([force_vector * inc for inc in increments])

if True:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # all nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=5, c="lightgray", label="all")
    # constrained
    ax.scatter(nodes[loc_mask, 0], nodes[loc_mask, 1], nodes[loc_mask, 2],
            s=30, c="tab:blue", label="constrained")
    # loaded
    ax.scatter(nodes[force_mask, 0], nodes[force_mask, 1], nodes[force_mask, 2],
            s=30, c="tab:red", marker="x", label="force")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    plt.show()



u, f, stress, F, state = model.solve(increments=increments,rtol=1e-5,method="spsolve",return_intermediate=True,verbose=True)

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
    },
    f"../../torchfem_dataset/rahmeneck_rebar_l.pt",
)

