## Introduction to the FEM Solver for Data Generation

*requirements: torch numpy scipy*

The Pytorch-based FEM solver was inspired by the Github repository **torch-fem** by *meyer-nils*; modifications were made with the following aims:
- Simplification of the repository for fast data generation
- Better applicability to structural engineering problems
- Improved meshing and stability
- Implementation of dynamic problems
- Updated dependencies and better compability with the main repo

### File structure:
torch_fem_module\
├─ README.md\
├─ src/torch_fem3d/\
│  ├─ __init__.py\
│  ├─ config.py                     # (dtype, device),dataclasses
│  ├─ utils/
│  │  ├─ linalg.py.                 # SPD solves, CG wrappers, spmv helpers
│  │  ├─ scatter.py                 # safe scatter_add + COO index builders
│  │  ├─ profiling.py
│  │  └─ visualization.py           # VTK/VTU export via meshio (optional)
│  ├─ mesh/
│  │  ├─ io.py                      # read/write, internal
│  │  ├─ topology.py                # connectivity, boundary faces/sets
│  │  ├─ quadrature.py           # Gauss rules (tet/hex), weights, points
│  │  ├─ shape_functions.py      # N, dN/dξ, Jacobians, B-matrix builders
│  │  └─ utils.py
│  ├─ materials/
│  │  ├─ base.py                 # Material interface
│  │  ├─ linear_elastic.py       # small-strain (E, ν, ρ), constitutive ops
│  │  ├─ neo_hookean.py          # large-strain (optional, later)
│  │  └─ damping.py              # Rayleigh α,β and custom C models
│  ├─ elements/
│  │  ├─ base.py                 # Element interface
│  │  ├─ tet4.py                 # Ke, Me, fint for 4-node tets
│  │  ├─ tet10.py                # (later)
│  │  ├─ hex8.py                 # (later)
│  │  └─ assembly.py             # global COO assembly (K, M, C, fext)
│  ├─ boundary/
│  │  ├─ sets.py                 # node/face/volume sets from tags or queries
│  │  ├─ dirichlet.py            # masks, elimination, or penalty
│  │  └─ neumann.py              # traction/pressure load vectors
│  ├─ dynamics/
│  │  ├─ system.py               # FEMSystem(nn.Module): holds mesh, mats, ops
│  │  ├─ integrators/
│  │  │  ├─ newmark.py           # implicit Newmark-β
│  │  │  ├─ generalized_alpha.py # implicit gen-α (better dissipation ctrl)
│  │  │  └─ central_difference.py# explicit (with mass lumping)
│  │  ├─ solvers/
│  │  │  ├─ linear.py            # torch CG/Cholesky wrappers (SPD)
│  │  │  └─ nonlinear.py         # Newton(-Krylov) w/ line search
│  │  └─ autodiff.py             # grad utilities, gradcheck helpers
│  ├─ losses/
│  │  ├─ observation.py          # dof→sensor mapping, probe extraction
│  │  └─ losses.py               # MSE, smooth L1, physics-in-loss, reg
│  └─ training/
│     ├─ loops.py                # BPTT over time, checkpointing
│     ├─ schedulers.py
│     └─ optim_utils.py
