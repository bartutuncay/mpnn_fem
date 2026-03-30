import torch
import numpy as np

## Material law
E = 210e3
nu = 0.3
sigma_y = 250
k = 2e3

def sigma_f(q):
    return sigma_y + k * q
# Derivative of the hardening function
def sigma_f_prime(q):
    return k

#####

def dev(A):
    return A - np.trace(A) / 3.0 * np.eye(3)

def mises_eq_from_dev(s):
    # sigma_eq = sqrt(3/2 * s:s)
    return np.sqrt(1.5 * np.tensordot(s, s))

def j2_return_map_linear_iso_hardening(eps, E, nu, sigma_y, k, q0=0.0):
    """
    eps: 3x3 small-strain tensor at the point
    returns: sigma (3x3), q1 (scalar), is_plastic (bool)
    """
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    tr_eps = np.trace(eps)
    eps_dev = dev(eps)

    # trial stress: sigma_tr = 2G*eps_dev + K*tr(eps)*I
    sigma_tr = 2.0 * G * eps_dev + K * tr_eps * np.eye(3)
    s_tr = dev(sigma_tr)

    sigma_eq_tr = mises_eq_from_dev(s_tr)
    sigma_f0 = sigma_y + k * q0
    f_tr = sigma_eq_tr - sigma_f0

    if f_tr <= 0.0 or sigma_eq_tr < 1e-14:
        return sigma_tr, q0, False

    # plastic correction
    dgamma = f_tr / (3.0 * G + k)  # linear hardening
    q1 = q0 + dgamma

    # radial return on deviatoric part
    scale = 1.0 - (3.0 * G * dgamma) / sigma_eq_tr
    s_new = scale * s_tr

    # hydrostatic part unchanged from trial
    p_tr = np.trace(sigma_tr) / 3.0
    sigma_new = s_new + p_tr * np.eye(3)

    return sigma_new, q1, True

def voigt_to_tensor_strain(epsv):
    # epsv: (6,) with engineering shear
    exx, eyy, ezz, gxy, gyz, gzx = epsv
    return np.array([
        [exx, 0.5*gxy, 0.5*gzx],
        [0.5*gxy, eyy, 0.5*gyz],
        [0.5*gzx, 0.5*gyz, ezz],
    ])

def tensor_to_voigt_stress(sig):
    # returns (6,) with physical shear stresses (not doubled)
    return np.array([sig[0,0], sig[1,1], sig[2,2], sig[0,1], sig[1,2], sig[2,0]])

def hex8_dN_dxi(ksi, eta, zet):
    # returns (8,3): dN/dksi, dN/deta, dN/dzet
    # node order must match your connectivity!
    # Standard trilinear Hex8 with natural coords in [-1,1]
    s = np.array([
        [-1,-1,-1],
        [ 1,-1,-1],
        [ 1, 1,-1],
        [-1, 1,-1],
        [-1,-1, 1],
        [ 1,-1, 1],
        [ 1, 1, 1],
        [-1, 1, 1],
    ], dtype=float)
    dN = np.zeros((8,3), dtype=float)
    for a in range(8):
        sa, ta, ua = s[a]
        dN[a,0] = 0.125 * sa * (1 + ta*eta) * (1 + ua*zet)  # dN/dksi
        dN[a,1] = 0.125 * ta * (1 + sa*ksi) * (1 + ua*zet)  # dN/deta
        dN[a,2] = 0.125 * ua * (1 + sa*ksi) * (1 + ta*eta)  # dN/dzet
    return dN

def hex8_B_matrix(x_e, ksi, eta, zet):
    """
    x_e: (8,3) nodal coordinates of element
    returns B: (6,24), detJ
    """
    dN_nat = hex8_dN_dxi(ksi, eta, zet)  # (8,3)

    # Jacobian J = sum_a x_a ⊗ grad_nat(N_a)  -> (3,3)
    J = x_e.T @ dN_nat  # (3,8)@(8,3)=(3,3)
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)

    # spatial gradients dN/dx = dN/dxi * invJ^T (or invJ depending on convention)
    # Here: grad_x = grad_nat @ invJ^T
    dN_xyz = dN_nat @ invJ.T  # (8,3)

    B = np.zeros((6, 24), dtype=float)
    for a in range(8):
        dNx, dNy, dNz = dN_xyz[a]
        col = 3*a
        # normal strains
        B[0, col+0] = dNx
        B[1, col+1] = dNy
        B[2, col+2] = dNz
        # engineering shear strains gamma_xy, gamma_yz, gamma_zx
        B[3, col+0] = dNy
        B[3, col+1] = dNx
        B[4, col+1] = dNz
        B[4, col+2] = dNy
        B[5, col+0] = dNz
        B[5, col+2] = dNx

    return B, detJ

def element_gp_stresses_hex8(x_e, u_e, E, nu, sigma_y, k):
    """
    x_e: (8,3)
    u_e: (8,3) displacements
    returns: sig_gp (8,6) stresses at 8 gauss points in Voigt stress,
             q_gp (8,) eq plastic strain (approx, starting from 0),
             plast_gp (8,) bool
    """
    a = 1.0 / np.sqrt(3.0)
    gps = [(i*a, j*a, l*a) for i in (-1,1) for j in (-1,1) for l in (-1,1)]

    ue_vec = u_e.reshape(-1)  # (24,)
    sig_gp = np.zeros((8,6), dtype=float)
    q_gp = np.zeros((8,), dtype=float)
    plast_gp = np.zeros((8,), dtype=bool)

    for g,(ksi,eta,zet) in enumerate(gps):
        B, detJ = hex8_B_matrix(x_e, ksi, eta, zet)
        epsv = B @ ue_vec  # (6,)
        eps = voigt_to_tensor_strain(epsv)

        sigma, q1, is_pl = j2_return_map_linear_iso_hardening(
            eps, E, nu, sigma_y, k, q0=0.0
        )
        sig_gp[g,:] = tensor_to_voigt_stress(sigma)
        q_gp[g] = q1
        plast_gp[g] = is_pl

    return sig_gp, q_gp, plast_gp

def accumulate_nodal_stress(n_nodes, conn, elem_stress, elem_vol):
    """
    conn: (n_elem,8) node indices
    elem_stress: (n_elem,6) e.g. element-averaged stress
    elem_vol: (n_elem,)
    returns nodal_stress: (n_nodes,6)
    """
    nodal = np.zeros((n_nodes,6), float)
    wsum = np.zeros((n_nodes,), float)

    for e in range(conn.shape[0]):
        Ve = elem_vol[e]
        for a in conn[e]:
            nodal[a] += Ve * elem_stress[e]
            wsum[a] += Ve

    nodal /= (wsum[:,None] + 1e-30)
    return nodal
