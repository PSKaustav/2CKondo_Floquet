#___1_______________________________________________________________________________________________________________
#1: The relevant imports so as to make my life easier
#__________________________________________________________________________________________________________________
import numpy as np 
from qlimb.classical.mps import MPS

from itertools import combinations
from qlimb.classical.utils import apply_svd

from qlimb.classical.gates import Gate

import matplotlib.pyplot as plt
#__________________________________________________________________________________________________________________
# This file is the one where the floquet step is: even odd UK (which doesnt match the previous work)
#==================================================================================================================
# Prepare the fermi sea using half filling in each spin sector in the unfolded representation of the chain with 
# impurity at the centre
#==================================================================================================================

def build_kspace_fermi_orbitals(N_bath, filling=0.5):
    """
    Parameters
    ----------
    N_bath : int
        Number of bath sites in ONE spin sector.

    filling : float
        Filling fraction for each spin sector.

    Returns
    -------
    orb : ndarray, shape (N_bath, N_occ)
        Columns are occupied plane-wave orbitals expressed in real space.

    k_occ : ndarray
        Occupied momenta.
    """

    x = np.arange(N_bath)

    if N_bath % 2 == 0:
        k_indices = np.arange(-N_bath // 2, N_bath // 2)
    else:
        k_indices = np.arange(-(N_bath // 2), N_bath // 2 + 1)

    k_vals = 2.0 * np.pi * k_indices / N_bath

    n_occ = int(round(filling * N_bath))

    # Fill states closest to k=0 (Fermi sea)
    energies = -2*np.cos(k_vals)

    order = np.argsort(energies)
    """order = np.argsort(np.abs(k_vals))"""
    k_occ = k_vals[order[:n_occ]]

    orb = np.zeros((N_bath, n_occ), dtype=complex)

    for col, k in enumerate(k_occ):
        orb[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)

    return orb, k_occ



#
# up chain : N bath sites
# impurity : 1 qubit
# down chain : N bath sites
#
# Total system size = 2*N + 1

def build_initial_orbitals(N_bath):
    """
    Half-filled Fermi sea in each spin sector.

    Returns
    -------
    orb_up, orb_down, n_up, n_down
    """

    orb_up, _ = build_kspace_fermi_orbitals(N_bath, filling=0.5)
    orb_down, _ = build_kspace_fermi_orbitals(N_bath, filling=0.5)

    n_up = orb_up.shape[1]
    n_down = orb_down.shape[1]

    return orb_up, orb_down, n_up, n_down


#==================================================================================================================
# 2: Two Utility Functions to map real space sites to qubits
#==================================================================================================================

def dictionary(N, spin_label, k_bath):
    """
    Map a bath-site index to the qubit index in the unfolded chain.

    Layout:

        0 ... N-1     N      N+1 ... 2N
          down      impurity      up

    spin_label = 0 -> spin up
    spin_label = 1 -> spin down

    k_bath runs from 0 to N-1.
    """

    if not isinstance(N, int):
        raise TypeError("N must be an integer")
    if not isinstance(spin_label, int):
        raise TypeError("spin_label must be either 0 or 1")
    if not isinstance(k_bath, int):
        raise TypeError("k_bath must be an integer between 0 and N-1")

    if spin_label not in (0, 1):
        raise ValueError("spin_label must be either 0 or 1")

    if k_bath < 0 or k_bath >= N:
        raise ValueError("k_bath must be an integer between 0 and N-1")

    if spin_label == 0:      # spin up, right of impurity
        return N + 1 + k_bath
    else:                    # spin down, left of impurity
        return k_bath

def k_imp_new(N):
    return N 

#=================================================================================================================
# Preparing the state and then the mps
#=================================================================================================================

def build_initial_state(N, orb_up, orb_down, n_up, n_down):

    psi = np.zeros(2**(2*N + 1), dtype=complex)

    impurity_qbit = np.array([1], dtype=int)

    for down_occ in combinations(range(N), n_down):

        spin_down_qbits = np.zeros(N, dtype=int)
        spin_down_qbits[list(down_occ)] = 1

        rows_down = np.where(spin_down_qbits == 1)[0]
        a_down = np.linalg.det(orb_down[rows_down, :])

        for up_occ in combinations(range(N), n_up):

            spin_up_qbits = np.zeros(N, dtype=int)
            spin_up_qbits[list(up_occ)] = 1

            rows_up = np.where(spin_up_qbits == 1)[0]
            a_up = np.linalg.det(orb_up[rows_up, :])

            full_state = np.concatenate((
                spin_down_qbits,
                impurity_qbit,
                spin_up_qbits
            ))

            idx = 0
            for bit in full_state:
                idx = (idx << 1) | bit

            psi[idx] = a_down * a_up

    return psi

def convert_psi_to_mps(psi, N, bond_dim=np.inf):
    d=2
    tensors = []
    T = psi.reshape([d]*(2*N+1))
    chi_left = 1

    for site in range(2*N):
        T = T.reshape(chi_left,d,-1,1)
        U, s, s_trunc, V = apply_svd(T, bond_dim=bond_dim,direction='right',preserve_norm=True, tol=1e-10)
        tensors.append(U)
        chi_left = U.shape[-1]
        T = V.reshape(chi_left,d,-1)
    tensors.append(T.reshape(chi_left, d, 1))
    return tensors

def state_mps(N,tensors):
    psi_mps = MPS(nqbits=2*N + 1, phys_dim=2, tensors=tensors,bond_dim=np.inf, preserve_norm=True, trunc_tol=1e-10)
    return psi_mps    

#===================================================================================================================
# Construct the interaction unitary involving all the terms as in the paper and then convert to Gate object
#===================================================================================================================

def fsim_matrix(theta):
    """fsim(theta, phi=0, beta=0)"""
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)

def build_fsim_gate(theta, left_site):
    """Return an fsim gate acting on neighbouring bath sites."""
    fsim_mat = fsim_matrix(theta)
    return Gate(matrix=fsim_mat, indices=[left_site, left_site + 1])

def build_UK(theta_K, theta_z):

    l1 = np.exp(1j * theta_z / 2)
    l2 = np.exp(-1j * theta_z / 2)
    c1 = np.cos(theta_K)
    s1 = np.sin(theta_K)

    mat = np.eye(8, dtype=np.complex128)

    mat[2, 2] = c1 * l1
    mat[2, 5] = 1j * s1 * l1
    mat[5, 2] = 1j * s1 * l1
    mat[5, 5] = c1 * l1

    # |011> = index 3, |100> = index 4: phase l2
    mat[3, 3] = l2
    mat[4, 4] = l2

    return mat

def build_UK_gate(N,theta_K, theta_z):
    UK_mat = build_UK(theta_K, theta_z)
    return Gate(matrix=UK_mat, indices=[N-1, N, N+1])

#====================================================================================================================
# Applying the gates to the circuit and then measuring the impurity magnetization at each step
#====================================================================================================================   

# =============================================================================
# Build the initial MPS
# =============================================================================

N = 6

theta =  np.pi/3
theta_K =  np.pi/4
theta_z = 0.5*np.sqrt(2)*(np.sqrt(2) - 1)*np.sin(theta)
no_floquet_steps = 100

orb_up, orb_down, n_up, n_down = build_initial_orbitals(N)

psi = build_initial_state(
    N,
    orb_up,
    orb_down,
    n_up,
    n_down
)

tensors = convert_psi_to_mps(
    psi,
    N,
    bond_dim=np.inf
)

psi_mps = state_mps(
    N,
    tensors
)


Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
impurity_mag = []
mz0 = psi_mps.measure_observable(Sz, [N])

impurity_mag.append(2*mz0)
for step in range(no_floquet_steps):

    # ==========================================================
    # First half of the kinetic evolution (even bonds)
    # ==========================================================

    # Down-spin bath
    for left in range(0, N-1, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps

    # Up-spin bath
    for left in range(N+1, 2*N, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps





    # ==========================================================
    # Second half of the kinetic evolution (odd bonds)
    # ==========================================================

    # Down-spin bath
    for left in range(1, N-1, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps

    # Up-spin bath
    for left in range(N+2, 2*N, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps

    # ==========================================================
    # Kondo gate
    # ==========================================================

    UK = build_UK(theta_K, theta_z)
    psi_mps.apply(UK, [N-1, N, N+1])        


    # ==========================================================
    # Measure impurity magnetization
    # ==========================================================

    mz = psi_mps.measure_observable(Sz, [N])
    impurity_mag.append(2*mz)

    #===============================================================================================================
    # Plotting the impurity magnetization as a function of time
    #===============================================================================================================

    print("Impurity magnetization:")
print(impurity_mag)

plt.figure(figsize=(12,8))
plt.plot(range(no_floquet_steps+1), impurity_mag)
plt.xlabel("Floquet step")
plt.ylabel(r"$\langle S_z^{imp}\rangle$")
plt.grid(True)
plt.show()

    


#


















































































































 
"""N = 6

orb_up, orb_down, n_up, n_down = build_initial_orbitals(N)

psi = build_initial_state(N, orb_up, orb_down, n_up, n_down)

print("Length of state vector =", len(psi))
print("Number of qubits =", int(np.log2(len(psi))))
tensors = convert_psi_to_mps(psi, N, bond_dim=np.inf)
psi_mps = state_mps(N, tensors)
print(psi_mps.nqbits)"""









