
#3
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

from qlimb.classical.mps import MPS
from qlimb.classical.gates import Gate
from qlimb.classical.utils import apply_svd

# Floquet cycle exactly matches the previous work.
# =================================================================================================================
# 1: Building Fermi-sea orbitals
# =================================================================================================================
 
def build_kspace_fermi_orbitals(N_bath):
    """Half-filled Fermi sea orbitals for one chain of length N_bath """
    x = np.arange(N_bath)
    m = N_bath / 2.0
    if m % 2 != 0:
        j_range = range(-int(m // 2), int(m // 2) + 1)
    else:
        j_range = range(-int(m // 2), int(m // 2))
    k_occ = np.array([2.0 * np.pi * j / N_bath for j in j_range])

    orb = np.zeros((N_bath, len(k_occ)), dtype=complex)
    for col, k in enumerate(k_occ):
        orb[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return orb, k_occ


def build_initial_orbitals(N_bath):
    orb_up, _ = build_kspace_fermi_orbitals(N_bath)
    orb_down, _ = build_kspace_fermi_orbitals(N_bath)
    n_up = orb_up.shape[1]
    n_down = orb_down.shape[1]
    return orb_up, orb_down, n_up, n_down


# =================================================================================================================
# 2: Building the initial statevector
# =================================================================================================================

def build_initial_state(N, orb_up, orb_down, n_up, n_down):
    psi = np.zeros(2 ** (2 * N + 1), dtype=complex)
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

            # up chain -> qubits 0..N-1, REVERSED
            phys_left = spin_up_qbits[::-1].copy()
            # down chain -> qubits N+1..2N, not reversed
            phys_right = spin_down_qbits.copy()

            #I am doing the following step because soumyadeep dada had done qc.x(list1): flip every bath qubit (everything except the impurity)
            phys_left = 1 - phys_left
            phys_right = 1 - phys_right

            full_state = np.concatenate((phys_left, impurity_qbit, phys_right))

            idx = 0
            for bit in full_state:
                idx = (idx << 1) | bit
            psi[idx] = a_down * a_up

    return psi


def convert_psi_to_mps(psi, N, bond_dim=np.inf):
    d = 2
    tensors = []
    T = psi.reshape([d] * (2 * N + 1))
    chi_left = 1
    for site in range(2 * N):
        T = T.reshape(chi_left, d, -1, 1)
        U, s, s_trunc, V = apply_svd(T, bond_dim=bond_dim, direction='right', preserve_norm=True, tol=1e-10)
        tensors.append(U)
        chi_left = U.shape[-1]
        T = V.reshape(chi_left, d, -1)
    tensors.append(T.reshape(chi_left, d, 1))
    return tensors


def state_mps(N, tensors, bond_dim=np.inf):
    return MPS(nqbits=2 * N + 1, phys_dim=2, tensors=tensors, bond_dim=bond_dim,
               preserve_norm=True, trunc_tol=1e-10)


# =================================================================================================================
# 3: Gates
# =================================================================================================================

def fsim_matrix(theta):
    """fsim(theta, phi=0, beta=0) -- identical to qiskit's fsim(theta,0,0)."""
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)


def build_fsim_gate(theta, left_site):
    return Gate(matrix=fsim_matrix(theta), indices=[left_site, left_site + 1])


def build_UK(theta_K, theta_z):

    l1 = np.exp(1j * theta_z / 2)
    l2 = np.exp(-1j * theta_z / 2)
    c1 = np.cos(theta_K)
    s1 = np.sin(theta_K)

    mat = np.eye(8, dtype=np.complex128)
    mat[1, 1] = c1 * l1
    mat[1, 6] = 1j * s1 * l1
    mat[6, 1] = 1j * s1 * l1
    mat[6, 6] = c1 * l1
    mat[3, 3] = l2
    mat[4, 4] = l2
    return mat


# =================================================================================================================
# 4: Building the initial MPS and running the Floquet evolution (decomposing the floquet unitary exactly as done 
#    in the previous paper on 1 channel kondo)
# =================================================================================================================

N = 6
theta = np.pi / 3
theta_K = np.pi / 4
theta_z = 0.5 * np.sqrt(2) * (np.sqrt(2) - 1) * np.sin(theta)
no_floquet_steps = 100
bond_dim =  np.inf  # NO truncation

orb_up, orb_down, n_up, n_down = build_initial_orbitals(N)
psi = build_initial_state(N, orb_up, orb_down, n_up, n_down)
tensors = convert_psi_to_mps(psi, N, bond_dim=bond_dim)
psi_mps = state_mps(N, tensors, bond_dim=bond_dim)

Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
impurity_mag = []
mz0 = psi_mps.measure_observable(Sz, [N])
impurity_mag.append(2 * mz0)

UK = build_UK(theta_K, theta_z)
bond_dim_array = []


#----------------------------------------------------------------------------------------------------------
# Directly exponentiating the kinetic part of the Hamiltonian and applying it to the MPS.
#----------------------------------------------------------------------------------------------------------

#====================================================================================================================
# Exact (non-Trotterized) reference using qlimb's MPS/MPO machinery instead of raw statevectors.
# H_kin is built as the FULL un-split XX+YY sum over all bonds (no even/odd bond splitting),
# exponentiated once as a dense matrix, then converted to an MPO via MPO.from_matrix.
# U_K is embedded on the full chain the same way and also converted to an MPO.
# Reuses N, theta, theta_K, theta_z, no_floquet_steps, psi, UK (from build_UK), Sz from your main script.
#====================================================================================================================

import numpy as np
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse import kron as skron, identity as sident

from qlimb.classical.mps import MPS
from qlimb.classical.mpo import MPO

total_qubits = 2 * N + 1
phys_dim = 2

# ---------------------------------------------------------------------------------------------------------------
# 1. Build H_kin_full as a sparse Pauli sum, ALL bonds in each chain at once (no brick splitting)
# ---------------------------------------------------------------------------------------------------------------

X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
hbond = skron(X, X, format='csr') + skron(Y, Y, format='csr')

def embed_two_qubit_sparse(op4, left_site, total_qubits):
    left_dim = 2 ** left_site
    right_dim = 2 ** (total_qubits - left_site - 2)
    return skron(skron(sident(left_dim, format='csr'), op4, format='csr'),
                 sident(right_dim, format='csr'), format='csr')

H_kin_full = sparse.csr_matrix((2 ** total_qubits, 2 ** total_qubits), dtype=complex)
for left in range(0, N - 1):               # ALL bonds in the up chain (qubits 0..N-1), not even/odd split
    H_kin_full = H_kin_full + embed_two_qubit_sparse(hbond, left, total_qubits)
for left in range(N + 1, 2 * N):           # ALL bonds in the down chain (qubits N+1..2N)
    H_kin_full = H_kin_full + embed_two_qubit_sparse(hbond, left, total_qubits)

H_kin_full = H_kin_full.toarray()

# ---------------------------------------------------------------------------------------------------------------
# 2. Dense unitaries: exact kinetic step at angle=theta, and U_K embedded on the full chain
# ---------------------------------------------------------------------------------------------------------------

eval, evec = np.linalg.eigh(H_kin_full)
U = evec@np.diag(np.exp(1j * (theta / 2) * eval))@evec.conj().T   

def embed_op_dense(op, first_qubit, total_qubits):
    k = int(round(np.log2(op.shape[0])))
    left_dim = 2 ** first_qubit
    right_dim = 2 ** (total_qubits - first_qubit - k)
    return np.kron(np.eye(left_dim), np.kron(op, np.eye(right_dim)))

U_K_dense = embed_op_dense(UK, N - 1, total_qubits)   # UK from build_UK(theta_K, theta_z), reused as-is

# ---------------------------------------------------------------------------------------------------------------
# 3. Convert both dense unitaries to MPOs (qlimb decomposes them via successive SVDs)
# ---------------------------------------------------------------------------------------------------------------

mpo_bond_dim = phys_dim ** total_qubits   # "no truncation" ceiling, same convention as your trivial-state test

U_kin_mpo = MPO.from_matrix(U, phys_dim=phys_dim, nqbits=total_qubits,
                             max_bond_dim=mpo_bond_dim, trunc_tol=1e-12)
U_K_mpo = MPO.from_matrix(U_K_dense, phys_dim=phys_dim, nqbits=total_qubits,
                           max_bond_dim=mpo_bond_dim, trunc_tol=1e-12)

# ---------------------------------------------------------------------------------------------------------------
# 4. Build the initial MPS from your existing statevector psi and run the exact Floquet evolution
# ---------------------------------------------------------------------------------------------------------------

mps_exact = psi_mps.copy()   # same initial state you fed into the MPS circuit

impurity_mag_exact = [2 * mps_exact.measure_observable(Sz, [N])]
bond_dims_exact = [mps_exact.get_bond_dimensions()]

for step in range(no_floquet_steps):
    mps_exact = U_kin_mpo @ mps_exact       # exact kinetic step, no even/odd splitting
    mps_exact = U_K_mpo @ mps_exact         # exact Kondo interaction, same as circuit's UK
    impurity_mag_exact.append(2 * mps_exact.measure_observable(Sz, [N]))
    bond_dims_exact.append(mps_exact.get_bond_dimensions())

print("Exact (MPO/MPS, non-Trotterized) impurity magnetization:")
print(impurity_mag_exact)
print("Bond dimensions after each step:")
print(bond_dims_exact)

# ---------------------------------------------------------------------------------------------------------------
# 5. Compare against your Trotterized circuit result (impurity_mag), trimmed to match indexing
# ---------------------------------------------------------------------------------------------------------------

impurity_mag_trimmed = impurity_mag[:-1]   # drop trailing boundary-closure entry (see earlier discussion)

max_diff = np.max(np.abs(np.array(impurity_mag_trimmed) - np.array(impurity_mag_exact)))
print("Max |Trotter - Exact| over all steps:", max_diff)

#----------------------------------------------------------------------------------------------------------
# directly exponentiating the kinetic part of the Hamiltonian and applying it to the statevector.
#----------------------------------------------------------------------------------------------------------

# =================================================================================================================
# 6: EXACT (non-Trotterized) reference: H_kin exponentiated directly, no even/odd brick splitting
# =================================================================================================================
'''from scipy import sparse
from scipy.sparse import kron as skron, identity as sident

total_qubits = 2 * N + 1
dim = 2 ** total_qubits

X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))

def embed_two_qubit_sparse(op4, left_site, total_qubits):
    left_dim = 2 ** left_site
    right_dim = 2 ** (total_qubits - left_site - 2)
    return skron(skron(sident(left_dim, format='csr'), op4, format='csr'),
                 sident(right_dim, format='csr'), format='csr')

hbond = skron(X, X, format='csr') + skron(Y, Y, format='csr')

H_kin_full = sparse.csr_matrix((dim, dim), dtype=complex)
for left in range(0, N - 1):          # ALL bonds (0,1)...(N-2,N-1) in the up chain, not split even/odd
    H_kin_full = H_kin_full + embed_two_qubit_sparse(hbond, left, total_qubits)
for left in range(N + 1, 2 * N):      # ALL bonds (N+1,N+2)...(2N-1,2N) in the down chain
    H_kin_full = H_kin_full + embed_two_qubit_sparse(hbond, left, total_qubits)

H_kin_full = H_kin_full.toarray()

# Diagonalize once (Hermitian) -> exponentiate for any angle cheaply, no repeated expm calls
evals, evecs = np.linalg.eigh(H_kin_full)
evecs_dag = evecs.conj().T

def U_kin_exact(angle):
    """exp(i * angle/2 * H_kin_full): the un-split analogue of Ukin at Floquet angle `angle`."""
    return evecs @ np.diag(np.exp(1j * (angle / 2) * evals)) @ evecs_dag

def embed_op_dense(op, first_qubit, total_qubits):
    k = int(round(np.log2(op.shape[0])))
    left_dim = 2 ** first_qubit
    right_dim = 2 ** (total_qubits - first_qubit - k)
    return np.kron(np.eye(left_dim), np.kron(op, np.eye(right_dim)))

UK_matrix = build_UK(theta_K, theta_z)        # already exact, reused as-is
U_K_full = embed_op_dense(UK_matrix, N - 1, total_qubits)

U_F_exact = U_K_full @ U_kin_exact(theta)     # Eq. (5): U_F = U_K * U_kin, U_kin now exact (no brick split)

def expectation_Sz(psi_vec, qubit, total_qubits):
    left = 2 ** qubit
    right = 2 ** (total_qubits - qubit - 1)
    psi_r = psi_vec.reshape(left, 2, right)
    p0 = np.sum(np.abs(psi_r[:, 0, :]) ** 2)
    p1 = np.sum(np.abs(psi_r[:, 1, :]) ** 2)
    return 0.5 * (p0 - p1)

psi_exact = psi.copy()   # same initial state you fed into the MPS circuit
impurity_mag_exact = [2 * expectation_Sz(psi_exact, N, total_qubits)]

for step in range(no_floquet_steps):
    psi_exact = U_F_exact @ psi_exact
    impurity_mag_exact.append(2 * expectation_Sz(psi_exact, N, total_qubits))

print("Exact (non-Trotterized) impurity magnetization:")
print(impurity_mag_exact)

ssdata = np.loadtxt("N = 6, theta = 1.05, theta_k = 0.79, t = 100_sz_tol.txt")
plt.figure(figsize=(12, 8))
plt.plot(range(len(impurity_mag)), impurity_mag,
         label=f"MPS Trotter circuit (max $\\chi$={bond_dim})", color='yellow', linewidth=2)
plt.plot(range(len(impurity_mag_exact)), impurity_mag_exact,
         label="Exact (H_kin exponentiated, no brick-splitting)", color='blue', linestyle='-.', linewidth=2)
plt.plot(range(len(ssdata)), ssdata[:, 1], label="SS plot (no MPS)", color='red', linestyle='--', linewidth=2)
plt.xlabel("Floquet step")
plt.ylabel(r"$\langle S_z^{imp}\rangle$")
plt.grid(True)
plt.legend()
plt.title("MPS Trotter Circuit vs Exact (non-Trotterized) Reference")
plt.show()'''

#----------------------------------------------------------------------------------------------------------

'''if no_floquet_steps > 0:

    # Initial half kinetic evolution (theta) 
    for left in range(0, N - 1, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps
    for left in range(N + 1, 2 * N, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps
    for left in range(1, N - 1, 2):
        psi_mps = build_fsim_gate(2*  theta, left) @ psi_mps
    for left in range(N + 2, 2 * N, 2):
        psi_mps = build_fsim_gate(2 * theta, left) @ psi_mps

    psi_mps.apply(UK, [N - 1, N, N + 1])

    mz = psi_mps.measure_observable(Sz, [N])
    impurity_mag.append(2 * mz)

    # Remaining full Floquet steps (2*theta) 
    for step in range(1, no_floquet_steps):
        for left in range(0, N - 1, 2):
            psi_mps = build_fsim_gate(2 * theta, left) @ psi_mps
        for left in range(N + 1, 2 * N, 2):
            psi_mps = build_fsim_gate(2 * theta, left) @ psi_mps
        for left in range(1, N - 1, 2):
            psi_mps = build_fsim_gate(2 * theta, left) @ psi_mps
        for left in range(N + 2, 2 * N, 2):
            psi_mps = build_fsim_gate(2 * theta, left) @ psi_mps

        psi_mps.apply(UK, [N - 1, N, N + 1])

        mz = psi_mps.measure_observable(Sz, [N])
        impurity_mag.append(2 * mz)

    # Final inverse half kinetic evolution (theta) 
    for left in range(0, N - 1, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps
    for left in range(N + 1, 2 * N, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps

    mz = psi_mps.measure_observable(Sz, [N])
    impurity_mag.append(2 * mz)

# =================================================================================================================
# 5: Plot
# =================================================================================================================

print("Impurity magnetization:")
print(impurity_mag)
print(psi_mps.get_bond_dimensions())

ssdata = np.loadtxt("N = 6, theta = 1.05, theta_k = 0.79, t = 100_sz_tol.txt")
plt.figure(figsize=(12, 8))
plt.plot(range(len(impurity_mag)), impurity_mag, label=f"MPS (max $\chi$ = {bond_dim})", color='yellow', linewidth=2)
plt.plot(range(len(ssdata)), ssdata[:, 1], label="SS plot (no MPS)", color='red', linestyle ='--', linewidth=2)
plt.xlabel("Floquet step")
plt.ylabel(r"$\langle S_z^{imp}\rangle$")
plt.grid(True)
if bond_dim == np.inf:    
    plt.title("Impurity Magnetization vs No. of Floquet Steps\n\n" f"$\chi$ = {psi_mps.get_bond_dimensions()}\n" f"Maximum $\chi$ = {bond_dim} --> No truncation", fontsize=14,fontweight='bold')
else:
    plt.title("Impurity Magnetization vs No. of Floquet Steps\n\n" f"$\chi$ = {psi_mps.get_bond_dimensions()}\n" f"Maximum $\chi$ = {bond_dim}", fontsize=14,fontweight='bold')
plt.legend()
plt.show()'''