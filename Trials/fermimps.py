#__________________________________________________________________________________________________________________
#1: The relevant imports so as to make my life easier
#__________________________________________________________________________________________________________________
import numpy as np 
from qlimb.classical.mps import MPS
import qlimb.classical.utils as utils
from qlimb.classical.mpo import MPO
from itertools import combinations
from qlimb.classical.utils import apply_svd
from scipy.linalg import expm
from qlimb.classical.gates import Gate
from openfermion.linalg.givens_rotations import givens_decomposition_square
#__________________________________________________________________________________________________________________

#===================================================================================================================
# Now checking the MPS machinery on a fermi sea state, the previous one in checkmps.py was on a random state
#===================================================================================================================

# I will take an N site system of fermions, with a simple hamiltonian (the bath hamiltonian)

#===================================================================================================================
# Fermi Sea preparation 
#===================================================================================================================

def build_fermi_sea(N,t0):

    h_block = np.zeros((N, N))
    for i in range(N - 1):
        h_block[i, i+1] = -t0 / 2
        h_block[i+1, i] = -t0 / 2

    evals, evecs = np.linalg.eigh(h_block)

    N_occ = N//2

    occupied_orbitals = evecs[:,:N_occ]

    dim = 2**N
    psi = np.zeros(dim, dtype=complex)

    for occ_sites in combinations(range(N), N_occ):

        A = occupied_orbitals[list(occ_sites), :]

        amp = np.linalg.det(A)

        idx = 0
        for site in occ_sites:
            idx |= (1 << (N - 1 - site))

        psi[idx] = amp
    psi /= np.linalg.norm(psi)
    return psi

#================================================================================================================
# Convert to MPS tensor and then MPS
#================================================================================================================

def statevec_to_mps_tensors(psi, N, bond_dim=np.inf):
    d = 2
    tensors = []
    T = psi.reshape([d]*N)
    chi_left = 1

    for site in range(N-1):
        T = T.reshape(chi_left,d,-1,1)
        U, s, s_trunc, V = apply_svd(T, bond_dim=bond_dim,direction='right',preserve_norm=True, tol=1e-10)
        tensors.append(U)
        chi_left = U.shape[-1]
        T = V.reshape(chi_left,d,-1)
    tensors.append(T.reshape(chi_left, d, 1))
    return tensors

def state_mps(N,tensors):
    psi_mps = MPS(nqbits=N, phys_dim=2, tensors=tensors,bond_dim=np.inf, preserve_norm=True, trunc_tol=1e-10)
    return psi_mps


#==================================================================================================================
# Construct c dagger c mpo and then construct the 1RDM
#==================================================================================================================


def build_cdag_c_mpo_simple(N, i, j):

    I2 = np.eye(2,dtype=complex)

    Z = np.array([[1,0],
                  [0,-1]],dtype=complex)

    sp  = np.array([[0,1],
                    [0,0]],dtype=complex)

    sm  = np.array([[0,0],
                    [1,0]],dtype=complex)

    if i == j:

        Nop = (I2 + Z)/2

        tensors = []

        for k in range(N):

            op = Nop if k==i else I2

            tensors.append(op.reshape(1,2,2,1))

        return MPO(nqbits=N, phys_dim=2, tensors=tensors)

    left = min(i,j)
    right = max(i,j)

    tensors = []

    if i == left:
        for k in range(N):

            if k == left:
                op = -sp

            elif k == right:
                op = sm

            elif left < k < right:
                op = Z

            else:
                op = I2

            tensors.append(op.reshape(1,2,2,1))

    else:
        for k in range(N):

            if k == left:
                op = -sm

            elif k == right:
                op = sp

            elif left < k < right:
                op = Z

            else:
                op = I2

            tensors.append(op.reshape(1,2,2,1))        

    mpo = MPO(nqbits=N, phys_dim=2, tensors=tensors)
    return mpo

def build_1rdm(psi_mps, N):

    C = np.zeros((N,N), dtype=complex)

    for i in range(N):
        for j in range(N):

            mpo = build_cdag_c_mpo_simple(N, i, j)

            C[i,j] = psi_mps @ (mpo @ psi_mps)
    
    return C


# Parameters
N = 6
t0 = 1.0

# Build Fermi sea state vector
psi = build_fermi_sea(N, t0)

# Convert state vector -> MPS tensors
tensors = statevec_to_mps_tensors(psi, N)

# Build MPS
psi_mps = state_mps(N, tensors)

# Compute 1-RDM
C = build_1rdm(psi_mps, N)

print("1-RDM:")
print(np.real_if_close(C))

#===================================================================================================================
# Diagonalise 1rdm, in other words, get the natural orbitals and get the givens decompositions
#===================================================================================================================

evals, U = np.linalg.eigh(C)
decomp, diagonal = givens_decomposition_square(U)

def build_givens_matrix(N,i,j,theta,phi):

    G = np.eye(N, dtype=complex)

    c = np.cos(theta)
    s = np.sin(theta)

    G[i,i] = c
    G[j,j] = np.exp(1j*phi)*c

    G[i,j] = -np.exp(1j*phi)*s
    G[j,i] = s

    return G

R = np.eye(N, dtype=complex)

for layer in decomp:

    for i,j,theta,phi in layer:

        G = build_givens_matrix(
            N,
            i,
            j,
            theta,
            phi
        )

        R = G @ R

def fermionic_givens_gate(i, j, theta, phi):

    c = np.cos(theta)

    s = np.sin(theta)

    G = np.array([

        [1, 0, 0, 0],

        [0, c, -np.exp(1j*phi)*s, 0],

        [0, s,  np.exp(1j*phi)*c, 0],

        [0, 0, 0, 1]

    ], dtype=complex)

    return Gate(matrix=G, indices=(i,j))

#==================================================================================================================
# Converting psi_mps, which is now in the real space basis, to the natural orbital basis and constructing the 1rdm
# thereafter
#==================================================================================================================

def apply_givens_circuit(psi_mps, decomp, diagonal):
    psi_no = psi_mps.copy()
  
    for site, phase in enumerate(diagonal):
        phase_gate = np.array([[1, 0],
                               [0, np.conj(phase)]], dtype=complex)
        gate = Gate(matrix=phase_gate, indices=(site,))
        psi_no = gate@psi_no


    for layer in reversed(decomp):
        for i, j, theta, phi in layer:
            gate = fermionic_givens_gate(i, j, theta, phi)
            psi_no = gate@psi_no

    return psi_no

def build_1rdm_no(psi_no, N):

    C_no = np.zeros((N,N), dtype=complex)

    for i in range(N):
        for j in range(N):

            mpo = build_cdag_c_mpo_simple(N, i, j)

            C_no[i,j] = psi_no @ (mpo @ psi_no)

    return C_no


psi_no = apply_givens_circuit(psi_mps, decomp, diagonal)


C_no = build_1rdm_no(psi_no, N)

print("\n1-RDM in natural orbital basis:")
print(np.real_if_close(C_no))
    



