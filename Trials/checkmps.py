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

#==================================================================================================================
# State Preperation 
#==================================================================================================================

def prepare_state(N):
    dim = 2**N
    psi = np.zeros(dim,dtype=complex)
    idx1 = int('101001', 2)
    idx2 = int('000111', 2)
    idx3 = int('011001', 2)
    idx4 = int('101010', 2)
    idx5 = int('001110', 2)
    idx6 = int('010101', 2)
    idx7 = int('100011', 2)

    psi[idx2] = 1/np.sqrt(7)
    psi[idx1] = 1/np.sqrt(7)
    psi[idx3] = 1/np.sqrt(7)
    psi[idx4] = 1/np.sqrt(7)
    psi[idx5] = 1/np.sqrt(7)
    psi[idx6] = 1/np.sqrt(7)
    psi[idx7] = 1/np.sqrt(7)
    return psi

def statevec_to_mps_tensors(state, N, bond_dim=np.inf):
    d = 2
    tensors = []
    T = state.reshape([d]*N)
    chi_left = 1

    for site in range(N-1):
        T = T.reshape(chi_left,d,-1,1)
        U, s, s_trunc, V = apply_svd(T, bond_dim=np.inf,direction='right',preserve_norm=True, tol=1e-10)
        tensors.append(U)
        chi_left = U.shape[-1]
        T = V.reshape(chi_left,d,-1)
    tensors.append(T.reshape(chi_left, d, 1))
    return tensors

def state_mps(N,tensors):
    psi_mps = MPS(nqbits=N, phys_dim=2, tensors=tensors,bond_dim=np.inf, preserve_norm=True, trunc_tol=1e-10)
    return psi_mps

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

N = 6

state = prepare_state(N)

tensors = statevec_to_mps_tensors(state, N)

psi_mps = state_mps(N, tensors)

C = build_1rdm(psi_mps, N)
print(C)
#======================================================================================================================
"""print(np.linalg.norm(C.conj().T - C))
print(np.trace(C))"""
#======================================================================================================================

evals, U = np.linalg.eigh(C)

C_no = U.conj().T @ C @ U

#======================================================================================================================
"""print(np.linalg.norm(U.conj().T@U - np.eye(N)))"""
#======================================================================================================================

decomp, diagonal = givens_decomposition_square(U)

#======================================================================================================================
"""print(decomp)"""

"""print(diagonal)"""

"""print(len(decomp))"""
#======================================================================================================================

def build_givens_matrix(N, i, j, theta, phi):

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

U_rec = np.diag(diagonal) @ R

"""print(np.linalg.norm(U - U_rec))"""


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

psi_no = apply_givens_circuit(psi_mps, decomp, diagonal)

def build_1rdm_no(psi_no, N):

    C_no = np.zeros((N,N), dtype=complex)

    for i in range(N):
        for j in range(N):

            mpo = build_cdag_c_mpo_simple(N, i, j)

            C_no[i,j] = psi_no @ (mpo @ psi_no)

    return C_no

C_no = build_1rdm_no(psi_no, N)
print(C_no)


    
  

















