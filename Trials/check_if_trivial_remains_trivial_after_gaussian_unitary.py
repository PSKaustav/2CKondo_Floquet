#====================================================================================================================
# This file is to check whether applying a gaussian unitary to a trivial state will still remain trivial or not.
#====================================================================================================================

#====================================================================================================================
# Relevant imports 
#====================================================================================================================

#from curses import window

import numpy as np
#from itertools import combinations
#import matplotlib.pyplot as plt

from qlimb.classical.mps import MPS
#from qlimb.classical.gates import Gate
#from qlimb.classical.utils import apply_svd
from qlimb.classical.mpo import MPO
#from openfermion.linalg.givens_rotations import givens_decomposition

#from openfermion.linalg.givens_rotations import givens_decomposition_square

from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from scipy.linalg import expm

#--------------------------------------------------------------------------------------------------------------------

phys_dim = 2
nqbits = 7

vec_in = np.zeros(phys_dim**nqbits, dtype=complex)
vec_in[int("111111", 2)] = 1.0   

mps_in = MPS.from_vec(vec_in, bond_dim=phys_dim**nqbits, phys_dim=phys_dim, trunc_tol=1e-12, nqbits=nqbits)

H_sp = np.zeros((nqbits, nqbits), dtype=complex)

for i in range(nqbits - 1):
    H_sp[i, i+1] = -1
    H_sp[i+1, i] = -1

fermion_H = FermionOperator()

for i in range(nqbits):
    for j in range(nqbits):
        if abs(H_sp[i, j]) > 1e-12:
            fermion_H += H_sp[i, j] * FermionOperator(((i, 1), (j, 0)))

qubit_H = jordan_wigner(fermion_H)

H_manybody = get_sparse_operator(qubit_H, n_qubits=nqbits).toarray()

U = expm(-1j * H_manybody)

print(U.shape)

mpo = MPO.from_matrix(U, phys_dim=phys_dim, nqbits=nqbits, max_bond_dim=np.inf, trunc_tol=1e-12)

mps_out = mpo@mps_in

print("Bond dimension of the output MPS: ", mps_out.get_bond_dimensions())
print("Bond dimension of the input MPS: ", mps_in.get_bond_dimensions())

# Bond dimension of the output MPS:  [1, 1, 1, 1, 1]
# Bond dimension of the  input MPS:  [1, 1, 1, 1, 1]


# The following lines of code are just to check if the function for constructing the mpo for c^\dagger_i c_j is working properly or not.

I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)

def build_cdag_c_mpo_simple(N_total, imp_qubit, i, j):
    if i == j:
        Nop = (I2 + Z) / 2
        tensors = [(Nop if k == i else I2).reshape(1, 2, 2, 1) for k in range(N_total)]
        return MPO(nqbits=N_total, phys_dim=2, tensors=tensors)

    left, right = min(i, j), max(i, j)
    tensors = []
    for k in range(N_total):
        if k == imp_qubit:
            op = I2
        elif k == left:
            op = -sp if i < j else -sm
        elif k == right:
            op = sm if i < j else sp
        elif left < k < right:
            op = Z
        else:
            op = I2
        tensors.append(op.reshape(1, 2, 2, 1))
    return MPO(nqbits=N_total, phys_dim=2, tensors=tensors)

# Compute the one-body correlation matrix C_ij = <c_i^\dagger c_j>

C = np.zeros((nqbits, nqbits), dtype=complex)

for i in range(nqbits):
    for j in range(nqbits):
        mpo_cdag_c = build_cdag_c_mpo_simple(nqbits, 3, i, j)
        C[i, j] = mps_in @ mpo_cdag_c @ mps_in

print("Correlation matrix:")
print(np.real_if_close(C))
        

