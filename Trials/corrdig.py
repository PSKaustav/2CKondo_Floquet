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
#__________________________________________________________________________________________________________________

#==================================================================================================================
# 2: Some Utility Functions
#==================================================================================================================

def dictionary(N, spin_label, k_bath): #dictionary to map the physical site k to a qubit index in the unfolded representation 
    if not isinstance(N,int):
        raise TypeError("N must be an integer")
    if not isinstance(spin_label,int):
        raise TypeError("spin_label must be either 0 or 1")
    if not isinstance(k_bath,int):
        raise TypeError("k must be an integer between 1 and N") 
    if spin_label < 0 or spin_label > 1:
        raise ValueError("spin_label must be either 0 or 1")      
    if k_bath < 1 or k_bath > N:
        raise ValueError("k must be an integer between 1 and N")
    if spin_label == 0:           # 0 means spin up
        k_bath_new = N - k_bath
    elif spin_label == 1:         # 1 means spin down
        k_bath_new = N + k_bath
    return k_bath_new

def k_imp_new(N):
    return N

#===============================================================================================================
# 3: Ground State Preparation 
#===============================================================================================================

def build_bath_and_get_orbitals(N, t0):
    n_bath = N
    
    # single spin hopping block
    h_block = np.zeros((n_bath, n_bath))
    for i in range(n_bath - 1):
        h_block[i, i+1] = -t0 / 2
        h_block[i+1, i] = -t0 / 2
    
    # full 2(N-1) x 2(N-1) single-particle Hamiltonian
    h1p = np.block([[h_block,                  np.zeros((n_bath, n_bath))],
                    [np.zeros((n_bath, n_bath)), h_block               ]])
    
    evals, evecs = np.linalg.eigh(h1p)

    # identify spin sector of each eigenvector
    up_mask = np.sum(np.abs(evecs[:n_bath, :])**2, axis=0)

    up_cols = np.where(up_mask > 0.5)[0]
    down_cols = np.where(up_mask < 0.5)[0]

    # half filling
    n_occ  = N - 1
    n_up   = n_occ // 2
    n_down = n_occ - n_up

    orb_up = evecs[:n_bath, up_cols[:n_up]]
    orb_down = evecs[n_bath:, down_cols[:n_down]]
    
    return evals, orb_up, orb_down, n_up, n_down

def build_state_vec(N, orb_up, orb_down, n_up, n_down):
    n_bath = N
    n_mps = 2*N + 1
    dim_mps = 2**n_mps

    state_vec = np.zeros(dim_mps, dtype=complex)

    for occ_up in combinations(range(n_bath), n_up):
        for occ_down in combinations(range(n_bath), n_down):
            amp_up = np.linalg.det(orb_up[list(occ_up),:])
            amp_down = np.linalg.det(orb_down[list(occ_down),:])
            amp = amp_up*amp_down

            occupied_qubits = []

            for k_bath in occ_up:
                q = dictionary(N,0,k_bath + 1)
                occupied_qubits.append(q)

            for k_bath in occ_down:
                q = dictionary(N,1,k_bath + 1)
                occupied_qubits.append(q)

            idx = sum(1 << (n_mps - 1 - q) for q in occupied_qubits)
            state_vec[idx] += amp
            

    state_vec /= np.linalg.norm(state_vec)
    return state_vec

# Now i need to compress this state_vec to a list of tensors. My goal is to use the class MPS from prof Thomas'
# repository to construct the MPS. To do the compression, i will use a hybrid code having both my code and a 
# method, called apply_svd, from Prof Thomas' repository.

def statevec_to_mps_tensors(state, n_mps, bond_dim=np.inf, trunc_tol=1e-10):
    d = 2
    tensors = []
    
    T = state.reshape([d] * n_mps)
    
    chi_L = 1
    for k in range(n_mps - 1):
        # reshape into 4-legged tensor (chi_L, d, d_rest, 1) for apply_svd
        T = T.reshape(chi_L, d, -1, 1)
        
        u_tensor, s, s_trunc, v_tensor = apply_svd(T, bond_dim, direction='right', 
                                                     preserve_norm=True, tol=trunc_tol)
        # u_tensor shape: (chi_L, d, chi_new)
        tensors.append(u_tensor)
        
        chi_new = u_tensor.shape[2]
        # v_tensor shape: (chi_new, d_rest, 1) — squeeze the last dim and continue
        T = v_tensor.reshape(chi_new, -1)
        chi_L = chi_new
    
    # last site
    tensors.append(T.reshape(chi_L, d, 1))
    return tensors

def build_mps(N,orb_up, orb_down, n_up, n_down):
    n_mps    = 2*N + 1
    state    = build_state_vec(N, orb_up, orb_down, n_up, n_down)
    tensors  = statevec_to_mps_tensors(state, n_mps, bond_dim=np.inf)
    psi      = MPS(nqbits=n_mps, phys_dim=2, tensors=tensors, bond_dim=np.inf)
    return psi

#===================================================================================================================
# 4: Jordan Wigner MPO construction 
#===================================================================================================================

def build_cdag_c_mpo(N, site_i, site_j,i_spin_label, j_spin_label):
    n_mps = 2*N + 1

    if i_spin_label == 0:
        site_i_qb = dictionary(N, 0, site_i)
    else:
        site_i_qb = dictionary(N, 1, site_i)

    if j_spin_label == 0:
        site_j_qb = dictionary(N, 0, site_j)
    else:
        site_j_qb = dictionary(N, 1, site_j)

    I2 = np.eye(2, dtype=complex)

    Z = np.array([[1, 0],
                  [0,-1]], dtype=complex)

    SP = np.array([[0,1],
                   [0,0]], dtype=complex)

    SM = np.array([[0,0],
                   [1,0]], dtype=complex)
    
    # Note that we use these definition of Sp and Sm so as to avoind the minus sign which comes when calculating 
    # ci dagger cj for i < j, which otherwise comes if we use the usual convention 
    if site_i_qb == site_j_qb:

        N_op = (I2 + Z)/2

        tensors = []

        for k in range(n_mps):

            if k == site_i_qb:
                local_op = N_op
            else:
                local_op = I2

            tensors.append(local_op.reshape(1,2,2,1))

        return MPO(
            nqbits=n_mps,
            phys_dim=2,
            tensors=tensors)
    left  = min(site_i_qb, site_j_qb)
    right = max(site_i_qb, site_j_qb)

    tensors = []

    if site_i_qb == left:
        for k in range(n_mps):

            if k == left:
                local_op = -SP

            elif k == right:
                local_op = SM

            elif left < k < right:
                local_op = Z

            else:
                local_op = I2

            tensors.append(
                local_op.reshape(1, 2, 2, 1)
            )
    else:
        for k in range(n_mps):

            if k == left:
                local_op = -SM

            elif k == right:
                local_op = SP

            elif left < k < right:
                local_op = Z

            else:
                local_op = I2

            tensors.append(
                local_op.reshape(1, 2, 2, 1)
            )        


    mpo = MPO(
        nqbits=n_mps,
        phys_dim=2,
        tensors=tensors
    )

    return mpo

#================================================================================================================
# 5: Constructing the 1-RDM
#================================================================================================================

def correlation_matrix(N, psi):

    n_bath = N 
    n_modes = 2*n_bath

    C = np.zeros((n_modes, n_modes), dtype=complex)

    for alpha in range(n_modes):

        if alpha < n_bath:
            site_i = alpha + 1
            spin_i = 0
        else:
            site_i = alpha - n_bath + 1
            spin_i = 1

        for beta in range(n_modes):

            if beta < n_bath:
                site_j = beta + 1
                spin_j = 0
            else:
                site_j = beta - n_bath + 1
                spin_j = 1

            mpo = build_cdag_c_mpo(
                N,
                site_i,
                site_j,
                spin_i,
                spin_j
            )

            C[alpha,beta] = psi @ (mpo @ psi)

    return C


#================================================================================================================
# 6: Some checks
#================================================================================================================

N = 9
t0 = 1.0

evals, orb_up, orb_down, n_up, n_down = build_bath_and_get_orbitals(N, t0)

vec1 = build_state_vec(N, orb_up, orb_down, n_up, n_down)

psi = build_mps(N, orb_up, orb_down, n_up, n_down)

vec2 = psi.to_vec()

print("||vec1 - vec2|| =", np.linalg.norm(vec1 - vec2))
print("norm(vec1) =", np.linalg.norm(vec1))
print("norm(vec2) =", np.linalg.norm(vec2))

#Output

#||vec1 - vec2|| = 1.044210238154656e-15
#norm(vec1) = 1.0000000000000002
#norm(vec2) = 0.9999999999999993

# Slater determinant construction is working.

# Mapping from occupations to qubit basis states is working.

# statevec_to_mps_tensors() is working.

# build_mps() is working.

# The MPS class is storing the state correctly.

print(psi.is_canonical())

#Output 

# True 

q = dictionary(N, 0, 1)
I2 = np.eye(2)
Z  = np.array([[1,0],
               [0,-1]])

N_op = (I2 + Z)/2

tensors = []

for k in range(psi.nqbits):

    if k == q:
        local_op = N_op
    else:
        local_op = I2

    tensors.append(
        local_op.reshape(1,2,2,1)
    )

mpo = MPO(
    nqbits=psi.nqbits,
    phys_dim=2,
    tensors=tensors
)

phi = mpo @ psi

val = psi @ phi

print(val)

#now calculating the expectation value by the usual way and then comparing with the above result
"""ops = []

for k in range(psi.nqbits):

    if k == q:
        ops.append(N_op)
    else:
        ops.append(I2)

O = ops[0]

for op in ops[1:]:
    O = np.kron(O, op)

vec = psi.to_vec()

val_dense = np.vdot(vec, O @ vec)

print(val_dense)

print("MPO value   =", val)
print("Dense value =", val_dense)
print("Difference  =", abs(val - val_dense))"""

# Output

#MPO value   = (0.49999999999999983+0j)
#Dense value = (0.4999999999999997+0j)
#Difference  = 1.1102230246251565e-16

# So success! Everything is working fine, we can move forwarddddddd!

# 'Some' checks done
# Some more checks

mpo = build_cdag_c_mpo(
    N,
    site_i=1,
    site_j=5,
    i_spin_label=0,
    j_spin_label=0
)

val = psi @ (mpo @ psi)

print(val)

C_up_exact = orb_up @ orb_up.conj().T       # sum over occupied states (phi n i times phi n j star)

print(C_up_exact[0,1])

print(abs(val - C_up_exact[0,1]))

#Output
#(0.4472135954999574+0j)
#0.44721359549995804
#6.661338147750939e-16

# Success!
#____________________________________________________________________________________________________________________
#--------------------------------------------------------------------------------------------------------------------


C = correlation_matrix(N, psi)

print(np.round(C, 6))
print(np.trace(C)) # = 8 for for N = 5
print(np.linalg.norm(C@C - C))   # = 0



#____________________________________________________________________________________________________________________
# Move all the subsequent parts to addkondo.py later.
#____________________________________________________________________________________________________________________

#====================================================================================================================
# The Frame matrix and its update
#====================================================================================================================

def initial_frame_exp(N):
    return np.zero((2*(N-1),2*(N-1)), dtype=complex)

def frame_update(F,h1p,T):
    frame_exp = expm(-1j*F)





