#==================================================================================================================
# Relevant imports
#==================================================================================================================
import numpy as np 
from qlimb.classical.mps import MPS
#import qlimb.classical.utils as utils
from qlimb.classical.mpo import MPO
from itertools import combinations
from qlimb.classical.utils import apply_svd
#from scipy.linalg import expm
from qlimb.classical.gates import Gate
from openfermion.linalg.givens_rotations import givens_decomposition_square
#import matplotlib.pyplot as plt

#__________________________________________________________________________________________________________________

#==================================================================================================================
# Define the Hamiltonian for the bath (the spin term in H1 excluded)
#==================================================================================================================

def bath_hopping_matrix(L, t0 = 1.0):           #change t0 later if necessary 
    """
    Returns the hopping matrix for a 1D chain of length L with hopping parameter t0.

    Parameters:
    L (int): Length of the chain.
    t0 (float): Hopping parameter.
    """
    H1 = np.zeros((L, L), dtype=np.complex128)
    for i in range(L - 1):
        H1[i, i + 1] = -t0/2
        H1[i + 1, i] = -t0/2

    # since Periodic boundary condition    
    H1[0, L - 1] = -t0/2  
    H1[L - 1, 0] = -t0/2  
    return H1

#==================================================================================================================
# Now I'll code for the initialization step (n=0)
#==================================================================================================================

# Ute can be constructed for the general step n, no need to define a special case for n=0 because the n in the exp
# term in the middle will automatically take care of the n=0 case.

def Ute(L, t0, t, n):
    """
    Constructs the unitary operator Ute for the time evolution of the bath.
    
    Parameters:
    L (int): Length of the chain.
    t0 (float): Hopping parameter.
    t (float): Time step for evolution.
    n (int): Current time step index.
    
    Returns:
    np.ndarray: The unitary operator Ute as a matrix.
    """
    H1 = bath_hopping_matrix(L, t0)
    
    ep, Uft = np.linalg.eigh(H1)
    Ute = Uft @ np.diag(np.exp(-1j * ep * t * n)) @ Uft.conj().T
    return Ute

# The unitary to transform into the natural orbital basis at n=0 is simply Uft

def U0(L, t0):
    """
    Constructs the unitary operator U0 for transforming into the natural orbital basis at n=0.
    
    Parameters:
    L (int): Length of the chain.
    t0 (float): Hopping parameter.
    
    Returns:
    np.ndarray: The unitary operator U0 as a matrix.
    """
    H1 = bath_hopping_matrix(L, t0)
    
    ep, Uft = np.linalg.eigh(H1)
    return Uft

M0Σ = 0         #active orbitals equal to zero at n=0

# Constructing the vector v0𝜎 (as in notes) which expresses operators in the natural orbital basis in terms of the real
# space basis. 

def v0Σ(L, t0, t, n=0):
    """
    Constructs the vector v0∑ for transforming operators from the real space basis to the natural orbital basis at n=0.
    
    Parameters:
    L (int): Length of the chain.
    t0 (float): Hopping parameter.
    t (float): Time step for evolution.
    n (int): Current time step index (default is 0).
    
    Returns:
    np.ndarray: The vector v0∑ 
    """
    Uft = U0(L, t0)
    Ute0 = Ute(L, t0, t, n)
    V = Ute0@Uft
    v0Σ = V[0,:].conj().T
    return v0Σ

#==================================================================================================================
# Constructing the initial MPS
#==================================================================================================================

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
# 2: Building the initial statevector and then its MPS representation
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

#===================================================================================================================
# 4: Jordan Wigner MPO construction 
#===================================================================================================================

def dictionary(N, spin_label, k_bath): #dictionary to map the physical site k to a qubit index in the unfolded representation 
    if not isinstance(N,int):
        raise TypeError("N must be an integer")
    if not isinstance(spin_label,int):
        raise TypeError("spin_label must be either 0 or 1")
    if not isinstance(k_bath,int):
        raise TypeError("k must be an integer between 0 and N - 1") 
    if spin_label < 0 or spin_label > 1:
        raise ValueError("spin_label must be either 0 or 1")      
    if k_bath < 0 or k_bath > N - 1:
        raise ValueError("k must be an integer between 0 and N - 1")
    if spin_label == 0:           # 0 means spin up
        k_bath_new = N - k_bath - 1
    elif spin_label == 1:         # 1 means spin down
        k_bath_new = N + k_bath + 1    
    return k_bath_new

def k_imp_new(N):
    return N 


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
                # The impurity site is a spin, not a fermionic mode.
                # Therefore it must not contribute a Jordan-Wigner parity string.
                if k == k_imp_new(N):
                    local_op = I2
                else:
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
                # The impurity site is a spin, not a fermionic mode.
                # Therefore it must not contribute a Jordan-Wigner parity string.
                if k == k_imp_new(N):
                    local_op = I2
                else:
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
            site_i = alpha
            spin_i = 0
        else:
            site_i = alpha - n_bath
            spin_i = 1

        for beta in range(n_modes):

            if beta < n_bath:
                site_j = beta
                spin_j = 0
            else:
                site_j = beta - n_bath
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



#=================================================================================================================
# Some checks
#=================================================================================================================

# Parameters
N = 6          # or whatever bath size you are using
t0 = 1.0

# Build the occupied orbitals
orb_up, orb_down, n_up, n_down = build_initial_orbitals(N)

# Construct the initial many-body state and the mps
psi_vec = build_initial_state(N, orb_up, orb_down, n_up, n_down)

tensors = convert_psi_to_mps(psi_vec, N)

psi = state_mps(N, tensors)

C = correlation_matrix(N, psi)


'''np.set_printoptions(precision=6, suppress=True)
print("Correlation matrix:")
print(C)'''

#=================================================================================================================
# I am interested to have a look at the 1rdm in the natural orbital basis, so that i know which orbitals are 
# filled, which are empty and which are active
#=================================================================================================================

evals, U = np.linalg.eigh(C)


decomp, diagonal = givens_decomposition_square(U)


def build_givens_matrix(N, i, j, theta, phi):

    G = np.eye(2*N, dtype=complex)

    c = np.cos(theta)
    s = np.sin(theta)

    G[i,i] = c
    G[j,j] = np.exp(1j*phi)*c

    G[i,j] = -np.exp(1j*phi)*s
    G[j,i] = s

    return G


R = np.eye(2*N, dtype=complex)

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


'''def apply_givens_circuit(psi_mps, decomp, diagonal):
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

psi_no = apply_givens_circuit(psi, decomp, diagonal)'''

def build_1rdm_no(psi_no, N):
    n_modes = 2 * N
    C_no = np.zeros((n_modes, n_modes), dtype=complex)

    for alpha in range(n_modes):
        if alpha < N:
            site_i = alpha
            spin_i = 0
        else:
            site_i = alpha - N
            spin_i = 1

        for beta in range(n_modes):
            if beta < N:
                site_j = beta
                spin_j = 0
            else:
                site_j = beta - N
                spin_j = 1

            mpo = build_cdag_c_mpo(
                N,
                site_i,
                site_j,
                spin_i,
                spin_j,
            )

            C_no[alpha, beta] = psi_no @ (mpo @ psi_no)

    return C_no

'''C_no = build_1rdm_no(psi_no, N)
print(C_no)'''

