
#** Copy this entire file to 1ckfbrmps.py later

from curses import window
from scipy.linalg import expm
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from qlimb.classical.mps import MPS
from qlimb.classical.gates import Gate
from qlimb.classical.utils import apply_svd
from qlimb.classical.mpo import MPO
from openfermion.linalg.givens_rotations import givens_decomposition
from openfermion.linalg.givens_rotations import givens_matrix_elements
from openfermion.linalg.givens_rotations import givens_decomposition_square

from Trials.corrdig import I2

# Floquet cycle exactly matches the previous work 
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


def convert_psi_to_tensors(psi, N, bond_dim=np.inf):
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
no_floquet_steps = 400
bond_dim =  np.inf  # NO truncation

orb_up, orb_down, n_up, n_down = build_initial_orbitals(N)
psi = build_initial_state(N, orb_up, orb_down, n_up, n_down)
tensors = convert_psi_to_tensors(psi, N, bond_dim=bond_dim)
psi_mps = state_mps(N, tensors, bond_dim=bond_dim)

Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
impurity_mag = []
mz0 = psi_mps.measure_observable(Sz, [N])
impurity_mag.append(2 * mz0)

UK = build_UK(theta_K, theta_z)
bond_dim_array = []

if no_floquet_steps > 0:

    # Initial half kinetic evolution (theta) 
    for left in range(0, N - 1, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps
    for left in range(N + 1, 2 * N, 2):
        psi_mps = build_fsim_gate(theta, left) @ psi_mps
    for left in range(1, N - 1, 2):
        psi_mps = build_fsim_gate(2 * theta, left) @ psi_mps
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

'''print("Impurity magnetization:")
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




# ==================================================================================================================
# Natural-orbital transformation of the bath (impurity qubit excluded from the mode space)
# ==================================================================================================================

N_total  = 2 * N + 1     # total physical qubits: N up + 1 impurity + N down
imp_qubit = N            # physical index of the impurity qubit

# Bath qubits in physical order, impurity excluded:
#   up-chain  -> physical qubits 0 .. N-1   (already reversed in your state prep)
#   down-chain-> physical qubits N+1 .. 2N
bath_qubits = [q for q in range(N_total) if q != imp_qubit]
n_bath = len(bath_qubits)     # = 2N

up_bath_qubits   = list(range(N - 1, -1, -1))     # [N-1, N-2, ..., 0] -- qubit N-1 is adjacent to impurity
down_bath_qubits = list(range(N + 1, 2 * N + 1))   # [N+1, N+2, ..., 2N] -- qubit N+1 is adjacent to impurity


def build_cdag_c_mpo_simple(N_total, imp_qubit, i, j):

    I2 = np.eye(2, dtype=complex)

    Z = np.array([[1, 0],
                  [0,-1]], dtype=complex)

    sp = np.array([[0,1],
                   [0,0]], dtype=complex)

    sm = np.array([[0,0],
                   [1,0]], dtype=complex)

    if i == j:
        Nop = (I2 + Z) / 2
        tensors = [
            (Nop if k == i else I2).reshape(1, 2, 2, 1) for k in range(N_total)
        ]
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


'''def build_cdag_c_mpo_simple(N_total, imp_qubit, i, j): #🔴

    I2 = np.eye(2, dtype=complex)

    Z = np.array([[1, 0],
                  [0,-1]], dtype=complex)

    sp = np.array([[0,1],
                   [0,0]], dtype=complex)

    sm = np.array([[0,0],
                   [1,0]], dtype=complex)

    # -------------------------
    # Number operator
    # -------------------------
    if i == j:

        Nop = (I2 + Z)/2

        tensors = []

        for k in range(N_total):

            if k < imp_qubit:

                if k == i:
                    op = Nop
                else:
                    op = I2
            
            else:

                op = I2

            tensors.append(op.reshape(1,2,2,1))

        return MPO(
            nqbits=N_total,
            phys_dim=2,
            tensors=tensors
        )

    left = min(i, j)
    right = max(i, j)

    tensors = []

    for k in range(N_total):
            
        if k < imp_qubit:

            # left endpoint
            if k == left:

                if i < j:
                    op = -sp
                else:
                    op = -sm

            # right endpoint
            elif k == right:

                if i < j:
                    op = sm
                else:
                    op = sp

            # Jordan-Wigner string
            elif left < k < right:

                    op = Z

            # everywhere else
            else:
                op = I2
        
        else:
            op = I2

        tensors.append(op.reshape(1,2,2,1))

    return MPO(
        nqbits=N_total,
        phys_dim=2,
        tensors=tensors
    )


def build_cdag_c_mpo_simple_up(N_total_up, i, j): #🔴

    I2 = np.eye(2, dtype=complex)

    Z = np.array([[1, 0],
                  [0,-1]], dtype=complex)

    sp = np.array([[0,1],
                   [0,0]], dtype=complex)

    sm = np.array([[0,0],
                   [1,0]], dtype=complex)

    # -------------------------
    # Number operator
    # -------------------------
    if i == j:

        Nop = (I2 + Z)/2

        tensors = []

        for k in range(N_total_up):

            

            if k == i:
                op = Nop
            else:
                op = I2
            
            

                

            tensors.append(op.reshape(1,2,2,1))

        return MPO(
            nqbits=N_total_up,
            phys_dim=2,
            tensors=tensors
        )

    left = min(i, j)
    right = max(i, j)

    tensors = []

    for k in range(N_total_up):
            
        

        # left endpoint
        if k == left:

            if i < j:
                op = -sp
            else:
                op = -sm

        # right endpoint
        elif k == right:

            if i < j:
                op = sm
            else:
                op = sp

        # Jordan-Wigner string
        elif left < k < right:

                op = Z

        # everywhere else
        else:
            op = I2
        
        tensors.append(op.reshape(1,2,2,1))

    return MPO(
        nqbits=N_total_up,
        phys_dim=2,
        tensors=tensors
    )


def build_cdag_c_mpo_simple_down(N_total_down, i, j): #🔴

    I2 = np.eye(2, dtype=complex)

    Z = np.array([[1, 0],
                  [0,-1]], dtype=complex)

    sp = np.array([[0,1],
                   [0,0]], dtype=complex)

    sm = np.array([[0,0],
                   [1,0]], dtype=complex)

    # -------------------------
    # Number operator
    # -------------------------
    if i == j:

        Nop = (I2 + Z)/2

        tensors = []

        for k in range(N_total_down):

            

            if k == i:
                op = Nop
            else:
                op = I2
            
            

            tensors.append(op.reshape(1,2,2,1))

        return MPO(
            nqbits=N_total_down,
            phys_dim=2,
            tensors=tensors
        )

    left = min(i, j)
    right = max(i, j)

    tensors = []

    for k in range(N_total_down):
            

        # left endpoint
        if k == left:

            if i < j:
                op = -sp
            else:
                op = -sm

        # right endpoint
        elif k == right:

            if i < j:
                op = sm
            else:
                op = sp

        # Jordan-Wigner string
        elif left < k < right:

                op = Z

        # everywhere else
        else:
            op = I2
    

    tensors.append(op.reshape(1,2,2,1))

    return MPO(
        nqbits=N_total_down,
        phys_dim=2,
        tensors=tensors
    )'''


# ---- 1) Correlation matrix C_ab = <psi| c_a^dagger c_b |psi>, a,b over bath qubits only ----
def build_1rdm_bath_up(psi_mps, up_bath_qubits, N_total):
    n = len(up_bath_qubits)
    C = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            mpo = build_cdag_c_mpo_simple(N_total, imp_qubit, up_bath_qubits[a], up_bath_qubits[b])
            C[a, b] = psi_mps @ (mpo @ psi_mps)
    return C

def build_1rdm_bath_down(psi_mps, down_bath_qubits, N_total):
    n = len(down_bath_qubits)
    C = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            mpo = build_cdag_c_mpo_simple(N_total, imp_qubit, down_bath_qubits[a], down_bath_qubits[b])
            C[a, b] = psi_mps @ (mpo @ psi_mps)
    return C



def fermionic_givens_gate(i, j, theta, phi):
    # i, j are now physical qubit indices
    c = np.cos(theta)
    s = np.sin(theta)
    ph = np.exp(1j * phi)
    G = np.array([
        [1, 0,     0,    0],
        [0, ph*c, -ph*s, 0],
        [0, s,     c,    0],
        [0, 0,     0,    ph]
    ], dtype=complex)
    return Gate(matrix=G, indices=(i, j))


# Updated apply_givens_circuit function
def apply_givens_circuit(psi_mps, decomp, diagonal, physical_qubits):
    """
    Apply the single-particle Givens decomposition only on the specified
    physical qubits. Qubits not listed in ``physical_qubits`` are left
    untouched automatically.
    """
    psi_no = psi_mps.copy()

    # Apply orbital phases
    for local_site, phase in enumerate(diagonal):
        phase_gate = np.array([[1, 0],
                               [0, phase]], dtype=complex)
        gate = Gate(matrix=phase_gate,
                    indices=(physical_qubits[local_site],))
        psi_no = gate @ psi_no

    # Apply Givens rotations
    for layer in reversed(decomp):
        for i, j, theta, phi in layer:
            qi = physical_qubits[i]
            qj = physical_qubits[j]
            gate = fermionic_givens_gate(qi, qj, theta, phi)
            psi_no = gate @ psi_no

    return psi_no


# ---------- Up-spin natural orbitals ----------
C_up = build_1rdm_bath_up(psi_mps, up_bath_qubits, N_total)
_, evecs_up = np.linalg.eigh(C_up)
decomp_up, diagonal_up = givens_decomposition_square(evecs_up)

psi_no = apply_givens_circuit(
    psi_mps,
    decomp_up,
    diagonal_up,
    up_bath_qubits,
)

# ---------- Down-spin natural orbitals ----------
C_down = build_1rdm_bath_down(psi_no, down_bath_qubits, N_total)
_, evecs_down = np.linalg.eigh(C_down)
decomp_down, diagonal_down = givens_decomposition_square(evecs_down)

psi_no = apply_givens_circuit(
    psi_no,
    decomp_down,
    diagonal_down,
    down_bath_qubits,
)


#=======================================================================================================================
# Now i want to build the 1-RDM of the bath qubits in the natural orbital basis, i.e. C_no_ab = <psi_no| c_a^dagger c_b |psi_no>, a,b over bath qubits only
#=======================================================================================================================

def build_1rdm_bath_no_up(psi_no, up_bath_qubits, N_total):
    n = len(up_bath_qubits)
    C = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            mpo = build_cdag_c_mpo_simple(
                N_total,
                imp_qubit,
                up_bath_qubits[a],
                up_bath_qubits[b],
            )
            C[a, b] = psi_no @ (mpo @ psi_no)
    return C


def build_1rdm_bath_no_down(psi_no, down_bath_qubits, N_total):
    n = len(down_bath_qubits)
    C = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            mpo = build_cdag_c_mpo_simple(
                N_total,
                imp_qubit,
                down_bath_qubits[a],
                down_bath_qubits[b],
            )
            C[a, b] = psi_no @ (mpo @ psi_no)
    return C

C_no_up = build_1rdm_bath_no_up(psi_no, up_bath_qubits, N_total)
C_no_down = build_1rdm_bath_no_down(psi_no, down_bath_qubits, N_total)
'''print(C_no_up, C_no_down)'''

#=======================================================================================================================
# Now the tasks from the blueprint
#=======================================================================================================================

#==================================================================================================================
# Define the Hamiltonian for the bath (the spin term in H1 excluded)
#==================================================================================================================

def bath_hopping_matrix(L, t0 = 1.0, boundary_condition = "periodic"):           #change t0 later if necessary 
    """
    Returns the hopping matrix for a 1D chain of length L with hopping parameter t0.

    Parameters:
    L (int): Length of the chain.
    t0 (float): Hopping parameter.
    boundary_condition (str): Type of boundary condition ('periodic' or 'open'). Default is 'periodic'.
    """
    H1 = np.zeros((L, L), dtype=np.complex128)
    for i in range(L - 1):
        H1[i, i + 1] = -t0/2
        H1[i + 1, i] = -t0/2

    if boundary_condition == "periodic":
        # since Periodic boundary condition    🔴 # how to include the spin term 
        H1[0, L - 1] = -t0/2  
        H1[L - 1, 0] = -t0/2  
        return H1
    else:
        return H1

#==================================================================================================================
# Now I'll code for a general step 
#==================================================================================================================

# Ute can be constructed for the general step n, no need to define a special case for n=0 because the n in the exp
# term in the middle will automatically take care of the n=0 case.

def Ute_n(L, t0, t, n):
    """
    Constructs the unitary operator Ute to transform the bath orbitals into the eigen basis of H1.
    
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

# The unitary to transform into the natural orbital basis at n=0 is simply Uft. I want to write a gen func

'''def U_n(psi, bath_qubits, N_total):   #🔴 WRONG 🔴 modified function (hopefully correct) right below
    """
    Constructs the unitary operator Un for transforming into the natural orbital basis at step n.
    
    Parameters:
    L (int): Length of the chain.
    t0 (float): Hopping parameter.
    n (int): Current time step index.
    
    Returns:
    np.ndarray: The unitary operator Un as a matrix.
    """

    C = build_1rdm_bath(psi, bath_qubits, N_total)      #🔴 this should be of the nth step (no of the nth step )
                                                        #🔴 psi should be in real space
    evals, Un = np.linalg.eigh(C)  
    perm = np.argsort(evals)[::-1]

    evals = evals[perm]

    Un = Un[:, perm]
    return evals, Un'''

#=======================================================================================================================
# Constructing U(n+1), equation 52 in the paper.
#=======================================================================================================================

def build_next_step_U_up(U_up_current_step, vec_filled, sub_vector, vec_empty):  # 🔴take qfull qempty | Done
    """Constructs the unitary operator U(n+1) for the next time step in the evolution. See equation 52 in the paper for details.
    Assuming we are the nth step. So current step = n and next step = n+1.
    Parameters:
    Returns:
    """
    def build_unitary_that_diag_next_step_C(n, psi, bath_qubits, N_total):
        """
        Constructs the unitary operator that diagonalizes the 1-RDM of the bath at step n+1. 🔴 I need to construct the 1rdm 
                                                                                                at step n+1 in the fewest body 
                                                                                                natural orbital basis of step n.
        Unitary will be constructed from the 1rdm of the bath of an individual spin sector.
        Parameters:
        n : int
            Current time step index.
        psi : MPS
            The MPS of the state at the n+1 th step (🔴🔴ask if psi(n+1) is before applying Us and Qs of the n+1 th step or after
                                                        since the basis is the fewest body natural orbital basis of step n, 
                                                        I think it should be before applying Us and Qs of the n+1 th step)
        bath_qubits : list[int]
            List of indices of the bath qubits.
        N_total : int
            Total number of qubits in the system.
        Returns:
        np.ndarray
            The unitary operator that diagonalizes the 1-RDM at step n+1.
        """
        C_up_next_step = build_1rdm_bath_up(psi, up_bath_qubits, N_total)       #note that the psi should be psi(n+1) in the fewest body natural orbital basis of step n, not psi(n) in the real space basis
                                                                                # i.e just after applying the mpo  
                                                                                #🔴 check the function build_1rdm if it actually 
                                                                                #   constructs the 1 rdm of the next step
       
        up_occupation, u_up = np.linalg.eigh(C_up_next_step)
        
        return u_up, up_occupation

    u_up, up_occupation = build_unitary_that_diag_next_step_C(n, psi, bath_qubits, N_total)

    filled_qbits_up, _ , empty_qbits_up = classify_orb(up_occupation) #🔴This should be of step n+1 right? (meaning next step)

    N_up = len(filled_qbits_up) + len(empty_qbits_up) + u_up.shape[0] 
    big_matrix_up = np.eye(N_up, dtype=complex)
    start_up = filled_qbits_up
    stop_up = filled_qbits_up + u_up.shape[0]

    big_matrix_up[start_up:stop_up] = u_up

    U_up_next_step = V_up_matrix_current_step(U_up_current_step, vec_filled, sub_vector, vec_empty) @ big_matrix_up    #Q_few_up is from the current step 

    return U_up_next_step

#print these matrices.


#----------------------------------------------------------------------------------------------------------------------------------------
# Constructing the first quantised matrix representation of Qfull, Qempty and Qfew, for updating the frame
#----------------------------------------------------------------------------------------------------------------------------------------

def build_Q_up_matrix(vec_filled, sub_vector, vec_empty):  #🔴 Note that sub_vector is the one defined in the middle of the function build_vec_and_mps_after_qfew, which is the vector with the active components plus 2 more components, one from the filled sector vector and another from the empty sector vector
    """
    Constructs the first quantized matrix representations of Q_full, Q_empty, and Q_few for the up-spin sector.
    Then constructs the entire Q matrix for the up-spin sector by combining Q_full, Q_empty, and Q_few. Refer to 
    the matrix just below equation 42 in the paper.

    Parameters
    ----------
    vec_filled : np.ndarray
        The filled component of the coupling vector.
    sub_vector : np.ndarray
        The active component of the coupling vector after applying Q_full and Q_empty.
    vec_empty : np.ndarray
        The empty component of the coupling vector.

    Returns
    -------
    Qfull_up : np.ndarray
        The first quantized matrix representation of Q_full for the up-spin sector.
    Qempty_up : np.ndarray
        The first quantized matrix representation of Q_empty for the up-spin sector.
    Qfew_up : np.ndarray
        The first quantized matrix representation of Q_few for the up-spin sector.
    """
    
    # Construct Qfull_up

    size_of_q_full = len(vec_filled)
    Q_full_matrix = np.eye(size_of_q_full, dtype=complex)
    for i in range(len(vec_filled) - 1, 0, -1):
        g_small = givens_matrix_elements(vec_filled[i], vec_filled[i - 1], which = 'right')
        G = np.eye(size_of_q_full, dtype=complex)
        G[np.ix_([i - 1, i], [i - 1, i])] = g_small
        Q_full_matrix = G @ Q_full_matrix
        vec_filled = G @ vec_filled

    # Construct Qempty_up

    size_of_q_empty = len(vec_empty)
    Q_empty_matrix = np.eye(size_of_q_empty, dtype=complex)
    for i in range(len(vec_empty) - 1, 0, -1):
        g_small = givens_matrix_elements(vec_empty[i-1], vec_empty[i], which = 'right')
        G = np.eye(size_of_q_empty, dtype=complex)
        G[np.ix_([i - 1, i], [i - 1, i])] = g_small
        Q_empty_matrix = G @ Q_empty_matrix
        vec_empty = G @ vec_empty


    # Construct Qfew_up
    
    size_of_q_few = len(sub_vector)
    Q_few_matrix = np.eye(size_of_q_few, dtype=complex)
    for i in range(len(sub_vector) - 1, 0, -1):
        g_small = givens_matrix_elements(sub_vector[i-1], sub_vector[i], which = 'right')
        G = np.eye(size_of_q_few, dtype=complex)
        G[np.ix_([i - 1, i], [i - 1, i])] = g_small
        Q_few_matrix = G @ Q_few_matrix    
        sub_vector = G @ sub_vector

    Q_up = block_diag(Q_full_matrix, np.eye(size_of_q_few-2, dtype=complex), Q_empty_matrix)@block_diag(np.eye(size_of_q_full-1, dtype=complex), Q_few_matrix, np.eye(size_of_q_empty-1, dtype=complex))

    return Q_up


def build_Q_down_matrix(vec_filled, sub_vector, vec_empty):  #🔴 Note that sub_vector is the one defined in the middle of the function build_vec_and_mps_after_qfew, which is the vector with the active components plus 2 more components, one from the filled sector vector and another from the empty sector vector
    """
    Constructs the first quantized matrix representations of Q_full, Q_empty, and Q_few for the down-spin sector.
    Then constructs the entire Q matrix for the down-spin sector by combining Q_full, Q_empty, and Q_few. Refer to 
    the matrix just below equation 42 in the paper.

    Parameters
    ----------
    vec_filled : np.ndarray
        The filled component of the coupling vector.
    sub_vector : np.ndarray
        The active component of the coupling vector after applying Q_full and Q_empty.
    vec_empty : np.ndarray
        The empty component of the coupling vector.

    Returns
    -------
    Qfull_down : np.ndarray
        The first quantized matrix representation of Q_full for the down-spin sector.
    Qempty_down : np.ndarray
        The first quantized matrix representation of Q_empty for the down-spin sector.
    Qfew_down : np.ndarray
        The first quantized matrix representation of Q_few for the down-spin sector.
    """
    
    # Construct Qfull_down

    size_of_q_full = len(vec_filled)
    Q_full_matrix = np.eye(size_of_q_full, dtype=complex)
    for i in range(len(vec_filled) - 1, 0, -1):
        g_small = givens_matrix_elements(vec_filled[i], vec_filled[i - 1], which = 'right')
        G = np.eye(size_of_q_full, dtype=complex)
        G[np.ix_([i - 1, i], [i - 1, i])] = g_small
        Q_full_matrix = G @ Q_full_matrix
        vec_filled = G @ vec_filled

    # Construct Qempty_down

    size_of_q_empty = len(vec_empty)
    Q_empty_matrix = np.eye(size_of_q_empty, dtype=complex)
    for i in range(len(vec_empty) - 1, 0, -1):
        g_small = givens_matrix_elements(vec_empty[i-1], vec_empty[i], which = 'right')
        G = np.eye(size_of_q_empty, dtype=complex)
        G[np.ix_([i - 1, i], [i - 1, i])] = g_small
        Q_empty_matrix = G @ Q_empty_matrix
        vec_empty = G @ vec_empty


    # Construct Qfew_down
    
    size_of_q_few = len(sub_vector)
    Q_few_matrix = np.eye(size_of_q_few, dtype=complex)
    for i in range(len(sub_vector) - 1, 0, -1):
        g_small = givens_matrix_elements(sub_vector[i-1], sub_vector[i], which = 'right')
        G = np.eye(size_of_q_few, dtype=complex)
        G[np.ix_([i - 1, i], [i - 1, i])] = g_small
        Q_few_matrix = G @ Q_few_matrix  
        sub_vector = G @ sub_vector

    Q_down = block_diag(Q_full_matrix, np.eye(size_of_q_few-2, dtype=complex), Q_empty_matrix)@block_diag(np.eye(size_of_q_full-1, dtype=complex), Q_few_matrix, np.eye(size_of_q_empty-1, dtype=complex))

    return Q_down


# V is the matrix that has been defined just before equation 44 in the notes 

def V_up_matrix_current_step(U_up_current_step, vec_filled, sub_vector, vec_empty):
    """Constructs the matrix V(n) for the current time step in the evolution. See equation 52 in the paper for details.
    Assuming we are the nth step. So current step = n and next step = n+1.
    Parameters:
    Returns:
    """
    Q_up_current_step = build_Q_up_matrix(vec_filled, sub_vector, vec_empty)  #🔴 vec_filled, sub_vector, vec_empty are from the current step
    V_up_current_step = U_up_current_step @ Q_up_current_step
    return V_up_current_step

def V_down_matrix_current_step(U_down_current_step, vec_filled, sub_vector, vec_empty):
    """Constructs the matrix V(n) for the current time step in the evolution. See equation 52 in the paper for details.
    Assuming we are the nth step. So current step = n and next step = n+1.
    Parameters:
    Returns:
    """
    Q_down_current_step = build_Q_down_matrix(vec_filled, sub_vector, vec_empty)  #🔴 vec_filled, sub_vector, vec_empty are from the current step
    V_down_current_step = U_down_current_step @ Q_down_current_step
    return V_down_current_step


def build_next_step_U_down(U_down_current_step, vec_filled, sub_vector, vec_empty):  # 🔴take qfull qempty | Done
    """Constructs the unitary operator U(n+1) for the next time step in the evolution. See equation 52 in the paper for details.
    Assuming we are the nth step. So current step = n and next step = n+1.
    Parameters:
    Returns:
    """
    def build_unitary_that_diag_next_step_C(n, psi, bath_qubits, N_total):
        """
        Constructs the unitary operator that diagonalizes the 1-RDM of the bath at step n+1. 🔴 I need to construct the 1rdm 
                                                                                                at step n+1 in the fewest body 
                                                                                                natural orbital basis of step n.
        Unitary will be constructed from the 1rdm of the bath of an individual spin sector.
        Parameters:
        n : int
            Current time step index.
        psi : MPS
            The MPS of the state at the n+1 th step (🔴🔴ask if psi(n+1) is before applying Us and Qs of the n+1 th step or after
                                                        since the basis is the fewest body natural orbital basis of step n, 
                                                        I think it should be before applying Us and Qs of the n+1 th step)
        bath_qubits : list[int]
            List of indices of the bath qubits.
        N_total : int
            Total number of qubits in the system.
        Returns:
        np.ndarray
            The unitary operator that diagonalizes the 1-RDM at step n+1.
        """
               #note that the psi should be psi(n+1) in the fewest body natural orbital basis of step n, not psi(n) in the real space basis
                                                                                # i.e just after applying the mpo  
        C_down_next_step = build_1rdm_bath_down(psi, down_bath_qubits, N_total)
        
        down_occupation , u_down = np.linalg.eigh(C_down_next_step)
        return  u_down, down_occupation
    
     
                   
    u_down, down_occupation = build_unitary_that_diag_next_step_C(n, psi, bath_qubits, N_total)

    filled_qbits_down, _ , empty_qbits_down = classify_orb(down_occupation) #🔴This should be of step n+1 right? (meaning next step)

    N_down = len(filled_qbits_down) + len(empty_qbits_down) + u_down.shape[0] 
    big_matrix_down = np.eye(N_down, dtype=complex)
    start_down = filled_qbits_down
    stop_down = filled_qbits_down + u_down.shape[0]

    big_matrix_down[start_down:stop_down] = u_down

    U_down_next_step = V_down_matrix_current_step(U_down_current_step, vec_filled, sub_vector, vec_empty) @ big_matrix_down    #Q_few_up is from the current step 

    return U_down_next_step


# The following function classifies the orb into full, empty and active orbs based on the evals of C
def classify_orb(evals, tol=1e-10):   #🔴 evals should be evals of C in natural orbitals of the nth step, but psimps of the n+1 step
    """
    Classify natural orbitals into filled, active and empty orbitals
    according to the eigenvalues of the one-particle density matrix.

    Parameters
    evals : Eigenvalues of the bath correlation matrix.
    tol : float | Numerical tolerance.

    Returns
    filled : list[int] | Indices of filled orbitals.
    active : list[int] | Indices of active orbitals.
    empty : list[int] | Indices of empty orbitals.
    """

    filled = []
    active = []
    empty = []

    for i, occ in enumerate(evals):

        if np.isclose(occ, 1.0, atol=tol):
            filled.append(i)

        elif np.isclose(occ, 0.0, atol=tol):
            empty.append(i)

        else:
            active.append(i)

    return filled, active, empty    #active orbitals equal to zero at n=0


# Constructing the vector vn𝜎 (as in notes) which expresses operators  in the natural orbital basis in terms of the real
# space basis. 

def v_n_Σ_up(L, t0, t, n, psi, bath_qubits, N_total, impurity_neighbor):
    """
    Construct the coupling vector v_{nΣ} for the bath site adjacent to the
    impurity, expressed in the natural-orbital basis.

    Parameters
    ----------
    impurity_neighbor : int
        Index (within the spin-resolved bath ordering) of the bath orbital
        adjacent to the impurity.

        For the current qubit ordering:
            up chain   -> impurity_neighbor = row N - 1
            down chain -> impurity_neighbor = row 0
    """

    U_up_next_step = build_next_step_U_up(U_up_current_step= , Q_few_up= )
    Unte = Ute_n(L, t0, t, n)

    V = Unte @ U_up_next_step

    return V[impurity_neighbor, :] #.conj()     #🔴 for the up_bath -> last row, depends on the size of the 1rdm

def v_n_Σ_down(L, t0, t, n, psi, bath_qubits, N_total, impurity_neighbor):
    """
    Construct the coupling vector v_{nΣ} for the bath site adjacent to the
    impurity, expressed in the natural-orbital basis.

    Parameters
    ----------
    impurity_neighbor : int
        Index (within the spin-resolved bath ordering) of the bath orbital
        adjacent to the impurity.

        For the current qubit ordering:
            up chain   -> impurity_neighbor = row N - 1
            down chain -> impurity_neighbor = row 0
    """

    U_down_next_step = build_next_step_U_down(U_down_current_step= , Q_few_down= )
    Unte = Ute_n(L, t0, t, n)

    V = Unte @ U_down_next_step

    return V[0, :] #.conj()     

def get_v_n_Σ__up_components(L, t0, t, n, psi, bath_qubits, N_total):
    """
    Construct v_{nΣ} and split it into filled, active and empty
    components.
    """

    evals, _ = build_1rdm_bath_up(psi, bath_qubits, N_total) #🔴this 1rdm should be diagonal
                                                                # because at the start of a step, we
                                                                # are in the natural orbital basis of
                                                                # that step because of Qopt in the last
                                                                # step of the previous floquet step.

    filled, active, empty = classify_orb(evals)

    v = v_n_Σ_up(L, t0, t, n, psi, bath_qubits, N_total)

    vec_filled = v[filled].copy()
    vec_active = v[active].copy()
    vec_empty  = v[empty].copy()

    return (
        vec_filled,
        vec_active,
        vec_empty,
        filled,
        active,
        empty,
        v,
    )


def get_v_n_Σ__down_components(L, t0, t, n, psi, bath_qubits, N_total):
    """
    Construct v_{nΣ} and split it into filled, active and empty
    components.
    """

    evals, _ = build_1rdm_bath_down(psi, bath_qubits, N_total) #🔴this 1rdm should be diagonal
                                                                # because at the start of a step, we
                                                                # are in the natural orbital basis of
                                                                # that step because of Qopt in the last
                                                                # step of the previous floquet step.

    filled, active, empty = classify_orb(evals)

    v = v_n_Σ_down(L, t0, t, n, psi, bath_qubits, N_total)

    vec_filled = v[filled].copy()
    vec_active = v[active].copy()
    vec_empty  = v[empty].copy()

    return (
        vec_filled,
        vec_active,
        vec_empty,
        filled,
        active,
        empty,
        v,
    )


def apply_givens_decomposition_to_vector(vector, decomposition):
    """
    Applies a Givens decomposition to a vector and simultaneously constructs
    the overall single-particle unitary.

    Parameters
    ----------
    vector : np.ndarray
        Input vector.

    decomposition : list[list[(i,j,theta,phi)]]
        Output of openfermion.givens_decomposition.

    Returns
    -------
    v : np.ndarray
        Transformed vector.

    U : np.ndarray
        Overall single-particle unitary satisfying

            v_out = U @ vector.
    """

    v = vector.astype(complex).copy()
    '''L = len(v)'''

    'U = np.eye(L, dtype=complex)'

    for parallel_group in decomposition:

        for (i, j, theta, phi) in parallel_group:

            c = np.cos(theta)
            s = np.sin(theta)
            phase = np.exp(1j * phi)

            # Same 2x2 rotation used on the vector
            G2 = np.array([
                [c, -phase * s],
                [s,  phase * c]
            ], dtype=complex)

            # Apply to the vector
            vi = v[i]
            vj = v[j]

            v[i] = G2[0,0] * vi + np.conj(G2[0,1]) * vj
            v[j] = G2[1,0] * vi + np.conj(G2[1,1]) * vj

            # Embed into the full Hilbert space
            '''Gfull = np.eye(L, dtype=complex)
            Gfull[np.ix_([i, j], [i, j])] = G2

            # Accumulate the total unitary
            U = Gfull @ U'''

    return v #U


def apply_reduction_as_gates(psi_mps, vector, physical_qubits, reduce_position='first'): #🔴 use this function for qfew because it is the only unitary among Qfull Qempty and Qfew that is needed to be applied on the MPS
    """
    vec_filled, vec_empty and vec_active will be passed into this function.
    """
    v_ordered = vector[::-1].copy() if reduce_position == 'last' else vector.copy()     #v_ordered since v can either be reversed or the unchanged one depending on reduce position
    order = physical_qubits[::-1] if reduce_position == 'last' else physical_qubits

    norm = np.linalg.norm(v_ordered)
    if norm < 1e-14:
        return psi_mps, vector.copy().astype(complex)

    v_normalized = (v_ordered / norm).reshape(1, -1)
    decomposition, left_unitary, diagonal = givens_decomposition(v_normalized)

    for parallel_group in decomposition:
        for (i, j, theta, phi) in parallel_group:
            G, (a, b) = fermionic_givens_gate_matrix(theta, phi, order[i], order[j])
            psi_mps = Gate(matrix=G, indices=(a, b)) @ psi_mps

    v_transformed = apply_givens_decomposition_to_vector(v_ordered,  decomposition)
    if reduce_position == 'last':
        v_transformed = v_transformed[::-1]

    return psi_mps, v_transformed

#🔴🔴🔴 No need to act qfull and qempty on psi_mps, only qfew, 3 body int and qopt
       

#---------------------------------------------------------------------------------------------------------------------------------------
'''def reduce_to_single_entry(vector, reduce_position='first'):
    """
    Collapses a row vector to a single nonzero entry via Givens rotations
    Parameters:
    vector : np.ndarray
        The input vector to be reduced.
    reduce_position : str, optional
        Specifies where the nonzero entry should be placed ('first' or 'last'). Default is 'first'.
    Returns:
    np.ndarray
        The reduced vector with a single nonzero entry.
    """
    v_reverse = vector[::-1].copy() if reduce_position == 'last' else vector.copy()

    norm = np.linalg.norm(v_reverse)
    if norm < 1e-14:
        return vector.copy().astype(complex), np.eye(len(vector), dtype=complex)  # nothing to reduce 

    v_normalized = (v_reverse / norm).reshape(1, -1)
    decomposition, left_unitary, diagonal = givens_decomposition(v_normalized)
    v_transformed, U = apply_givens_decomposition_to_vector(v_reverse, decomposition)

    if reduce_position == 'last':
        v_transformed = v_transformed[::-1]
    return v_transformed, U'''

def build_vec_and_mps_after_qfull(psi_mps, vec_filled, physical_qubits):
    psi_mps_after_qfull, filled_reduced = apply_reduction_as_gates(
        psi_mps,
        vec_filled,
        physical_qubits,
        reduce_position='last'
    )
    return psi_mps_after_qfull, filled_reduced

#🔴 How to make them parallel 

def build_vec_and_mps_after_qempty(psi_mps, vec_empty, physical_qubits):
    psi_mps_after_qempty, empty_reduced = apply_reduction_as_gates(
        psi_mps,
        vec_empty,
        physical_qubits,
        reduce_position='first'
    )
    return psi_mps_after_qempty, empty_reduced    #🔴This mps will have both qfull and qempty right?
                                                  #i still need to do qfew here



'''def get_qfull(vec_filled, vec_active, vec_empty):
    
    _, Qsmall = reduce_to_single_entry(
        vec_filled,
        reduce_position="last"
    )

    L = len(vec_filled) + len(vec_active) + len(vec_empty)
    Qfull = np.eye(L, dtype=complex)

    nfilled = len(vec_filled)
    Qfull[:nfilled, :nfilled] = Qsmall

    return Qfull'''


'''def get_qempty(vec_filled, vec_active, vec_empty):
    
    _, Qsmall = reduce_to_single_entry(
        vec_empty,
        reduce_position="first"
    )

    L = len(vec_filled) + len(vec_active) + len(vec_empty)
    Qempty = np.eye(L, dtype=complex)

    nfilled = len(vec_filled)
    nactive = len(vec_active)
    start = nfilled + nactive

    Qempty[start:, start:] = Qsmall

    return Qempty'''

    
def build_vec_and_mps_after_qfew(psi_mps, filled_reduced, vec_active, empty_reduced, physical_qubits):
    """
    Applies Givnes rotations to the vector transformed by givens rotation corresponding to Qfull and Qempty
    to find Qfew

    Example
    -------
    Input:
        [0,0,0,a,b,c,d,e,f,0,0,0]

    Output:
        [0,0,0,0,0,0,0,0,alpha,0,0,0]
    """

    vector = np.concatenate((filled_reduced, vec_active, empty_reduced))
    vector = vector.astype(complex).copy()

    nz = np.flatnonzero(np.abs(vector) > 1e-12)
    if len(nz) == 0:
        return psi_mps, vector

    start = nz[0]
    end = nz[-1] + 1

    sub_vector = vector[start:end]

    sub_qubits = physical_qubits[start:end]
    psi_mps_after_qfew, sub_vector_transformed = apply_reduction_as_gates(
        psi_mps,
        sub_vector,
        sub_qubits,
        reduce_position="last"
    )

    final_vector = np.zeros_like(vector)
    final_vector[start:end] = sub_vector_transformed

    return psi_mps_after_qfew, final_vector, sub_vector


'''def get_qfew(vector):
    """
    Applies Givnes rotations to the vector transformed by givens rotation corresponding to Qfull and Qempty
    to find Qfew

    Example
    -------
    Input:
        [0,0,0,a,b,c,d,e,f,0,0,0]

    Output:
        [0,0,0,alpha,0,0,0,0,0,0,0,0]
    """

    vector = vector.astype(complex).copy()

    L = len(vector)
    Qfew = np.eye(L, dtype=complex)

    nz = np.flatnonzero(np.abs(vector) > 1e-12)
    if len(nz) == 0:
        return Qfew

    start = nz[0]
    end = nz[-1] + 1

    sub_vector = vector[start:end]

    _, Qsmall = reduce_to_single_entry(
        sub_vector,
        reduce_position="first"
    )

    Qfew[start:end, start:end] = Qsmall

    return Qfew'''





def fermionic_givens_gate_matrix(theta, phi, phyindex_i, phyindex_j ):
    c, s = np.cos(theta), np.sin(theta)
    ph = np.exp(1j * phi)
    G = np.array([
        [1, 0,     0,    0],
        [0, ph*c, -ph*s, 0],
        [0, s,     c,    0],
        [0, 0,     0,    ph]
    ], dtype=complex)
    return G , (phyindex_i, phyindex_j)

SWAP_MAT = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)



# ======================================================================
#  A: Analyse the current MPS (no  modification)
# ======================================================================
def build_spin_data(psi_mps, spin_bath_qubits, L, t0, t, n, N_total):
    """Per-spin: measure 1RDM directly from the MPS, classify, get v*_a."""
    evals, Un = U_n(psi_mps, spin_bath_qubits, N_total)
    filled, active, empty = classify_orb(evals)
    impurity_neighbor = N - 1 if spin_bath_qubits == up_bath_qubits else 0
    v = v_n_Σ(
        L,
        t0,
        t,
        n,
        psi_mps,
        spin_bath_qubits,
        N_total,
        impurity_neighbor,
    )
    vec_filled = v[filled]
    vec_active = v[active]
    vec_empty  = v[empty]
    return vec_filled, vec_active, vec_empty, filled, active, empty

# ======================================================================
#  B: Compress one spin sector with Q_full, Q_empty and Q_few.
# Returns the updated MPS together with the surviving effective orbital.
# ======================================================================
def apply_qfull_qempty_qfew_one_spin(psi_mps, vec_filled, vec_active, vec_empty, spin_bath_qubits):
    psi_mps, filled_reduced = build_vec_and_mps_after_qfull(psi_mps, vec_filled, spin_bath_qubits)
    psi_mps, empty_reduced = build_vec_and_mps_after_qempty(psi_mps, vec_empty, spin_bath_qubits)
    psi_mps, final_vector, _ = build_vec_and_mps_after_qfew(
        psi_mps,
        filled_reduced,
        vec_active,
        empty_reduced,
        spin_bath_qubits,
    )

    n_filled = len(vec_filled)
    c_eff_local_index = n_filled - 1
    c_eff_physical_qubit = spin_bath_qubits[c_eff_local_index]
    K_tilde = final_vector[c_eff_local_index]

    return psi_mps, c_eff_physical_qubit, K_tilde


X_GATE = np.array([[0,1],[1,0]], dtype=complex)

def build_H2_gate(K_tilde, theta_z, T):

    theta_K_eff = abs(K_tilde) * T
    l1 = np.exp(1j*theta_z/2) 
    l2 = np.exp(-1j*theta_z/2)
    c1, s1 = np.cos(theta_K_eff), np.sin(theta_K_eff)
    mat = np.eye(8, dtype=np.complex128)
    mat[1,1] = c1*l1
    mat[1,6] = 1j*s1*l1
    mat[6,1] = 1j*s1*l1
    mat[6,6] = c1*l1
    mat[3,3] = l2
    mat[4,4] = l2
    return mat


#----------------------------------------------------------------------------------------------------------------------------------------
# Constructing the matrix representation of the hamiltonian H2 bar, equation 29 in the notes and subsequently 
# converting into a three qubit gate
#----------------------------------------------------------------------------------------------------------------------------------------


Sz_op = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
Sp_op = np.array([[0, 1], [0, 0]], dtype=complex)   # S^+ raises |down> -> |up>
Sm_op = np.array([[0, 0], [1, 0]], dtype=complex)
Sx_op = 0.5 * (Sp_op + Sm_op)
Sy_op = -0.5j * (Sp_op - Sm_op)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def build_Hbar2_matrix(Jx, Jy, Jz, h, n, T):
    """
    Builds the 8x8 matrix of Hbar2(n), Eq. (29), on the 3-qubit effective
    space with basis ordering (c_eff_up, impurity, c_eff_down)

    Parameters:
    Jx, Jy, Jz : float   -- bare Kondo couplings from Eq. (2) (Jx=Jy=J typically)
    h, T       : float   -- Zeeman field and Floquet period from Eq. (1)/(4)
    n          : int     -- Floquet step index

    Returns:
    H : (8,8) complex ndarray, Hermitian
    """
    theta = h * n * T
    Sx_n =  np.cos(theta) * Sx_op + np.sin(theta) * Sy_op
    Sy_n = -np.sin(theta) * Sx_op + np.cos(theta) * Sy_op
    Sz_n =  Sz_op

    cdag_up, c_up = Sp_op * 0 + np.array([[0, 1], [0, 0]], dtype=complex), np.array([[0, 0], [1, 0]], dtype=complex)
    cdag_dn, c_dn = cdag_up, c_up   # same local ladder operator convention on both fermionic qubits

    op_x = kron3(cdag_up, np.eye(2, dtype=complex), c_dn) + kron3(c_up, np.eye(2, dtype=complex), cdag_dn)
    op_y = -1j * kron3(cdag_up, np.eye(2, dtype=complex), c_dn) + 1j * kron3(c_up, np.eye(2, dtype=complex), cdag_dn)
    n_up = cdag_up @ c_up
    n_dn = cdag_dn @ c_dn
    op_z = kron3(n_up, np.eye(2, dtype=complex), np.eye(2, dtype=complex)) - kron3(np.eye(2, dtype=complex), np.eye(2, dtype=complex), n_dn)

    H = (Jx * op_x @ kron3(np.eye(2, dtype=complex), Sx_n, np.eye(2, dtype=complex))
       + Jy * op_y @ kron3(np.eye(2, dtype=complex), Sy_n, np.eye(2, dtype=complex))
       + Jz * op_z @ kron3(np.eye(2, dtype=complex), Sz_n, np.eye(2, dtype=complex)))
    return H



# Hbar2(0) built once (Eq. 29 at n=0) -- reuse build_Hbar2_matrix from before with n=0
H_bar2_0 = build_Hbar2_matrix(Jx, Jy, Jz, h, n=0, T=T)
U_bar2_0 = expm(-1j * H_bar2_0 * T)          # fixed gate, computed ONCE, reused every step


def build_impurity_Z_rotation(h, n, T):
    """R(n) = e^{i h n T Sz}, acting only on the impurity qubit."""
    return expm(1j * h * n * T * Sz_op)      # 2x2, impurity-qubit matrix only


def build_Hbar2_gate_via_sandwich(U_bar2_0, h, n, T, up_qubit, imp_qubit, down_qubit):
    """
    Constructs the 3-qubit gate U(n) = R(n)^\dagger U_bar2(0) R(n), where R(n) is the impurity Z rotation at step n.
    The gate acts on the qubits (up_qubit, imp_qubit, down_qubit) in that order, which must be adjacent in the MPS.
    """
    Rz = build_impurity_Z_rotation(h, n, T)      # 2x2
    R_full = kron3(I2, Rz, I2)                   # embed on the impurity factor of the 3-qubit gate
    U_n = R_full.conj().T @ U_bar2_0 @ R_full       
    return Gate(matrix=U_n, indices=(up_qubit, imp_qubit, down_qubit))        #🔴 I need to make sure that up_qubit, imp_qubit and down_qubit are adjacent       


#----------------------------------------------------------------------------------------------------------------------------------------
# Constructing Qopt that will act on the MPS after it has been acted upon by the Kondo interaction gate, Hbar2, and the MPS has been 
# updated to the next step. Qopt is the unitary that diagonalizes the 1-RDM of the bath at step n+1 in the fewest body natural orbital 
# basis of step n. So MPS will be the one from the step n+1.
#----------------------------------------------------------------------------------------------------------------------------------------

def build_Qopt_up(psi):
    """
    Constructs the unitary Qopt that diagonalizes the 1-RDM of the bath at step n+1 in the fewest body natural orbital basis of step n.
    This is done by computing the 1-RDM of the bath from the MPS psi, and then performing an eigenvalue decomposition to obtain the
    unitary that diagonalizes it.

    Parameters
    ----------
    psi : MPS
        The MPS of the state at step n+1.

    Returns
    -------
    Qopt : np.ndarray
        The unitary that diagonalizes the 1-RDM of the bath at step n+1 in the fewest body natural orbital basis of step n.
    """
    C_bath = build_1rdm_bath_up(psi)  #🔴 This function should compute the 1-RDM of the bath from the MPS psi
    evals, evecs = np.linalg.eigh(C_bath)
    Qopt_up = evecs.conj().T  # The unitary that diagonalizes C_bath #🔴 should i conjugate it?
    return Qopt_up

def build_Qopt_down(psi):
    """
    Constructs the unitary Qopt that diagonalizes the 1-RDM of the bath at step n+1 in the fewest body natural orbital basis of step n.
    This is done by computing the 1-RDM of the bath from the MPS psi, and then performing an eigenvalue decomposition to obtain the
    unitary that diagonalizes it.

    Parameters
    ----------
    psi : MPS
        The MPS of the state at step n+1.

    Returns
    -------
    Qopt : np.ndarray
        The unitary that diagonalizes the 1-RDM of the bath at step n+1 in the fewest body natural orbital basis of step n.
    """
    C_bath = build_1rdm_bath_down(psi)  #🔴 This function should compute the 1-RDM of the bath from the MPS psi
    evals, evecs = np.linalg.eigh(C_bath)
    Qopt_down = evecs.conj().T  # The unitary that diagonalizes C_bath #🔴 should i conjugate it?
    return Qopt_down





    






    












