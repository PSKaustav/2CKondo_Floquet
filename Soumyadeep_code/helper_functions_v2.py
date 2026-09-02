import numpy as np
from scipy.linalg import block_diag, expm
from itertools import combinations
import matplotlib.pyplot as plt

from openfermion.linalg.givens_rotations import givens_decomposition, givens_decomposition_square
from qlimb.classical.gates import Gate
from qlimb.classical.mpo import MPO
from qlimb.classical.mps import MPS
from qlimb.classical.utils import apply_svd

# =======================================================================================
# 1. Core Operators and Matrices
# =======================================================================================
I2 = np.eye(2, dtype=complex)
Z_OP = np.array([[1, 0], [0, -1]], dtype=complex)
SP_OP = np.array([[0, 1], [0, 0]], dtype=complex)   
SM_OP = np.array([[0, 0], [1, 0]], dtype=complex)
_SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
FSWAP_MAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=complex)

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[-1, 0], [0, 1]], dtype=complex)

def kron3(a, b, c): 
    """Computes the Kronecker product of three matrices sequentially.

    Args:
        a (np.ndarray): First matrix.
        b (np.ndarray): Second matrix.
        c (np.ndarray): Third matrix.

    Returns:
        np.ndarray: The resulting tensor product matrix.
    """
    return np.kron(np.kron(a, b), c)

# =======================================================================================
# 2. Dynamic MPS Resizing & Safe Truncation
# =======================================================================================
def augment_mps_safe(psi_mps, insert_idx, state_type):
    """Re-inserts a truncated filled/empty pure-state qubit into the MPS.

    Implements the augmentation of the few-body wave function.
    Reference: Eq. (63)[2].

    Args:
        psi_mps (MPS): The active Matrix Product State.
        insert_idx (int): The integer index at which to insert the tensor.
        state_type (int): 1 for a filled orbital, 0 for an empty orbital.

    Returns:
        MPS: The newly augmented Matrix Product State.
    """
    if insert_idx > 0:
        vL = psi_mps.tensors[insert_idx - 1].shape[2]
    elif insert_idx < psi_mps.nqbits:
        vL = psi_mps.tensors[insert_idx].shape[0]
    else:
        vL = 1
    T = np.zeros((vL, 2, vL), dtype=complex)
    for v in range(vL):
        T[v, state_type, v] = 1.0  
    psi_mps.tensors.insert(insert_idx, T)
    psi_mps.tags.insert(insert_idx, "NA")
    psi_mps.nqbits += 1
    psi_mps.center_idx = None 
    return psi_mps

def truncate_mps_safe(psi_mps, indices_to_remove):
    """Removes filled/empty natural-orbital qubits from the active MPS window.

    Projects the unentangled states back into the adjacent tensors once their 
    occupation reaches 0 or 1. Reference: Eq. (63)[2].

    Args:
        psi_mps (MPS): The active Matrix Product State.
        indices_to_remove (list of tuple): List of ``(idx, state)`` tuples 
            to be extracted and absorbed.

    Returns:
        MPS: The safely truncated Matrix Product State.
    """
    for (idx, state) in sorted(indices_to_remove, key=lambda x: x[0], reverse=True):
        T = psi_mps.tensors[idx]
        M = T[:, state, :] 
        if idx < psi_mps.nqbits - 1:
            T_next = psi_mps.tensors[idx + 1]
            psi_mps.tensors[idx + 1] = np.tensordot(M, T_next, axes=([1], [0]))
        elif idx > 0:
            T_prev = psi_mps.tensors[idx - 1]
            psi_mps.tensors[idx - 1] = np.tensordot(T_prev, M, axes=([2], [0]))
        psi_mps.tensors.pop(idx)
        psi_mps.tags.pop(idx)
        psi_mps.nqbits -= 1
    psi_mps.center_idx = None
    psi_mps.move_center(0) 
    sq_norm = psi_mps.norm()
    if sq_norm > 0:
        psi_mps.tensors[psi_mps.center_idx] /= np.sqrt(sq_norm)
    return psi_mps

# =======================================================================================
# 3. Initialization & Classical Single-Particle Tools
# =======================================================================================
def get_free_propagator(L, theta, n, use_trotter=True, boundary='open'):
    """Returns the single-particle bath propagator matrix.

    Evaluates the free evolution under the hopping Hamiltonian.
    Reference: Eq. (8)[2].

    Args:
        L (int): Number of sites in the chain.
        theta (float): The hopping parameter angle.
        n (int): The current Floquet step index.
        use_trotter (bool, optional): Whether to use discrete brickwall evolution. Defaults to True.
        boundary (str, optional): Boundary conditions ('open' or 'periodic'). Defaults to 'open'.

    Returns:
        np.ndarray: The L x L unitary propagator matrix.
    """
    def bond_rotation(i, angle):
        c = np.cos(angle)
        s = 1j * np.sin(angle)
        R = np.array([[c, s], [s, c]], dtype=complex)
        U = np.eye(L, dtype=complex)
        U[i:i+2, i:i+2] = R
        return U

    def wrap_rotation(angle):
        c, s = np.cos(angle), 1j * np.sin(angle)
        U = np.eye(L, dtype=complex)
        i, j = L - 1, 0
        U[i, i], U[i, j], U[j, i], U[j, j] = c, s, s, c
        return U

    if use_trotter:
        U_even_half = np.eye(L, dtype=complex)
        for i in range(0, L-1, 2): U_even_half = bond_rotation(i, theta) @ U_even_half
        U_odd_full = np.eye(L, dtype=complex)
        for i in range(1, L-1, 2): U_odd_full = bond_rotation(i, 2*theta) @ U_odd_full
        if boundary == 'periodic' and L % 2 == 0: U_odd_full = wrap_rotation(2*theta) @ U_odd_full
        U_even_full = np.eye(L, dtype=complex)
        for i in range(0, L-1, 2): U_even_full = bond_rotation(i, 2*theta) @ U_even_full

        U_initial = U_odd_full @ U_even_half
        U_step = U_odd_full @ U_even_full
        return U_initial if n == 0 else np.linalg.matrix_power(U_step, n) @ U_initial
    else:
        A = np.eye(L, k=1) + np.eye(L, k=-1)
        if boundary == 'periodic': A[0, L-1] = A[L-1, 0] = 1
        U_step = expm(-1j * 2 * theta * A)
        return np.eye(L, dtype=complex) if n == 0 else np.linalg.matrix_power(U_step, n)

def build_initial_orbitals(N_bath):
    """Constructs the initial momentum-space orbitals for the non-interacting bath.

    Args:
        N_bath (int): Number of bath sites.

    Returns:
        tuple: Two np.ndarray matrices containing the Up and Down bath orbitals.
    """
    x, m = np.arange(N_bath), N_bath / 2.0
    j_range = range(-int(m // 2), int(m // 2) + 1) if m % 2 != 0 else range(-int(m // 2), int(m // 2))
    k_occ = np.array([2.0 * np.pi * j / N_bath for j in j_range])
    orb = np.zeros((N_bath, len(k_occ)), dtype=complex)
    for col, k in enumerate(k_occ): orb[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return orb, orb

def build_initial_state(N, orb_up, orb_down, n_up, n_down):
    """Builds the explicit initial Fermi sea many-body state.

    Constructs the ground state of the free Hamiltonian. Conventionally,
    1 denotes a filled orbital and 0 denotes an unfilled orbital.
    Reference: Eq. (11)[2].

    Args:
        N (int): Number of bath sites per spin channel.
        orb_up (np.ndarray): The up-spin orbital matrix.
        orb_down (np.ndarray): The down-spin orbital matrix.
        n_up (int): Number of filled up-spin orbitals.
        n_down (int): Number of filled down-spin orbitals.

    Returns:
        np.ndarray: The 2^(2N+1) dense statevector array.
    """
    psi = np.zeros(2 ** (2 * N + 1), dtype=complex)
    for down_occ in combinations(range(N), n_down):
        s_down = np.zeros(N, dtype=int); s_down[list(down_occ)] = 1
        a_down = np.linalg.det(orb_down[np.where(s_down == 1)[0], :])
        for up_occ in combinations(range(N), n_up):
            s_up = np.zeros(N, dtype=int); s_up[list(up_occ)] = 1
            a_up = np.linalg.det(orb_up[np.where(s_up == 1)[0], :])
            full_state = np.concatenate((s_up, [0], s_down))
            idx = 0
            for bit in full_state: idx = (idx << 1) | bit
            psi[idx] = a_down * a_up
    return psi

def state_mps(psi, N, bond_dim=np.inf):
    """Encodes a dense statevector into a Matrix Product State (MPS) via SVD.

    Args:
        psi (np.ndarray): The 2^(2N+1) dense statevector.
        N (int): The number of bath sites per spin.
        bond_dim (int, optional): The maximum virtual bond dimension. Defaults to np.inf.

    Returns:
        MPS: The resulting Matrix Product State.
    """
    tensors, chi_left, T = [], 1, psi.reshape([2] * (2 * N + 1))
    for _ in range(2 * N):
        U, _, _, V = apply_svd(T.reshape(chi_left, 2, -1, 1), bond_dim=bond_dim, direction='right', preserve_norm=True, tol=1e-10)
        tensors.append(U); chi_left = U.shape[-1]; T = V.reshape(chi_left, 2, -1)
    tensors.append(T.reshape(chi_left, 2, 1))
    return MPS(nqbits=2 * N + 1, phys_dim=2, tensors=tensors, bond_dim=bond_dim, preserve_norm=True, trunc_tol=1e-10)

def classify_orb(evals, tol=1e-10):
    """Classifies natural orbitals into filled, active, or empty sets.

    Orbitals are categorized based on their 1RDM eigenvalues.
    Reference: Eq. (37)[2].

    Args:
        evals (np.ndarray): The array of 1RDM eigenvalues.
        tol (float, optional): The numerical tolerance for occupation. Defaults to 1e-10.

    Returns:
        tuple: Three lists containing indices for filled, active, and empty orbitals.
    """
    filled, active, empty = [], [], []
    for i, occ in enumerate(evals):
        if occ >= 1.0 - tol: filled.append(i)
        elif occ <= tol: empty.append(i)
        else: active.append(i)
    return filled, active, empty

def build_1rdm_bath(psi_mps, bath_qubits, imp_qubit):
    """Computes the 1-body Reduced Density Matrix (1RDM) from the MPS.

    Evaluates the correlation matrix for the active natural orbitals.
    Reference: Eq. (35)[2].

    Args:
        psi_mps (MPS): The active Matrix Product State.
        bath_qubits (list): The list of physical qubit indices for the bath.
        imp_qubit (int): The physical index of the impurity qubit.

    Returns:
        np.ndarray: The computed correlation matrix.
    """
    N_total, n = psi_mps.nqbits, len(bath_qubits)
    C = np.zeros((n, n), dtype=complex)
    Nop = (I2 - Z_OP) / 2  
    
    for a in range(n):
        for b in range(a, n):
            i, j = bath_qubits[a], bath_qubits[b]
            left, right = min(i, j), max(i, j)
            tensors = []
            for k in range(N_total):
                if i == j: op = Nop if k == i else I2
                else:
                    if k == imp_qubit: op = I2
                    elif k == left: op = SM_OP if i < j else SP_OP
                    elif k == right: op = SP_OP if i < j else SM_OP
                    elif left < k < right: op = -Z_OP
                    else: op = I2
                tensors.append(op.reshape(1, 2, 2, 1))
            val = psi_mps @ (MPO(nqbits=N_total, phys_dim=2, tensors=tensors) @ psi_mps)
            C[a, b] = val
            if a != b: C[b, a] = np.conj(val)
    return C

# =======================================================================================
# 4. Givens Rotations & Classical Reductions
# =======================================================================================
def apply_two_qubit_gate(psi_mps, mat, qi, qj):
    """Applies a 2-qubit gate to the MPS, automatically handling internal FSWAPs.

    Args:
        psi_mps (MPS): The active Matrix Product State.
        mat (np.ndarray): The 4x4 unitary matrix representing the operation.
        qi (int): Physical index of the first qubit.
        qj (int): Physical index of the second qubit.

    Returns:
        MPS: The updated Matrix Product State.

    Raises:
        ValueError: If the qubits are not strictly adjacent.
    """
    if qj == qi + 1: return Gate(matrix=mat, indices=(qi, qj)) @ psi_mps
    elif qi == qj + 1: return Gate.FSWAP(indices=(qj, qi), phys_dim=2) @ Gate(matrix=mat, indices=(qj, qi)) @ Gate.FSWAP(indices=(qj, qi), phys_dim=2) @ psi_mps
    else: raise ValueError(f"qubits {qi},{qj} not adjacent")

def fermionic_givens_gate_matrix(theta, phi):
    """Generates a 4x4 fermionic parity-preserving Givens gate matrix.

    Args:
        theta (float): The rotational angle.
        phi (float): The phase parameter.

    Returns:
        np.ndarray: The 4x4 unitary block matrix.
    """
    c, s, ph = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    phc = np.conj(ph)
    return np.array([[1, 0, 0, 0], [0, c, -phc * s, 0], [0, s, phc * c, 0], [0, 0, 0, phc]], dtype=complex)

def fermionic_givens_gate_matrix_square(theta, phi):
    """Generates a 4x4 fermionic Givens gate specific for the frame update.

    Args:
        theta (float): The rotational angle.
        phi (float): The phase parameter.

    Returns:
        np.ndarray: The 4x4 unitary block matrix.
    """
    c, s, ph = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    phc = np.conj(ph)
    return np.array([[1, 0, 0, 0], [0, c, -phc * s, 0], [0, s, phc * c, 0], [0, 0, 0, phc]], dtype=complex)

def get_actual_two_site_gate(theta, phi, q_left, q_right):
    """Generates the appropriately mapped 4x4 gate for targeted application.

    Automatically handles internal swap ordering depending on directional targeting.

    Args:
        theta (float): The rotational angle.
        phi (float): The phase parameter.
        q_left (int): Physical index of the left qubit.
        q_right (int): Physical index of the right qubit.

    Returns:
        np.ndarray: The localized 4x4 unitary matrix.

    Raises:
        ValueError: If qubits are not adjacent.
    """
    G4 = fermionic_givens_gate_matrix(theta, phi)
    if q_right == q_left + 1: return G4
    if q_left == q_right + 1: return _SWAP4 @ G4 @ _SWAP4
    raise ValueError(f'qubits {q_left} and {q_right} are not adjacent')

def extract_single_particle_block_from_gate(G4):
    """Extracts the 2x2 single-particle block from a 4x4 unitary gate.

    Args:
        G4 (np.ndarray): The 4x4 unitary interaction block.

    Returns:
        np.ndarray: The 2x2 extracted single-particle mapping.
    """
    return np.array([[G4[2, 2], G4[2, 1]], [G4[1, 2], G4[1, 1]]], dtype=complex)

def _classical_givens_reduction(vec, target_pos='last'):
    """Constructs a classical N x N Givens reduction matrix.

    Operates strictly on the classical coefficients to isolate a target orbital.
    Reference: Eq. (46)[2].

    Args:
        vec (np.ndarray): The vector of coefficients to be reduced.
        target_pos (str, optional): 'last' or 'first', dictating where the 
            amplitude should accumulate. Defaults to 'last'.

    Returns:
        np.ndarray: The resulting classical unitary matrix.
    """
    dim = len(vec)
    if dim <= 1:
        return np.eye(dim, dtype=complex)
        
    v_ord = vec[::-1].copy() if target_pos == 'last' else vec.copy()
    norm = np.linalg.norm(v_ord)
    if norm < 1e-14:
        return np.eye(dim, dtype=complex)
        
    decomp, _, _ = givens_decomposition((v_ord / norm).reshape(1, -1))
    
    Q = np.eye(dim, dtype=complex)
    order = list(range(dim))[::-1] if target_pos == 'last' else list(range(dim))
    
    for grp in decomp:
        for (i, j, theta, phi) in grp:
            phys_i, phys_j = order[i], order[j]
            
            c, s = np.cos(theta), np.sin(theta)
            phc = np.exp(-1j * phi)
            U2 = np.array([[c, -phc * s], [s, phc * c]], dtype=complex)
            
            G = np.eye(dim, dtype=complex)
            G[np.ix_([phys_i, phys_j], [phys_i, phys_j])] = U2
            Q = G @ Q
            
    return Q

def build_qfullempty_classically(v_f, v_e, chain_type):
    """Builds the Q_full and Q_empty matrices classically based on coefficient vectors.

    Uses the ordering convention derived from the up and down spin chain layouts.
    Reference: Eq. (46)[2].

    Args:
        v_f (np.ndarray): The coefficient vector for the filled orbitals.
        v_e (np.ndarray): The coefficient vector for the empty orbitals.
        chain_type (str): 'up' or 'down', dictating the target position based on 
            the proximity to the impurity.

    Returns:
        tuple: The Q_full and Q_empty classical matrices.
    """
    target_pos = 'last' if chain_type == 'up' else 'first'
    Q_full = _classical_givens_reduction(v_f, target_pos=target_pos)
    Q_empty = _classical_givens_reduction(v_e, target_pos=target_pos)
    return Q_full, Q_empty

def build_qfew(v_target, physical_qubits, target_pos='last'):
    """Builds the classical Q_few matrix and the corresponding quantum gates.

    Generates the sequence of 4x4 quantum gates required to physically route 
    the target orbital to the impurity boundary. Reference: Eq. (46)[2].

    Args:
        v_target (np.ndarray): The composite vector representing the effective orbital.
        physical_qubits (list): The list of contiguous physical qubits mapped to the vector.
        target_pos (str, optional): 'last' or 'first', dictating the extraction boundary. Defaults to 'last'.

    Returns:
        tuple: The classical N x N Q_few matrix, and a list of (qi, qj, Gate) tuples.
    """
    dim = len(v_target)
    Q_few_classical = np.eye(dim, dtype=complex)
    quantum_gates = []
    
    if dim <= 1:
        return Q_few_classical, quantum_gates
        
    v_ord = v_target[::-1].copy() if target_pos == 'last' else v_target.copy()
    order = physical_qubits[::-1] if target_pos == 'last' else physical_qubits
    local_order = list(range(dim))[::-1] if target_pos == 'last' else list(range(dim))
    
    norm = np.linalg.norm(v_ord)
    if norm < 1e-14:
        return Q_few_classical, quantum_gates
        
    decomp, _, _ = givens_decomposition((v_ord / norm).reshape(1, -1))
    
    for grp in decomp:
        for (i, j, theta, phi) in grp:
            phys_i, phys_j = order[i], order[j]
            loc_i, loc_j = local_order[i], local_order[j]
            
            # Classical Matrix Construction
            c, s = np.cos(theta), np.sin(theta)
            phc = np.exp(-1j * phi)
            U2 = np.array([[c, -phc * s], [s, phc * c]], dtype=complex)
            
            G_sp = np.eye(dim, dtype=complex)
            G_sp[np.ix_([loc_i, loc_j], [loc_i, loc_j])] = U2
            Q_few_classical = G_sp @ Q_few_classical
            
            # Quantum Gate Generation
            G4 = fermionic_givens_gate_matrix(theta, phi)
            quantum_gates.append((phys_i, phys_j, G4))
            
    return Q_few_classical, quantum_gates

def apply_qfew_quantum_gates(psi_mps, quantum_gates):
    """Applies the sequence of pre-computed Q_few quantum gates directly to the active MPS.

    Reference: Eq. (46)[2].

    Args:
        psi_mps (MPS): The active Matrix Product State.
        quantum_gates (list): A list of tuples ``(qi, qj, G4)`` representing the Givens rotations.

    Returns:
        MPS: The updated Matrix Product State.
    """
    for (qi, qj, G4) in quantum_gates:
        psi_mps = apply_two_qubit_gate(psi_mps, G4, qi, qj)
    return psi_mps

#def move_mode_adjacent_to_impurity(psi_mps, curr_q, chain_qs):
    """Routes an effective mode strictly adjacent to the impurity via FSWAPs.

    Args:
        psi_mps (MPS): The active Matrix Product State.
        curr_q (int): The current physical index of the mode.
        chain_qs (list): The localized chain indices.

    Returns:
        tuple: The updated MPS, the final target index, and the sequence permutation list.
    """
    #target_q = chain_qs[0] 
    #if curr_q == target_q: return psi_mps, curr_q, []
    #pos, step, perm = chain_qs.index(curr_q), -1 if chain_qs.index(curr_q) > 0 else 1, []
    #for p in range(pos, 0, step):
    #    psi_mps = apply_two_qubit_gate(psi_mps, FSWAP_MAT, curr_q, chain_qs[p + step])
    #    perm.append((curr_q, chain_qs[p + step]))
    #    curr_q = chain_qs[p + step]
    #return psi_mps, curr_q, perm

#def apply_column_swaps(V, permutation, local_idx_map):
    """Applies the recorded permutations natively to the classical frame matrix.

    Args:
        V (np.ndarray): The classical frame matrix.
        permutation (list): The list of ``(qa, qb)`` swap instructions.
        local_idx_map (dict): A dictionary mapping physical indices to their local matrix equivalents.

    Returns:
        np.ndarray: The updated classical frame matrix.
    """
    #for (qa, qb) in permutation:
    #    ia, ib = local_idx_map[qa], local_idx_map[qb]
    #    V[:, [ia, ib]] = V[:, [ib, ia]]
    #return V

# =======================================================================================
# 5. Natural Orbital Frame Advance & Kondo Gates
# =======================================================================================
def apply_givens_circuit(psi_mps, decomp, diagonal, physical_qubits):
    """Applies a decomposed sequence of Givens rotations natively to the MPS.

    Args:
        psi_mps (MPS): The active Matrix Product State.
        decomp (list): The output list of layered Givens operations.
        diagonal (list): The vector of localized phase shifts.
        physical_qubits (list): The mapped list of physical interacting qubits.

    Returns:
        MPS: The updated Matrix Product State.
    """
    psi_no = psi_mps.copy()
    for local_site, phase in enumerate(diagonal):
        psi_no = Gate(matrix=np.array([[1, 0], [0, phase]], dtype=complex), indices=(physical_qubits[local_site],)) @ psi_no
    for layer in reversed(decomp):
        for (i, j, theta, phi) in layer:
            G = fermionic_givens_gate_matrix_square(theta, phi)
            psi_no = apply_two_qubit_gate(psi_no, G, physical_qubits[i], physical_qubits[j])
    return psi_no

def advance_natural_orbital_frame(psi_mps, V_sigma, bath_qubits, imp_qubit, L_total, active_indices_in_L, chain_type):
    """Diagonalizes the 1RDM to update the natural orbital frame.

    Classifies and physically sweeps trivial states to the outer boundaries 
    of the tensor network based on the chain type directional sorting. 
    Reference: Eq. (59) and Eq. (61)[2].

    Args:
        psi_mps (MPS): The active Matrix Product State.
        V_sigma (np.ndarray): The classical frame matrix from the previous step.
        bath_qubits (list): The list of physical qubit indices for this chain.
        imp_qubit (int): The physical index of the impurity qubit.
        L_total (int): The total number of bath sites in this chain.
        active_indices_in_L (list): The global indices corresponding to the active window.
        chain_type (str): 'up' or 'down', dictating the required spatial sorting.

    Returns:
        tuple: The updated MPS, the updated U_sigma(n+1) matrix, and the sorted eigenvalues.

    Raises:
        ValueError: If chain_type is not 'up' or 'down'.
    """
    C = build_1rdm_bath(psi_mps, bath_qubits, imp_qubit)
    evals, evecs = np.linalg.eigh(C)
    
    C_diag = evecs.conj().T @ C @ evecs
    off_diag_norm = np.linalg.norm(C_diag - np.diag(np.diagonal(C_diag)))
    assert off_diag_norm < 1e-10, f"1RDM not strictly diagonalized! Off-diagonal norm: {off_diag_norm}"

    order_desc = np.argsort(evals)[::-1]
    evals_desc = evals[order_desc]
    evecs_desc = evecs[:, order_desc]
    
    filled, active, empty = classify_orb(evals_desc, tol=1e-10)
    
    if chain_type == 'up':
        smart_order = filled[::-1] + empty[::-1] + active[::-1]
    elif chain_type == 'down':
        smart_order = active + empty + filled
    else:
        raise ValueError("chain_type must be 'up' or 'down'")

    evals_sorted = evals_desc[smart_order]
    evecs_sorted = evecs_desc[:, smart_order]
    
    decomp, diagonal = givens_decomposition_square(evecs_sorted.conj())
    psi_mps = apply_givens_circuit(psi_mps, decomp, diagonal, bath_qubits)
    
    U_embed = np.eye(L_total, dtype=complex)
    for idx_local, idx_L in enumerate(active_indices_in_L):
        for jdx_local, jdx_L in enumerate(active_indices_in_L):
            U_embed[idx_L, jdx_L] = evecs_sorted[idx_local, jdx_local]
            
    U_sigma_new = V_sigma @ U_embed
    return psi_mps, U_sigma_new, evals_sorted

def build_kondo_gate(Jk, Jz, h, T, n):
    """Builds the 3-qubit Kondo gate exactly isolating interactions inside the few-body basis.

    Strictly enforces the S_z conserving basis: |n_up, impurity, n_down>.
    Reference: Eq. (14)[cite: 1] and Eq. (12)[2].

    Args:
        Jk (float): The transverse Kondo coupling parameter.
        Jz (float): The longitudinal Kondo coupling parameter.
        h (float): The magnetic field applied to the impurity.
        T (float): The Floquet time period step size.
        n (int): The current step index.

    Returns:
        np.ndarray: The 8x8 unitary gate matrix.
    """
    theta_K = Jk * T / 2.0
    theta_z = Jz * T / 2.0
    
    UK_0 = np.zeros((8, 8), dtype=complex)
    UK_0[0,0] = UK_0[2,2] = UK_0[5,5] = UK_0[7,7] = 1.0
    
    c_k, s_k = np.cos(theta_K), np.sin(theta_K)
    p_z, m_z = np.exp(1j * theta_z / 2.0), np.exp(-1j * theta_z / 2.0)
    
    UK_0[1,1] = UK_0[6,6] = m_z 
    UK_0[3,3] = UK_0[4,4] = p_z * c_k
    UK_0[3,4] = UK_0[4,3] = -1j * p_z * s_k
    
    phi = h * T * n / 2.0
    R_z = np.array([[np.exp(1j * phi), 0], [0, np.exp(-1j * phi)]], dtype=complex)
    K_n = kron3(I2, R_z, I2)
    return K_n.conj().T @ UK_0 @ K_n

def apply_3_qubit_gate_custom(psi_mps, mat, q1, q2, q3):
    """Applies a 3-qubit gate strictly on adjacent MPS tensors.

    Args:
        psi_mps (MPS): The active Matrix Product State.
        mat (np.ndarray): The 8x8 unitary matrix.
        q1 (int): First qubit index.
        q2 (int): Second qubit index.
        q3 (int): Third qubit index.

    Returns:
        MPS: The updated Matrix Product State.
    """
    return Gate(matrix=mat, indices=(q1, q2, q3)) @ psi_mps

# =======================================================================================
# 6. Primary Loop Step Execution
# =======================================================================================
# sergio_step to be added later