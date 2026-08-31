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
    return np.kron(np.kron(a, b), c)

# =======================================================================================
# 2. Dynamic MPS Resizing & Safe Truncation
# =======================================================================================
def augment_mps_safe(psi_mps, insert_idx, state_type):
    """
    Re-inserts a truncated filled/empty pure-state qubit into the MPS.
    Implements the augmentation of the few-body wave function.
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
    """
    Removes filled/empty natural-orbital qubits from the active MPS window once 
    their occupation is near 0 or 1.
    Reference: Eq. (63)[1].
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
    """
    Returns U_TE(n), the L x L single-particle bath propagator.
    Reference: Eq. (5) and free evolution under H1[1].
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
    x, m = np.arange(N_bath), N_bath / 2.0
    j_range = range(-int(m // 2), int(m // 2) + 1) if m % 2 != 0 else range(-int(m // 2), int(m // 2))
    k_occ = np.array([2.0 * np.pi * j / N_bath for j in j_range])
    orb = np.zeros((N_bath, len(k_occ)), dtype=complex)
    for col, k in enumerate(k_occ): orb[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return orb, orb

def build_initial_state(N, orb_up, orb_down, n_up, n_down):
    """
    Builds the explicit initial Fermi sea state. 1 = filled, 0 = unfilled.
    Impurity initialized to ket 0 (Down).
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
    tensors, chi_left, T = [], 1, psi.reshape([2] * (2 * N + 1))
    for _ in range(2 * N):
        U, _, _, V = apply_svd(T.reshape(chi_left, 2, -1, 1), bond_dim=bond_dim, direction='right', preserve_norm=True, tol=1e-10)
        tensors.append(U); chi_left = U.shape[-1]; T = V.reshape(chi_left, 2, -1)
    tensors.append(T.reshape(chi_left, 2, 1))
    return MPS(nqbits=2 * N + 1, phys_dim=2, tensors=tensors, bond_dim=bond_dim, preserve_norm=True, trunc_tol=1e-10)

def classify_orb(evals, tol=1e-10):
    filled, active, empty = [], [], []
    for i, occ in enumerate(evals):
        if occ >= 1.0 - tol: filled.append(i)
        elif occ <= tol: empty.append(i)
        else: active.append(i)
    return filled, active, empty

def build_1rdm_bath(psi_mps, bath_qubits, imp_qubit):
    """Computes 1RDM C^sigma_ij(n) from active MPS. Reference: Eq. (7)[1]."""
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
    if qj == qi + 1: return Gate(matrix=mat, indices=(qi, qj)) @ psi_mps
    elif qi == qj + 1: return Gate.FSWAP(indices=(qj, qi), phys_dim=2) @ Gate(matrix=mat, indices=(qj, qi)) @ Gate.FSWAP(indices=(qj, qi), phys_dim=2) @ psi_mps
    else: raise ValueError(f"qubits {qi},{qj} not adjacent")

def fermionic_givens_gate_matrix(theta, phi):
    c, s, ph = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    phc = np.conj(ph)
    return np.array([[1, 0, 0, 0], [0, c, -phc * s, 0], [0, s, phc * c, 0], [0, 0, 0, phc]], dtype=complex)

def fermionic_givens_gate_matrix_square(theta, phi):
    c, s, ph = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    phc = np.conj(ph)
    return np.array([[1, 0, 0, 0], [0, c, -phc * s, 0], [0, s, phc * c, 0], [0, 0, 0, phc]], dtype=complex)

def get_actual_two_site_gate(theta, phi, q_left, q_right):
    G4 = fermionic_givens_gate_matrix(theta, phi)
    if q_right == q_left + 1: return G4
    if q_left == q_right + 1: return _SWAP4 @ G4 @ _SWAP4
    raise ValueError(f'qubits {q_left} and {q_right} are not adjacent')

def extract_single_particle_block_from_gate(G4):
    return np.array([[G4[2, 2], G4[2, 1]], [G4[1, 2], G4[1, 1]]], dtype=complex)

def reduce_vector_classically(vector, reduce_position='first'):
    """Classical reduction yielding Givens targets without touching the MPS."""
    v_ord = vector[::-1].copy() if reduce_position == 'last' else vector.copy()
    norm = np.linalg.norm(v_ord)
    if norm < 1e-14: return vector.copy().astype(complex)
    decomp, _, _ = givens_decomposition((v_ord / norm).reshape(1, -1))
    v = v_ord.copy()
    for parallel_group in decomp:
        for (i, j, theta, phi) in parallel_group:
            G4 = fermionic_givens_gate_matrix(theta, phi)
            U2 = extract_single_particle_block_from_gate(G4)
            vi, vj = v[i], v[j]
            v[i], v[j] = U2[0, 0] * vi + U2[0, 1] * vj, U2[1, 0] * vi + U2[1, 1] * vj
    return v[::-1] if reduce_position == 'last' else v

def apply_reduction_as_gates(psi_mps, vector, physical_qubits, reduce_position='first'):
    """Applies Q_few targeting exactly the extracted rank-1 bath mode."""
    v_ord = vector[::-1].copy() if reduce_position == 'last' else vector.copy()
    order = physical_qubits[::-1] if reduce_position == 'last' else physical_qubits
    norm = np.linalg.norm(v_ord)
    if norm < 1e-14: return psi_mps, vector.copy().astype(complex)
    decomp, _, _ = givens_decomposition((v_ord / norm).reshape(1, -1))
    for grp in decomp:
        for (i, j, theta, phi) in grp:
            G4 = get_actual_two_site_gate(theta, phi, order[i], order[j])
            psi_mps = apply_two_qubit_gate(psi_mps, G4, order[i], order[j])
            v_ord[i], v_ord[j] = extract_single_particle_block_from_gate(G4) @ np.array([v_ord[i], v_ord[j]], dtype=complex)
    return psi_mps, (v_ord[::-1] if reduce_position == 'last' else v_ord)

def build_Q_matrix(N, n_f, n_a, n_e, vec_f, sub_v, vec_e, w_start, w_end):
    """Builds the N x N classical matrix Q(n) representing Q_few Q_empty Q_full. Reference: Eq. (48)[1]."""
    def reduce_block(block_vec, reduce_position, dim):
        v = block_vec.astype(complex).copy()
        v_ord = v[::-1].copy() if reduce_position == 'last' else v.copy()
        decomp, _, _ = givens_decomposition((v_ord / np.linalg.norm(v_ord)).reshape(1, -1))
        Q = np.eye(dim, dtype=complex)
        for grp in decomp:
            for (i, j, theta, phi) in grp:
                G4 = get_actual_two_site_gate(theta, phi, i, j)
                U2 = extract_single_particle_block_from_gate(G4)
                G = np.eye(dim, dtype=complex)
                G[np.ix_([i, j], [i, j])] = U2
                Q = G @ Q
                v_ord[i], v_ord[j] = U2 @ np.array([v_ord[i], v_ord[j]], dtype=complex)
        return Q, (v_ord[::-1] if reduce_position == 'last' else v_ord)

    Q_f, _ = reduce_block(vec_f, 'last', n_f) if n_f > 0 else (np.eye(n_f, dtype=complex), vec_f)
    Q_e, _ = reduce_block(vec_e, 'first', n_e) if n_e > 0 else (np.eye(n_e, dtype=complex), vec_e)

    M1 = block_diag(Q_f, np.eye(n_a, dtype=complex), Q_e)
    win_len = w_end - w_start
    Q_few = np.eye(win_len, dtype=complex)
    if win_len > 0 and np.linalg.norm(sub_v) > 1e-14:
        sub_v_copy = sub_v.astype(complex).copy()
        decomp, _, _ = givens_decomposition((sub_v_copy / np.linalg.norm(sub_v_copy)).reshape(1, -1))
        for grp in decomp:
            for (i, j, theta, phi) in grp:
                G4 = get_actual_two_site_gate(theta, phi, i, j)
                U2 = extract_single_particle_block_from_gate(G4)
                G = np.eye(win_len, dtype=complex)
                G[np.ix_([i, j], [i, j])] = U2
                Q_few = G @ Q_few
                sub_v_copy[i], sub_v_copy[j] = U2 @ np.array([sub_v_copy[i], sub_v_copy[j]], dtype=complex)

    M2 = np.eye(N, dtype=complex)
    if win_len > 0: M2[w_start:w_end, w_start:w_end] = Q_few
    return M1 @ M2

def apply_qfull_qempty_qfew_one_spin(psi_mps, v_f, v_a, v_e, phys_qubits):
    """
    Applies Q_few^dag mapping the augmented MPS to the fewest-body ordered basis.
    Extracts the rank-1 coupling yielding N_active + 2 orbitals.
    Reference: Eq. (46), (47), and (48)[1].
    """
    f_red = reduce_vector_classically(v_f, 'last')
    e_red = reduce_vector_classically(v_e, 'first')
    
    vec = np.concatenate((f_red, v_a, e_red)).astype(complex)
    nz = np.flatnonzero(np.abs(vec) > 1e-12)
    start, end = (nz[0], nz[-1] + 1) if len(nz) > 0 else (0, 0)
    
    if len(nz) > 0:
        psi_mps, _sub_trans = apply_reduction_as_gates(psi_mps, vec[start:end], phys_qubits, 'last')
    
    c_idx = len(phys_qubits) - 1 if len(nz) > 0 else 0
    Q_sigma = build_Q_matrix(len(v_f) + len(v_a) + len(v_e), len(v_f), len(v_a), len(v_e), v_f, vec[start:end], v_e, start, end)
    return psi_mps, phys_qubits[c_idx], Q_sigma, (start, end)

def move_mode_adjacent_to_impurity(psi_mps, curr_q, chain_qs):
    target_q = chain_qs[0] 
    if curr_q == target_q: return psi_mps, curr_q, []
    pos, step, perm = chain_qs.index(curr_q), -1 if chain_qs.index(curr_q) > 0 else 1, []
    for p in range(pos, 0, step):
        psi_mps = apply_two_qubit_gate(psi_mps, FSWAP_MAT, curr_q, chain_qs[p + step])
        perm.append((curr_q, chain_qs[p + step]))
        curr_q = chain_qs[p + step]
    return psi_mps, curr_q, perm

def apply_column_swaps(V, permutation, local_idx_map):
    for (qa, qb) in permutation:
        ia, ib = local_idx_map[qa], local_idx_map[qb]
        V[:, [ia, ib]] = V[:, [ib, ia]]
    return V

# =======================================================================================
# 5. Natural Orbital Frame Advance & Kondo Gates
# =======================================================================================
def apply_givens_circuit(psi_mps, decomp, diagonal, physical_qubits):
    psi_no = psi_mps.copy()
    for local_site, phase in enumerate(diagonal):
        psi_no = Gate(matrix=np.array([[1, 0], [0, phase]], dtype=complex), indices=(physical_qubits[local_site],)) @ psi_no
    for layer in reversed(decomp):
        for (i, j, theta, phi) in layer:
            G = fermionic_givens_gate_matrix_square(theta, phi)
            psi_no = apply_two_qubit_gate(psi_no, G, physical_qubits[i], physical_qubits[j])
    return psi_no

def advance_natural_orbital_frame(psi_mps, V_sigma, bath_qubits, imp_qubit, L_total, active_indices_in_L):
    """
    Diagonalizes 1RDM to find u_sigma(n+1) and natively updates U_sigma(n+1) via classical multiplication.
    Reference: Eq. (59), (60), (61)[1].
    """
    C = build_1rdm_bath(psi_mps, bath_qubits, imp_qubit)
    evals, evecs = np.linalg.eigh(C)
    
    C_diag = evecs.conj().T @ C @ evecs
    off_diag_norm = np.linalg.norm(C_diag - np.diag(np.diagonal(C_diag)))
    assert off_diag_norm < 1e-10, f"1RDM not strictly diagonalized! Off-diagonal norm: {off_diag_norm}"

    order = np.argsort(evals)[::-1]
    evals_sorted, evecs_sorted = evals[order], evecs[:, order]
    
    decomp, diagonal = givens_decomposition_square(evecs_sorted.conj())
    psi_mps = apply_givens_circuit(psi_mps, decomp, diagonal, bath_qubits)
    
    U_embed = np.eye(L_total, dtype=complex)
    for idx_local, idx_L in enumerate(active_indices_in_L):
        for jdx_local, jdx_L in enumerate(active_indices_in_L):
            U_embed[idx_L, jdx_L] = evecs_sorted[idx_local, jdx_local]
            
    U_sigma_new = V_sigma @ U_embed
    return psi_mps, U_sigma_new, evals_sorted

def build_kondo_gate(Jk, Jz, h, T, n):
    """
    Builds the 3-qubit Kondo gate exactly isolating interactions inside the 
    few-body natural orbital basis.
    Reference: Eq. (5) and Eq. (6)[1].
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
    return Gate(matrix=mat, indices=(q1, q2, q3)) @ psi_mps

# =======================================================================================
# 6. Primary Loop Step Execution
# =======================================================================================
def sergio_step(psi_mps, U_up, U_down, V_up, V_down, evals_up, evals_down, n, Jk, Jz, h, T, theta, L_bath, tol=1e-12):
    """
    Executes a single step (n -> n + 1) explicitly enforcing the Q_few^dag augmentation.
    Takes in V_up, V_down and outputs them for the next step alongside U(n+1)[1].
    """
    # 1. Update Free Evolution
    U_TE = get_free_propagator(L_bath, theta, n, use_trotter=True)
    
    # 2. Extract specific coupling modes corresponding to site 0
    v_up_conj = (U_TE @ U_up)[0, :]
    v_down_conj = (U_TE @ U_down)[0, :]
    v_up, v_down = v_up_conj.conj(), v_down_conj.conj()

    # 3. Classify Orbitals & Split
    filled_u, active_u, empty_u = classify_orb(evals_up, tol)
    filled_d, active_d, empty_d = classify_orb(evals_down, tol)
    
    v_f_u, v_a_u, v_e_u = v_up[filled_u], v_up[active_u], v_up[empty_u]
    v_f_d, v_a_d, v_e_d = v_down[filled_d], v_down[active_d], v_down[empty_d]

    M_u, M_d = len(active_u), len(active_d)
    
    # 4. Augment MPS exactly preserving active layout
    psi_mps = augment_mps_safe(psi_mps, 0, state_type=1)        
    psi_mps = augment_mps_safe(psi_mps, M_u + 1, state_type=0)  
    
    imp_q = M_u + 2
    psi_mps = augment_mps_safe(psi_mps, imp_q + 1, state_type=1)        
    psi_mps = augment_mps_safe(psi_mps, imp_q + M_d + 2, state_type=0)  
    
    # 5. Apply Q_few^dag to extract the rank-1 coupling
    up_phys_qubits = list(range(0, M_u + 2))
    psi_mps, c_eff_up_q, Q_up, (start_u, end_u) = apply_qfull_qempty_qfew_one_spin(psi_mps, v_f_u, v_a_u, v_e_u, up_phys_qubits)
    
    down_phys_qubits = list(range(imp_q + 1, imp_q + M_d + 3))
    psi_mps, c_eff_down_q, Q_down, (start_d, end_d) = apply_qfull_qempty_qfew_one_spin(psi_mps, v_f_d, v_a_d, v_e_d, down_phys_qubits)
    
    # 6. FSWAP effective modes directly adjacent to the Impurity
    psi_mps, c_eff_up_q, perm_up = move_mode_adjacent_to_impurity(psi_mps, c_eff_up_q, up_phys_qubits[::-1])
    psi_mps, c_eff_down_q, perm_down = move_mode_adjacent_to_impurity(psi_mps, c_eff_down_q, down_phys_qubits)
    
    Q_up = apply_column_swaps(Q_up, perm_up, {q: i for i, q in enumerate(up_phys_qubits)})
    Q_down = apply_column_swaps(Q_down, perm_down, {q: i for i, q in enumerate(down_phys_qubits)})
    
    # 7. Step classical frame V_sigma(n) = U_sigma(n) Q(n)[1]
    V_up_new, V_down_new = U_up @ Q_up, U_down @ Q_down
    
    # 8. Apply 3-Qubit Interaction Gate [Eq. (57)][1]
    UK_n = build_kondo_gate(Jk, Jz, h, T, n)
    psi_mps = apply_3_qubit_gate_custom(psi_mps, UK_n, imp_q - 1, imp_q, imp_q + 1)
    
    # 9. Advance Natural Orbital Frame: U_sigma(n+1) = V_sigma(n) @ u_sigma(n+1) [Eq. (61)][1]
    active_indices_u = list(range(start_u, end_u))
    psi_mps, U_up_next, evals_up_next = advance_natural_orbital_frame(psi_mps, V_up_new, up_phys_qubits, imp_q, L_bath, active_indices_u)
    
    active_indices_d = list(range(start_d, end_d))
    psi_mps, U_down_next, evals_down_next = advance_natural_orbital_frame(psi_mps, V_down_new, down_phys_qubits, imp_q, L_bath, active_indices_d)
    
    # 10. Classify and Truncate Inactive Subspaces
    filled_u_next, active_u_next, empty_u_next = classify_orb(evals_up_next, tol)
    filled_d_next, active_d_next, empty_d_next = classify_orb(evals_down_next, tol)
    
    to_remove_up = [(up_phys_qubits[i], 1) for i in filled_u_next] + [(up_phys_qubits[i], 0) for i in empty_u_next]
    to_remove_down = [(down_phys_qubits[i], 1) for i in filled_d_next] + [(down_phys_qubits[i], 0) for i in empty_d_next]
    psi_mps = truncate_mps_safe(psi_mps, to_remove_up + to_remove_down)
    
    return psi_mps, U_up_next, U_down_next, V_up_new, V_down_new, evals_up_next, evals_down_next, len(active_u_next), len(active_d_next)