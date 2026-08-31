## A file of helper functions and examples only, to be used for prompting and for making the main algorithm modular and more readable.

from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from openfermion.linalg.givens_rotations import (
    givens_decomposition,
    givens_decomposition_square,
)
from scipy.linalg import block_diag, expm

from qlimb.classical.gates import Gate
from qlimb.classical.mpo import MPO
from qlimb.classical.mps import MPS
from qlimb.classical.utils import apply_svd

#For Claude: For functions imported from qlimb or openfermion, the input and output arguments and data types will be provided below for reference. DO NOT implement these functions separately.
# MPO from qlimb: inputs -> nqbits: int, phys_dim: int, tensors: list of np.ndarray, bond_dim: int (default infty), preserve_norm: bool (default True), trunc_tol (default 1e-10): float; output -> MPO object
# MPS from qlimb: inputs -> nqbits: int, phys_dim: int, tensors: list of np.ndarray, bond_dim: int (default infty), preserve_norm: bool (default True), trunc_tol (default 1e-10): float; output -> MPS object
# GATE from qlimb: inputs -> gate_matrix: np.ndarray, position array: np.ndarray (row vector); output -> Gate object
# apply_svd from qlimb: inputs -> tensor: np.ndarray, max_bond_dim: int, direction: str ('left' or 'right'), preserve_norm: bool (default True), trunc_tol (default 1e-10): float; output -> U, S, S_truncated Vh (np.ndarray)
# givens_decomposition_square from openfermion: inputs -> matrix: np.ndarray (Unitary); output -> list of Givens rotation matrices and diagonal phases (np.ndarray)

#For examples as to how these functions are used, please refer to the checkmps.py and mpo_demo.py files attached. openfermion is a public library, so givens_decomposition_square can be checked directly from the openfermion documentation. The other functions are part of the qlimb library, which is private.

#### HELPER FUNCTIONS ####
# =======================================================================================
# 1. Core Operators and Matrices
# =======================================================================================
I2 = np.eye(2, dtype=complex)
Z_OP = np.array([[1, 0], [0, -1]], dtype=complex)
SP_OP = np.array([[0, 1], [0, 0]], dtype=complex)   
SM_OP = np.array([[0, 0], [1, 0]], dtype=complex)
_SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
FSWAP_MAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=complex)

Sz_op = 0.5 * Z_OP
Sp_op = np.array([[0, 1], [0, 0]], dtype=complex)   
Sm_op = np.array([[0, 0], [1, 0]], dtype=complex)
Sx_op = 0.5 * (Sp_op + Sm_op)
Sy_op = -0.5j * (Sp_op - Sm_op)
# Pauli matrices for the impurity spin
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a, b, c): return np.kron(np.kron(a, b), c)

# =======================================================================================
# 2. Safe Dynamic MPS Resizing (unchanged)
# =======================================================================================
def augment_mps_safe(psi_mps, insert_idx, state_type):
    """
    Re-inserts one previously-truncated filled/empty pure-state qubit into
    the MPS. Implements the "restore" half of the Eq. (43)/(54) ansatz
    |Psi~(n)> ~ |filled> (x) |psi(n)> (x) |empty>: the two filled and two
    empty spectator modes referenced in Sec. VI.D.1 ("Augment the MPS for
    the few body wave function by multiplying two filled and two empty
    orbitals") are stored as trivial product states outside the active
    window and spliced back in here at the start of each Floquet step.
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
    Removes filled/empty natural-orbital qubits from the active MPS window
    once their occupation is 0 or 1 (Eq. 32a/32c classification -> Eq. 54
    truncation step, Sec. VI.D.3): "If some lambda_a^sigma(n+1) is in
    [0,eps] union [1-eps,1], truncate the MPS ... |Psi(n+1)> ~ |filled> (x)
    |psi(n+1)> (x) |empty>". The removed product-state amplitude is
    absorbed into the neighboring tensor and the state is renormalized,
    consistent with the notes' remark that this projection may require
    renormalizing (or dividing final observables by the norm).
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


def update_chain_bookkeeping(up_bath, down_bath, imp_q, idx, op):
    if op == 'insert':
        up_bath = [p + 1 if p >= idx else p for p in up_bath]
        down_bath = [p + 1 if p >= idx else p for p in down_bath]
        if idx < imp_q:
            imp_q += 1
            pos = idx
            insert_at = 0
            while insert_at < len(up_bath) and up_bath[insert_at] > pos:
                insert_at += 1
            up_bath.insert(insert_at, pos)
        elif idx > imp_q:
            pos = idx
            insert_at = 0
            while insert_at < len(down_bath) and down_bath[insert_at] < pos:
                insert_at += 1
            down_bath.insert(insert_at, pos)
        return up_bath, down_bath, imp_q

    if op == 'remove':
        if idx < imp_q:
            up_bath = [p - 1 if p > idx else p for p in up_bath]
            if idx in up_bath:
                up_bath.remove(idx)
            imp_q -= 1
        elif idx > imp_q:
            down_bath = [p - 1 if p > idx else p for p in down_bath]
            if idx in down_bath:
                down_bath.remove(idx)
        return up_bath, down_bath, imp_q

    raise ValueError(f'unknown bookkeeping op {op}')

# =======================================================================================
# 3. Initialization & SERGIO Core (unchanged)
# =======================================================================================
# Initial state |Psi(0)> = |FS>_up (x) |Down_arrow> (x) |FS>_down (text
# preceding Eq. 1): builds the single-particle k-space Fermi-sea orbitals
# used to Slater-determinant-fill each half-chain bath at half filling.
def build_initial_orbitals(N_bath):
    x, m = np.arange(N_bath), N_bath / 2.0
    j_range = range(-int(m // 2), int(m // 2) + 1) if m % 2 != 0 else range(-int(m // 2), int(m // 2))
    k_occ = np.array([2.0 * np.pi * j / N_bath for j in j_range])
    orb = np.zeros((N_bath, len(k_occ)), dtype=complex)
    for col, k in enumerate(k_occ): orb[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return orb, orb


def build_initial_orbital_frame(N_bath, reverse=False):
    """Canonical full single-particle Fourier frame used at step 0."""
    x = np.arange(N_bath)[::-1] if reverse else np.arange(N_bath)
    k_vals = [2.0 * np.pi * j / N_bath for j in range(-N_bath // 2, N_bath // 2)]
    basis = np.zeros((N_bath, N_bath), dtype=complex)
    for col, k in enumerate(k_vals):
        basis[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return basis

# Builds |Psi(0)> as an explicit 2^(2N+1) statevector: Slater determinants
# of the Fermi-sea orbitals (build_initial_orbitals) on each half-chain,
# impurity fixed to |Down_arrow>, in the unfolded up|imp|down qubit layout
# used throughout (matches latest.py's build_initial_state exactly).
def build_initial_state(N, orb_up, orb_down, n_up, n_down):
    psi = np.zeros(2 ** (2 * N + 1), dtype=complex)
    for down_occ in combinations(range(N), n_down):
        s_down = np.zeros(N, dtype=int); s_down[list(down_occ)] = 1
        a_down = np.linalg.det(orb_down[np.where(s_down == 1)[0], :])
        for up_occ in combinations(range(N), n_up):
            s_up = np.zeros(N, dtype=int); s_up[list(up_occ)] = 1
            a_up = np.linalg.det(orb_up[np.where(s_up == 1)[0], :])
            full_state = np.concatenate((1 - s_up[::-1], [1], 1 - s_down))
            idx = 0
            for bit in full_state: idx = (idx << 1) | bit
            psi[idx] = a_down * a_up
    return psi

# Converts the explicit |Psi(0)> statevector into an MPS via sequential
# SVD -- the initial "|psi(0)> as a (rather trivial) MPS" step of Sec.
# VI.C ("Initialization on the computer").
def state_mps(psi, N, bond_dim=np.inf):
    tensors, chi_left, T = [], 1, psi.reshape([2] * (2 * N + 1))
    for _ in range(2 * N):
        U, _, _, V = apply_svd(T.reshape(chi_left, 2, -1, 1), bond_dim=bond_dim, direction='right', preserve_norm=True, tol=1e-10)
        tensors.append(U); chi_left = U.shape[-1]; T = V.reshape(chi_left, 2, -1)
    tensors.append(T.reshape(chi_left, 2, 1))
    return MPS(nqbits=2 * N + 1, phys_dim=2, tensors=tensors, bond_dim=bond_dim, preserve_norm=True, trunc_tol=1e-10)

def build_1rdm_bath(psi_mps, bath_qubits, imp_qubit):
    """
    Computes C^sigma_ij(n) = <Psi_IA(n)| c_i,sigma^dagger c_j,sigma |Psi_IA(n)>,
    Eq. (30), for one channel (up or down). Diagonalizing this (done in
    advance_natural_orbital_frame) gives the natural orbital basis, Eq. (37).
    Includes the required Jordan-Wigner Z-string between sites i and j for
    off-diagonal (i != j) elements -- this is the one place a JW string is
    genuinely needed, since i and j are both real bath sites in the same
    channel with possible other occupied sites between them.
    """
    N_total, n = psi_mps.nqbits, len(bath_qubits)
    C, Nop = np.zeros((n, n), dtype=complex), (I2 + Z_OP) / 2
    for a in range(n):
        for b in range(a, n):
            i, j = bath_qubits[a], bath_qubits[b]
            left, right = min(i, j), max(i, j)
            tensors = []
            for k in range(N_total):
                if i == j:
                    op = Nop if k == i else I2
                else:
                    if k == imp_qubit:
                        op = I2
                    elif k == left:
                        op = SP_OP if i < j else SM_OP
                    elif k == right:
                        op = SM_OP if i < j else SP_OP
                    elif left < k < right:
                        op = -Z_OP
                    else:
                        op = I2
                tensors.append(op.reshape(1, 2, 2, 1))
            val = psi_mps @ (MPO(nqbits=N_total, phys_dim=2, tensors=tensors) @ psi_mps)
            C[a, b] = val
            if a != b: C[b, a] = np.conj(val)
    return C

# Eq. (32a)-(32c): classify natural-orbital occupations lambda^sigma_a
# into filled ([1-eps,1]), active ([eps,1-eps]), empty ([0,eps]).
def classify_orb(evals, tol=0.1):
    filled, active, empty = [], [], []
    for i, occ in enumerate(evals):
        if occ >= 1.0 - tol:
            filled.append(i)
        elif occ <= tol:
            empty.append(i)
        else:
            active.append(i)
    return filled, active, empty

# Circuit-level primitive: applies a 2-qubit matrix to adjacent MPS sites,
# orienting it correctly regardless of which of qi,qj is physically left.
# Used to realize the Gaussian (single-particle) rotations of Sec. V as
# nearest-neighbor gate sequences (per the notes' open request in Sec. II:
# "I didn't start thinking about how to implement certain Givens rotations").
'''def apply_two_qubit_gate(psi_mps, mat, qi, qj):
    if qj == qi + 1: return Gate(matrix=mat, indices=(qi, qj)) @ psi_mps
    elif qi == qj + 1: return Gate(matrix=_SWAP4 @ mat @ _SWAP4, indices=(qj, qi)) @ psi_mps
    else: raise ValueError(f"qubits {qi},{qj} not adjacent")'''

def apply_two_qubit_gate(psi_mps, mat, qi, qj):
    if qj == qi + 1: return Gate(matrix=mat, indices=(qi, qj)) @ psi_mps
    elif qi == qj + 1: return Gate.FSWAP(indices=(qj, qi), phys_dim=2) @ Gate(matrix=mat, indices=(qj, qi)) @ Gate.FSWAP(indices=(qj, qi), phys_dim=2) @ psi_mps
    else: raise ValueError(f"qubits {qi},{qj} not adjacent")


# One fermionic (particle-number-conserving) Givens rotation gate, the
# elementary building block of any single-particle Gaussian unitary V in
# Sec. V ([V]_ij = [e^{-h}]_ij), decomposed via openfermion's
# givens_decomposition / givens_decomposition_square.
def fermionic_givens_gate_matrix(theta, phi):
    # FIX (verified via a from-scratch, controlled 2-qubit unit test): the
    # physical gate implementing openfermion's returned (theta,phi) for a
    # VECTOR reduction must be built from conj(G), not G, where
    # G = [[cos(theta), -e^{i phi} sin(theta)], [sin(theta), e^{i phi} cos(theta)]].
    c, s, ph = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    phc = np.conj(ph)
    return np.array([[1, 0, 0, 0], [0, c, -phc * s, 0], [0, s, phc * c, 0], [0, 0, 0, phc]], dtype=complex)


def fermionic_givens_gate_matrix_square(theta, phi):
    # SEPARATE convention for the givens_decomposition_square consumer
    # (apply_givens_circuit / advance_natural_orbital_frame): verified
    # empirically and must not be mixed with the reduction-gate convention.
    c, s, ph = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    phc = np.conj(ph)
    return np.array([[1, 0, 0, 0], [0, c, -phc * s, 0], [0, s, phc * c, 0], [0, 0, 0, phc]], dtype=complex)


def apply_givens_circuit(psi_mps, decomp, diagonal, physical_qubits):
    """
    Applies a full Gaussian unitary u_sigma(n+1) (decomposed by openfermion
    into diagonal phases + Givens rotation layers) to the MPS as a gate
    circuit. This is the concrete circuit realization of Eq. (51)/(53):
    u_sigma(n+1) dagger acting on the natural-orbital-basis wavefunction,
    i.e. |psi~(n+1)> = u_sigma(n+1)^dagger |psi(n+1))).
    """
    psi_no = psi_mps.copy()
    for local_site, phase in enumerate(diagonal):
        psi_no = Gate(matrix=np.array([[1, 0], [0, phase]], dtype=complex), indices=(physical_qubits[local_site],)) @ psi_no
    for layer in reversed(decomp):
        for (i, j, theta, phi) in layer:
            G = fermionic_givens_gate_matrix_square(theta, phi)
            psi_no = apply_two_qubit_gate(psi_no, G, physical_qubits[i], physical_qubits[j])
    return psi_no


def advance_natural_orbital_frame(psi_mps, V_sigma, bath_qubits, imp_qubit):
    """
    "Finding the new basis of natural orbitals", Sec. VI.D.2: diagonalizes
    the 1-RDM (Eq. 30/37) to get the sorted eigenvalues lambda^sigma_a(n+1)
    and eigenvectors u_sigma(n+1) (Eq. 50), applies u_sigma(n+1)^dagger to
    the MPS as a Givens circuit (Eq. 51/53), and updates the real-space
    frame matrix U_sigma(n+1) = V_sigma @ u_sigma(n+1) (Eq. 52). Used both
    for the very first natural-orbital rotation (Sec. VI.C, Eq. 45) and at
    the end of every Floquet step.
    """
    evals, evecs = np.linalg.eigh(build_1rdm_bath(psi_mps, bath_qubits, imp_qubit))
    order = np.argsort(evals)[::-1]
    evals_sorted, evecs_sorted = evals[order], evecs[:, order]
    decomp, diagonal = givens_decomposition_square(evecs_sorted.conj())
    psi_mps = apply_givens_circuit(psi_mps, decomp, diagonal, bath_qubits)
    V_sigma_new = V_sigma @ evecs_sorted
    return psi_mps, V_sigma_new, evals_sorted

def two_mode_unitary_to_fermionic_gate(U2):
    """
    Embed a 2x2 single-particle unitary U2 into the corresponding 4x4
    number-conserving fermionic gate in basis [|00>,|01>,|10>,|11>].
    """
    G4 = np.zeros((4, 4), dtype=complex)
    G4[0, 0] = 1.0
    # One-particle subspace ordering is [|10>, |01>] -> indices [2, 1]
    G4[2, 2], G4[2, 1] = U2[0, 0], U2[0, 1]
    G4[1, 2], G4[1, 1] = U2[1, 0], U2[1, 1]
    # Two-particle sector picks det(U2) for Gaussian number-conserving maps.
    G4[3, 3] = np.linalg.det(U2)
    return G4


def get_actual_two_site_gate(theta, phi, q_left, q_right):
    G4 = fermionic_givens_gate_matrix(theta, phi)
    if q_right == q_left + 1:
        return G4
    if q_left == q_right + 1:
        return _SWAP4 @ G4 @ _SWAP4
    raise ValueError(f'qubits {q_left} and {q_right} are not adjacent')


def extract_single_particle_block_from_gate(G4):
    return np.array([[G4[2, 2], G4[2, 1]], [G4[1, 2], G4[1, 1]]], dtype=complex)


def apply_givens_decomposition_to_vector(vector, decomposition, physical_qubits=None):
    v = vector.astype(complex).copy()
    for parallel_group in decomposition:
        for (i, j, theta, phi) in parallel_group:
            if physical_qubits is None:
                G4 = fermionic_givens_gate_matrix(theta, phi)
            else:
                G4 = get_actual_two_site_gate(theta, phi, physical_qubits[i], physical_qubits[j])
            U2 = extract_single_particle_block_from_gate(G4)
            vi, vj = v[i], v[j]
            v[i], v[j] = U2[0, 0] * vi + U2[0, 1] * vj, U2[1, 0] * vi + U2[1, 1] * vj
    return v


def reduce_vector_classically(vector, reduce_position='first'):
    v_ord = vector[::-1].copy() if reduce_position == 'last' else vector.copy()
    norm = np.linalg.norm(v_ord)
    if norm < 1e-14:
        return vector.copy().astype(complex)
    decomp, _, _ = givens_decomposition((v_ord / norm).reshape(1, -1))
    v = apply_givens_decomposition_to_vector(v_ord, decomp, physical_qubits=np.arange(len(v_ord)))
    return v[::-1] if reduce_position == 'last' else v


def apply_reduction_as_gates(psi_mps, vector, physical_qubits, reduce_position='first'):
    v_ord = vector[::-1].copy() if reduce_position == 'last' else vector.copy()
    order = physical_qubits[::-1] if reduce_position == 'last' else physical_qubits
    norm = np.linalg.norm(v_ord)
    if norm < 1e-14:
        return psi_mps, vector.copy().astype(complex)
    decomp, _, _ = givens_decomposition((v_ord / norm).reshape(1, -1))
    for grp in decomp:
        for (i, j, theta, phi) in grp:
            G4 = get_actual_two_site_gate(theta, phi, order[i], order[j])
            psi_mps = apply_two_qubit_gate(psi_mps, G4, order[i], order[j])
            v_ord[i], v_ord[j] = extract_single_particle_block_from_gate(G4) @ np.array([v_ord[i], v_ord[j]], dtype=complex)
    v_trans = v_ord
    return psi_mps, (v_trans[::-1] if reduce_position == 'last' else v_trans)


def build_Q_matrix(N, n_f, n_a, n_e, vec_f, sub_v, vec_e, w_start, w_end):
    """
    Constructs the full single-particle Q_sigma(n) matrix, right below Eq. (42),
    from the three separate Givens circuits Q_full, Q_empty, and Q_few.
    """
    def reduce_block(block_vec, reduce_position, dim):
        v = block_vec.astype(complex).copy()
        if reduce_position == 'last':
            v_ord = v[::-1].copy()
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
            return Q, v_ord[::-1]

        v_ord = v.copy()
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
        return Q, v_ord

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
    if win_len > 0:
        M2[w_start:w_end, w_start:w_end] = Q_few
    return M1 @ M2

def apply_qfull_qempty_qfew_one_spin(psi_mps, v_f, v_a, v_e, phys_qubits):
    """
    One spin channel's full Sec. VI.D.1 "time evolution step" prelude:
    reduces the filled subspace to f_0,sigma and the empty subspace to
    e_{M+1,sigma} (Eq. 17a/17b, via reduce_vector_classically), concatenates
    with the active-orbital amplitudes v_a, then applies that combined
    window as a Givens gate circuit (apply_reduction_as_gates) realizing
    Q_few (Eq. 41-42) on the MPS -- collapsing [f_0, a_1..a_M, e_{M+1}] into
    a single effective mode c_eff_sigma. Returns the updated MPS, the
    physical qubit index now holding c_eff_sigma, and the single-particle
    Q_sigma matrix (Eq. 40, via build_Q_matrix) needed to update U_sigma.
    """
    f_red = reduce_vector_classically(v_f, 'last')
    e_red = reduce_vector_classically(v_e, 'first')
    
    vec = np.concatenate((f_red, v_a, e_red)).astype(complex)
    nz = np.flatnonzero(np.abs(vec) > 1e-12)
    start, end = (nz[0], nz[-1] + 1) if len(nz) > 0 else (0, 0)
    
    if len(nz) > 0:
        psi_mps, _sub_trans = apply_reduction_as_gates(psi_mps, vec[start:end], phys_qubits[start:end], 'last')
    
    c_idx = end - 1 if len(nz) > 0 else 0
    Q_sigma = build_Q_matrix(len(phys_qubits), len(v_f), len(v_a), len(v_e), v_f, vec[start:end], v_e, start, end)
    return psi_mps, phys_qubits[c_idx], Q_sigma

def move_mode_adjacent_to_impurity(psi_mps, curr_q, chain_qs):
    """
    Physical bookkeeping step required so that H2bar(n) (Eq. 27/47), a
    3-qubit gate, can act on adjacent MPS sites [c_eff_up, imp, c_eff_down]:
    fermionic-swaps (FSWAP, so no extra JW sign is introduced) the effective
    mode c_eff_sigma from wherever apply_qfull_qempty_qfew_one_spin left it
    up to the qubit nearest the impurity. Not separately numbered in the
    notes but implied by the requirement that H2bar(n) is genuinely 3-qubit
    local (Eq. 29).
    """
    target_q = chain_qs[0] 
    if curr_q == target_q: return psi_mps, curr_q, []
    pos, step, perm = chain_qs.index(curr_q), -1 if chain_qs.index(curr_q) > 0 else 1, []
    for p in range(pos, 0, step):
        psi_mps = apply_two_qubit_gate(psi_mps, FSWAP_MAT, curr_q, chain_qs[p + step])
        perm.append((curr_q, chain_qs[p + step])); curr_q = chain_qs[p + step]
    return psi_mps, curr_q, perm

# Keeps the real-space frame matrix U_sigma consistent with the FSWAPs
# just applied to the MPS in move_mode_adjacent_to_impurity, by permuting
# the corresponding columns of U_sigma the same way.
def apply_column_swaps(V, permutation, local_idx_map):
    for (qa, qb) in permutation:
        ia, ib = local_idx_map[qa], local_idx_map[qb]
        V[:, [ia, ib]] = V[:, [ib, ia]]
    return V

# =======================================================================================
# 4. Corrected Free Propagator (Trotter/Exact switch)
# =======================================================================================
def get_free_propagator(L, theta, n, use_trotter=True, boundary='open'):
    """
    Returns U_TE(n) = [e^{-i H1 T n}]_ji, the L x L single-particle bath
    propagator of Eq. (8), used in Eq. (15)/(55) to get v*_a,sigma(n).

    use_trotter=True: the discrete "brickwall" H1 explicitly flagged in
    the notes ("In principle, instead of H1 we should consider a
    brickwall, as in Sarma et al."). This is NOT an approximation of the
    continuum hopping model here -- it is, by construction, exactly the
    same even/odd fSim-gate brickwork used for the bath in latest.py.

    use_trotter=False: the continuum single-particle hopping propagator
    exp(-i * 2*theta * A * n) for a nearest-neighbor adjacency matrix A.
    NOTE: this is the thing worth testing next -- the notes' comment about
    the brickwall being introduced "instead of H1" reads as flagging an
    ALTERNATIVE discretization, not a statement that the SS benchmark
    itself was generated with the brickwall. If SS was generated from
    continuous H1 directly (e.g. plain exact diagonalization/expm, no
    digital gates at all), use_trotter=True will never match it regardless
    of how correct the rest of the pipeline is.

    boundary='open'     -> bonds (0,1),...,(L-2,L-1)  [original]
    boundary='periodic' -> adds the wraparound bond (L-1,0), consistent
      with the periodic-k-space Fermi sea initial state and the notes'
      closed-BC assumption (Sec. I). For the Trotter path (L even) this
      wraparound bond is folded into the odd-bond layer, completing the
      two-color perfect-matching partition of the ring graph. For the
      exact continuum path it's just one extra adjacency-matrix entry.
    """
    def bond_rotation(i, angle):
        c = np.cos(angle)
        s = 1j * np.sin(angle)   # fsim uses i*sin
        R = np.array([[c, s], [s, c]], dtype=complex)
        U = np.eye(L, dtype=complex)
        U[i:i+2, i:i+2] = R
        return U

    def wrap_rotation(angle):
        # wraparound bond (L-1, 0) as its own commuting rotation
        c = np.cos(angle)
        s = 1j * np.sin(angle)
        U = np.eye(L, dtype=complex)
        i, j = L - 1, 0
        U[i, i], U[i, j], U[j, i], U[j, j] = c, s, s, c
        return U

    if use_trotter:
        # Trotterized free evolution matching the benchmark (latest.py)
        U_even_half = np.eye(L, dtype=complex)
        for i in range(0, L-1, 2):
            U_even_half = bond_rotation(i, theta) @ U_even_half

        U_odd_full = np.eye(L, dtype=complex)
        for i in range(1, L-1, 2):
            U_odd_full = bond_rotation(i, 2*theta) @ U_odd_full
        if boundary == 'periodic' and L % 2 == 0:
            U_odd_full = wrap_rotation(2*theta) @ U_odd_full

        U_even_full = np.eye(L, dtype=complex)
        for i in range(0, L-1, 2):
            U_even_full = bond_rotation(i, 2*theta) @ U_even_full

        U_initial = U_odd_full @ U_even_half
        U_step = U_odd_full @ U_even_full

        if n == 0:
            return U_initial
        else:
            return np.linalg.matrix_power(U_step, n) @ U_initial
    else:
        # Exact free evolution: exp(-i * 2*theta * adjacency * n)
        A = np.eye(L, k=1) + np.eye(L, k=-1)
        if boundary == 'periodic':
            A[0, L-1] = 1
            A[L-1, 0] = 1
        U_step = expm(-1j * 2 * theta * A)
        if n == 0:
            return np.eye(L, dtype=complex)
        else:
            return np.linalg.matrix_power(U_step, n)

# =======================================================================================
# 5. Build the time‑dependent 3‑qubit Kondo gate (Eq. (10), (27))
# =======================================================================================
#def build_kondo_gate(Jk, Jz, h, T, n): #Here Jk = J_x = J_y, see paper "Design and Benchmarks..." PDF, Eq. (1c)
    
    
    # Notice that S^mu(n) is just S^mu conjugated with e^{-i h T n S^z}, which is a rotation around the z-axis by angle h T n.
    # We need exp (-iT H2(n)) = e^{-iT H2(0)} conjugated with e^{-i h T n S^z} (essentially changing the impurity spin to interaction picture). We have the exact 8 by 8 analytical matrix for e^{-iT H2(0)}, given as Eq.(14) in the "Design and Benchmarks..." PDF.

    # Complete the rest of the build_kondo_gate code based on this

#For every helper function, first check if the function is correctly implemented (have assert statements or test cases. If not, write a test case for it). The build_kondo_gate need not be checked in the current workflow and may be checked separately latter. Also make the description of each helper function uniform in format and style.
# Only construct a new helper function if it is extremely helpful for the final algorithm and is not present here yet.
# While making the final sergio_floquet_code, ensure there are assert statements present in between steps to ensure the correctness of the intermediate results. If any intermediate result is not correct, raise an error and stop the execution of the code.

###### End of helper_functions.py #######