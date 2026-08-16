

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag, expm, logm

from qlimb.classical.gates import Gate
from qlimb.classical.mpo import MPO
from qlimb.classical.mps import MPS
from qlimb.classical.utils import apply_svd
from scipy import sparse
from scipy.sparse import kron as skron, identity as sident

# =======================================================================================
# 1. Core operators (UNCHANGED)
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
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


# =======================================================================================
# 2. Exact-rotation primitives (UNCHANGED -- validated, not flagged)
# =======================================================================================
def _build_generator_dense(h_gen, n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for a in range(n):
        for b in range(n):
            coeff = h_gen[a, b]
            if abs(coeff) < 1e-13:
                continue
            if a == b:
                Nop = (I2 + Z_OP) / 2
                ops = [Nop if k == a else I2 for k in range(n)]
            else:
                left, right = min(a, b), max(a, b)
                ops = []
                for k in range(n):
                    if k == left:
                        ops.append(SP_OP if a < b else SM_OP)
                    elif k == right:
                        ops.append(SM_OP if a < b else SP_OP)
                    elif left < k < right:
                        ops.append(-Z_OP)
                    else:
                        ops.append(I2)
            term = ops[0]
            for o in ops[1:]:
                term = np.kron(term, o)
            H += coeff * term
    return H


def apply_exact_gaussian_rotation(psi_mps, W, qubit_positions, n_total):
    n = len(qubit_positions)
    if n <= 1:
        return psi_mps

    h_gen = 1j * logm(W)
    h_gen = (h_gen + h_gen.conj().T) / 2

    order = np.argsort(qubit_positions)
    sorted_positions = [qubit_positions[o] for o in order]
    h_gen_sorted = h_gen[np.ix_(order, order)]

    H_small = _build_generator_dense(h_gen_sorted, n)
    U_small = expm(-1j * H_small)

    mpo_small = MPO.from_matrix(U_small, phys_dim=2, nqbits=n, max_bond_dim=np.inf, trunc_tol=1e-13)

    full_tensors = [np.eye(2, dtype=complex).reshape(1, 2, 2, 1) for _ in range(n_total)]
    for k, q in enumerate(sorted_positions):
        full_tensors[q] = mpo_small.tensors[k]
    mpo_full = MPO(nqbits=n_total, phys_dim=2, tensors=full_tensors)
    return mpo_full @ psi_mps


def householder_unitary(v, target_idx=0):
    n = len(v)
    v = np.asarray(v, dtype=complex)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.eye(n, dtype=complex)
    vn = v / norm
    e = np.zeros(n, dtype=complex)
    e[target_idx] = 1.0
    phase = vn[target_idx] / abs(vn[target_idx]) if abs(vn[target_idx]) > 1e-14 else 1.0
    w = vn - phase * e
    wnorm = np.linalg.norm(w)
    if wnorm < 1e-12:
        return np.eye(n, dtype=complex)
    w = w / wnorm
    H = np.eye(n, dtype=complex) - 2.0 * np.outer(w, w.conj())
    return H


# =======================================================================================
# 3. Initialization (UNCHANGED)
# =======================================================================================
def build_initial_orbitals(N_bath):
    x, m = np.arange(N_bath), N_bath / 2.0
    j_range = range(-int(m // 2), int(m // 2) + 1) if m % 2 != 0 else range(-int(m // 2), int(m // 2))
    k_occ = np.array([2.0 * np.pi * j / N_bath for j in j_range])
    orb = np.zeros((N_bath, len(k_occ)), dtype=complex)
    for col, k in enumerate(k_occ):
        orb[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return orb, orb


def build_initial_orbital_frame(N_bath, reverse=False):
    x = np.arange(N_bath)[::-1] if reverse else np.arange(N_bath)
    k_vals = [2.0 * np.pi * j / N_bath for j in range(-N_bath // 2, N_bath // 2)]
    basis = np.zeros((N_bath, N_bath), dtype=complex)
    for col, k in enumerate(k_vals):
        basis[:, col] = np.exp(-1j * k * x) / np.sqrt(N_bath)
    return basis


def build_initial_state(N, orb_up, orb_down, n_up, n_down):
    psi = np.zeros(2 ** (2 * N + 1), dtype=complex)
    for down_occ in combinations(range(N), n_down):
        s_down = np.zeros(N, dtype=int); s_down[list(down_occ)] = 1
        a_down = np.linalg.det(orb_down[np.where(s_down == 1)[0], :])
        for up_occ in combinations(range(N), n_up):
            s_up = np.zeros(N, dtype=int); s_up[list(up_occ)] = 1
            a_up = np.linalg.det(orb_up[np.where(s_up == 1)[0], :])
            full_state = np.concatenate((s_up[::-1], [1], s_down))
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


# =======================================================================================
# 4. 1RDM (UNCHANGED)
# =======================================================================================
def build_1rdm_bath(psi_mps, bath_qubits, imp_qubit):
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


def classify_orb(evals, tol=1e-10):
    filled, active, empty = [], [], []
    for i, occ in enumerate(evals):
        if occ >= 1.0 - tol:
            filled.append(i)
        elif occ <= tol:
            empty.append(i)
        else:
            active.append(i)
    return filled, active, empty


# =======================================================================================
# 5. Natural-orbital-frame update (UNCHANGED)
# =======================================================================================
def advance_natural_orbital_frame(psi_mps, V_sigma, bath_qubits, imp_qubit):
    C = build_1rdm_bath(psi_mps, bath_qubits, imp_qubit)
    evals, V = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals_sorted, V_sorted = evals[order], V[:, order]

    W = V_sorted.conj().T
    psi_mps = apply_exact_gaussian_rotation(psi_mps, W, bath_qubits, psi_mps.nqbits)
    V_sigma_new = V_sigma @ W
    return psi_mps, V_sigma_new, evals_sorted


# =======================================================================================
# 6. Filled/empty/few-body reduction (UNCHANGED)
# =======================================================================================
def reduce_and_apply(psi_mps, vector, physical_qubits, target_idx):
    n = len(vector)
    if n == 0:
        return psi_mps, np.zeros((0, 0), dtype=complex)
    Q = householder_unitary(vector, target_idx=target_idx)
    if np.allclose(Q, np.eye(n), atol=1e-13):
        return psi_mps, Q
    psi_mps = apply_exact_gaussian_rotation(psi_mps, Q, physical_qubits, psi_mps.nqbits)
    return psi_mps, Q


def apply_qfull_qempty_qfew_one_spin(psi_mps, v_f, v_a, v_e, phys_qubits):
    n_f, n_a, n_e = len(v_f), len(v_a), len(v_e)
    N = len(phys_qubits)

    f_phys = phys_qubits[:n_f]
    psi_mps, Q_f = reduce_and_apply(psi_mps, v_f, f_phys, target_idx=n_f - 1) if n_f > 0 else (psi_mps, np.zeros((0, 0), dtype=complex))
    f_red = Q_f @ v_f if n_f > 0 else v_f

    e_phys = phys_qubits[n_f + n_a:]
    psi_mps, Q_e = reduce_and_apply(psi_mps, v_e, e_phys, target_idx=0) if n_e > 0 else (psi_mps, np.zeros((0, 0), dtype=complex))
    e_red = Q_e @ v_e if n_e > 0 else v_e

    vec = np.concatenate((f_red, v_a, e_red)).astype(complex)
    nz = np.flatnonzero(np.abs(vec) > 1e-12)
    start, end = (nz[0], nz[-1] + 1) if len(nz) > 0 else (0, 0)
    win_len = end - start

    Q_few = np.eye(win_len, dtype=complex)
    if win_len > 0:
        sub_v = vec[start:end]
        win_phys = phys_qubits[start:end]
        psi_mps, Q_few = reduce_and_apply(psi_mps, sub_v, win_phys, target_idx=win_len - 1)

    c_idx = end - 1 if win_len > 0 else 0

    M1 = block_diag(
        Q_f if n_f > 0 else np.zeros((0, 0), dtype=complex),
        np.eye(n_a, dtype=complex),
        Q_e if n_e > 0 else np.zeros((0, 0), dtype=complex),
    )
    M2 = np.eye(N, dtype=complex)
    if win_len > 0:
        M2[start:end, start:end] = Q_few
    Q_sigma = M2 @ M1
    return psi_mps, phys_qubits[c_idx], Q_sigma


def move_mode_adjacent_to_impurity(psi_mps, curr_q, chain_qs):
    target_q = chain_qs[0]
    if curr_q == target_q: return psi_mps, curr_q, []
    pos, step, perm = chain_qs.index(curr_q), -1, []
    for p in range(pos, 0, step):
        gate = FSWAP_MAT
        qi, qj = curr_q, chain_qs[p + step]
        if qj == qi + 1:
            psi_mps = Gate(matrix=gate, indices=(qi, qj)) @ psi_mps
        else:
            psi_mps = Gate(matrix=_SWAP4 @ gate @ _SWAP4, indices=(qj, qi)) @ psi_mps
        perm.append((curr_q, chain_qs[p + step])); curr_q = chain_qs[p + step]
    return psi_mps, curr_q, perm


def apply_column_swaps(V, permutation, local_idx_map):
    for (qa, qb) in permutation:
        ia, ib = local_idx_map[qa], local_idx_map[qb]
        V[:, [ia, ib]] = V[:, [ib, ia]]
    return V


# =======================================================================================
# 7. U_TE(n) -- closed BC
# =======================================================================================
def get_free_propagator(L, t0, n):
# not used 
    A = np.eye(L, k=1) + np.eye(L, k=-1)
    A[0, L - 1] = 1.0
    A[L - 1, 0] = 1.0   # closed boundary condition -- the wraparound bond
    H1_sp = -(t0/2) * A
    if n == 0:
        return np.eye(L, dtype=complex)
    U_step = expm(-1j * H1_sp)
    return np.linalg.matrix_power(U_step, n)


def build_kondo_gate(Jx, Jy, Jz, h, T, n):
    
    cos_phi = np.cos(h * n * T); sin_phi = np.sin(h * n * T)
    Sx_n = cos_phi * Sx_op + sin_phi * Sy_op
    Sy_n = -sin_phi * Sx_op + cos_phi * Sy_op
    Sz_n = Sz_op

    def M_mu(sigma):
        s00, s01, s10, s11 = sigma[0, 0], sigma[0, 1], sigma[1, 0], sigma[1, 1]
        n_up = np.diag([0, 0, 1, 1]).astype(complex)
        n_down = np.diag([0, 1, 0, 1]).astype(complex)
        c_up_dag_c_down = np.zeros((4, 4), dtype=complex)
        c_up_dag_c_down[2, 1] = -1.0
        c_down_dag_c_up = c_up_dag_c_down.T.conj()
        return s00 * n_up + s01 * c_up_dag_c_down + s10 * c_down_dag_c_up + s11 * n_down

    M_x, M_y, M_z = M_mu(sigma_x), M_mu(sigma_y), M_mu(sigma_z)
    H2 = (Jx * np.kron(M_x, Sx_n) + Jy * np.kron(M_y, Sy_n) + Jz * np.kron(M_z, Sz_n))
    perm = np.zeros((8, 8), dtype=complex)
    for n_up in (0, 1):
        for n_down in (0, 1):
            for s in (0, 1):
                perm[(n_up * 4 + s * 2 + n_down), (n_up * 4 + n_down * 2 + s)] = 1.0
    return expm(-1j * T * (perm @ H2 @ perm.T))


def build_epsilon_and_frame(L, t0):

    A = np.eye(L, k=1) + np.eye(L, k=-1)
    A[0, L - 1] = 1.0
    A[L - 1, 0] = 1.0   # closed boundary condition
    H1_sp = -(t0/2) * A
    epsilon_p, W_eig = np.linalg.eigh(H1_sp)
    return epsilon_p, W_eig


def get_free_propagator_from_spectrum(epsilon_p, W_eig, n):
    """U_TE(n) = exp(-i n T epsilon_p)"""
    if n == 0:
        return np.eye(len(epsilon_p), dtype=complex)
    phases = np.exp(-1j * n * epsilon_p)
    return (W_eig * phases) @ W_eig.conj().T


# =======================================================================================
# 8. Exact benchmark reference (UNCHANGED) -- literal Eq. (3) gate order (Kondo first, hop
#    second), zero-truncation diagnostic for run_sergio_floquet.
# =======================================================================================

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


def embed_op_dense(op, first_qubit, total_qubits):
    k = int(round(np.log2(op.shape[0])))
    left_dim = 2 ** first_qubit
    right_dim = 2 ** (total_qubits - first_qubit - k)
    return np.kron(np.eye(left_dim), np.kron(op, np.eye(right_dim)))


def run_exact_benchmark_reference(N, no_floquet_steps, theta, theta_K, theta_z):
 
    orb_up, orb_down = build_initial_orbitals(N)
    psi = build_initial_state(N, orb_up, orb_down, orb_up.shape[1], orb_down.shape[1])
    psi_mps = state_mps(psi, N)

    total_qubits = 2 * N + 1
    phys_dim = 2

    X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
    hbond = skron(X, X, format='csr') + skron(Y, Y, format='csr')

    def embed_two_qubit_sparse(op4, left_site, total_qubits):
        left_dim = 2 ** left_site
        right_dim = 2 ** (total_qubits - left_site - 2)
        return skron(skron(sident(left_dim, format='csr'), op4, format='csr'),
                     sident(right_dim, format='csr'), format='csr')

    H_kin_full = sparse.csr_matrix((2 ** total_qubits, 2 ** total_qubits), dtype=complex)
    for left in range(0, N - 1):
        H_kin_full = H_kin_full + embed_two_qubit_sparse(hbond, left, total_qubits)
    for left in range(N + 1, 2 * N):
        H_kin_full = H_kin_full + embed_two_qubit_sparse(hbond, left, total_qubits)
    H_kin_full = H_kin_full.toarray()

    U_kin_dense = expm(1j * (theta / 2) * H_kin_full)

    UK_matrix = build_UK(theta_K, theta_z)
    U_K_dense = embed_op_dense(UK_matrix, N - 1, total_qubits)

    mpo_bond_dim = phys_dim ** total_qubits
    U_kin_mpo = MPO.from_matrix(U_kin_dense, phys_dim=phys_dim, nqbits=total_qubits,
                                 max_bond_dim=mpo_bond_dim, trunc_tol=1e-12)
    U_K_mpo = MPO.from_matrix(U_K_dense, phys_dim=phys_dim, nqbits=total_qubits,
                               max_bond_dim=mpo_bond_dim, trunc_tol=1e-12)

    impurity_mag = [2 * psi_mps.measure_observable(Sz_op, [N])]
    for step in range(no_floquet_steps):
        psi_mps = U_kin_mpo @ psi_mps
        psi_mps = U_K_mpo @ psi_mps
        impurity_mag.append(2 * psi_mps.measure_observable(Sz_op, [N]))

    return np.asarray(impurity_mag, dtype=float)


# =======================================================================================

# =======================================================================================
def run_sergio_floquet(N, no_floquet_steps, t0, Jx, Jy, Jz, h, T,
                        classify_tol=1e-10, verbose=False):
    orb_up, orb_down = build_initial_orbitals(N)
    psi = build_initial_state(N, orb_up, orb_down, orb_up.shape[1], orb_down.shape[1])
    psi_mps = state_mps(psi, N)

    imp_q = N
    up_bath = list(range(N - 1, -1, -1))
    down_bath = list(range(N + 1, 2 * N + 1))
    loc_map_up = {q: i for i, q in enumerate(up_bath)}
    loc_map_dn = {q: i for i, q in enumerate(down_bath)}

    U_FT_up = build_initial_orbital_frame(N, reverse=True)
    U_FT_dn = build_initial_orbital_frame(N, reverse=False)
    psi_mps = apply_exact_gaussian_rotation(psi_mps, U_FT_up.conj().T, up_bath, psi_mps.nqbits)
    psi_mps = apply_exact_gaussian_rotation(psi_mps, U_FT_dn.conj().T, down_bath, psi_mps.nqbits)

    # --- Store Q_sigma(0) and V_sigma(0) = U_FT Q_sigma(0), per Eqs. (46a)/(46b) and the
    #     "Initialization on the computer" box ("Store L x L matrices Q_sigma(0) and
    #     V_sigma(0) = U_FT Q_sigma(0)"). v_{b,sigma}(0) = 1/sqrt(L) for both spins
    #     (Sec. VI.C); at half filling M_0^sigma = 0 (no active orbitals yet), so the first
    #     N/2 bath modes are "full" and the remaining N/2 are "empty".
    v0 = np.ones(N, dtype=complex) / np.sqrt(N)
    n_full0 = N // 2
    f0, a0, e0 = list(range(n_full0)), [], list(range(n_full0, N))

    psi_mps, c_up_q0, Q_up_0 = apply_qfull_qempty_qfew_one_spin(psi_mps, v0[f0], v0[a0], v0[e0], up_bath)
    psi_mps, c_dn_q0, Q_dn_0 = apply_qfull_qempty_qfew_one_spin(psi_mps, v0[f0], v0[a0], v0[e0], down_bath)

    U_up = U_FT_up @ Q_up_0     # V_sigma(0) = U_FT Q_sigma(0)
    U_dn = U_FT_dn @ Q_dn_0

    evals_up0 = np.linalg.eigvalsh(build_1rdm_bath(psi_mps, up_bath, imp_q))[::-1]
    evals_dn0 = np.linalg.eigvalsh(build_1rdm_bath(psi_mps, down_bath, imp_q))[::-1]

    impurity_mag = [2 * psi_mps.measure_observable(Sz_op, [imp_q])]

    # Store epsilon_p and consequently U_TE, per the "Initialization on the computer" box.
    epsilon_p, W_eig = build_epsilon_and_frame(N, t0)

    for n in range(no_floquet_steps):
        imp_q = N
        up_bath = list(range(N - 1, -1, -1))
        down_bath = list(range(N + 1, 2 * N + 1))

        U_TE = get_free_propagator_from_spectrum(epsilon_p, W_eig, n)
        v_up = (U_TE @ U_up)[0, :]
        v_dn = (U_TE @ U_dn)[0, :]

        f_up, a_up, e_up = classify_orb(evals_up0, tol=classify_tol)
        f_dn, a_dn, e_dn = classify_orb(evals_dn0, tol=classify_tol)

        psi_mps, c_up_q, Q_up = apply_qfull_qempty_qfew_one_spin(psi_mps, v_up[f_up], v_up[a_up], v_up[e_up], up_bath)
        psi_mps, c_dn_q, Q_dn = apply_qfull_qempty_qfew_one_spin(psi_mps, v_dn[f_dn], v_dn[a_dn], v_dn[e_dn], down_bath)
        U_up = U_up @ Q_up
        U_dn = U_dn @ Q_dn

        psi_mps, c_up_q, perm_up = move_mode_adjacent_to_impurity(psi_mps, c_up_q, up_bath)
        psi_mps, c_dn_q, perm_dn = move_mode_adjacent_to_impurity(psi_mps, c_dn_q, down_bath)
        U_up = apply_column_swaps(U_up, perm_up, loc_map_up)
        U_dn = apply_column_swaps(U_dn, perm_dn, loc_map_dn)

        gate = build_kondo_gate(Jx, Jy, Jz, h, T, n)
        psi_mps.apply(gate, [c_up_q, imp_q, c_dn_q])

        psi_mps, U_up, evals_up0 = advance_natural_orbital_frame(psi_mps, U_up, up_bath, imp_q)
        psi_mps, U_dn, evals_dn0 = advance_natural_orbital_frame(psi_mps, U_dn, down_bath, imp_q)

        impurity_mag.append(2 * psi_mps.measure_observable(Sz_op, [imp_q]))
        if verbose:
            print(f"step {n}: Sz_imp={impurity_mag[-1]:.6f}")

    return np.asarray(impurity_mag, dtype=float)


# =======================================================================================
# Run + plot
# =======================================================================================
if __name__ == '__main__':
    N = 6
    steps = 100
    t0 = np.pi / 3
    Jx = Jy = Jz = np.pi / 4
    h, T = 0.0, 1.0

    theta = np.pi / 3
    theta_K = np.pi / 4
    theta_z = 0.5 * np.sqrt(2) * (np.sqrt(2) - 1) * np.sin(theta)

    mag_sergio = run_sergio_floquet(N, steps, t0, Jx, Jy, Jz, h, T, classify_tol=1e-14)
    mag_exact = run_exact_benchmark_reference(N, steps, theta, theta_K, theta_z)

    print("step  SERGIO(tol=1e-14)   exact_benchmark(Script1)   |diff (meaningless, diff models)|")
    for n in range(steps + 1):
        print(f"{n:4d}  {mag_sergio[n]:+.15f}  {mag_exact[n]:+.15f}  {abs(mag_sergio[n]-mag_exact[n]):.2e}")

    n_axis = np.arange(steps + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(n_axis, mag_exact, '-', lw=2, label='Script 1 exact MPO (theta, theta_K, theta_z)')
    plt.plot(n_axis, mag_sergio, '--', lw=1.5, label='SERGIO (t0, Jx, Jy, Jz, h)')
    plt.xlabel('Floquet step $n$')
    plt.ylabel(r'$\langle S^z_{\rm imp}(n)\rangle$')
    plt.title(f'Impurity magnetization: two different Kondo parameterizations (N={N})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('impurity_magnetization.png', dpi=150)
    plt.show()