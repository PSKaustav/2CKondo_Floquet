"""Full-chain SERGIO evolution for the one-channel Kondo model.

The physical MPS layout is

    [up bath, far -> near] -- impurity -- [down bath, near -> far].

Unlike the compressed pseudo-algorithm in Sec. VI of ``SERGIO_notes.pdf``,
this implementation never inserts or removes tensors.  Filled, active, and
empty natural orbitals are represented by labels on a fixed ``2 * N + 1``
site MPS.  The effective mode ``c_bar[0, sigma]`` is placed at site ``N - 1``
for spin up and site ``N + 1`` for spin down, so the Kondo gate is always a
local three-site gate.

The module deliberately imports ``helper_functions_v2`` lazily.  This keeps
the dense N=6 reference calculation usable in environments that do not have
the private ``qlimb`` package, while the MPS path still uses the supplied MPS,
MPO, Gate, and local-gate helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from types import ModuleType
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass
class SergioResult:
    """Output of :func:`sergio_step_floquet`."""

    psi_mps: object
    U_up: Array
    U_down: Array
    V_up: Array
    V_down: Array
    M_up: int
    M_down: int
    magnetization: Array
    active_orbitals: Array
    exact_magnetization: Array | None = None


def _load_helpers(helper_module: ModuleType | None = None) -> ModuleType:
    if helper_module is not None:
        return helper_module
    try:
        import helper_functions_v2 as helpers
    except ImportError as exc:  # pragma: no cover - depends on private qlimb
        raise ImportError(
            "Place sergio_full_mps.py next to helper_functions_v2.py (or pass "
            "helper_module=...) and use the environment containing qlimb and "
            "openfermion."
        ) from exc
    return helpers


def _validate_boundary(boundary: str) -> str:
    boundary = str(boundary).lower()
    if boundary not in {"obc", "pbc"}:
        raise ValueError("boundary must be 'obc' or 'pbc'")
    return boundary


def _validate_floquet_order(floquet_order: str) -> str:
    aliases = {
        "interaction_then_free": "interaction_then_free",
        "kondo_then_kinetic": "interaction_then_free",
        "free_then_interaction": "free_then_interaction",
        "kinetic_then_kondo": "free_then_interaction",
    }
    key = str(floquet_order).lower()
    if key not in aliases:
        raise ValueError(
            "floquet_order must be 'interaction_then_free' or "
            "'free_then_interaction'"
        )
    return aliases[key]


def build_h1(N: int, t0: float = 1.0, boundary: str = "obc") -> Array:
    """Return H1 for open (``obc``) or periodic (``pbc``) boundaries."""

    if N <= 0:
        raise ValueError("N must be positive")
    boundary = _validate_boundary(boundary)
    H1 = np.zeros((N, N), dtype=complex)
    if N > 1:
        links = np.arange(N - 1)
        H1[links, links + 1] = -0.5 * t0
        H1[links + 1, links] = -0.5 * t0
        if boundary == "pbc":
            # Addition is intentional: for N=2 the two directed ring bonds
            # connect the same pair and contribute twice.
            H1[0, N - 1] += -0.5 * t0
            H1[N - 1, 0] += -0.5 * t0
    return H1


def build_h1_open(N: int, t0: float = 1.0) -> Array:
    """Backward-compatible alias for :func:`build_h1` with OBC."""

    return build_h1(N, t0=t0, boundary="obc")


def diagonalize_h1(H1: Array, T: float = 1.0) -> tuple[Array, Array, Array]:
    """Diagonalize H1 and return energies, eigenvectors, and exp(-i H1 T)."""

    H1 = np.asarray(H1, dtype=complex)
    if H1.ndim != 2 or H1.shape[0] != H1.shape[1]:
        raise ValueError("H1 must be square")
    if not np.allclose(H1, H1.conj().T, atol=1e-12):
        raise ValueError("H1 must be Hermitian")
    energies, orbitals = np.linalg.eigh(H1)
    free_step = (orbitals * np.exp(-1j * energies * T)) @ orbitals.conj().T
    return energies, orbitals, free_step


def build_fermi_sea_orbitals(
    N: int,
    n_filled: int | None = None,
    t0: float = 1.0,
    boundary: str = "obc",
) -> tuple[Array, Array, Array]:
    """Return occupied H1 eigenmodes, the full eigenframe, and energies.

    Rows are real-space sites ordered by distance from the impurity
    (site 0 is adjacent to it).  Columns of ``occupied`` are the lowest-energy
    one-particle orbitals.  ``boundary`` must be ``"obc"`` or ``"pbc"``.
    """

    boundary = _validate_boundary(boundary)
    if n_filled is None:
        if N % 2:
            raise ValueError("Half filling requires even N")
        n_filled = N // 2
    if not 0 <= n_filled <= N:
        raise ValueError("n_filled must lie between 0 and N")
    energies, frame = np.linalg.eigh(build_h1(N, t0=t0, boundary=boundary))
    order = np.argsort(energies)
    energies = energies[order]
    frame = frame[:, order]
    return frame[:, :n_filled], frame, energies


def build_open_chain_fermi_sea_orbitals(
    N: int,
    n_filled: int | None = None,
    t0: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Backward-compatible OBC wrapper for :func:`build_fermi_sea_orbitals`."""

    return build_fermi_sea_orbitals(
        N, n_filled=n_filled, t0=t0, boundary="obc"
    )


def _interleave(first: Sequence[int], second: Sequence[int]) -> list[int]:
    out: list[int] = []
    for i in range(max(len(first), len(second))):
        if i < len(first):
            out.append(int(first[i]))
        if i < len(second):
            out.append(int(second[i]))
    return out


def _classify_values(values: Array, tol: float) -> tuple[list[int], list[int], list[int]]:
    filled = [i for i, x in enumerate(values) if x >= 1.0 - tol]
    empty = [i for i, x in enumerate(values) if x <= tol]
    active = [i for i, x in enumerate(values) if tol < x < 1.0 - tol]
    return filled, active, empty


def _physical_natural_order(values: Array, spin: str, tol: float) -> list[int]:
    """Order natural modes in the fixed mirrored layout from the supplied sketch."""

    values = np.asarray(values, dtype=float)
    filled, active, empty = _classify_values(values, tol)
    active.sort(key=lambda i: values[i], reverse=True)
    filled.sort(key=lambda i: values[i], reverse=True)
    empty.sort(key=lambda i: values[i])

    # From the impurity outwards: active, e_0, f_0, e_1, f_1, ... .
    near_to_far = active + _interleave(empty, filled)
    if spin == "up":
        return near_to_far[::-1]
    if spin == "down":
        return near_to_far
    raise ValueError("spin must be 'up' or 'down'")


def _labels_for_values(values: Array, spin: str, tol: float) -> list[str]:
    """Create labels with index zero nearest the impurity in every category."""

    labels = [""] * len(values)
    positions = range(len(values) - 1, -1, -1) if spin == "up" else range(len(values))
    counts = {"filled": 0, "active": 0, "empty": 0}
    for pos in positions:
        value = float(values[pos])
        if value >= 1.0 - tol:
            kind = "filled"
        elif value <= tol:
            kind = "empty"
        else:
            kind = "active"
        labels[pos] = f"{kind}_{spin}_{counts[kind]}"
        counts[kind] += 1
    return labels


def _unitary_with_target_column(v_star: Array, target: int) -> Array:
    """Build Q such that v_star @ Q is the row vector e_target."""

    v_star = np.asarray(v_star, dtype=complex).reshape(-1)
    N = len(v_star)
    norm = np.linalg.norm(v_star)
    if norm == 0.0:
        raise ValueError("the interaction-picture site-0 mode has zero norm")
    v_star = v_star / norm
    target_vector = v_star.conj()

    # Complete target_vector to an orthonormal basis.  Removing the standard
    # basis vector with its largest component guarantees a full-rank seed.
    pivot = int(np.argmax(np.abs(target_vector)))
    seed = [target_vector]
    for j in range(N):
        if j != pivot:
            e_j = np.zeros(N, dtype=complex)
            e_j[j] = 1.0
            seed.append(e_j)
    complement, _ = np.linalg.qr(np.column_stack(seed))
    complement[:, 0] = target_vector

    Q = np.empty((N, N), dtype=complex)
    other_columns = [j for j in range(N) if j != target]
    Q[:, target] = target_vector
    Q[:, other_columns] = complement[:, 1:]
    if not np.allclose(Q.conj().T @ Q, np.eye(N), atol=2e-12):
        raise RuntimeError("failed to complete the effective mode to a unitary Q")
    if not np.allclose(v_star @ Q, np.eye(N)[target], atol=2e-12):
        raise RuntimeError("Q does not route c_bar[0,sigma] to the target site")
    return Q


def _build_structured_q(
    v_star: Array,
    occupations: Array,
    spin: str,
    tol: float,
    target: int,
) -> Array:
    """Construct Q = Q_bar Q_few with the structure of Eqs. (17)-(33).

    ``Q_bar`` mixes filled orbitals only with filled orbitals and empty only
    with empty, singling out ``f_0`` and ``e_0`` next to the active window.
    ``Q_few`` then mixes just ``f_0``, the active modes, and ``e_0``.  The
    unspecified orthogonal complements are completed deterministically.
    """

    v_star = np.asarray(v_star, dtype=complex).reshape(-1)
    occupations = np.asarray(occupations, dtype=float).reshape(-1)
    if v_star.shape != occupations.shape:
        raise ValueError("v_star and occupations must have the same length")
    N = len(v_star)
    filled, active, empty = _classify_values(occupations, tol)
    Q_bar = np.eye(N, dtype=complex)
    selected: list[int] = list(active)

    for group in (filled, empty):
        if not group:
            continue
        group = sorted(group)
        # The category's site nearest the impurity is f_0 or e_0.
        slot = max(group) if spin == "up" else min(group)
        norm = np.linalg.norm(v_star[group])
        if norm > 1e-14:
            local_target = group.index(slot)
            block = _unitary_with_target_column(v_star[group] / norm, local_target)
            Q_bar[np.ix_(group, group)] = block
            selected.append(slot)

    reduced = v_star @ Q_bar
    selected = sorted(set(selected))
    if target not in selected:
        # This only occurs in a limiting case where one category has exactly
        # zero coupling weight.  Including the boundary identity column keeps
        # the prescribed c_eff location without changing the state.
        selected.append(target)
        selected.sort()
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise RuntimeError("the f_0-active-e_0 Q_few window is not contiguous")

    Q_few = np.eye(N, dtype=complex)
    local_target = selected.index(target)
    block = _unitary_with_target_column(reduced[selected], local_target)
    Q_few[np.ix_(selected, selected)] = block
    Q = Q_bar @ Q_few
    normalized = v_star / np.linalg.norm(v_star)
    if not np.allclose(normalized @ Q, np.eye(N)[target], atol=3e-12):
        raise RuntimeError("structured Q failed to isolate the effective mode")
    return Q


def _adjacent_unitary_decomposition(W: Array) -> tuple[Array, list[tuple[int, int, Array]]]:
    """Decompose W into phases and adjacent two-mode state unitaries.

    If ``rotations`` are generated in elimination order, then

        W = G_1^dag ... G_m^dag D.

    Consequently an MPS is acted on by D first and by the stored G matrices in
    reverse order, using ``G.conj().T`` as each two-mode state unitary.
    """

    work = np.asarray(W, dtype=complex).copy()
    N = work.shape[0]
    if work.shape != (N, N):
        raise ValueError("W must be square")
    if not np.allclose(work.conj().T @ work, np.eye(N), atol=2e-10):
        raise ValueError("W must be unitary")

    rotations: list[tuple[int, int, Array]] = []
    for col in range(N - 1):
        for row in range(N - 1, col, -1):
            a = work[row - 1, col]
            b = work[row, col]
            radius = np.hypot(abs(a), abs(b))
            if radius < 1e-15:
                continue
            if abs(a) < 1e-15:
                c = 0.0
                s = np.conj(b) / abs(b)
            else:
                c = abs(a) / radius
                s = (a / abs(a)) * np.conj(b) / radius
            G = np.array([[c, s], [-np.conj(s), c]], dtype=complex)
            work[[row - 1, row], :] = G @ work[[row - 1, row], :]
            rotations.append((row - 1, row, G))

    off_diagonal = work - np.diag(np.diag(work))
    if np.linalg.norm(off_diagonal) > 5e-10:
        raise RuntimeError("adjacent Givens elimination did not diagonalize W")
    phases = np.diag(work)
    phases = np.divide(phases, np.abs(phases), out=np.ones_like(phases), where=np.abs(phases) > 1e-15)
    return phases, rotations


def _two_mode_state_gate(U2: Array) -> Array:
    """Lift a 2x2 one-particle state unitary to the two-mode Fock space."""

    U2 = np.asarray(U2, dtype=complex)
    gate = np.zeros((4, 4), dtype=complex)
    gate[0, 0] = 1.0
    # Computational basis is |00>, |01>, |10>, |11>; the mode-ordered
    # one-particle basis is |10>, |01>.
    gate[np.ix_([2, 1], [2, 1])] = U2
    gate[3, 3] = np.linalg.det(U2)
    return gate


def _apply_gaussian_mps(psi_mps: object, W: Array, sites: Sequence[int], helpers: ModuleType) -> object:
    """Apply the second quantization of W to consecutive MPS mode sites."""

    sites = list(map(int, sites))
    if sites != list(range(sites[0], sites[0] + len(sites))):
        raise ValueError("Gaussian circuit sites must be consecutive and increasing")
    phases, rotations = _adjacent_unitary_decomposition(W)
    out = psi_mps
    for local, phase in enumerate(phases):
        phase_gate = np.diag([1.0, phase]).astype(complex)
        out = helpers.Gate(matrix=phase_gate, indices=(sites[local],)) @ out
    for left, right, G in reversed(rotations):
        out = helpers.apply_two_qubit_gate(
            out,
            _two_mode_state_gate(G.conj().T),
            sites[left],
            sites[right],
        )
    return out


def build_1rdm_bath_fixed(
    psi_mps: object,
    bath_qubits: Sequence[int],
    imp_qubit: int,
    helper_module: ModuleType | None = None,
) -> Array:
    """Corrected version of ``helper_functions_v2.build_1rdm_bath``.

    For the helper convention ``n=(I-Z)/2``, the Jordan-Wigner string in
    ``c_i^dag c_j`` is ``Z`` on intervening fermion modes.  The supplied helper
    uses ``-Z`` at every intervening site, which changes off-diagonal elements
    by a separation-dependent sign.  The impurity is a spin, not a fermionic
    mode, and is therefore excluded from the string.
    """

    helpers = _load_helpers(helper_module)
    bath_qubits = list(map(int, bath_qubits))
    total = int(psi_mps.nqbits)
    C = np.zeros((len(bath_qubits), len(bath_qubits)), dtype=complex)
    number = (helpers.I2 - helpers.Z_OP) / 2.0

    for a, i in enumerate(bath_qubits):
        for b in range(a, len(bath_qubits)):
            j = bath_qubits[b]
            left, right = min(i, j), max(i, j)
            tensors = []
            for site in range(total):
                if i == j:
                    op = number if site == i else helpers.I2
                elif site == imp_qubit:
                    op = helpers.I2
                elif site == left:
                    op = helpers.SM_OP if i < j else helpers.SP_OP
                elif site == right:
                    op = helpers.SP_OP if i < j else helpers.SM_OP
                elif left < site < right:
                    op = helpers.Z_OP
                else:
                    op = helpers.I2
                tensors.append(op.reshape(1, 2, 2, 1))
            mpo = helpers.MPO(nqbits=total, phys_dim=2, tensors=tensors)
            value = psi_mps @ (mpo @ psi_mps)
            C[a, b] = value
            if a != b:
                C[b, a] = np.conj(value)
    return 0.5 * (C + C.conj().T)


def _natural_frame_advance_mps(
    psi_mps: object,
    V_sigma_n: Array,
    bath_sites: Sequence[int],
    imp_site: int,
    spin: str,
    tol: float,
    helpers: ModuleType,
) -> tuple[object, Array, Array, int]:
    C = build_1rdm_bath_fixed(psi_mps, bath_sites, imp_site, helpers)
    raw_values, raw_vectors = np.linalg.eigh(C)
    order = _physical_natural_order(raw_values, spin, tol)
    values = np.clip(raw_values[order].real, 0.0, 1.0)
    eigenvectors = raw_vectors[:, order]

    # Eq. (59): u^T C u* = diag(lambda), hence u = eigenvectors*.
    # State coordinates transform with u^dag = eigenvectors^T.
    psi_next = _apply_gaussian_mps(psi_mps, eigenvectors.T, bath_sites, helpers)
    U_next = V_sigma_n @ eigenvectors.conj()
    _, active, _ = _classify_values(values, tol)
    return psi_next, U_next, values, len(active)


tol_val = 1e-10  # try different values of tol to see if the results change significantly


def sergio_step(
    U_up_n: Array,
    U_down_n: Array,
    V_up_n: Array,
    V_down_n: Array,
    Jk: float,
    Jz: float,
    M_up_n: int,
    M_down_n: int,
    psi_mps_n: object,
    *,
    n: int,
    free_step: Array,
    T: float = 1.0,
    tol: float = tol_val,
    helper_module: ModuleType | None = None,
) -> tuple[Array, Array, Array, Array, int, int, object]:
    """Advance the fixed-size MPS from SERGIO step n to n+1.

    Parameters named ``U_*_n``, ``V_*_n``, ``M_*_n``, and ``psi_mps_n`` are
    returned in the same order for step n+1.  ``free_step`` is the logical
    real-space one-particle matrix ``exp(-i H1 T)``.  It and ``n`` are required
    by Eq. (64) to construct ``V_sigma(n+1)``; they cannot be inferred from the
    listed state variables alone.
    """

    helpers = _load_helpers(helper_module)
    U_up_n = np.asarray(U_up_n, dtype=complex)
    U_down_n = np.asarray(U_down_n, dtype=complex)
    V_up_n = np.asarray(V_up_n, dtype=complex)
    V_down_n = np.asarray(V_down_n, dtype=complex)
    free_step = np.asarray(free_step, dtype=complex)
    N = U_up_n.shape[0]
    expected_shape = (N, N)
    for name, matrix in (
        ("U_up_n", U_up_n),
        ("U_down_n", U_down_n),
        ("V_up_n", V_up_n),
        ("V_down_n", V_down_n),
        ("free_step", free_step),
    ):
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if psi_mps_n.nqbits != 2 * N + 1:
        raise ValueError("the full-chain MPS must always have 2*N+1 sites")
    if n < 0:
        raise ValueError("n must be non-negative")

    tags = list(getattr(psi_mps_n, "tags", []))
    if tags:
        tagged_up = sum(str(x).startswith("active_up_") for x in tags[:N])
        tagged_down = sum(str(x).startswith("active_down_") for x in tags[N + 1 :])
        if (tagged_up, tagged_down) != (M_up_n, M_down_n):
            raise ValueError(
                "M_sigma(n) disagrees with the fixed-MPS orbital labels: "
                f"labels={(tagged_up, tagged_down)}, arguments={(M_up_n, M_down_n)}"
            )

    up_sites = list(range(N))
    down_sites = list(range(N + 1, 2 * N + 1))
    imp_site = N

    # Q_sigma(n) = U_sigma(n)^dag V_sigma(n).  Acting with Q_sigma^dag
    # changes state coordinates from the natural to the fewest-body basis.
    Q_up_n = U_up_n.conj().T @ V_up_n
    Q_down_n = U_down_n.conj().T @ V_down_n
    psi_few = _apply_gaussian_mps(psi_mps_n, Q_up_n.conj().T, up_sites, helpers)
    psi_few = _apply_gaussian_mps(psi_few, Q_down_n.conj().T, down_sites, helpers)

    # Eq. (14) of Sarma-Koenig, permuted to the supplied physical ordering
    # [up, impurity, down].  The gate is time independent because h=0 here.
    interaction = build_kondo_gate_eq14(Jk, Jz, T)
    psi_after_H2 = helpers.apply_3_qubit_gate_custom(
        psi_few, interaction, N - 1, N, N + 1
    )

    psi_next, U_up_next, occ_up, M_up_next = _natural_frame_advance_mps(
        psi_after_H2, V_up_n, up_sites, imp_site, "up", tol, helpers
    )
    psi_next, U_down_next, occ_down, M_down_next = _natural_frame_advance_mps(
        psi_next, V_down_n, down_sites, imp_site, "down", tol, helpers
    )

    U_TE_next = np.linalg.matrix_power(free_step, n + 1)
    v_up_star_next = (U_TE_next @ U_up_next)[0, :]
    v_down_star_next = (U_TE_next @ U_down_next)[0, :]
    Q_up_next = _build_structured_q(v_up_star_next, occ_up, "up", tol, N - 1)
    Q_down_next = _build_structured_q(v_down_star_next, occ_down, "down", tol, 0)
    V_up_next = U_up_next @ Q_up_next
    V_down_next = U_down_next @ Q_down_next

    if hasattr(psi_next, "tags"):
        psi_next.tags = (
            _labels_for_values(occ_up, "up", tol)
            + ["impurity"]
            + _labels_for_values(occ_down, "down", tol)
        )
    if psi_next.nqbits != 2 * N + 1:
        raise RuntimeError("SERGIO step changed the fixed MPS length")

    return (
        U_up_next,
        U_down_next,
        V_up_next,
        V_down_next,
        M_up_next,
        M_down_next,
        psi_next,
    )


def initialize_sergio_mps(
    N: int,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    bond_dim: float = np.inf,
    tol: float = tol_val,
    helper_module: ModuleType | None = None,
) -> tuple[Array, Array, Array, Array, int, int, object, Array]:
    """Initialize the full MPS in the step-0 natural-orbital Fermi sea."""

    if N % 2:
        raise ValueError("N must be even for the requested half-filled initialization")
    helpers = _load_helpers(helper_module)
    boundary = _validate_boundary(boundary)
    H1 = build_h1(N, t0=t0, boundary=boundary)
    energies, eigenvectors, free_step = diagonalize_h1(H1, T=T)
    occupations_raw = (energies < 0.0).astype(float)
    if int(occupations_raw.sum()) != N // 2:
        # This also gives a deterministic convention if a finite system has a
        # one-particle level exactly at the Fermi energy.
        occupations_raw[:] = 0.0
        occupations_raw[np.argsort(energies)[: N // 2]] = 1.0

    order_up = _physical_natural_order(occupations_raw, "up", tol)
    order_down = _physical_natural_order(occupations_raw, "down", tol)
    U_up = eigenvectors[:, order_up]
    U_down = eigenvectors[:, order_down]
    occ_up = occupations_raw[order_up]
    occ_down = occupations_raw[order_down]

    # Paper [1] convention: |1>_imp is spin down.
    bits = np.concatenate((occ_up.astype(int), [1], occ_down.astype(int)))
    state = np.zeros(2 ** (2 * N + 1), dtype=complex)
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    state[index] = 1.0
    psi_mps = helpers.state_mps(state, N, bond_dim=bond_dim)
    psi_mps.tags = (
        _labels_for_values(occ_up, "up", tol)
        + ["impurity"]
        + _labels_for_values(occ_down, "down", tol)
    )

    Q_up = _build_structured_q(U_up[0, :], occ_up, "up", tol, N - 1)
    Q_down = _build_structured_q(U_down[0, :], occ_down, "down", tol, 0)
    V_up = U_up @ Q_up
    V_down = U_down @ Q_down
    return U_up, U_down, V_up, V_down, 0, 0, psi_mps, free_step


def sergio_step_floquet(
    N: int,
    no_floquet_steps: int,
    Jk: float,
    Jz: float,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    tol: float = tol_val,
    bond_dim: float = np.inf,
    validate_exact: bool = True,
    helper_module: ModuleType | None = None,
) -> SergioResult:
    """Iterate :func:`sergio_step` and optionally run the dense N=6 benchmark."""

    if no_floquet_steps < 0:
        raise ValueError("no_floquet_steps must be non-negative")
    boundary = _validate_boundary(boundary)
    helpers = _load_helpers(helper_module)
    U_up, U_down, V_up, V_down, M_up, M_down, psi, free_step = initialize_sergio_mps(
        N,
        T=T,
        t0=t0,
        boundary=boundary,
        bond_dim=bond_dim,
        tol=tol,
        helper_module=helpers,
    )

    impurity_z = np.diag([1.0, -1.0]).astype(complex)
    magnetization = [float(np.real(psi.measure_observable(impurity_z, [N])))]
    active = [(M_up, M_down)]
    for n in range(no_floquet_steps):
        U_up, U_down, V_up, V_down, M_up, M_down, psi = sergio_step(
            U_up,
            U_down,
            V_up,
            V_down,
            Jk,
            Jz,
            M_up,
            M_down,
            psi,
            n=n,
            free_step=free_step,
            T=T,
            tol=tol,
            helper_module=helpers,
        )
        magnetization.append(float(np.real(psi.measure_observable(impurity_z, [N]))))
        active.append((M_up, M_down))

    exact = None
    if validate_exact:
        if N > 6:
            raise ValueError("dense validation is intentionally limited to N <= 6")
        exact = direct_floquet_dense(
            N,
            no_floquet_steps,
            Jk,
            Jz,
            T=T,
            t0=t0,
            boundary=boundary,
        )
        if not np.allclose(magnetization, exact, atol=2e-8, rtol=2e-8):
            error = float(np.max(np.abs(np.asarray(magnetization) - exact)))
            raise AssertionError(f"SERGIO/direct magnetization mismatch: max error {error:.3e}")

    return SergioResult(
        psi_mps=psi,
        U_up=U_up,
        U_down=U_down,
        V_up=V_up,
        V_down=V_down,
        M_up=M_up,
        M_down=M_down,
        magnetization=np.asarray(magnetization),
        active_orbitals=np.asarray(active, dtype=int),
        exact_magnetization=exact,
    )


# ---------------------------------------------------------------------------
# Dense reference implementation (no qlimb/openfermion dependency)
# ---------------------------------------------------------------------------


def _apply_dense_gate(state: Array, gate: Array, sites: Sequence[int], total: int) -> Array:
    sites = list(map(int, sites))
    rest = [i for i in range(total) if i not in sites]
    permutation = sites + rest
    inverse = np.argsort(permutation)
    tensor = state.reshape([2] * total).transpose(permutation)
    updated = np.asarray(gate) @ tensor.reshape(2 ** len(sites), -1)
    return updated.reshape([2] * total).transpose(inverse).reshape(-1)


def _apply_gaussian_dense(state: Array, W: Array, sites: Sequence[int], total: int) -> Array:
    phases, rotations = _adjacent_unitary_decomposition(W)
    out = state
    sites = list(sites)
    for local, phase in enumerate(phases):
        out = _apply_dense_gate(out, np.diag([1.0, phase]), [sites[local]], total)
    for left, right, G in reversed(rotations):
        out = _apply_dense_gate(
            out,
            _two_mode_state_gate(G.conj().T),
            [sites[left], sites[right]],
            total,
        )
    return out


def _dense_1rdm(state: Array, sites: Sequence[int], total: int) -> Array:
    I = np.eye(2, dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    create = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    annihilate = create.conj().T
    number = np.diag([0.0, 1.0]).astype(complex)
    sites = list(map(int, sites))
    C = np.zeros((len(sites), len(sites)), dtype=complex)
    for a, i in enumerate(sites):
        for b in range(a, len(sites)):
            j = sites[b]
            if i == j:
                ket = _apply_dense_gate(state, number, [i], total)
            else:
                left, right = min(i, j), max(i, j)
                operators: list[tuple[int, Array]] = []
                operators.append((left, create if i < j else annihilate))
                operators.extend((k, Z) for k in range(left + 1, right))
                operators.append((right, annihilate if i < j else create))
                ket = state
                for site, operator in operators:
                    ket = _apply_dense_gate(ket, operator, [site], total)
            value = np.vdot(state, ket)
            C[a, b] = value
            if a != b:
                C[b, a] = np.conj(value)
    return 0.5 * (C + C.conj().T)


def build_kondo_gate_eq14(Jk: float, Jz: float, T: float = 1.0) -> Array:
    """Eq. (14) of [1] in physical order [up, impurity, down].

    The matrix printed in [1] uses another ordering of the central three
    qubits.  After permutation to the supplied mirror-symmetric chain, the
    exchange block occupies computational-basis indices 1 and 6.  This is the
    same convention as ``build_UK`` in the original 1CK simulation.
    """

    theta_k = Jk * T / 2.0
    theta_z = Jz * T / 2.0
    gate = np.eye(8, dtype=complex)
    p_z = np.exp(1j * theta_z / 2.0)
    m_z = np.exp(-1j * theta_z / 2.0)
    gate[1, 1] = gate[6, 6] = p_z * np.cos(theta_k)
    gate[1, 6] = gate[6, 1] = 1j * p_z * np.sin(theta_k)
    gate[3, 3] = gate[4, 4] = m_z
    return gate


def _slater_state(
    N: int,
    occupied_up_physical: Array,
    occupied_down_physical: Array,
    impurity_bit: int = 1,
) -> Array:
    n_up = occupied_up_physical.shape[1]
    n_down = occupied_down_physical.shape[1]
    state = np.zeros(2 ** (2 * N + 1), dtype=complex)
    for down_occ in combinations(range(N), n_down):
        down_rows = np.asarray(down_occ, dtype=int)
        amp_down = np.linalg.det(occupied_down_physical[down_rows, :])
        for up_occ in combinations(range(N), n_up):
            up_rows = np.asarray(up_occ, dtype=int)
            amp_up = np.linalg.det(occupied_up_physical[up_rows, :])
            bits = np.zeros(2 * N + 1, dtype=int)
            bits[up_rows] = 1
            bits[N] = impurity_bit
            bits[N + 1 + down_rows] = 1
            index = 0
            for bit in bits:
                index = (index << 1) | int(bit)
            state[index] = amp_up * amp_down
    state /= np.linalg.norm(state)
    return state


def direct_floquet_dense(
    N: int,
    no_floquet_steps: int,
    Jk: float,
    Jz: float,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    initial_boundary: str | None = None,
    floquet_order: str = "interaction_then_free",
    return_states: bool = False,
) -> Array | tuple[Array, list[Array]]:
    """Evaluate (exp(-i H1 T) exp(-i H2 T))^n |FS,down> for N <= 6."""

    if N > 6:
        raise ValueError("the dense reference is intentionally limited to N <= 6")
    boundary = _validate_boundary(boundary)
    initial_boundary = _validate_boundary(
        boundary if initial_boundary is None else initial_boundary
    )
    floquet_order = _validate_floquet_order(floquet_order)
    occupied, _, _ = build_fermi_sea_orbitals(
        N, t0=t0, boundary=initial_boundary
    )
    # Up physical sites run far -> near, while logical H1 rows run near -> far.
    state = _slater_state(N, occupied[::-1, :], occupied, impurity_bit=1)
    H1 = build_h1(N, t0=t0, boundary=boundary)
    _, _, free = diagonalize_h1(H1, T=T)
    reverse = np.eye(N)[::-1]
    free_up_physical = reverse @ free @ reverse
    total = 2 * N + 1
    up_sites = list(range(N))
    down_sites = list(range(N + 1, total))
    interaction = build_kondo_gate_eq14(Jk, Jz, T)
    sigma_z = np.diag([1.0, -1.0])

    magnetization = []
    states = []
    for step in range(no_floquet_steps + 1):
        states.append(state.copy())
        z_state = _apply_dense_gate(state, sigma_z, [N], total)
        magnetization.append(float(np.vdot(state, z_state).real))
        if step == no_floquet_steps:
            break
        if floquet_order == "interaction_then_free":
            # UF = exp(-i H1 T) exp(-i H2 T): H2 acts first.
            state = _apply_dense_gate(
                state, interaction, [N - 1, N, N + 1], total
            )
            state = _apply_gaussian_dense(
                state, free_up_physical, up_sites, total
            )
            state = _apply_gaussian_dense(state, free, down_sites, total)
        else:
            # latest.py convention: bath hopping first, then the Kondo gate.
            state = _apply_gaussian_dense(
                state, free_up_physical, up_sites, total
            )
            state = _apply_gaussian_dense(state, free, down_sites, total)
            state = _apply_dense_gate(
                state, interaction, [N - 1, N, N + 1], total
            )
    values = np.asarray(magnetization)
    return (values, states) if return_states else values


def sergio_floquet_dense(
    N: int,
    no_floquet_steps: int,
    Jk: float,
    Jz: float,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    tol: float = tol_val,
    return_states: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, list[Array]]:
    """Dependency-free dense realization of the same SERGIO basis recursion."""

    if N > 6:
        raise ValueError("the dense reference is intentionally limited to N <= 6")
    boundary = _validate_boundary(boundary)
    H1 = build_h1(N, t0=t0, boundary=boundary)
    energies, eigenvectors, free_step = diagonalize_h1(H1, T=T)
    raw_occ = np.zeros(N)
    raw_occ[np.argsort(energies)[: N // 2]] = 1.0
    order_up = _physical_natural_order(raw_occ, "up", tol)
    order_down = _physical_natural_order(raw_occ, "down", tol)
    U_up = eigenvectors[:, order_up]
    U_down = eigenvectors[:, order_down]
    occ_up = raw_occ[order_up]
    occ_down = raw_occ[order_down]
    bits = np.concatenate((occ_up.astype(int), [1], occ_down.astype(int)))
    state = np.zeros(2 ** (2 * N + 1), dtype=complex)
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    state[index] = 1.0

    Q_up = _build_structured_q(U_up[0, :], occ_up, "up", tol, N - 1)
    Q_down = _build_structured_q(U_down[0, :], occ_down, "down", tol, 0)
    V_up = U_up @ Q_up
    V_down = U_down @ Q_down
    total = 2 * N + 1
    up_sites = list(range(N))
    down_sites = list(range(N + 1, total))
    interaction = build_kondo_gate_eq14(Jk, Jz, T)
    sigma_z = np.diag([1.0, -1.0])
    magnetization = []
    active_counts = []
    physical_states = []
    reverse = np.eye(N)[::-1]

    for n in range(no_floquet_steps + 1):
        free_n = np.linalg.matrix_power(free_step, n)
        state_physical = _apply_gaussian_dense(
            state, reverse @ free_n @ U_up, up_sites, total
        )
        state_physical = _apply_gaussian_dense(
            state_physical, free_n @ U_down, down_sites, total
        )
        physical_states.append(state_physical)
        z_state = _apply_dense_gate(state, sigma_z, [N], total)
        magnetization.append(float(np.vdot(state, z_state).real))
        C_up_now = _dense_1rdm(state, up_sites, total)
        C_down_now = _dense_1rdm(state, down_sites, total)
        values_up_now = np.linalg.eigvalsh(C_up_now)
        values_down_now = np.linalg.eigvalsh(C_down_now)
        active_counts.append(
            (
                len(_classify_values(values_up_now, tol)[1]),
                len(_classify_values(values_down_now, tol)[1]),
            )
        )
        if n == no_floquet_steps:
            break

        Q_up = U_up.conj().T @ V_up
        Q_down = U_down.conj().T @ V_down
        state = _apply_gaussian_dense(state, Q_up.conj().T, up_sites, total)
        state = _apply_gaussian_dense(state, Q_down.conj().T, down_sites, total)
        state = _apply_dense_gate(state, interaction, [N - 1, N, N + 1], total)

        C_up = _dense_1rdm(state, up_sites, total)
        values, vectors = np.linalg.eigh(C_up)
        order = _physical_natural_order(values, "up", tol)
        occ_up = np.clip(values[order].real, 0.0, 1.0)
        state = _apply_gaussian_dense(state, vectors[:, order].T, up_sites, total)
        U_up = V_up @ vectors[:, order].conj()

        C_down = _dense_1rdm(state, down_sites, total)
        values, vectors = np.linalg.eigh(C_down)
        order = _physical_natural_order(values, "down", tol)
        occ_down = np.clip(values[order].real, 0.0, 1.0)
        state = _apply_gaussian_dense(state, vectors[:, order].T, down_sites, total)
        U_down = V_down @ vectors[:, order].conj()

        U_TE_next = np.linalg.matrix_power(free_step, n + 1)
        Q_up = _build_structured_q(
            (U_TE_next @ U_up)[0, :], occ_up, "up", tol, N - 1
        )
        Q_down = _build_structured_q(
            (U_TE_next @ U_down)[0, :], occ_down, "down", tol, 0
        )
        V_up = U_up @ Q_up
        V_down = U_down @ Q_down

    values = np.asarray(magnetization)
    counts = np.asarray(active_counts, dtype=int)
    return (values, counts, physical_states) if return_states else (values, counts)


def audit_fermi_sea_helpers(
    N: int = 6,
    *,
    t0: float = 1.0,
    helper_module: ModuleType | None = None,
) -> dict[str, float | bool]:
    """Numerically demonstrate why the supplied orbital helper is not OBC-correct."""

    helpers = _load_helpers(helper_module)
    supplied_up, supplied_down = helpers.build_initial_orbitals(N)
    corrected, _, _ = build_open_chain_fermi_sea_orbitals(N, t0=t0)
    H1 = build_h1_open(N, t0=t0)

    C_supplied = supplied_up @ supplied_up.conj().T
    C_corrected = corrected @ corrected.conj().T
    supplied_commutator = np.linalg.norm(H1 @ C_supplied - C_supplied @ H1)
    corrected_commutator = np.linalg.norm(H1 @ C_corrected - C_corrected @ H1)

    state = helpers.build_initial_state(
        N,
        corrected[::-1, :],
        corrected,
        corrected.shape[1],
        corrected.shape[1],
    )
    state_tensor = state.reshape(2**N, 2, 2**N)
    impurity_one_probability = float(np.sum(np.abs(state_tensor[:, 1, :]) ** 2))
    return {
        "supplied_orbitals_are_obc_eigenstate": bool(supplied_commutator < 1e-10),
        "supplied_obc_commutator_norm": float(supplied_commutator),
        "corrected_obc_commutator_norm": float(corrected_commutator),
        "corrected_state_norm": float(np.vdot(state, state).real),
        "helper_uses_paper_impurity_down_bit": bool(impurity_one_probability > 1.0 - 1e-12),
        "helper_impurity_bit_one_probability": impurity_one_probability,
        "up_down_orbitals_equal": bool(np.allclose(supplied_up, supplied_down)),
    }


def n6_dense_validation(
    no_floquet_steps: int = 8,
    Jk: float = 0.8,
    Jz: float = 0.3,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
) -> dict[str, Array | float]:
    """Run the requested N=6 direct/SERGIO comparison without private packages."""

    boundary = _validate_boundary(boundary)
    direct, direct_states = direct_floquet_dense(
        6,
        no_floquet_steps,
        Jk,
        Jz,
        T=T,
        t0=t0,
        boundary=boundary,
        return_states=True,
    )
    sergio, active, sergio_states = sergio_floquet_dense(
        6,
        no_floquet_steps,
        Jk,
        Jz,
        T=T,
        t0=t0,
        boundary=boundary,
        return_states=True,
    )
    fidelities = np.asarray(
        [abs(np.vdot(exact, reduced)) ** 2 for exact, reduced in zip(direct_states, sergio_states)]
    )
    return {
        "direct_magnetization": direct,
        "sergio_magnetization": sergio,
        "active_orbitals": active,
        "max_abs_error": float(np.max(np.abs(direct - sergio))),
        "state_fidelity": fidelities,
        "max_state_infidelity": max(0.0, float(np.max(1.0 - fidelities))),
    }


if __name__ == "__main__":
    check = n6_dense_validation()
    print("direct  :", check["direct_magnetization"])
    print("SERGIO  :", check["sergio_magnetization"])
    print("M_up/dn :", check["active_orbitals"].tolist())
    print("max err :", f"{check['max_abs_error']:.3e}")
    print("1-F     :", f"{check['max_state_infidelity']:.3e}")
