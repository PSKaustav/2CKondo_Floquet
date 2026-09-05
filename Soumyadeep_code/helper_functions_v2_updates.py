"""Minimal SERGIO helpers to append to ``helper_functions_v2.py``.

The functions in this file deliberately reuse the existing helpers in
``helper_functions_v2.py``.  When they are copied into that module, the optional
``backend`` argument can be omitted.  In ``sergio_qlimb_openfermion.py`` the
argument is used to point at either the real helper module or the test-only mock
backend.

Two corrections relative to the supplied helper file are included here:

* the Jordan--Wigner string in the bath 1-RDM is a product of ``Z_OP`` (not
  ``-Z_OP``); and
* if ``evecs`` are the columns returned by ``numpy.linalg.eigh(C)``, the
  first-quantized frame update is ``U_new = V @ evecs.conj()``.  This follows
  from ``u.T @ C @ u.conj() = diag`` with ``u = evecs.conj()``.
* the supplied Q helpers accumulate rotations for left action on a coefficient
  column, whereas Eq. (31) requires right action on the row ``v_star``.  They
  also omit the final phase returned by a complex Givens reduction.  The
  phase-complete reconstruction below fixes both conventions.

Only ``Q_few^dagger`` is returned as a quantum circuit.  ``Q_full`` and
``Q_empty`` participate in the classical matrix ``Q_total`` but are never
applied to the MPS.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def build_1rdm_bath_jw(
    psi_mps: Any,
    bath_qubits: Sequence[int],
    imp_qubit: int,
    *,
    backend: Optional[Any] = None,
) -> np.ndarray:
    """Return ``C[i,j] = <c_i^dagger c_j>`` on the ordered bath sites.

    This is a corrected version of ``build_1rdm_bath`` in the supplied helper
    file.  It uses the same qlimb ``MPO`` interface, but the intermediate
    Jordan--Wigner factors are ``Z`` rather than ``-Z``.
    """

    if backend is None:
        backend = sys.modules[__name__]

    bath_qubits = [int(q) for q in bath_qubits]
    nqbits = int(psi_mps.nqbits)
    nbath = len(bath_qubits)
    C = np.zeros((nbath, nbath), dtype=complex)

    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    annihilate = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    create = annihilate.conj().T
    number = create @ annihilate

    for i, qi in enumerate(bath_qubits):
        for j in range(i, nbath):
            qj = bath_qubits[j]
            local_ops = []
            if i == j:
                local_ops = [number if q == qi else I2 for q in range(nqbits)]
            else:
                lo, hi = sorted((qi, qj))
                for q in range(nqbits):
                    if q == int(imp_qubit):
                        op = I2
                    elif q == qi:
                        op = create
                    elif q == qj:
                        op = annihilate
                    elif lo < q < hi:
                        op = Z
                    else:
                        op = I2
                    local_ops.append(op)

            tensors = [op.reshape(1, 2, 2, 1) for op in local_ops]
            mpo = backend.MPO(nqbits=nqbits, phys_dim=2, tensors=tensors)
            value = psi_mps @ (mpo @ psi_mps)
            C[i, j] = value
            C[j, i] = np.conjugate(value)

    return C


def build_q_block_openfermion(
    vec: np.ndarray,
    *,
    target_pos: str,
    physical_qubits: Optional[Sequence[int]] = None,
    make_quantum_gates: bool = False,
    tol: float = 1.0e-12,
    backend: Optional[Any] = None,
) -> Tuple[np.ndarray, Sequence[Any]]:
    """Build one Q block using OpenFermion's exact Givens convention.

    The returned matrix satisfies ``(vec / ||vec||) @ Q = e_target``.  If
    ``make_quantum_gates`` is true, the second return value implements
    ``Q^dagger`` on the MPS.  Each gate is ``(sites, matrix)``; ``sites`` has
    length one for the residual phase and length two for a Givens rotation.

    OpenFermion documents ``V A U^dagger = D`` and ``U = G_k ... G_1``.  Thus
    the desired row-action matrix is ``U^dagger`` followed by a one-mode phase
    correction.  This is not the left-accumulated matrix returned by the
    original ``_classical_givens_reduction`` helper.
    """

    if backend is None:
        backend = sys.modules[__name__]
    if target_pos not in {"first", "last"}:
        raise ValueError("target_pos must be 'first' or 'last'")

    vec = np.asarray(vec, dtype=complex).reshape(-1)
    dim = vec.size
    if physical_qubits is None:
        physical_qubits = list(range(dim))
    physical_qubits = [int(q) for q in physical_qubits]
    if len(physical_qubits) != dim:
        raise ValueError("physical_qubits must have the same length as vec")
    if make_quantum_gates and dim and not physical_qubits:
        raise ValueError("physical_qubits are required for quantum gates")
    if dim == 0:
        return np.eye(0, dtype=complex), []

    norm = np.linalg.norm(vec)
    if norm < tol:
        return np.eye(dim, dtype=complex), []

    order = list(range(dim - 1, -1, -1)) if target_pos == "last" else list(range(dim))
    ordered_vec = vec[order] / norm
    decomposition, _, _ = backend.givens_decomposition(
        ordered_vec.reshape(1, -1)
    )

    # OpenFermion returns U=G_k...G_1.  Iterating in the returned order and
    # left-multiplying reconstructs U; Q_ordered starts as U^dagger.
    U = np.eye(dim, dtype=complex)
    quantum_gates = []
    for layer in decomposition:
        for i, j, theta, phi in layer:
            cosine, sine = np.cos(theta), np.sin(theta)
            phase = np.exp(1j * phi)
            G2 = np.array(
                [
                    [cosine, -phase * sine],
                    [sine, phase * cosine],
                ],
                dtype=complex,
            )
            embedded = np.eye(dim, dtype=complex)
            embedded[np.ix_([i, j], [i, j])] = G2
            U = embedded @ U

            if make_quantum_gates:
                # Second quantization of G2 in computational order
                # |00>,|01>,|10>,|11>.  The one-particle mode order is
                # |10> (first site), |01> (second site).
                G4 = np.zeros((4, 4), dtype=complex)
                G4[0, 0] = 1.0
                G4[np.ix_([2, 1], [2, 1])] = G2
                G4[3, 3] = np.linalg.det(G2)
                quantum_gates.append(
                    (
                        (physical_qubits[order[i]], physical_qubits[order[j]]),
                        G4,
                    )
                )

    Q_ordered = U.conj().T
    routed_ordered = ordered_vec @ Q_ordered
    residual = routed_ordered[0]
    if abs(residual) < 1.0 - 100 * tol:
        raise RuntimeError("OpenFermion Givens reduction did not isolate the row")
    residual /= abs(residual)

    phase_fix = np.eye(dim, dtype=complex)
    phase_fix[0, 0] = np.conjugate(residual)
    Q_ordered = Q_ordered @ phase_fix
    if make_quantum_gates and not np.isclose(residual, 1.0, atol=tol):
        # Q_ordered^dagger = phase_fix^dagger U.  Since gates are applied from
        # right to left on a ket, U's Givens gates come first and this phase is
        # appended last.
        quantum_gates.append(
            (
                (physical_qubits[order[0]],),
                np.diag([1.0, residual]).astype(complex),
            )
        )

    permutation = np.eye(dim, dtype=complex)[:, order]
    Q = permutation @ Q_ordered @ permutation.T
    target = 0 if target_pos == "first" else dim - 1
    if not np.allclose(
        (vec / norm) @ Q, np.eye(dim, dtype=complex)[target], atol=100 * tol
    ):
        raise RuntimeError("Phase-complete OpenFermion Q reconstruction failed")
    return Q, quantum_gates


def apply_qfew_dagger_gates(
    psi_mps: Any,
    quantum_gates: Sequence[Any],
    *,
    backend: Optional[Any] = None,
) -> Any:
    """Apply only the phase-complete ``Q_few^dagger`` circuit to an MPS."""

    if backend is None:
        backend = sys.modules[__name__]
    out = psi_mps
    for sites, matrix in quantum_gates:
        sites = tuple(int(q) for q in sites)
        if len(sites) == 1:
            out = backend.Gate(matrix=np.asarray(matrix), indices=sites) @ out
        elif len(sites) == 2:
            out = backend.apply_two_qubit_gate(
                out, np.asarray(matrix), sites[0], sites[1]
            )
        else:
            raise ValueError("A Q_few gate must act on one or two sites")
    return out


def apply_three_qubit_gate_mps(
    psi_mps: Any,
    matrix: np.ndarray,
    q1: int,
    q2: int,
    q3: int,
    *,
    backend: Optional[Any] = None,
) -> Any:
    """Apply an adjacent 8x8 gate to a qlimb MPS using two SVD splits.

    The current qlimb ``Gate @ MPS`` implementation handles one- and two-site
    gates only; treating every multi-site gate as a two-site gate attempts to
    reshape an 8x8 matrix into four qubit legs.  This helper performs the
    required three-tensor contraction explicitly while retaining qlimb's own
    ``apply_svd`` truncation and norm-preservation policy.
    """

    if backend is None:
        backend = sys.modules[__name__]
    q1, q2, q3 = int(q1), int(q2), int(q3)
    if (q2, q3) != (q1 + 1, q1 + 2):
        raise ValueError("The three-qubit gate sites must be increasing and adjacent")
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (8, 8):
        raise ValueError("matrix must have shape (8, 8)")

    # Dense MockMPS objects intentionally have no tensor list.
    if not hasattr(psi_mps, "tensors"):
        return backend.apply_3_qubit_gate_custom(
            psi_mps, matrix, q1, q2, q3
        )

    out = psi_mps.copy()
    out.move_center(q1, method="qr")
    A, B, C = out.tensors[q1], out.tensors[q2], out.tensors[q3]
    theta = np.tensordot(A, B, axes=([2], [0]))
    theta = np.tensordot(theta, C, axes=([-1], [0]))
    # theta: (chi_left, in1, in2, in3, chi_right)
    physical_dim = int(out.phys_dim)
    gate_tensor = matrix.reshape([physical_dim] * 6)
    updated = np.tensordot(
        gate_tensor, theta, axes=([3, 4, 5], [1, 2, 3])
    )
    # tensordot result: (out1,out2,out3,chi_left,chi_right)
    updated = np.ascontiguousarray(updated.transpose(3, 0, 1, 2, 4))

    chi_left, _, _, _, chi_right = updated.shape
    first_split = updated.reshape(
        chi_left, physical_dim, physical_dim**2, chi_right
    )
    A_new, _, _, remainder = backend.apply_svd(
        first_split,
        bond_dim=out.bond_dim,
        direction="right",
        preserve_norm=out.preserve_norm,
        tol=out.trunc_tol,
    )
    second_split = remainder.reshape(
        remainder.shape[0], physical_dim, physical_dim, chi_right
    )
    B_new, _, _, C_new = backend.apply_svd(
        second_split,
        bond_dim=out.bond_dim,
        direction="right",
        preserve_norm=out.preserve_norm,
        tol=out.trunc_tol,
    )
    out.tensors[q1], out.tensors[q2], out.tensors[q3] = A_new, B_new, C_new
    out.tags[q1], out.tags[q2], out.tags[q3] = "L", "L", "N"
    out.center_idx = q3
    return out


def apply_square_gaussian_mps(
    psi_mps: Any,
    one_particle_unitary: np.ndarray,
    physical_qubits: Sequence[int],
    *,
    tol: float = 1.0e-10,
    backend: Optional[Any] = None,
) -> Any:
    """Apply a square one-particle unitary using OpenFermion and qlimb.

    For OpenFermion's ``W = D G_k ... G_1`` convention, the ket circuit applies
    ``G_1`` through ``G_k`` first and the diagonal phases last.  The supplied
    ``apply_givens_circuit`` does the opposite (phases first and reversed
    rotations), so it does not implement the requested natural-frame update.
    """

    if backend is None:
        backend = sys.modules[__name__]
    W = np.asarray(one_particle_unitary, dtype=complex)
    physical_qubits = [int(q) for q in physical_qubits]
    dim = len(physical_qubits)
    if W.shape != (dim, dim):
        raise ValueError("one_particle_unitary has the wrong shape")
    if not np.allclose(W.conj().T @ W, np.eye(dim), atol=100 * tol):
        raise ValueError("one_particle_unitary must be unitary")

    decomposition, diagonal = backend.givens_decomposition_square(W)
    # The dense mock encodes the desired matrix directly in its marker.
    if isinstance(decomposition, dict) and "matrix" in decomposition:
        return backend.apply_givens_circuit(
            psi_mps, decomposition, diagonal, physical_qubits
        )

    out = psi_mps
    U = np.eye(dim, dtype=complex)
    for layer in decomposition:
        for i, j, theta, phi in layer:
            cosine, sine = np.cos(theta), np.sin(theta)
            phase = np.exp(1j * phi)
            G2 = np.array(
                [
                    [cosine, -phase * sine],
                    [sine, phase * cosine],
                ],
                dtype=complex,
            )
            embedded = np.eye(dim, dtype=complex)
            embedded[np.ix_([i, j], [i, j])] = G2
            U = embedded @ U

            G4 = np.zeros((4, 4), dtype=complex)
            G4[0, 0] = 1.0
            G4[np.ix_([2, 1], [2, 1])] = G2
            G4[3, 3] = np.linalg.det(G2)
            out = backend.apply_two_qubit_gate(
                out, G4, physical_qubits[i], physical_qubits[j]
            )

    diagonal = np.asarray(diagonal, dtype=complex)
    reconstructed = np.diag(diagonal) @ U
    if not np.allclose(reconstructed, W, atol=500 * tol, rtol=0.0):
        raise RuntimeError("OpenFermion square-Givens reconstruction failed")
    for local, phase in enumerate(diagonal):
        if not np.isclose(phase, 1.0, atol=tol):
            out = backend.Gate(
                matrix=np.diag([1.0, phase]).astype(complex),
                indices=(physical_qubits[local],),
            ) @ out
    return out


def compose_q_frame_and_qfew_gates(
    v_star: np.ndarray,
    occupations: np.ndarray,
    bath_qubits: Sequence[int],
    chain_type: str,
    *,
    tol: float = 1.0e-8,
    backend: Optional[Any] = None,
) -> Tuple[np.ndarray, Sequence[Any], int, Dict[str, Any]]:
    """Construct the classical ``Q`` frame and only the physical Q_few gates.

    ``Q_total = Q_bar @ Q_few``, where ``Q_bar`` contains the filled and empty
    OpenFermion reductions.  Those two rotations are classical bookkeeping
    only.  The returned gate list implements only ``Q_few^dagger``.

    The bath-site ordering must be the fixed mirrored SERGIO ordering: far to
    near on the up chain and near to far on the down chain.  Consequently the
    fewest-ordered window is contiguous and ends/starts next to the impurity.
    """

    if backend is None:
        backend = sys.modules[__name__]
    if chain_type not in {"up", "down"}:
        raise ValueError("chain_type must be 'up' or 'down'")

    v_star = np.asarray(v_star, dtype=complex).reshape(-1)
    occupations = np.asarray(occupations, dtype=float).reshape(-1)
    bath_qubits = [int(q) for q in bath_qubits]
    n_orb = v_star.size
    if occupations.size != n_orb or len(bath_qubits) != n_orb:
        raise ValueError("v_star, occupations, and bath_qubits must have equal length")
    if not np.isclose(np.linalg.norm(v_star), 1.0, atol=50 * tol):
        raise ValueError("v_star must be normalized")

    filled, active, empty = backend.classify_orb(occupations, tol=tol)
    target_pos = "last" if chain_type == "up" else "first"
    Q_full, _ = build_q_block_openfermion(
        v_star[filled], target_pos=target_pos, backend=backend
    )
    Q_empty, _ = build_q_block_openfermion(
        v_star[empty], target_pos=target_pos, backend=backend
    )

    Q_bar = np.eye(n_orb, dtype=complex)
    Q_bar[np.ix_(filled, filled)] = Q_full
    Q_bar[np.ix_(empty, empty)] = Q_empty
    v_ordered = v_star @ Q_bar

    if chain_type == "up":
        f0 = max(filled) if filled else None
        e0 = max(empty) if empty else None
        expected_effective_local = n_orb - 1
    else:
        f0 = min(filled) if filled else None
        e0 = min(empty) if empty else None
        expected_effective_local = 0

    few_local = sorted(set(active + [x for x in (f0, e0) if x is not None]))
    if not few_local:
        # The only remaining mathematical case is a fully filled or fully empty
        # chain.  Its boundary orbital is sufficient for the effective mode.
        few_local = [expected_effective_local]
    if few_local != list(range(few_local[0], few_local[-1] + 1)):
        raise ValueError(
            "The f0/e0/active window is not contiguous.  Check the fixed mirrored "
            "tensor ordering and the natural-orbital relabeling."
        )

    few_qubits = [bath_qubits[i] for i in few_local]
    Q_few_local, qfew_gates = build_q_block_openfermion(
        v_ordered[few_local],
        target_pos=target_pos,
        physical_qubits=few_qubits,
        make_quantum_gates=True,
        backend=backend,
    )
    effective_qubit = few_qubits[-1] if target_pos == "last" else few_qubits[0]

    Q_few = np.eye(n_orb, dtype=complex)
    Q_few[np.ix_(few_local, few_local)] = Q_few_local
    Q_total = Q_bar @ Q_few

    effective_local = bath_qubits.index(int(effective_qubit))
    if effective_local != expected_effective_local:
        raise ValueError(
            "The fewest-ordered bath mode is not adjacent to the impurity: "
            f"got bath-local site {effective_local}, expected {expected_effective_local}."
        )

    target = np.zeros(n_orb, dtype=complex)
    target[expected_effective_local] = 1.0
    routed = v_star @ Q_total
    if not np.allclose(routed, target, atol=100 * tol, rtol=0.0):
        raise ValueError(
            "The helper Q convention did not route v_star to the impurity-adjacent "
            "site.  Inspect build_qfullempty_classically/build_qfew conventions."
        )

    info = {
        "filled": filled,
        "active": active,
        "empty": empty,
        "f0": f0,
        "e0": e0,
        "few_local": few_local,
        "few_qubits": few_qubits,
        "Q_bar": Q_bar,
        "Q_few": Q_few,
    }
    return Q_total, qfew_gates, int(effective_qubit), info


def advance_full_natural_orbital_frame(
    psi_mps: Any,
    V: np.ndarray,
    bath_qubits: Sequence[int],
    imp_qubit: int,
    chain_type: str,
    *,
    tol: float = 1.0e-8,
    backend: Optional[Any] = None,
) -> Tuple[Any, np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Rotate a full fixed-length MPS into the next natural-orbital frame.

    OpenFermion's ``givens_decomposition_square`` supplies the circuit, and the
    supplied ``apply_givens_circuit`` helper applies it with qlimb gates.  No MPS
    tensors are added, removed, or copied for active-space truncation.
    """

    if backend is None:
        backend = sys.modules[__name__]
    if chain_type not in {"up", "down"}:
        raise ValueError("chain_type must be 'up' or 'down'")

    C = build_1rdm_bath_jw(
        psi_mps, bath_qubits, imp_qubit, backend=backend
    )
    evals, evecs = np.linalg.eigh(C)
    filled, active, empty = backend.classify_orb(evals, tol=tol)

    # eigh is ascending.  Within the full/active/empty sectors we retain that
    # deterministic order, then arrange the inactive pairs away from the
    # impurity.  The down chain is the mirror of the up chain.
    active_order = list(active)
    filled_near = list(reversed(filled))
    empty_near = list(empty)
    inactive_near = []
    for k in range(max(len(filled_near), len(empty_near))):
        if k < len(empty_near):
            inactive_near.append(empty_near[k])
        if k < len(filled_near):
            inactive_near.append(filled_near[k])
    near_to_far = active_order + inactive_near
    order = list(reversed(near_to_far)) if chain_type == "up" else near_to_far
    if sorted(order) != list(range(len(evals))):
        raise RuntimeError("Natural-orbital ordering did not form a permutation")

    evals_ordered = np.real_if_close(evals[order]).real
    evecs_ordered = evecs[:, order]

    # u = evecs.conj() satisfies u.T C u.conj() = diag.  The ket must therefore
    # be acted on by u^dagger=evecs.T.
    psi_next = apply_square_gaussian_mps(
        psi_mps,
        evecs_ordered.T,
        list(bath_qubits),
        tol=tol,
        backend=backend,
    )
    U_next = np.asarray(V, dtype=complex) @ evecs_ordered.conj()
    M_next = len(active)

    info = {
        "C": C,
        "order": order,
        "filled": [i for i, x in enumerate(evals_ordered) if x >= 1.0 - tol],
        "active": [i for i, x in enumerate(evals_ordered) if tol < x < 1.0 - tol],
        "empty": [i for i, x in enumerate(evals_ordered) if x <= tol],
    }
    return psi_next, U_next, evals_ordered, M_next, info
