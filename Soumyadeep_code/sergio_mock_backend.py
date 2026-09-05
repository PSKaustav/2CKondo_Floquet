"""Dense test double for ``sergio_qlimb_openfermion.py``.

This module is not part of the qlimb implementation and is not intended to be
copied into ``helper_functions_v2.py``.  It only mirrors the small interface
used by the production code so that orchestration and small-system physics can
be checked on a machine without qlimb/OpenFermion.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np

import sergio_full_mps as dense


class MockMPS:
    def __init__(self, state: np.ndarray, nqbits: int):
        self.state = np.asarray(state, dtype=complex).reshape(-1)
        self.nqbits = int(nqbits)
        self.tags = ["NA"] * self.nqbits

    def copy(self) -> "MockMPS":
        result = MockMPS(self.state.copy(), self.nqbits)
        result.tags = list(self.tags)
        return result

    def __matmul__(self, other: "MockMPS") -> complex:
        return np.vdot(self.state, other.state)

    def measure_observable(self, operator: np.ndarray, sites: Sequence[int]) -> complex:
        ket = dense._apply_dense_gate(
            self.state, np.asarray(operator), list(sites), self.nqbits
        )
        return np.vdot(self.state, ket)


class MockGate:
    def __init__(self, matrix: np.ndarray, indices: Sequence[int]):
        self.matrix = np.asarray(matrix, dtype=complex)
        self.indices = [int(i) for i in indices]

    def __matmul__(self, state: MockMPS) -> MockMPS:
        result = MockMPS(
            dense._apply_dense_gate(
                state.state, self.matrix, self.indices, state.nqbits
            ),
            state.nqbits,
        )
        result.tags = list(state.tags)
        return result


class MockMPO:
    def __init__(self, nqbits: int, phys_dim: int, tensors: Sequence[np.ndarray]):
        if int(phys_dim) != 2:
            raise ValueError("MockMPO only supports qubits")
        self.nqbits = int(nqbits)
        self.operators = [np.asarray(t).reshape(2, 2) for t in tensors]

    def __matmul__(self, state: MockMPS) -> MockMPS:
        vector = state.state
        for site, operator in enumerate(self.operators):
            vector = dense._apply_dense_gate(
                vector, operator, [site], state.nqbits
            )
        result = MockMPS(vector, state.nqbits)
        result.tags = list(state.tags)
        return result


MPS = MockMPS
MPO = MockMPO
Gate = MockGate

I2 = np.eye(2, dtype=complex)
Z_OP = np.diag([1.0, -1.0]).astype(complex)
SP_OP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SM_OP = SP_OP.conj().T
sigma_z = np.diag([-1.0, 1.0]).astype(complex)


def state_mps(psi: np.ndarray, N: int, bond_dim: float = np.inf) -> MockMPS:
    del bond_dim
    return MockMPS(psi, 2 * int(N) + 1)


def classify_orb(evals: np.ndarray, tol: float = 1.0e-10):
    filled, active, empty = [], [], []
    for i, occ in enumerate(np.real_if_close(evals).real):
        if occ >= 1.0 - tol:
            filled.append(i)
        elif occ <= tol:
            empty.append(i)
        else:
            active.append(i)
    return filled, active, empty


def build_qfullempty_classically(
    v_f: np.ndarray, v_e: np.ndarray, chain_type: str
):
    def block(vector, use_last):
        vector = np.asarray(vector, dtype=complex).reshape(-1)
        if vector.size == 0:
            return np.eye(0, dtype=complex)
        if vector.size == 1:
            if abs(vector[0]) < 1.0e-14:
                return np.eye(1, dtype=complex)
            return np.array([[np.conjugate(vector[0] / abs(vector[0]))]])
        target_index = vector.size - 1 if use_last else 0
        return dense._unitary_with_target_column(vector, target_index)

    target = len(v_f) - 1 if chain_type == "up" else 0
    del target
    Q_full = block(v_f, chain_type == "up")
    Q_empty = block(v_e, chain_type == "up")
    return Q_full, Q_empty


def build_qfew(
    v_target: np.ndarray,
    physical_qubits: Sequence[int],
    target_pos: str = "last",
):
    target = len(v_target) - 1 if target_pos == "last" else 0
    vector = np.asarray(v_target, dtype=complex).reshape(-1)
    if vector.size == 1:
        Q_few = np.array(
            [[1.0 if abs(vector[0]) < 1.0e-14 else np.conjugate(vector[0] / abs(vector[0]))]],
            dtype=complex,
        )
    else:
        Q_few = dense._unitary_with_target_column(vector, target)
    effective_qubit = int(physical_qubits[target])
    # A marker used only by apply_qfew_quantum_gates below.
    gates = [("MOCK_QFEW", list(physical_qubits), Q_few)]
    del effective_qubit
    return Q_few, gates


def apply_qfew_quantum_gates(psi_mps: MockMPS, qfew_gates: Sequence[Any]):
    out = psi_mps
    for kind, sites, Q_few in qfew_gates:
        if kind != "MOCK_QFEW":
            raise ValueError("Unrecognized mock Q_few gate marker")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vector = dense._apply_gaussian_dense(
                out.state, np.asarray(Q_few).conj().T, sites, out.nqbits
            )
        result = MockMPS(vector, out.nqbits)
        result.tags = list(out.tags)
        out = result
    return out


def givens_decomposition(matrix: np.ndarray):
    """Minimal one-row equivalent of OpenFermion's decomposition for tests."""

    current = np.asarray(matrix, dtype=complex).copy()
    if current.ndim != 2 or current.shape[0] != 1:
        raise ValueError("The mock givens_decomposition supports one row")
    decomposition = []
    for j in range(current.shape[1] - 1, 0, -1):
        a, b = current[0, j - 1], current[0, j]
        if abs(b) < 1.0e-14:
            continue
        if abs(a) < 1.0e-14:
            cosine, sine = 0.0, -1.0
            phase = b / abs(b)
        else:
            radius = np.hypot(abs(a), abs(b))
            cosine, sine = abs(a) / radius, -abs(b) / radius
            phase = (b / abs(b)) / (a / abs(a))
        theta, phi = np.arctan2(sine, cosine), np.angle(phase)
        G2 = np.array(
            [
                [cosine, -phase * sine],
                [sine, phase * cosine],
            ],
            dtype=complex,
        )
        current[:, [j - 1, j]] = current[:, [j - 1, j]] @ G2.conj().T
        decomposition.append(((j - 1, j, theta, phi),))
    return decomposition, np.eye(1, dtype=complex), np.array([current[0, 0]])


def givens_decomposition_square(matrix: np.ndarray):
    """Return a marker consumed by the dense mock circuit application."""

    matrix = np.asarray(matrix, dtype=complex)
    return {"matrix": matrix}, np.ones(matrix.shape[0], dtype=complex)


def apply_givens_circuit(
    psi_mps: MockMPS,
    decomposition: Any,
    diagonal: np.ndarray,
    physical_qubits: Sequence[int],
):
    del diagonal
    if not isinstance(decomposition, dict) or "matrix" not in decomposition:
        raise ValueError("The mock backend expected a matrix decomposition marker")
    W = np.asarray(decomposition["matrix"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vector = dense._apply_gaussian_dense(
            psi_mps.state, W, list(physical_qubits), psi_mps.nqbits
        )
    result = MockMPS(vector, psi_mps.nqbits)
    result.tags = list(psi_mps.tags)
    return result


def apply_two_qubit_gate(
    psi_mps: MockMPS, mat: np.ndarray, qi: int, qj: int
) -> MockMPS:
    return MockGate(mat, (qi, qj)) @ psi_mps


def apply_3_qubit_gate_custom(
    psi_mps: MockMPS, mat: np.ndarray, q1: int, q2: int, q3: int
) -> MockMPS:
    return MockGate(mat, (q1, q2, q3)) @ psi_mps


def build_kondo_gate(Jk: float, Jz: float, h: float, T: float, n: int):
    """Dense test copy of the supplied helper's Eq. (14) convention."""

    theta_k = float(Jk) * float(T) / 2.0
    theta_z = float(Jz) * float(T) / 2.0
    gate = np.zeros((8, 8), dtype=complex)
    gate[0, 0] = gate[2, 2] = gate[5, 5] = gate[7, 7] = 1.0
    p_z = np.exp(1j * theta_z / 2.0)
    m_z = np.exp(-1j * theta_z / 2.0)
    gate[1, 1] = gate[6, 6] = m_z
    gate[3, 3] = gate[4, 4] = p_z * np.cos(theta_k)
    gate[3, 4] = gate[4, 3] = -1j * p_z * np.sin(theta_k)
    phi = float(h) * float(T) * int(n) / 2.0
    rz = np.diag([np.exp(1j * phi), np.exp(-1j * phi)])
    kn = np.kron(np.kron(I2, rz), I2)
    return kn.conj().T @ gate @ kn
