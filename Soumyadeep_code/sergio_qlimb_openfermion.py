"""Full-MPS SERGIO implementation using qlimb and OpenFermion.

This is the production implementation.  ``sergio_full_mps.py`` remains a
dense, dependency-free reference check only.

Set the single switch below to ``False`` in an environment containing qlimb
and OpenFermion.  Keep this file, ``helper_functions_v2.py``, and
``helper_functions_v2_updates.py`` in the same directory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Backend switch requested for local testing versus the real implementation.
# ---------------------------------------------------------------------------
USE_MOCK_BACKEND = False  # <-- set True to use the dense MockMPS backend
if "SERGIO_USE_MOCK" in os.environ:
    USE_MOCK_BACKEND = os.environ["SERGIO_USE_MOCK"].strip().lower() not in {
        "0",
        "false",
        "no",
    }

if USE_MOCK_BACKEND:
    import sergio_mock_backend as hf

    MPS = hf.MPS
    MPO = hf.MPO
    Gate = hf.Gate
    givens_decomposition = hf.givens_decomposition
    givens_decomposition_square = hf.givens_decomposition_square
else:
    import helper_functions_v2 as hf
    from openfermion.linalg.givens_rotations import (
        givens_decomposition,
        givens_decomposition_square,
    )
    from qlimb.classical.gates import Gate
    from qlimb.classical.mpo import MPO
    from qlimb.classical.mps import MPS

    # Make the dependency used by the supplied helper functions explicit.  In
    # the supplied file these names are already imported from OpenFermion; the
    # assignments prevent an accidentally shadowed implementation being used.
    hf.givens_decomposition = givens_decomposition
    hf.givens_decomposition_square = givens_decomposition_square

from helper_functions_v2_updates import (
    advance_full_natural_orbital_frame,
    apply_qfew_dagger_gates,
    apply_three_qubit_gate_mps,
    build_1rdm_bath_jw,
    compose_q_frame_and_qfew_gates,
)


def build_h1(N: int, t0: float = 1.0, boundary: str = "obc") -> np.ndarray:
    """Build the one-particle bath Hamiltonian for OBC or PBC."""

    N = int(N)
    boundary = str(boundary).lower()
    if N <= 0:
        raise ValueError("N must be positive")
    if boundary not in {"obc", "pbc"}:
        raise ValueError("boundary must be 'obc' or 'pbc'")

    H1 = np.zeros((N, N), dtype=complex)
    if N > 1:
        links = np.arange(N - 1)
        H1[links, links + 1] = -0.5 * float(t0)
        H1[links + 1, links] = -0.5 * float(t0)
        if boundary == "pbc":
            # Addition is needed for N=2, where the two ring bonds coincide.
            H1[0, N - 1] += -0.5 * float(t0)
            H1[N - 1, 0] += -0.5 * float(t0)
    return H1


def diagonalize_h1(
    H1: np.ndarray, T: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``D``, ``V``, and ``V exp(-i D T) V^dagger``."""

    H1 = np.asarray(H1, dtype=complex)
    if H1.ndim != 2 or H1.shape[0] != H1.shape[1]:
        raise ValueError("H1 must be square")
    if not np.allclose(H1, H1.conj().T, atol=1.0e-12):
        raise ValueError("H1 must be Hermitian")
    energies, eigenvectors = np.linalg.eigh(H1)
    free_step = (
        eigenvectors * np.exp(-1j * energies * float(T))
    ) @ eigenvectors.conj().T
    return energies, eigenvectors, free_step


def _normalize_floquet_order(floquet_order: str) -> Tuple[str, int]:
    """Return the canonical Floquet-order name and interaction-picture offset."""

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
    canonical = aliases[key]
    return canonical, int(canonical == "free_then_interaction")


def _initial_mirrored_frame(
    eigenvectors: np.ndarray, n_filled: int, chain_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Order Fermi-sea orbitals in the fixed mirrored tensor layout."""

    N = eigenvectors.shape[0]
    filled = list(range(n_filled))
    empty = list(range(n_filled, N))
    filled_near = list(reversed(filled))
    empty_near = list(empty)
    down_near_to_far = []
    for k in range(max(len(filled_near), len(empty_near))):
        if k < len(empty_near):
            down_near_to_far.append(empty_near[k])
        if k < len(filled_near):
            down_near_to_far.append(filled_near[k])
    order = (
        list(reversed(down_near_to_far))
        if chain_type == "up"
        else down_near_to_far
    )
    frame = eigenvectors[:, order]
    occupations = np.array([1.0 if i < n_filled else 0.0 for i in order])
    return frame, occupations


def _product_state_mps(
    occupations_up: np.ndarray,
    occupations_down: np.ndarray,
    impurity_bit: int,
    bond_dim: float,
    trunc_tol: float = 1.0e-10,
) -> Any:
    N = len(occupations_up)
    bits = np.concatenate(
        (
            np.rint(occupations_up).astype(int),
            [int(impurity_bit)],
            np.rint(occupations_down).astype(int),
        )
    )

    if not USE_MOCK_BACKEND:
        # This state is already a bond-dimension-one MPS.  Constructing a
        # length-2**(2*N+1) one-hot vector first is exponentially wasteful and
        # makes otherwise easy N=14/18 initializations impossible.
        tensors = []
        for bit in bits:
            tensor = np.zeros((1, 2, 1), dtype=complex)
            tensor[0, int(bit), 0] = 1.0
            tensors.append(tensor)
        return MPS(
            nqbits=2 * N + 1,
            phys_dim=2,
            tensors=tensors,
            bond_dim=bond_dim,
            preserve_norm=True,
            trunc_tol=float(trunc_tol),
        )

    # The dense mock is intentionally limited to small validation systems.
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    state = np.zeros(2 ** (2 * N + 1), dtype=complex)
    state[index] = 1.0
    return hf.state_mps(state, N, bond_dim=bond_dim)


def _chain_tags(occupations: np.ndarray, chain_type: str, tol: float):
    filled, active, empty = hf.classify_orb(occupations, tol=tol)
    categories = {}
    for i in filled:
        categories[i] = "filled"
    for i in active:
        categories[i] = "active"
    for i in empty:
        categories[i] = "empty"

    tags = [""] * len(occupations)
    counters = {"filled": 0, "active": 0, "empty": 0}
    near_to_far = (
        range(len(occupations) - 1, -1, -1)
        if chain_type == "up"
        else range(len(occupations))
    )
    for local in near_to_far:
        category = categories[local]
        tags[local] = f"{category}_{counters[category]}_{chain_type}"
        counters[category] += 1
    return tags


def _set_full_mps_tags(
    psi_mps: Any,
    occupations_up: np.ndarray,
    occupations_down: np.ndarray,
    tol: float,
) -> None:
    if not hasattr(psi_mps, "tags"):
        return
    psi_mps.tags = (
        _chain_tags(occupations_up, "up", tol)
        + ["impurity"]
        + _chain_tags(occupations_down, "down", tol)
    )


def initialize_sergio_mps(
    N: int,
    Jk: float,
    Jz: float,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    initial_boundary: Optional[str] = None,
    floquet_order: str = "interaction_then_free",
    n_up: Optional[int] = None,
    n_down: Optional[int] = None,
    impurity_bit: int = 0,
    bond_dim: float = np.inf,
    tol: float = 1.0e-8,
) -> Dict[str, Any]:
    """Initialize the complete ``2*N+1``-site SERGIO MPS.

    The bath tensors are an alternating filled/empty Fermi sea and the down
    chain is the mirror of the up chain.  All tensors remain in the MPS for the
    entire evolution.  Active-space truncation is represented only by tags.
    """

    N = int(N)
    if N % 2:
        raise ValueError("The requested mirrored half-filled layout requires even N")
    if n_up is None:
        n_up = N // 2
    if n_down is None:
        n_down = N // 2
    if not (0 <= n_up <= N and 0 <= n_down <= N):
        raise ValueError("n_up and n_down must lie between 0 and N")

    boundary = str(boundary).lower()
    initial_boundary = (
        boundary if initial_boundary is None else str(initial_boundary).lower()
    )
    floquet_order, free_power_offset = _normalize_floquet_order(floquet_order)
    H1 = build_h1(N, t0=t0, boundary=boundary)
    energies, eigenvectors, free_step = diagonalize_h1(H1, T=T)
    if initial_boundary != boundary:
        initial_H1 = build_h1(N, t0=t0, boundary=initial_boundary)
        _, eigenvectors, _ = diagonalize_h1(initial_H1, T=T)
    U_up, occupations_up = _initial_mirrored_frame(
        eigenvectors, int(n_up), "up"
    )
    U_down, occupations_down = _initial_mirrored_frame(
        eigenvectors, int(n_down), "down"
    )
    psi_mps = _product_state_mps(
        occupations_up,
        occupations_down,
        impurity_bit,
        bond_dim,
        trunc_tol=tol,
    )

    up_qubits = list(range(N))
    imp_qubit = N
    down_qubits = list(range(N + 1, 2 * N + 1))

    # Step n=0.  Q_full/Q_empty are used only in these classical V matrices;
    # the Q_few circuit is applied later inside sergio_step.
    free_at_first_interaction = np.linalg.matrix_power(
        free_step, free_power_offset
    )
    v_up = (free_at_first_interaction @ U_up)[0, :]
    v_down = (free_at_first_interaction @ U_down)[0, :]
    Q_up, _, qeff_up, _ = compose_q_frame_and_qfew_gates(
        v_up,
        occupations_up,
        up_qubits,
        "up",
        tol=tol,
        backend=hf,
    )
    Q_down, _, qeff_down, _ = compose_q_frame_and_qfew_gates(
        v_down,
        occupations_down,
        down_qubits,
        "down",
        tol=tol,
        backend=hf,
    )
    if qeff_up != N - 1 or qeff_down != N + 1:
        raise RuntimeError("Effective bath modes are not adjacent to the impurity")
    V_up = U_up @ Q_up
    V_down = U_down @ Q_down
    _set_full_mps_tags(psi_mps, occupations_up, occupations_down, tol)

    return {
        "n": 0,
        "N": N,
        "U_up": U_up,
        "V_up": V_up,
        "U_down": U_down,
        "V_down": V_down,
        "Jk": float(Jk),
        "Jz": float(Jz),
        "T": float(T),
        "M_up": 0,
        "M_down": 0,
        "psi_mps": psi_mps,
        "H1": H1,
        "energies": energies,
        "free_step": free_step,
        "boundary": boundary,
        "initial_boundary": initial_boundary,
        "floquet_order": floquet_order,
        "free_power_offset": free_power_offset,
        "occupations_up": occupations_up,
        "occupations_down": occupations_down,
    }


def _natural_occupations(
    psi_mps: Any,
    bath_qubits: Sequence[int],
    imp_qubit: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    C = build_1rdm_bath_jw(
        psi_mps, bath_qubits, imp_qubit, backend=hf
    )
    off_diagonal = C - np.diag(np.diag(C))
    if np.linalg.norm(off_diagonal) > 200 * tol:
        raise ValueError(
            "Input MPS is not in a natural-orbital frame; the bath 1-RDM is "
            "not diagonal."
        )
    occupations = np.clip(np.real(np.diag(C)), 0.0, 1.0)
    return occupations, C


TOL_VAL = 1e-10  # try different values of tol to see if the results change significantly


def sergio_step(
    U_up_n: np.ndarray,
    V_up_n: np.ndarray,
    U_down_n: np.ndarray,
    V_down_n: np.ndarray,
    Jk: float,
    Jz: float,
    T: float,
    M_up_n: int,
    M_down_n: int,
    psi_mps_n: Any,
    *,
    n: int,
    free_step: np.ndarray,
    free_power_offset: int = 0,
    tol: float = TOL_VAL,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    int,
    int,
    Any,
]:
    """Advance the full qlimb MPS and its SERGIO frames from n to n+1.

    The positional output has the same variables as the positional input, now
    at step n+1.  ``Jk``, ``Jz``, and ``T`` are returned unchanged.

    Crucially, the only Q-family gates applied to ``psi_mps_n`` are the gates
    returned by ``hf.build_qfew``.  The filled/empty Q blocks remain entirely
    within the classical ``V_sigma = U_sigma @ Q_sigma`` update.
    """

    N = int(np.asarray(U_up_n).shape[0])
    if np.asarray(U_up_n).shape != (N, N):
        raise ValueError("U_up_n must be square")
    for name, matrix in {
        "V_up_n": V_up_n,
        "U_down_n": U_down_n,
        "V_down_n": V_down_n,
        "free_step": free_step,
    }.items():
        if np.asarray(matrix).shape != (N, N):
            raise ValueError(f"{name} must have shape {(N, N)}")
    if int(psi_mps_n.nqbits) != 2 * N + 1:
        raise ValueError("psi_mps_n must retain all 2*N+1 sites")

    up_qubits = list(range(N))
    imp_qubit = N
    down_qubits = list(range(N + 1, 2 * N + 1))
    occupations_up, _ = _natural_occupations(
        psi_mps_n, up_qubits, imp_qubit, tol
    )
    occupations_down, _ = _natural_occupations(
        psi_mps_n, down_qubits, imp_qubit, tol
    )
    measured_M_up = len(hf.classify_orb(occupations_up, tol=tol)[1])
    measured_M_down = len(hf.classify_orb(occupations_down, tol=tol)[1])
    if measured_M_up != int(M_up_n) or measured_M_down != int(M_down_n):
        raise ValueError(
            "M_up_n/M_down_n do not match the natural occupations stored in the MPS"
        )

    free_power_offset = int(free_power_offset)
    if free_power_offset not in {0, 1}:
        raise ValueError("free_power_offset must be 0 or 1")
    free_n = np.linalg.matrix_power(
        np.asarray(free_step), int(n) + free_power_offset
    )
    v_up = (free_n @ np.asarray(U_up_n))[0, :]
    v_down = (free_n @ np.asarray(U_down_n))[0, :]
    Q_up, qfew_up, qeff_up, qinfo_up = compose_q_frame_and_qfew_gates(
        v_up,
        occupations_up,
        up_qubits,
        "up",
        tol=tol,
        backend=hf,
    )
    Q_down, qfew_down, qeff_down, qinfo_down = compose_q_frame_and_qfew_gates(
        v_down,
        occupations_down,
        down_qubits,
        "down",
        tol=tol,
        backend=hf,
    )

    expected_V_up = np.asarray(U_up_n) @ Q_up
    expected_V_down = np.asarray(U_down_n) @ Q_down
    if not np.allclose(V_up_n, expected_V_up, atol=200 * tol, rtol=0.0):
        raise ValueError("V_up_n is inconsistent with U_up_n and the prescribed Q_up")
    if not np.allclose(V_down_n, expected_V_down, atol=200 * tol, rtol=0.0):
        raise ValueError(
            "V_down_n is inconsistent with U_down_n and the prescribed Q_down"
        )
    if qeff_up != N - 1 or qeff_down != N + 1:
        raise RuntimeError("cbar_0,sigma is not adjacent to the impurity")

    # This is the only physical Q operation.  There is deliberately no call
    # that applies Q_full or Q_empty to the MPS.
    psi_few = apply_qfew_dagger_gates(psi_mps_n, qfew_up, backend=hf)
    psi_few = apply_qfew_dagger_gates(psi_few, qfew_down, backend=hf)

    kondo_gate = hf.build_kondo_gate(Jk, Jz, 0.0, T, int(n))
    psi_interacting = apply_three_qubit_gate_mps(
        psi_few,
        kondo_gate,
        qeff_up,
        imp_qubit,
        qeff_down,
        backend=hf,
    )
    if int(psi_interacting.nqbits) != 2 * N + 1:
        raise RuntimeError("The Kondo update changed the MPS length")

    psi_up, U_up_next, occ_up_next, M_up_next, frame_up = (
        advance_full_natural_orbital_frame(
            psi_interacting,
            expected_V_up,
            up_qubits,
            imp_qubit,
            "up",
            tol=tol,
            backend=hf,
        )
    )
    psi_next, U_down_next, occ_down_next, M_down_next, frame_down = (
        advance_full_natural_orbital_frame(
            psi_up,
            expected_V_down,
            down_qubits,
            imp_qubit,
            "down",
            tol=tol,
            backend=hf,
        )
    )

    free_next = np.linalg.matrix_power(
        np.asarray(free_step), int(n) + 1 + free_power_offset
    )
    v_up_next = (free_next @ U_up_next)[0, :]
    v_down_next = (free_next @ U_down_next)[0, :]
    Q_up_next, _, _, _ = compose_q_frame_and_qfew_gates(
        v_up_next,
        occ_up_next,
        up_qubits,
        "up",
        tol=tol,
        backend=hf,
    )
    Q_down_next, _, _, _ = compose_q_frame_and_qfew_gates(
        v_down_next,
        occ_down_next,
        down_qubits,
        "down",
        tol=tol,
        backend=hf,
    )
    V_up_next = U_up_next @ Q_up_next
    V_down_next = U_down_next @ Q_down_next
    _set_full_mps_tags(psi_next, occ_up_next, occ_down_next, tol)

    # Useful for interactive inspection without changing the required return
    # signature.  qlimb MPS objects normally permit ordinary attributes.
    try:
        psi_next.sergio_diagnostics = {
            "q_up": qinfo_up,
            "q_down": qinfo_down,
            "frame_up": frame_up,
            "frame_down": frame_down,
            "occupations_up": occ_up_next,
            "occupations_down": occ_down_next,
        }
    except Exception:
        pass

    return (
        U_up_next,
        V_up_next,
        U_down_next,
        V_down_next,
        float(Jk),
        float(Jz),
        float(T),
        int(M_up_next),
        int(M_down_next),
        psi_next,
    )


def sergio_step_floquet(
    N: int,
    no_floquet_steps: int,
    Jk: float,
    Jz: float,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    initial_boundary: Optional[str] = None,
    floquet_order: str = "interaction_then_free",
    initial_data: Optional[Dict[str, Any]] = None,
    bond_dim: float = np.inf,
    tol: float = TOL_VAL,
) -> Dict[str, Any]:
    """Iterate ``sergio_step`` without ever changing the MPS site count."""

    if int(no_floquet_steps) < 0:
        raise ValueError("no_floquet_steps must be non-negative")
    floquet_order, requested_offset = _normalize_floquet_order(floquet_order)
    data = (
        initialize_sergio_mps(
            N,
            Jk,
            Jz,
            T=T,
            t0=t0,
            boundary=boundary,
            initial_boundary=initial_boundary,
            floquet_order=floquet_order,
            bond_dim=bond_dim,
            tol=tol,
        )
        if initial_data is None
        else dict(initial_data)
    )
    if int(data["N"]) != int(N):
        raise ValueError("initial_data has a different N")
    stored_order = str(data.get("floquet_order", "interaction_then_free"))
    if stored_order != floquet_order:
        raise ValueError("initial_data has a different floquet_order")
    free_power_offset = int(data.get("free_power_offset", requested_offset))
    if free_power_offset != requested_offset:
        raise ValueError("initial_data has an inconsistent free_power_offset")

    U_up, V_up = data["U_up"], data["V_up"]
    U_down, V_down = data["U_down"], data["V_down"]
    M_up, M_down = int(data["M_up"]), int(data["M_down"])
    psi_mps = data["psi_mps"]
    free_step = data["free_step"]
    magnetization = []
    active_history = []

    for n in range(int(no_floquet_steps) + 1):
        magnetization.append(
            float(
                np.real(
                    psi_mps.measure_observable(
                        hf.sigma_z, (int(N),)
                    )
                )
            )
        )
        active_history.append((M_up, M_down))
        if n == int(no_floquet_steps):
            break
        (
            U_up,
            V_up,
            U_down,
            V_down,
            _,
            _,
            _,
            M_up,
            M_down,
            psi_mps,
        ) = sergio_step(
            U_up,
            V_up,
            U_down,
            V_down,
            Jk,
            Jz,
            T,
            M_up,
            M_down,
            psi_mps,
            n=n,
            free_step=free_step,
            free_power_offset=free_power_offset,
            tol=tol,
        )

    if int(psi_mps.nqbits) != 2 * int(N) + 1:
        raise RuntimeError("SERGIO evolution changed the fixed MPS length")
    return {
        "n": int(no_floquet_steps),
        "N": int(N),
        "U_up": U_up,
        "V_up": V_up,
        "U_down": U_down,
        "V_down": V_down,
        "Jk": float(Jk),
        "Jz": float(Jz),
        "T": float(T),
        "M_up": M_up,
        "M_down": M_down,
        "psi_mps": psi_mps,
        "free_step": free_step,
        "boundary": str(boundary).lower(),
        "initial_boundary": str(
            data.get("initial_boundary", boundary)
        ).lower(),
        "floquet_order": floquet_order,
        "free_power_offset": free_power_offset,
        "magnetization": np.asarray(magnetization),
        "active_history": np.asarray(active_history, dtype=int),
    }


if __name__ == "__main__":
    result = sergio_step_floquet(
        N=6,
        no_floquet_steps=4,
        Jk=0.8,
        Jz=0.3,
        T=1.0,
        boundary="obc",
    )
    print("backend:", "mock" if USE_MOCK_BACKEND else "qlimb/OpenFermion")
    print("MPS sites:", result["psi_mps"].nqbits)
    print("impurity magnetization:", result["magnetization"])
    print("active orbitals:", result["active_history"])
