"""Exact Floquet evolution of the 1CK chain with qlimb MPS/MPO objects.

The fixed physical ordering is

    [up bath: far -> near] -- impurity -- [down bath: near -> far].

Unlike the SERGIO evolution, this reference never changes basis or discards an
orbital.  One Floquet period is applied as

    U_F = exp(-i H_1 T) exp(-i H_2 T),

where the interaction propagator is a three-site MPO and the two spin copies
of the free propagator are exact N-site MPOs.  All MPO/MPS decompositions use
``max_bond_dim = bond_dim = np.inf`` and ``trunc_tol = 0``.

The only dense statevector operation is the one-time construction of the
Fermi-sea initial MPS.  Every Floquet step thereafter is an MPO-on-MPS
contraction; no dense statevector is formed during the evolution.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Sequence

import numpy as np

import helper_functions_v2 as hf
from qlimb.classical.mpo import MPO
from qlimb.classical.mps import MPS

from sergio_qlimb_openfermion import build_h1, diagonalize_h1


def _validate_inputs(N: int, no_floquet_steps: int, boundary: str) -> tuple[int, int, str]:
    N = int(N)
    no_floquet_steps = int(no_floquet_steps)
    boundary = str(boundary).lower()
    if N <= 0 or N % 2:
        raise ValueError("N must be a positive even integer")
    if no_floquet_steps < 0:
        raise ValueError("no_floquet_steps must be non-negative")
    if boundary not in {"obc", "pbc"}:
        raise ValueError("boundary must be 'obc' or 'pbc'")
    return N, no_floquet_steps, boundary


def _normalize_floquet_order(floquet_order: str) -> str:
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


def _occupied_sites(index: int, n_sites: int) -> list[int]:
    """Return occupied modes for a big-endian computational-basis index."""

    return [
        site
        for site in range(n_sites)
        if (int(index) >> (n_sites - 1 - site)) & 1
    ]


def second_quantized_unitary(one_particle_unitary: np.ndarray) -> np.ndarray:
    """Lift an N-mode one-particle unitary to its 2**N Fock-space unitary.

    For input occupation set J and output occupation set I, the matrix element
    is ``det(W[I, J])``.  Different particle-number sectors do not mix.
    """

    W = np.asarray(one_particle_unitary, dtype=complex)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("one_particle_unitary must be square")
    n_sites = W.shape[0]
    if not np.allclose(W.conj().T @ W, np.eye(n_sites), atol=2.0e-10):
        raise ValueError("one_particle_unitary must be unitary")

    dim = 1 << n_sites
    occupations = [_occupied_sites(index, n_sites) for index in range(dim)]
    sectors: dict[int, list[int]] = {}
    for index, occupied in enumerate(occupations):
        sectors.setdefault(len(occupied), []).append(index)

    fock_unitary = np.zeros((dim, dim), dtype=complex)
    for particle_number, indices in sectors.items():
        if particle_number == 0:
            fock_unitary[0, 0] = 1.0
            continue
        for output_index in indices:
            output_sites = occupations[output_index]
            for input_index in indices:
                input_sites = occupations[input_index]
                fock_unitary[output_index, input_index] = np.linalg.det(
                    W[np.ix_(output_sites, input_sites)]
                )
    return fock_unitary


def _slater_statevector(
    N: int,
    occupied_up_physical: np.ndarray,
    occupied_down_physical: np.ndarray,
    impurity_bit: int = 0,
) -> np.ndarray:
    """Build the normalized Fermi sea in the fixed full-chain ordering."""

    occupied_up_physical = np.asarray(occupied_up_physical, dtype=complex)
    occupied_down_physical = np.asarray(occupied_down_physical, dtype=complex)
    if occupied_up_physical.shape[0] != N or occupied_down_physical.shape[0] != N:
        raise ValueError("occupied orbital matrices must have N rows")
    if impurity_bit not in {0, 1}:
        raise ValueError("impurity_bit must be 0 or 1")

    n_up = occupied_up_physical.shape[1]
    n_down = occupied_down_physical.shape[1]
    state = np.zeros(1 << (2 * N + 1), dtype=complex)
    for down_occ_tuple in combinations(range(N), n_down):
        down_occ = np.asarray(down_occ_tuple, dtype=int)
        amp_down = np.linalg.det(occupied_down_physical[down_occ, :])
        for up_occ_tuple in combinations(range(N), n_up):
            up_occ = np.asarray(up_occ_tuple, dtype=int)
            amp_up = np.linalg.det(occupied_up_physical[up_occ, :])
            bits = np.zeros(2 * N + 1, dtype=int)
            bits[up_occ] = 1
            bits[N] = impurity_bit
            bits[N + 1 + down_occ] = 1
            index = 0
            for bit in bits:
                index = (index << 1) | int(bit)
            state[index] = amp_up * amp_down

    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise RuntimeError("Fermi-sea construction produced the zero vector")
    return state / norm


def _exact_mps_from_vector(vector: np.ndarray, n_sites: int) -> MPS:
    """Convert an initialization vector to an MPS without rank truncation."""

    psi_mps = MPS.from_vec(
        np.asarray(vector, dtype=complex),
        bond_dim=np.inf,
        phys_dim=2,
        trunc_tol=0.0,
        nqbits=int(n_sites),
    )
    # qlimb.from_vec produces a left-canonical tensor list whose center is the
    # last site, but older qlimb versions do not update center_idx accordingly.
    psi_mps.center_idx = int(n_sites) - 1
    psi_mps.tags = (int(n_sites) - 1) * ["L"] + ["N"]
    psi_mps.bond_dim = np.inf
    psi_mps.trunc_tol = 0.0
    psi_mps.preserve_norm = False
    return psi_mps


def _identity_mpo_tensor() -> np.ndarray:
    tensor = np.zeros((1, 2, 2, 1), dtype=complex)
    tensor[0, :, :, 0] = np.eye(2, dtype=complex)
    return tensor


def _embed_local_mpo(local_mpo: MPO, sites: Sequence[int], total_sites: int) -> MPO:
    """Embed a contiguous local MPO into a full-chain identity MPO."""

    sites = [int(site) for site in sites]
    if not sites:
        raise ValueError("sites cannot be empty")
    if sites != list(range(sites[0], sites[0] + len(sites))):
        raise ValueError("local MPO sites must be contiguous and increasing")
    if len(sites) != int(local_mpo.nqbits):
        raise ValueError("number of sites does not match local MPO length")
    if sites[0] < 0 or sites[-1] >= int(total_sites):
        raise ValueError("local MPO sites lie outside the full chain")

    tensors = [_identity_mpo_tensor() for _ in range(int(total_sites))]
    for local_index, site in enumerate(sites):
        tensors[site] = np.asarray(local_mpo.tensors[local_index], dtype=complex)
    return MPO(
        nqbits=int(total_sites),
        phys_dim=2,
        tensors=tensors,
        active_sites=sites,
    )


def _exact_operator_mpo(
    operator: np.ndarray, sites: Sequence[int], total_sites: int
) -> MPO:
    """Factor a local operator exactly and embed it in the full MPS chain."""

    sites = list(map(int, sites))
    local_mpo = MPO.from_matrix(
        np.asarray(operator, dtype=complex),
        phys_dim=2,
        nqbits=len(sites),
        max_bond_dim=np.inf,
        trunc_tol=0.0,
    )
    return _embed_local_mpo(local_mpo, sites, total_sites)


def initialize_exact_floquet_mps(
    N: int,
    Jk: float,
    Jz: float,
    *,
    T: float = 1.0,
    t0: float = 1.0,
    boundary: str = "obc",
    initial_boundary: str | None = None,
    floquet_order: str = "interaction_then_free",
) -> Dict[str, Any]:
    """Create the exact Fermi-sea MPS and the three full-chain propagator MPOs."""

    N, _, boundary = _validate_inputs(N, 0, boundary)
    initial_boundary = (
        boundary if initial_boundary is None else str(initial_boundary).lower()
    )
    if initial_boundary not in {"obc", "pbc"}:
        raise ValueError("initial_boundary must be 'obc' or 'pbc'")
    floquet_order = _normalize_floquet_order(floquet_order)
    total_sites = 2 * N + 1
    H1 = build_h1(N, t0=t0, boundary=boundary)
    energies, orbitals, free_step = diagonalize_h1(H1, T=T)
    if initial_boundary == boundary:
        initial_orbitals = orbitals
    else:
        initial_H1 = build_h1(N, t0=t0, boundary=initial_boundary)
        _, initial_orbitals, _ = diagonalize_h1(initial_H1, T=T)
    occupied = initial_orbitals[:, : N // 2]

    # Logical bath row 0 is impurity-adjacent.  The physical up chain is
    # reversed, whereas the physical down chain keeps the logical ordering.
    initial_vector = _slater_statevector(
        N,
        occupied[::-1, :],
        occupied,
        impurity_bit=0,
    )
    psi_mps = _exact_mps_from_vector(initial_vector, total_sites)

    reverse = np.eye(N, dtype=complex)[::-1]
    free_up_physical = reverse @ free_step @ reverse
    free_down_physical = free_step
    free_up_mpo = _exact_operator_mpo(
        second_quantized_unitary(free_up_physical),
        range(N),
        total_sites,
    )
    free_down_mpo = _exact_operator_mpo(
        second_quantized_unitary(free_down_physical),
        range(N + 1, total_sites),
        total_sites,
    )

    interaction_gate = hf.build_kondo_gate(
        float(Jk), float(Jz), 0.0, float(T), 0
    )
    interaction_mpo = _exact_operator_mpo(
        interaction_gate,
        [N - 1, N, N + 1],
        total_sites,
    )
    return {
        "N": N,
        "Jk": float(Jk),
        "Jz": float(Jz),
        "T": float(T),
        "t0": float(t0),
        "boundary": boundary,
        "initial_boundary": initial_boundary,
        "floquet_order": floquet_order,
        "psi_mps": psi_mps,
        "H1": H1,
        "energies": energies,
        "free_step": free_step,
        "interaction_mpo": interaction_mpo,
        "free_up_mpo": free_up_mpo,
        "free_down_mpo": free_down_mpo,
    }


def exact_floquet_step_mps(
    psi_mps: MPS,
    interaction_mpo: MPO,
    free_up_mpo: MPO,
    free_down_mpo: MPO,
    *,
    floquet_order: str = "interaction_then_free",
) -> MPS:
    """Apply one exact Floquet period as three MPO-on-MPS contractions."""

    floquet_order = _normalize_floquet_order(floquet_order)
    if floquet_order == "interaction_then_free":
        # U_F = exp(-i H1 T) exp(-i H2 T): H2 acts first on the ket.
        result = interaction_mpo @ psi_mps
        result = free_up_mpo @ result
        result = free_down_mpo @ result
    else:
        # latest.py convention: bath hopping first, then the Kondo gate.
        result = free_up_mpo @ psi_mps
        result = free_down_mpo @ result
        result = interaction_mpo @ result
    result.bond_dim = np.inf
    result.trunc_tol = 0.0
    result.preserve_norm = False
    return result


def exact_mps_floquet(
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
    initial_data: Dict[str, Any] | None = None,
    return_states: bool = False,
) -> Dict[str, Any]:
    """Evaluate ``U_F**n |FS, down>`` exactly with qlimb MPS/MPO objects.

    The argument names intentionally mirror :func:`sergio_step_floquet`.
    ``initial_data`` may be supplied to reuse the expensive exact MPO setup.
    """

    N, no_floquet_steps, boundary = _validate_inputs(
        N, no_floquet_steps, boundary
    )
    floquet_order = _normalize_floquet_order(floquet_order)
    requested_initial_boundary = (
        boundary if initial_boundary is None else str(initial_boundary).lower()
    )
    data = (
        initialize_exact_floquet_mps(
            N,
            Jk,
            Jz,
            T=T,
            t0=t0,
            boundary=boundary,
            initial_boundary=requested_initial_boundary,
            floquet_order=floquet_order,
        )
        if initial_data is None
        else dict(initial_data)
    )
    if int(data["N"]) != N:
        raise ValueError("initial_data has a different N")
    for key, requested in (("Jk", Jk), ("Jz", Jz), ("T", T), ("t0", t0)):
        if key in data and not np.isclose(float(data[key]), float(requested)):
            raise ValueError(f"initial_data has a different {key}")
    if str(data.get("boundary", boundary)).lower() != boundary:
        raise ValueError("initial_data has a different boundary")
    if (
        str(data.get("initial_boundary", boundary)).lower()
        != requested_initial_boundary
    ):
        raise ValueError("initial_data has a different initial_boundary")
    if str(data.get("floquet_order", "interaction_then_free")) != floquet_order:
        raise ValueError("initial_data has a different floquet_order")

    psi_mps = data["psi_mps"]
    if int(psi_mps.nqbits) != 2 * N + 1:
        raise ValueError("initial_data MPS has the wrong site count")
    interaction_mpo = data["interaction_mpo"]
    free_up_mpo = data["free_up_mpo"]
    free_down_mpo = data["free_down_mpo"]

    magnetization: list[float] = []
    bond_dimensions: list[list[int]] = []
    states: list[MPS] = []
    impurity_z = np.diag([-1.0, 1.0]).astype(complex)
    for step in range(no_floquet_steps + 1):
        magnetization.append(
            float(np.real(psi_mps.measure_observable(impurity_z, (N,))))
        )
        bond_dimensions.append(list(map(int, psi_mps.get_bond_dimensions())))
        if return_states:
            states.append(psi_mps.copy())
        if step < no_floquet_steps:
            psi_mps = exact_floquet_step_mps(
                psi_mps,
                interaction_mpo,
                free_up_mpo,
                free_down_mpo,
                floquet_order=floquet_order,
            )

    result: Dict[str, Any] = {
        **data,
        "n": no_floquet_steps,
        "psi_mps": psi_mps,
        "magnetization": np.asarray(magnetization),
        "bond_dimensions": np.asarray(bond_dimensions, dtype=int),
    }
    if return_states:
        result["states"] = states
    return result


if __name__ == "__main__":
    output = exact_mps_floquet(
        N=6,
        no_floquet_steps=2,
        Jk=0.8,
        Jz=0.8,
        T=1.0,
        boundary="obc",
    )
    print("impurity magnetization:", output["magnetization"])
    print("maximum exact bond dimension:", output["bond_dimensions"].max())
