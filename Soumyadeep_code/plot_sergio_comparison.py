"""Run the SERGIO and exact Floquet evolutions and make two comparison plots.

The first figure compares impurity magnetization from the qlimb/OpenFermion
SERGIO implementation with either the exact qlimb-MPS implementation or the
dense statevector reference.  The second figure shows the number of active up
and down natural orbitals retained by SERGIO at every Floquet step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

import sergio_qlimb_openfermion as sergio_impl


sergio_step_floquet = sergio_impl.sergio_step_floquet
SERGIO_DEFAULT_TOL = float(getattr(sergio_impl, "TOL_VAL", 1.0e-8))


def _run_exact_reference(
    exact_backend: str,
    N: int,
    no_floquet_steps: int,
    Jk: float,
    Jz: float,
    T: float,
    t0: float,
    boundary: str,
    initial_boundary: str | None,
    floquet_order: str,
    exact_initial_data: Dict[str, Any] | None,
) -> tuple[np.ndarray, Dict[str, Any] | None, str]:
    backend = str(exact_backend).lower()
    if backend not in {"mps", "dense"}:
        raise ValueError("exact_backend must be 'mps' or 'dense'")

    if backend == "mps":
        from sergio_exact_mps_qlimb import exact_mps_floquet

        result = exact_mps_floquet(
            N,
            no_floquet_steps,
            Jk,
            Jz,
            T=T,
            t0=t0,
            boundary=boundary,
            initial_boundary=initial_boundary,
            floquet_order=floquet_order,
            initial_data=exact_initial_data,
        )
        return (
            np.asarray(result["magnetization"], dtype=float),
            result,
            "Exact qlimb MPS",
        )

    if exact_initial_data is not None:
        raise ValueError("exact_initial_data is only supported by exact_backend='mps'")
    from sergio_full_mps import direct_floquet_dense

    magnetization = direct_floquet_dense(
        N,
        no_floquet_steps,
        Jk,
        Jz,
        T=T,
        t0=t0,
        boundary=boundary,
        initial_boundary=initial_boundary,
        floquet_order=floquet_order,
    )
    return np.asarray(magnetization, dtype=float), None, "Exact dense statevector"


def plot_sergio_comparison(
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
    bond_dim: float = np.inf,
    tol: float = SERGIO_DEFAULT_TOL,
    exact_backend: str = "mps",
    sergio_initial_data: Dict[str, Any] | None = None,
    exact_initial_data: Dict[str, Any] | None = None,
    output_dir: str | Path = ".",
    filename_prefix: str = "sergio",
    show: bool = True,
) -> Dict[str, Any]:
    """Run both algorithms and save magnetization and active-orbital plots.

    Parameters through ``tol`` match :func:`sergio_step_floquet`.
    Set ``exact_backend='dense'`` to use ``sergio_full_mps.py`` when the exact
    MPO/MPS evolution is slower than desired.  The dense reference is limited
    to the small sizes supported by that module (currently N <= 6).
    """

    boundary = str(boundary).lower()
    if boundary not in {"obc", "pbc"}:
        raise ValueError("boundary must be 'obc' or 'pbc'")
    initial_boundary = (
        boundary if initial_boundary is None else str(initial_boundary).lower()
    )
    if initial_boundary not in {"obc", "pbc"}:
        raise ValueError("initial_boundary must be 'obc' or 'pbc'")
    if int(no_floquet_steps) < 0:
        raise ValueError("no_floquet_steps must be non-negative")

    sergio = sergio_step_floquet(
        N,
        no_floquet_steps,
        Jk,
        Jz,
        T=T,
        t0=t0,
        boundary=boundary,
        initial_boundary=initial_boundary,
        floquet_order=floquet_order,
        initial_data=sergio_initial_data,
        bond_dim=bond_dim,
        tol=tol,
    )
    exact_magnetization, exact_result, exact_label = _run_exact_reference(
        exact_backend,
        N,
        no_floquet_steps,
        Jk,
        Jz,
        T,
        t0,
        boundary,
        initial_boundary,
        floquet_order,
        exact_initial_data,
    )

    sergio_magnetization = np.asarray(sergio["magnetization"], dtype=float)
    active_history = np.asarray(sergio["active_history"], dtype=int)
    expected_shape = (int(no_floquet_steps) + 1,)
    if sergio_magnetization.shape != expected_shape:
        raise RuntimeError("SERGIO returned an unexpected magnetization shape")
    if exact_magnetization.shape != expected_shape:
        raise RuntimeError("exact evolution returned an unexpected magnetization shape")
    if active_history.shape != (expected_shape[0], 2):
        raise RuntimeError("SERGIO returned an unexpected active_history shape")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = str(filename_prefix).strip() or "sergio"
    magnetization_path = output_path / f"{prefix}_magnetization_{boundary}.png"
    active_path = output_path / f"{prefix}_active_orbitals_{boundary}.png"
    steps = np.arange(expected_shape[0], dtype=int)
    tick_stride = max(1, int(np.ceil(expected_shape[0] / 11)))
    x_ticks = steps[::tick_stride]
    if x_ticks[-1] != steps[-1]:
        x_ticks = np.append(x_ticks, steps[-1])
    marker_stride = max(1, int(np.ceil(expected_shape[0] / 25)))

    fig_m, ax_m = plt.subplots(figsize=(7.2, 4.6))
    ax_m.plot(
        steps,
        exact_magnetization,
        color="black",
        marker="o",
        linewidth=2.0,
        markersize=5,
        markevery=marker_stride,
        label=exact_label,
    )
    ax_m.plot(
        steps,
        sergio_magnetization,
        color="tab:blue",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=5,
        markevery=marker_stride,
        label="SERGIO qlimb/OpenFermion",
    )
    ax_m.set_xlabel("Floquet step, n")
    ax_m.set_ylabel(r"Impurity magnetization, $\langle \sigma_z \rangle$")
    ax_m.set_title(f"Exact vs SERGIO magnetization ({boundary.upper()}, N={N})")
    ax_m.set_xticks(x_ticks)
    ax_m.grid(alpha=0.25)
    ax_m.legend(frameon=False)
    fig_m.tight_layout()
    fig_m.savefig(magnetization_path, dpi=180, bbox_inches="tight")

    fig_a, ax_a = plt.subplots(figsize=(7.2, 4.6))
    ax_a.plot(
        steps,
        active_history[:, 0],
        color="tab:red",
        marker="o",
        linewidth=2.0,
        markevery=marker_stride,
        label=r"Up chain, $M_n^\uparrow$",
    )
    ax_a.plot(
        steps,
        active_history[:, 1],
        color="tab:green",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markevery=marker_stride,
        label=r"Down chain, $M_n^\downarrow$",
    )
    ax_a.set_xlabel("Floquet step, n")
    ax_a.set_ylabel("Number of active orbitals")
    ax_a.set_title(f"SERGIO active space ({boundary.upper()}, N={N})")
    ax_a.set_xticks(x_ticks)
    ax_a.set_yticks(np.arange(0, int(N) + 1, dtype=int))
    ax_a.set_ylim(-0.2, int(N) + 0.2)
    ax_a.grid(alpha=0.25)
    ax_a.legend(frameon=False)
    fig_a.tight_layout()
    fig_a.savefig(active_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_m)
        plt.close(fig_a)

    return {
        "sergio_result": sergio,
        "exact_result": exact_result,
        "exact_backend": str(exact_backend).lower(),
        "exact_magnetization": exact_magnetization,
        "max_abs_magnetization_error": float(
            np.max(np.abs(sergio_magnetization - exact_magnetization))
        ),
        "magnetization_plot": magnetization_path,
        "active_orbitals_plot": active_path,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4, dest="no_floquet_steps")
    parser.add_argument("--Jk", type=float, default=0.8)
    parser.add_argument("--Jz", type=float, default=0.8)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--boundary", choices=("obc", "pbc"), default="obc")
    parser.add_argument("--initial-boundary", choices=("obc", "pbc"), default=None)
    parser.add_argument(
        "--floquet-order",
        choices=("interaction_then_free", "free_then_interaction"),
        default="interaction_then_free",
    )
    parser.add_argument("--tol", type=float, default=SERGIO_DEFAULT_TOL)
    parser.add_argument("--bond-dim", type=float, default=np.inf)
    parser.add_argument("--exact-backend", choices=("mps", "dense"), default="mps")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--filename-prefix", default="sergio")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    results = plot_sergio_comparison(
        N=args.N,
        no_floquet_steps=args.no_floquet_steps,
        Jk=args.Jk,
        Jz=args.Jz,
        T=args.T,
        t0=args.t0,
        boundary=args.boundary,
        initial_boundary=args.initial_boundary,
        floquet_order=args.floquet_order,
        bond_dim=args.bond_dim,
        tol=args.tol,
        exact_backend=args.exact_backend,
        output_dir=args.output_dir,
        filename_prefix=args.filename_prefix,
        show=not args.no_show,
    )
    print("magnetization plot:", results["magnetization_plot"])
    print("active-orbital plot:", results["active_orbitals_plot"])
    print(
        "max |SERGIO - exact|:",
        f"{results['max_abs_magnetization_error']:.3e}",
    )
