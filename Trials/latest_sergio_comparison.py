"""Reproduce the ``Trials/latest.py`` benchmark and compare it with SERGIO.

This keeps the benchmark's N, angle, and step-count format, but avoids forming
an 8192 x 8192 full-chain unitary.  The exact reference uses the equivalent
spin-chain propagators from ``sergio_full_mps.py`` (or, optionally, the exact
qlimb MPO/MPS backend), followed by the Kondo interaction as in ``latest.py``.

``latest.py`` prepares a periodic momentum-space Fermi sea and then evolves
with open-chain bath hopping.  Those are deliberately separate choices here:
``initial_boundary='pbc'`` and ``boundary='obc'``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


TRIALS_DIR = Path(__file__).resolve().parent
CODE_DIR = TRIALS_DIR.parent / "Soumyadeep_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from plot_sergio_comparison import SERGIO_DEFAULT_TOL, plot_sergio_comparison


# Keep the benchmark parameter block in the same form as latest.py.
N = 6
theta = np.pi / 3
theta_K = np.pi / 4
theta_z = 0.5 * np.sqrt(2) * (np.sqrt(2) - 1) * np.sin(theta)
no_floquet_steps = 100
bond_dim = np.inf

# latest.py used exp(-i H T/2).  Taking its T_latest=2 gives the same physical
# period as the current convention exp(-i H T) with CURRENT_T=1.
LATEST_T = 2.0
CURRENT_T = 1.0


def latest_angles_to_current_couplings(
    theta_value: float,
    theta_k_value: float,
    theta_z_value: float,
) -> tuple[float, float, float]:
    """Map latest.py angles to ``(Jk, Jz, t0)`` at current ``T=1``.

    In current conventions, the gate angles are ``Jk*T/2``, ``Jz*T/2``,
    and ``t0*T/2``.  Thus, with T fixed to one, all three couplings are twice
    the corresponding angle used by latest.py.
    """

    Jk = 2.0 * float(theta_k_value) / CURRENT_T
    Jz = 2.0 * float(theta_z_value) / CURRENT_T
    t0 = 2.0 * float(theta_value) / CURRENT_T
    return Jk, Jz, t0


def _saved_trotter_data() -> np.ndarray | None:
    path = TRIALS_DIR / (
        "N = 6, theta = 1.05, theta_k = 0.79, t = 100_sz_tol.txt"
    )
    if not path.is_file():
        return None
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected saved benchmark format in {path}")
    return data


def _replace_magnetization_plot(
    result: Dict[str, Any],
    reversed_result: Dict[str, Any] | None,
    output_path: Path,
    saved_trotter: np.ndarray | None,
    show: bool,
) -> None:
    exact = np.asarray(result["exact_magnetization"], dtype=float)
    sergio = np.asarray(
        result["sergio_result"]["magnetization"], dtype=float
    )
    steps = np.arange(exact.size)
    tick_stride = max(1, int(np.ceil(exact.size / 11)))
    ticks = steps[::tick_stride]
    if ticks[-1] != steps[-1]:
        ticks = np.append(ticks, steps[-1])

    figure, axis = plt.subplots(figsize=(11.5, 6.0))
    axis.plot(
        steps,
        exact,
        color="black",
        linewidth=2.1,
        label="Exact: bath then Kondo",
    )
    axis.plot(
        steps,
        sergio,
        color="tab:blue",
        linestyle="--",
        linewidth=1.7,
        label="SERGIO: bath then Kondo",
    )
    if reversed_result is not None:
        exact_reversed = np.asarray(
            reversed_result["exact_magnetization"], dtype=float
        )
        sergio_reversed = np.asarray(
            reversed_result["sergio_result"]["magnetization"], dtype=float
        )
        axis.plot(
            steps,
            exact_reversed,
            color="tab:orange",
            linewidth=2.0,
            label="Exact: Kondo then bath",
        )
        axis.plot(
            steps,
            sergio_reversed,
            color="tab:purple",
            linestyle="--",
            linewidth=1.6,
            label="SERGIO: Kondo then bath",
        )
    if saved_trotter is not None:
        axis.plot(
            saved_trotter[:, 0],
            saved_trotter[:, 1],
            color="tab:red",
            linestyle=":",
            linewidth=1.35,
            alpha=0.8,
            label="Saved brick-split/Trotter benchmark",
        )
    axis.set_xlabel("Floquet step, n")
    axis.set_ylabel(r"Impurity magnetization, $\langle \sigma_z \rangle$")
    axis.set_title(
        "latest.py parameters: both Floquet orderings\n"
        r"$\theta=\pi/3$, $\theta_K=\pi/4$"
    )
    axis.set_xticks(ticks)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=1)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def _replace_active_orbital_plot(
    result: Dict[str, Any],
    reversed_result: Dict[str, Any] | None,
    output_path: Path,
    show: bool,
) -> None:
    active = np.asarray(result["sergio_result"]["active_history"], dtype=int)
    steps = np.arange(len(active))
    tick_stride = max(1, int(np.ceil(len(active) / 11)))
    ticks = steps[::tick_stride]
    if ticks[-1] != steps[-1]:
        ticks = np.append(ticks, steps[-1])
    marker_stride = max(1, int(np.ceil(len(active) / 25)))

    figure, axis = plt.subplots(figsize=(10.2, 5.6))
    axis.plot(
        steps,
        active[:, 0],
        color="tab:red",
        linewidth=2.2,
        marker="o",
        markevery=marker_stride,
        label=r"Bath$\to$Kondo, $M_n^\uparrow$",
    )
    axis.plot(
        steps,
        active[:, 1],
        color="tab:green",
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markevery=marker_stride,
        label=r"Bath$\to$Kondo, $M_n^\downarrow$",
    )
    if reversed_result is not None:
        active_reversed = np.asarray(
            reversed_result["sergio_result"]["active_history"], dtype=int
        )
        axis.plot(
            steps,
            active_reversed[:, 0],
            color="tab:orange",
            linewidth=1.7,
            linestyle=":",
            marker="^",
            markevery=marker_stride,
            label=r"Kondo$\to$bath, $M_n^\uparrow$",
        )
        axis.plot(
            steps,
            active_reversed[:, 1],
            color="tab:purple",
            linewidth=1.5,
            linestyle="-.",
            marker="d",
            markevery=marker_stride,
            label=r"Kondo$\to$bath, $M_n^\downarrow$",
        )
    axis.set_xlabel("Floquet step, n")
    axis.set_ylabel("Number of active orbitals")
    axis.set_title("SERGIO active orbitals for both Floquet orderings")
    axis.set_xticks(ticks)
    axis.set_yticks(np.arange(0, N + 1))
    axis.set_ylim(-0.2, N + 0.2)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def run_latest_sergio_comparison(
    *,
    exact_backend: str = "mps",
    tol: float = 1e-2,#SERGIO_DEFAULT_TOL,
    output_dir: str | Path | None = None,
    include_saved_trotter: bool = True,
    include_reversed_order: bool = True,
    show: bool = True,
) -> Dict[str, Any]:
    """Run latest.py and reversed-order comparisons at current ``T=1``."""

    Jk, Jz, t0 = latest_angles_to_current_couplings(
        theta, theta_K, theta_z
    )
    destination = (
        TRIALS_DIR / "latest_sergio_plots"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    result = plot_sergio_comparison(
        N=N,
        no_floquet_steps=no_floquet_steps,
        Jk=Jk,
        Jz=Jz,
        T=CURRENT_T,
        t0=t0,
        boundary="obc",
        initial_boundary="pbc",
        floquet_order="free_then_interaction",
        bond_dim=bond_dim,
        tol=tol,
        exact_backend=exact_backend,
        output_dir=destination,
        filename_prefix="latest_sergio",
        show=False,
    )

    reversed_result = None
    if include_reversed_order:
        reversed_result = plot_sergio_comparison(
            N=N,
            no_floquet_steps=no_floquet_steps,
            Jk=Jk,
            Jz=Jz,
            T=CURRENT_T,
            t0=t0,
            boundary="obc",
            initial_boundary="pbc",
            floquet_order="interaction_then_free",
            bond_dim=bond_dim,
            tol=tol,
            exact_backend=exact_backend,
            output_dir=destination,
            filename_prefix="latest_reversed",
            show=False,
        )

    saved_trotter = _saved_trotter_data() if include_saved_trotter else None
    _replace_magnetization_plot(
        result,
        reversed_result,
        Path(result["magnetization_plot"]),
        saved_trotter,
        show,
    )
    _replace_active_orbital_plot(
        result,
        reversed_result,
        Path(result["active_orbitals_plot"]),
        show,
    )
    result["mapped_parameters"] = {
        "Jk": Jk,
        "Jz": Jz,
        "t0": t0,
        "T": CURRENT_T,
        "boundary": "obc",
        "initial_boundary": "pbc",
        "floquet_order": "free_then_interaction",
    }
    result["reversed_order_result"] = reversed_result
    if saved_trotter is not None:
        common = min(
            len(saved_trotter), len(result["exact_magnetization"])
        )
        result["saved_trotter_max_abs_difference"] = float(
            np.max(
                np.abs(
                    saved_trotter[:common, 1]
                    - result["exact_magnetization"][:common]
                )
            )
        )
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact-backend",
        choices=("dense", "mps"),
        default="mps",
        help="Use dense exact evolution (fast at N=6) or exact qlimb MPO/MPS.",
    )
    parser.add_argument("--tol", type=float, default=SERGIO_DEFAULT_TOL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--without-saved-trotter", action="store_true")
    parser.add_argument("--without-reversed-order", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    comparison = run_latest_sergio_comparison(
        exact_backend=arguments.exact_backend,
        tol=arguments.tol,
        output_dir=arguments.output_dir,
        include_saved_trotter=not arguments.without_saved_trotter,
        include_reversed_order=not arguments.without_reversed_order,
        show=not arguments.no_show,
    )
    mapped = comparison["mapped_parameters"]
    print(
        "Mapped parameters: "
        f"Jk={mapped['Jk']:.16g}, Jz={mapped['Jz']:.16g}, "
        f"t0={mapped['t0']:.16g}, T={mapped['T']:.1f}"
    )
    print(
        "Max |SERGIO - exact|:",
        f"{comparison['max_abs_magnetization_error']:.3e}",
    )
    if comparison["reversed_order_result"] is not None:
        print(
            "Max |SERGIO - exact| (Kondo then bath):",
            f"{comparison['reversed_order_result']['max_abs_magnetization_error']:.3e}",
        )
    if "saved_trotter_max_abs_difference" in comparison:
        print(
            "Max |saved Trotter - exact|:",
            f"{comparison['saved_trotter_max_abs_difference']:.3e}",
        )
    print("Magnetization plot:", comparison["magnetization_plot"])
    print("Active-orbital plot:", comparison["active_orbitals_plot"])
