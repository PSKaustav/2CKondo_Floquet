"""Benchmark SERGIO active-space growth for progressively larger baths.

The default parameters are fixed to the mapped ``latest.py`` protocol:

* T = 1
* Jk = pi/2
* Jz = sqrt(2) * (sqrt(2) - 1) * sin(pi/3)
* t0 = 2*pi/3
* initial PBC Fermi sea followed by OBC evolution
* Kondo interaction first, followed by bath hopping
* natural-occupation tolerance = 1e-10

Each size stops as soon as both spin sectors reach M=N.  A second safety guard
stops after a measured MPS bond dimension reaches 256; on the reference laptop
this avoids entering the much slower next step for N=14 and N=18.  Change the
limits with ``--post-full-steps`` and ``--max-bond-abort`` when desired.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

import sergio_qlimb_openfermion as sergio


DEFAULT_SIZES = (6, 10, 14, 18)
DEFAULT_MAX_STEPS = 20
DEFAULT_POST_FULL_STEPS = 0
DEFAULT_MAX_BOND_ABORT = 256
FIXED_TOL = 1.0e-10
FIXED_T = 1.0
FIXED_JK = np.pi / 2.0
FIXED_JZ = np.sqrt(2.0) * (np.sqrt(2.0) - 1.0) * np.sin(np.pi / 3.0)
FIXED_T0 = 2.0 * np.pi / 3.0
FIXED_BOUNDARY = "obc"
FIXED_INITIAL_BOUNDARY = "pbc"
FIXED_FLOQUET_ORDER = "interaction_then_free"


def _maximum_bond_dimension(psi_mps: Any) -> int:
    if not hasattr(psi_mps, "get_bond_dimensions"):
        return 1
    dimensions = list(psi_mps.get_bond_dimensions())
    return int(max(dimensions, default=1))


def run_one_size(
    N: int,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    post_full_steps: int = DEFAULT_POST_FULL_STEPS,
    stop_when_full: bool = True,
    max_bond_abort: int | None = DEFAULT_MAX_BOND_ABORT,
) -> Dict[str, Any]:
    """Run one SERGIO size and return per-step physics and timing data."""

    N = int(N)
    max_steps = int(max_steps)
    if N <= 0 or N % 2:
        raise ValueError("Every N must be a positive even integer")
    if max_steps < 0 or int(post_full_steps) < 0:
        raise ValueError("Step counts must be non-negative")

    initialization_start = perf_counter()
    data = sergio.initialize_sergio_mps(
        N=N,
        Jk=FIXED_JK,
        Jz=FIXED_JZ,
        T=FIXED_T,
        t0=FIXED_T0,
        boundary=FIXED_BOUNDARY,
        initial_boundary=FIXED_INITIAL_BOUNDARY,
        floquet_order=FIXED_FLOQUET_ORDER,
        bond_dim=np.inf,
        tol=FIXED_TOL,
    )
    initialization_seconds = perf_counter() - initialization_start

    U_up, V_up = data["U_up"], data["V_up"]
    U_down, V_down = data["U_down"], data["V_down"]
    M_up, M_down = int(data["M_up"]), int(data["M_down"])
    psi_mps = data["psi_mps"]
    free_step = data["free_step"]
    free_power_offset = int(data.get("free_power_offset", 0))

    steps: list[int] = []
    magnetization: list[float] = []
    active_up: list[int] = []
    active_down: list[int] = []
    max_bonds: list[int] = []
    step_seconds: list[float] = []
    full_at_step: int | None = None
    stop_reason = "max_steps"
    impurity_z = np.diag([-1.0, 1.0]).astype(complex)
    evolution_start = perf_counter()

    for step in range(max_steps + 1):
        steps.append(step)
        magnetization.append(
            float(np.real(psi_mps.measure_observable(impurity_z, (N,))))
        )
        active_up.append(M_up)
        active_down.append(M_down)
        current_max_bond = _maximum_bond_dimension(psi_mps)
        max_bonds.append(current_max_bond)
        if step == 0:
            step_seconds.append(0.0)

        print(
            f"N={N:2d}, step={step:2d}, "
            f"M_up={M_up:2d}, M_down={M_down:2d}, "
            f"max_chi={current_max_bond:4d}, "
            f"mag={magnetization[-1]: .8f}",
            flush=True,
        )

        if M_up == N and M_down == N and full_at_step is None:
            full_at_step = step
        if (
            stop_when_full
            and full_at_step is not None
            and step >= full_at_step + int(post_full_steps)
        ):
            stop_reason = "full_active_space"
            break
        if max_bond_abort is not None and current_max_bond >= int(max_bond_abort):
            stop_reason = "max_bond_abort"
            break
        if step == max_steps:
            break

        step_start = perf_counter()
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
        ) = sergio.sergio_step(
            U_up,
            V_up,
            U_down,
            V_down,
            FIXED_JK,
            FIXED_JZ,
            FIXED_T,
            M_up,
            M_down,
            psi_mps,
            n=step,
            free_step=free_step,
            free_power_offset=free_power_offset,
            tol=FIXED_TOL,
        )
        step_seconds.append(perf_counter() - step_start)

    evolution_seconds = perf_counter() - evolution_start
    return {
        "N": N,
        "steps": np.asarray(steps, dtype=int),
        "magnetization": np.asarray(magnetization, dtype=float),
        "M_up": np.asarray(active_up, dtype=int),
        "M_down": np.asarray(active_down, dtype=int),
        "max_bond_dimension": np.asarray(max_bonds, dtype=int),
        "step_seconds": np.asarray(step_seconds, dtype=float),
        "initialization_seconds": float(initialization_seconds),
        "evolution_seconds": float(evolution_seconds),
        "total_seconds": float(initialization_seconds + evolution_seconds),
        "full_at_step": full_at_step,
        "stop_reason": stop_reason,
        "last_step": int(steps[-1]),
        "final_M_up": int(active_up[-1]),
        "final_M_down": int(active_down[-1]),
        "peak_bond_dimension": int(max(max_bonds)),
    }


def _plot_results(results: Sequence[Dict[str, Any]], output_dir: Path) -> Dict[str, Path]:
    colors = plt.get_cmap("tab10").colors

    active_path = output_dir / "sergio_scaling_active_orbitals.png"
    ncols = 2
    nrows = int(np.ceil(len(results) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.2, 4.0 * nrows),
        squeeze=False,
        sharex=True,
    )
    for index, result in enumerate(results):
        axis = axes.flat[index]
        color = colors[index % len(colors)]
        axis.plot(
            result["steps"],
            result["M_up"],
            color=color,
            marker="o",
            linewidth=2.0,
            label=r"$M^\uparrow$",
        )
        axis.plot(
            result["steps"],
            result["M_down"],
            color=color,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label=r"$M^\downarrow$",
        )
        axis.axhline(
            result["N"], color="0.35", linestyle=":", linewidth=1.3,
            label=r"full space, $M=N$",
        )
        status = (
            f"full at n={result['full_at_step']}"
            if result["full_at_step"] is not None
            else f"stopped at n={result['last_step']} (safety cutoff)"
        )
        axis.set_title(f"N={result['N']}: {status}")
        axis.set_ylim(-0.5, result["N"] + 1.0)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="upper left")
    for index in range(len(results), nrows * ncols):
        axes.flat[index].set_visible(False)
    figure.supxlabel("Floquet step, n")
    figure.supylabel("Number of active orbitals")
    figure.suptitle(
        f"SERGIO active-space growth, tolerance={FIXED_TOL:.0e}", y=1.01
    )
    figure.tight_layout()
    figure.savefig(active_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    magnetization_path = output_dir / "sergio_scaling_magnetization.png"
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    for index, result in enumerate(results):
        axis.plot(
            result["steps"],
            result["magnetization"],
            color=colors[index % len(colors)],
            marker="o",
            linewidth=1.8,
            label=f"N={result['N']}",
        )
    axis.set_xlabel("Floquet step, n")
    axis.set_ylabel(r"Impurity magnetization, $\langle\sigma_z\rangle$")
    axis.set_title("SERGIO magnetization over the safely measured Floquet steps")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(magnetization_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    efficiency_path = output_dir / "sergio_scaling_efficiency.png"
    figure, primary = plt.subplots(figsize=(8.4, 5.2))
    sizes = np.asarray([result["N"] for result in results])
    runtimes = np.asarray([result["total_seconds"] for result in results])
    peak_bonds = np.asarray([result["peak_bond_dimension"] for result in results])
    primary.plot(sizes, runtimes, color="tab:blue", marker="o", linewidth=2.0)
    primary.set_xlabel("Bath orbitals per spin, N")
    primary.set_ylabel("Total wall time (seconds)", color="tab:blue")
    primary.tick_params(axis="y", labelcolor="tab:blue")
    primary.set_xticks(sizes)
    primary.grid(alpha=0.25)
    secondary = primary.twinx()
    secondary.plot(
        sizes,
        peak_bonds,
        color="tab:red",
        marker="s",
        linestyle="--",
        linewidth=1.8,
    )
    secondary.set_ylabel("Peak MPS bond dimension", color="tab:red")
    secondary.tick_params(axis="y", labelcolor="tab:red")
    primary.set_title("Measured cost up to saturation or the bond-dimension cutoff")
    figure.tight_layout()
    figure.savefig(efficiency_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {
        "active_orbitals_plot": active_path,
        "magnetization_plot": magnetization_path,
        "efficiency_plot": efficiency_path,
    }


def _write_data(results: Sequence[Dict[str, Any]], output_dir: Path) -> Dict[str, Path]:
    csv_path = output_dir / "sergio_scaling_steps.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "N",
                "step",
                "magnetization",
                "M_up",
                "M_down",
                "max_bond_dimension",
                "step_seconds",
            ]
        )
        for result in results:
            for values in zip(
                result["steps"],
                result["magnetization"],
                result["M_up"],
                result["M_down"],
                result["max_bond_dimension"],
                result["step_seconds"],
            ):
                writer.writerow([result["N"], *values])

    summary_path = output_dir / "sergio_scaling_summary.json"
    summary = {
        "parameters": {
            "T": FIXED_T,
            "Jk": float(FIXED_JK),
            "Jz": float(FIXED_JZ),
            "t0": float(FIXED_T0),
            "tol": FIXED_TOL,
            "boundary": FIXED_BOUNDARY,
            "initial_boundary": FIXED_INITIAL_BOUNDARY,
            "floquet_order": FIXED_FLOQUET_ORDER,
        },
        "sizes": [
            {
                key: result[key]
                for key in (
                    "N",
                    "last_step",
                    "final_M_up",
                    "final_M_down",
                    "full_at_step",
                    "peak_bond_dimension",
                    "initialization_seconds",
                    "evolution_seconds",
                    "total_seconds",
                    "stop_reason",
                )
            }
            for result in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {"step_data": csv_path, "summary": summary_path}


def run_scaling_benchmark(
    sizes: Sequence[int] = DEFAULT_SIZES,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    post_full_steps: int = DEFAULT_POST_FULL_STEPS,
    stop_when_full: bool = True,
    max_bond_abort: int | None = DEFAULT_MAX_BOND_ABORT,
    output_dir: str | Path = "sergio_scaling_results",
) -> Dict[str, Any]:
    """Run all sizes, save plots/data, and return the full in-memory results."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = [
        run_one_size(
            N,
            max_steps=max_steps,
            post_full_steps=post_full_steps,
            stop_when_full=stop_when_full,
            max_bond_abort=max_bond_abort,
        )
        for N in sizes
    ]
    paths = {
        **_plot_results(results, destination),
        **_write_data(results, destination),
    }
    return {"results": results, "paths": paths}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--post-full-steps", type=int, default=DEFAULT_POST_FULL_STEPS
    )
    parser.add_argument("--no-stop-full", action="store_true")
    parser.add_argument(
        "--max-bond-abort", type=int, default=DEFAULT_MAX_BOND_ABORT
    )
    parser.add_argument("--output-dir", default="sergio_scaling_results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    benchmark = run_scaling_benchmark(
        sizes=arguments.sizes,
        max_steps=arguments.max_steps,
        post_full_steps=arguments.post_full_steps,
        stop_when_full=not arguments.no_stop_full,
        max_bond_abort=arguments.max_bond_abort,
        output_dir=arguments.output_dir,
    )
    print("\nSaturation summary")
    for size_result in benchmark["results"]:
        print(
            f"N={size_result['N']:2d}: "
            f"final M=({size_result['final_M_up']}, {size_result['final_M_down']}), "
            f"full_at={size_result['full_at_step']}, "
            f"peak_chi={size_result['peak_bond_dimension']}, "
            f"time={size_result['total_seconds']:.3f}s, "
            f"stop={size_result['stop_reason']}"
        )
    for label, path in benchmark["paths"].items():
        print(f"{label}: {path}")
