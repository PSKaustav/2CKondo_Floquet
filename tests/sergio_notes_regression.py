import os
import sys
import numpy as np

# Allow importing from Trials/ without packaging changes
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRIALS_DIR = os.path.join(REPO_ROOT, "Trials")
if TRIALS_DIR not in sys.path:
    sys.path.insert(0, TRIALS_DIR)

from sergio_notes_cli import canonical_benchmark_angles, run_sergio_notes, rmse  # noqa: E402


def main():
    data_path = os.path.join(TRIALS_DIR, "N = 6, theta = 1.05, theta_k = 0.79, t = 100_sz_tol.txt")
    ss = np.loadtxt(data_path)
    benchmark = ss[:, 1] if ss.ndim > 1 else ss

    theta, theta_k, theta_z = canonical_benchmark_angles()
    T = 1.0
    Jx = theta_k / T
    Jy = theta_k / T
    Jz = theta_z / T
    h = 0.0

    mags = run_sergio_notes(
        N=6,
        no_floquet_steps=len(benchmark) - 1,
        theta=theta,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        h=h,
        T=T,
        bond_dim=np.inf,
        use_trotter=True,
        boundary="open",
        classify_tol=1e-10,
        do_orbital_truncation=False,
        verbose=False,
    )

    value = rmse(mags, benchmark)
    threshold = 1e-10
    print(f"Regression RMSE(no_trunc, benchmark) = {value:.6e}")
    if value > threshold:
        raise SystemExit(f"FAIL: RMSE {value:.6e} exceeded threshold {threshold:.1e}")

    print("PASS")


if __name__ == "__main__":
    main()
