import numpy as np
import matplotlib.pyplot as plt

# Import everything from the fixed SERGIO implementation without triggering
# its __main__ block (that block only runs when the file is executed
# directly, so a plain import is safe here).
from Trials.test_householder_aka_givens import run_sergio_floquet

# ---- Parameters (must match the benchmark data file) ----
N = 6
theta = 1.05
theta_K = 0.79
theta_z = 0.5 * np.sqrt(2) * (np.sqrt(2) - 1) * np.sin(theta)

T = 1.0
Jx = theta_K / T
Jy = theta_K / T
Jz = theta_z / T
h = 0.0   # no magnetic field (fixed)

# ---- Load benchmark ----
data_filename = "N = 6, theta = 1.05, theta_k = 0.79, t = 100_sz_tol.txt"
ssdata = np.loadtxt(data_filename)
ss_steps = ssdata[:, 0]
ss_mag = ssdata[:, 1]
no_floquet_steps = len(ss_mag) - 1

# ---- Run ONLY the no-truncation branch ----
print("Running SERGIO, Trotter, open BC, NO TRUNCATION (classify_tol=1e-14) ...")
no_trunc_mag = run_sergio_floquet(
    N=N, no_floquet_steps=no_floquet_steps,
    theta=theta, Jx=Jx, Jy=Jy, Jz=Jz, h=h, T=T,
    use_trotter=True, boundary='open',
    classify_tol=1e-14,   # truncation effectively off
    verbose=True
)

# ---- Compare ----
no_trunc_mag = np.array(no_trunc_mag)
diff = np.abs(no_trunc_mag - ss_mag[:len(no_trunc_mag)])
print(f"\nMax |SERGIO - benchmark| = {diff.max():.3e}")
print(f"Mean |SERGIO - benchmark| = {diff.mean():.3e}")

plt.figure(figsize=(11, 6.5))
plt.plot(ss_steps, ss_mag, label="Benchmark (exact)", color="red", linestyle="--", linewidth=2.5)
plt.plot(np.arange(len(no_trunc_mag)), no_trunc_mag, label="SERGIO (no truncation, h=0)", color="black", linewidth=1.6)
plt.xlabel("Floquet Step ($n$)")
plt.ylabel(r"$\langle S_z^{imp} \rangle$")
plt.title(f"No-truncation SERGIO vs Benchmark (N={N}, theta={theta}, theta_K={theta_K})")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("no_truncation_check.png", dpi=150)
plt.show()