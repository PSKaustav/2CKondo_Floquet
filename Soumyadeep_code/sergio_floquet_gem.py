import numpy as np
import matplotlib.pyplot as plt
from helper_functions_gem import *

def exact_evolution_benchmark(N, Jk, Jz, h, T, N_f):
    """
    Directly evaluates \psi_n = U_F^n \psi_0 using dense matrices for a small system.
    References U_F structure from Eq. (3)[cite: 2].
    """
    dim = 2 ** (2 * N + 1)
    # Build initial state directly
    orb_up, orb_down = build_initial_orbitals(N)
    psi_0 = build_initial_state(N, orb_up, orb_down, N//2, N//2)
    
    # Build exact UK(0) globally
    #from helper_functions import build_kondo_gate
    UK_0_local = build_kondo_gate(Jk, Jz, 0, T, 0)
    
    # Pad to full Hilbert Space (assuming impurity at center)
    I_left = np.eye(2 ** (N - 1))
    I_right = np.eye(2 ** (N - 1))
    UK_global = np.kron(np.kron(I_left, UK_0_local), I_right)
    
    # Free evolution H1 (Simplified hopping)
    H1 = np.zeros((dim, dim), dtype=complex)
    # ... Populate H1 with -t0/2 hopping across the chains ...
    U_kin = expm(-1j * H1 * T)
    
    U_F = UK_global @ U_kin
    
    mz_exact = []
    psi = psi_0.copy()
    
    # Observable Z_imp
    Z_imp = np.kron(np.kron(np.eye(2**N), sigma_z), np.eye(2**N))
    
    for n in range(N_f):
        mz = np.real(psi.conj().T @ Z_imp @ psi)
        mz_exact.append(mz)
        
        # Exact step
        psi = U_F @ psi
        
    return mz_exact

def run_sergio():
    N = 4 # Small system for benchmark comparison
    N_f = 20
    Jk, Jz, h, T = 1.0, 1.0, 0.0, 1.0
    tol = 1e-12

    # Initialization
    orb_up, orb_down = build_initial_orbitals(N)
    psi_vec = build_initial_state(N, orb_up, orb_down, N//2, N//2)
    psi_mps = state_mps(psi_vec, N, bond_dim=256)
    
    V_up = np.eye(N, dtype=complex)
    V_down = np.eye(N, dtype=complex)
    
    mz_sergio = []
    active_orbitals_tracker = []

    print("Starting SERGIO Execution...")
    for n in range(N_f):
        # Extract Magnetization via MPS
        imp_q = N
        T_imp = psi_mps.tensors[imp_q]
        mz = np.trace(np.einsum('v i w, v j w -> i j', T_imp.conj(), T_imp) @ sigma_z).real
        mz_sergio.append(mz)
        
        psi_mps, V_up, V_down, n_act_u, n_act_d = sergio_step(
            psi_mps, V_up, V_down, n, Jk, Jz, h, T, tol
        )
        active_orbitals_tracker.append((n_act_u, n_act_d))
        print(f"Step {n}: mz = {mz:.4f}, Active Orbs = (Up: {n_act_u}, Down: {n_act_d})")

    # Run Exact Benchmark
    mz_exact = exact_evolution_benchmark(N, Jk, Jz, h, T, N_f)
    
    # Plotting comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(N_f), mz_sergio, label='SERGIO MPS', marker='o')
    plt.plot(range(N_f), mz_exact, label='Exact $U_F^n$', linestyle='--')
    plt.xlabel('Floquet Steps ($N_s$)')
    plt.ylabel('Magnetization $\langle S_z \rangle$')
    plt.title('Benchmark: Magnetization Dynamics')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    active_u = [x[0] for x in active_orbitals_tracker]
    plt.plot(range(N_f), active_u, label='Active Orbitals (Up)', color='red')
    plt.xlabel('Floquet Steps ($N_s$)')
    plt.ylabel('Count')
    plt.title('Active Orbital Saturation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sergio()