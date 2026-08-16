# ==========================================
# Step 1 : Import all the libraries
# ==========================================

import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import combinations
import warnings

# Qiskit specific imports
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, SparsePauliOp

# Suppress minor Qiskit deprecation warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# Step 2 : Construction of the Gates           
# ==========================================

def fsim_theta(theta):
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    mat = np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    return sp.csr_matrix(mat)

def build_UK(theta_K, theta_z):
    mat = np.eye(8, dtype=np.complex128)
    phase_plus = np.exp(1j * theta_z / 2)
    phase_minus = np.exp(-1j * theta_z / 2)
    c_K = np.cos(theta_K)
    s_K = 1j * np.sin(theta_K)
    
    mat[2, 2] = phase_plus * c_K
    mat[2, 5] = phase_plus * s_K
    mat[5, 2] = phase_plus * s_K
    mat[5, 5] = phase_plus * c_K
    mat[3, 3] = phase_minus
    mat[4, 4] = phase_minus
    return sp.csr_matrix(mat)

# ==============================================
# Step 3 : Efficient Fermi Sea & State Generator
# ==============================================

def get_fermi_sea_amplitudes(M, N_f, boundary='periodic', selection='symmetric'): ##################################
    """
    Slater Determinant generator for the Fermi Sea.

    Parameters
    ----------
    M : int
        Number of sites in one spin block.
    N_f : int
        Number of spinless fermions.
    boundary : str
        'open' → sine waves (open BC),
        'periodic' → plane waves (PBC).
    selection : str
        For PBC only:
        'energy_sorted' → fill lowest single-particle energies (default).
        'symmetric' → choose k-set as in the recursive test code (array_k1 logic).
                      This may break degeneracies by picking negative k first.
    """
    if boundary == 'open':
        H_sp = np.zeros((M, M))
        for i in range(M - 1):
            H_sp[i, i+1] = -1.0
            H_sp[i+1, i] = -1.0
        energies, vecs = np.linalg.eigh(H_sp)
        P = vecs[:, :N_f]
    elif boundary == 'periodic':
        if selection == 'symmetric':
            # Replicate array_k1 logic exactly
            m_val = M / 2.0
            if m_val % 2 != 0:  # m_val is not integer when M is odd, but we check float modulo
                j_vals = np.arange(-int(m_val // 2), int(m_val // 2) + 1)
            else:
                j_vals = np.arange(-int(m_val // 2), int(m_val // 2))
            ks_occ = 2.0 * np.pi * j_vals / M
            # Now we have a list of ks, but we need exactly N_f of them.
            # The recursive code uses the first N_f entries (in the order generated).
            # For the standard half-filling case N_f = M//2, the length of j_vals equals N_f.
            if len(ks_occ) > N_f:
                ks_occ = ks_occ[:N_f]   # take the first N_f (likely the more negative ones)
            # Build plane-wave matrix
            x = np.arange(M)
            P = np.exp(1j * np.outer(x, ks_occ))
        else:  # energy_sorted
            ks = 2 * np.pi * np.arange(M) / M
            energies = -2 * np.cos(ks)
            idx_sort = np.argsort(energies)
            occupied_idx = idx_sort[:N_f]
            ks_occ = ks[occupied_idx]
            x = np.arange(M)
            P = np.exp(1j * np.outer(x, ks_occ))
    else:
        raise ValueError("boundary must be 'open' or 'periodic'")

    amplitudes = {}
    for occupied_sites in combinations(range(M), N_f):
        submatrix = P[list(occupied_sites), :]
        amp = np.linalg.det(submatrix)
        if np.abs(amp) > 1e-12:
            bit_integer = sum(2**s for s in occupied_sites)
            amplitudes[bit_integer] = amp + 0.0j

    norm = np.sqrt(sum(np.abs(v)**2 for v in amplitudes.values()))
    for k in amplitudes:
        amplitudes[k] /= norm
    return amplitudes


def get_initial_state(N, start_site=1, boundary='open', selection='energy_sorted',
                      flip_bath=False, reverse_up=False):
    """
    Build the initial state for the unfolded Kondo chain.

    Parameters
    ----------
    N : int
        Number of bath sites per spin.
    start_site : int (1 or 2)
        If 2, project out site 1 (nearest neighbour) from the Fermi sea.
    boundary : str
        'open' or 'periodic'.
    selection : str
        For PBC: 'energy_sorted' or 'symmetric'.
    flip_bath : bool
        If True, apply Pauli X on all bath qubits (particle‑hole transform),
        matching the recursive code's initial state.
    reverse_up : bool
        If True, reverse the bit order of the up (right) block, so that
        site 0 → qubit N-1 (adjacent to impurity), site N-1 → qubit 0.
        This matches the recursive code's qubit assignment.

    Returns
    -------
    ms : np.ndarray
        Statevector of length 2**L, L = 2*N + 1.
    """
    L = 2 * N + 1
    dim = 2**L
    ms = np.zeros(dim, dtype=np.complex128)

    N_f = N // 2
    full_sea = get_fermi_sea_amplitudes(N, N_f, boundary=boundary, selection=selection)

    if start_site == 2:
        projected = {}
        for bits, amp in full_sea.items():
            if not (bits & 1):
                projected[bits] = amp
        norm = np.sqrt(sum(abs(v)**2 for v in projected.values()))
        if norm == 0:
            raise ValueError("Projected state has zero norm.")
        for k in projected:
            projected[k] /= norm
        sea_amplitudes = projected
    else:
        sea_amplitudes = full_sea

    # Helper to reverse bits within N bits
    def reverse_bits(x, n_bits):
        rev = 0
        for i in range(n_bits):
            if (x >> i) & 1:
                rev |= (1 << (n_bits - 1 - i))
        return rev

    center_index = 1 << N               # impurity bit
    shift = L - N                        # N + 1, start of down block

    for down_bits, down_amp in sea_amplitudes.items():
        for up_bits, up_amp in sea_amplitudes.items():
            total_amp = down_amp * up_amp

            # Place down (left) block: identical to before
            shifted_down = down_bits << shift

            # Place up (right) block:
            if reverse_up:
                # Reverse bit order so that site 0 -> qubit N-1
                shifted_up = reverse_bits(up_bits, N)
            else:
                shifted_up = up_bits   # original mapping

            target = shifted_down + center_index + shifted_up
            ms[target] = total_amp

    # Apply bath flip if requested (same as recursive qc.x on all bath qubits)
    if flip_bath:
        # Flip bits of all bath qubits (all except impurity qubit N)
        ms_flipped = np.zeros_like(ms)
        for idx, amp in enumerate(ms):
            if amp == 0:
                continue
            # Flip bath bits: invert all bits except bit N
            bath_mask = (1 << L) - 1 - (1 << N)   # all bits except N
            flipped_idx = idx ^ bath_mask         # XOR flips bath bits
            ms_flipped[flipped_idx] = amp
        ms = ms_flipped

    return ms

# ==========================================
# Step 4: Qiskit Circuit Architecture
# ==========================================

def build_qiskit_U_kin(N, theta_kin):
    L = 2 * N + 1
    qc = QuantumCircuit(L)
    
    fsim_matrix = fsim_theta(theta_kin).toarray()
    fsim_gate = UnitaryGate(fsim_matrix, label="fSim")

    # Layer 1: Even-Odd pairs
    for j in range(0, L - 1, 2):
        if j == N - 1 or j == N:
            continue # Do not bridge the impurity
        qc.append(fsim_gate, [j, j+1])
        
    # Layer 2: Odd-Even pairs
    for j in range(1, L - 1, 2):
        if j == N - 1 or j == N:
            continue
        qc.append(fsim_gate, [j, j+1])
        
    return qc

def build_qiskit_U_F(N, theta_kin, theta_K, theta_z):
    L = 2 * N + 1
    qc_kin = build_qiskit_U_kin(N, theta_kin)
    
    qc_K = QuantumCircuit(L)
    UK_matrix = build_UK(theta_K, theta_z).toarray()
    UK_gate = UnitaryGate(UK_matrix, label="U_K")
    qc_K.append(UK_gate, [N - 1, N, N + 1])
    
    return qc_kin.compose(qc_K)

def build_kinetic_energy_observable(N):
    r"""
    Constructs the Qiskit SparsePauliOp for the free fermion kinetic energy:
    H_kin = -t \sum (c^\dagger_j c_{j+1} + h.c.)
    """
    L = 2 * N + 1
    pauli_strings = []
    coeffs = []
    
    # Standard hopping amplitude t = 1.0
    t_hop = 1.0 
    
    for j in range(L - 1):
        # We strictly skip the impurity connections, identical to U_kin
        if j == N - 1 or j == N:
            continue
            
        # X_j X_{j+1} term
        x_str = ['I'] * L
        x_str[j] = 'X'
        x_str[j+1] = 'X'
        pauli_strings.append("".join(x_str))
        coeffs.append(-0.5 * t_hop)
        
        # Y_j Y_{j+1} term
        y_str = ['I'] * L
        y_str[j] = 'Y'
        y_str[j+1] = 'Y'
        pauli_strings.append("".join(y_str))
        coeffs.append(-0.5 * t_hop)
        
    return SparsePauliOp(pauli_strings, coeffs)

# ==========================================
# Step 5: Statevector Time Evolution
# ==========================================

def run_qiskit_simulation(N, steps, theta_kin, theta_K, theta_z,
                          boundary='periodic', start_site=1, selection='symmetric', flip_bath=False, reverse_up=False):
    L = 2 * N + 1
    print(f"Building Qiskit Circuit for N={N} ({L} qubits)...")
    qc_F = build_qiskit_U_F(N, theta_kin, theta_K, theta_z)

    # ... Z_imp and H_kin_obs definitions unchanged ...
        # Define the Z observable exactly on the Impurity (Index N)
    z_string = ['I'] * L
    z_string[N] = 'Z'
    z_string = "".join(z_string)
    Z_imp = SparsePauliOp(z_string)
    
    H_kin_obs = build_kinetic_energy_observable(N)
    print("Initializing Ground State Fermi Sea ...")
    initial_array = get_initial_state(N, start_site=1, boundary='periodic',
                                  selection='symmetric',
                                  flip_bath=True, reverse_up=True)  # ← pass selection

    sv = Statevector(initial_array)
    sv = sv.reverse_qargs()

    trajectory_mz = []
    trajectory_heat = []
    print("Beginning Floquet Evolution...")
    for step in range(steps):
        mz = sv.expectation_value(Z_imp).real
        heat = sv.expectation_value(H_kin_obs).real
        trajectory_mz.append(mz)
        trajectory_heat.append(heat)               
        if step % 10 == 0:
            print(f"  Step {step}/{steps} completed. mz = {mz:.4f}")
        sv = sv.evolve(qc_F)

    return trajectory_mz, trajectory_heat

# ==========================================
# Step 6: Execution & Plotting
# ==========================================

if __name__ == "__main__":
    # We strictly use ODD values of N to preserve particle-hole symmetry
    # Default values
    N_system = 5                                                         ###############################################################################
    theta = np.pi / 3
    theta_k = np.pi / 4
    t_steps = 100

    # Override if command-line arguments are provided
    if len(sys.argv) > 1:
        N_system = int(sys.argv[1])
        theta = float(sys.argv[2])
        theta_k = float(sys.argv[3])
        t_steps = int(sys.argv[4])

    # In the isotropic case, theta_z = theta_k
    theta_z = theta_k 

    print(f"Starting Qiskit Simulation for N={N_system} (Total Qubits: {2*N_system + 1})...")
    
    # Run the simulation and unpack BOTH observables
    mz_data, heat_data = run_qiskit_simulation(
    N_system, t_steps, theta, theta_k, theta_z,
    boundary='periodic', start_site=1, selection='symmetric'
) ###########################
    
    print("\nSimulation Complete. Generating Dual-Panel Plot...")

    # Calculate Delta E (Absorbed Floquet Heating)
    heat_array = np.array(heat_data)
    delta_heat = heat_array - heat_array[0]

    # Generate the Publication-Quality Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Panel: Magnetization
    ax1.plot(range(t_steps), mz_data, marker='o', markersize=3, linestyle='-', color='darkblue', label=r'$\langle Z_{imp} \rangle$')
    ax1.set_title(f"Floquet Dynamics of the 1CK Model (N={N_system})", fontsize=14)
    ax1.set_ylabel(r"Magnetization $\langle Z_{imp} \rangle$", fontsize=12)
    ax1.axhline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc="upper right")

    # Bottom Panel: Floquet Heating
    ax2.plot(range(t_steps), delta_heat, marker='s', markersize=3, linestyle='-', color='darkred', label=r'$\Delta E(t)$')
    ax2.set_xlabel("Floquet Step ($N_s$)", fontsize=12)
    ax2.set_ylabel(r"Absorbed Energy $\Delta E$", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()