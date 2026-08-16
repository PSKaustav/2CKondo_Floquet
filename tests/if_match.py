import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import combinations
import warnings

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, SparsePauliOp

warnings.filterwarnings("ignore")

def get_fermi_sea_amplitudes(N):
    m = N / 2
    array_k = []
    if m % 2 != 0:
        for j in range(-int(m//2), int(m//2)+1):
            array_k.append(2*np.pi*j/N)
    else:
        for j in range(-int(m//2), int(m//2)):
            array_k.append(2*np.pi*j/N)

    N_f = len(array_k)
    coeff_array = np.array([
        [np.exp(-1j*k*x) for x in range(N)]
        for k in array_k
    ])

    amplitudes = {}
    for occupied_sites in combinations(range(N), N_f):
        submatrix = coeff_array[:, list(occupied_sites)]
        amp = np.linalg.det(submatrix)
        if np.abs(amp) > 1e-12:
            bit_integer = sum(2**s for s in occupied_sites)
            amplitudes[bit_integer] = amp + 0.0j

    norm = np.sqrt(sum(np.abs(v)**2 for v in amplitudes.values()))
    for k in amplitudes:
        amplitudes[k] /= norm
    return amplitudes

def get_initial_state(N, empty_adjacent=False):
    L = 2 * N + 1
    ms = np.zeros(2**L, dtype=np.complex128)

    full_sea = get_fermi_sea_amplitudes(N)

    if empty_adjacent:
        # Project out configurations where site 0 (adjacent to impurity) is occupied
        projected = {bits: amp for bits, amp in full_sea.items() if not (bits & 1)}
        norm = np.sqrt(sum(abs(v)**2 for v in projected.values()))
        for k in projected: projected[k] /= norm
        sea_amplitudes = projected
    else:
        sea_amplitudes = full_sea

    center_index = 1 << N
    shift = L - N

    for down_bits, down_amp in sea_amplitudes.items():
        for up_bits, up_amp in sea_amplitudes.items():
            shifted_down = down_bits << shift
            target = shifted_down + center_index + up_bits
            ms[target] = down_amp * up_amp

    return ms

def fsim_matrix(theta):
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return np.array([[1,0,0,0],[0,c,s,0],[0,s,c,0],[0,0,0,1]], dtype=np.complex128)

def build_UK(theta_K, theta_z):
    """Sarma's kondo_unitary_2, applied to [imp, up_bath, down_bath]."""
    l1 = np.exp(1j * theta_z / 2)
    l2 = np.exp(-1j * theta_z / 2)
    c1 = np.cos(theta_K)
    s1 = np.sin(theta_K)
    mat = np.eye(8, dtype=np.complex128)
    mat[2, 2] = c1 * l1
    mat[2, 5] = 1j * s1 * l1
    mat[5, 2] = 1j * s1 * l1
    mat[5, 5] = c1 * l1
    mat[3, 3] = l2
    mat[4, 4] = l2
    return mat

def build_kin_layer(N, theta_even, theta_odd):
    L = 2 * N + 1
    qc = QuantumCircuit(L)
    gate_e = UnitaryGate(fsim_matrix(theta_even), label=f"fSim_e")
    gate_o = UnitaryGate(fsim_matrix(theta_odd), label=f"fSim_o")
    for j in range(0, L-1, 2):
        if j == N-1 or j == N: continue
        qc.append(gate_e, [j, j+1])
    for j in range(1, L-1, 2):
        if j == N-1 or j == N: continue
        qc.append(gate_o, [j, j+1])
    return qc

def build_floquet_operator(N, theta, theta_K, theta_z): #use this to get the correct magnetisation plot
    """
    Single fixed Floquet period (for sequential evolution):
    half_kin(theta, 2*theta) -> UK -> inv_half_kin(theta, even only)
    """
    L = 2 * N + 1

    # half step: even=theta, odd=2*theta
    qc = build_kin_layer(N, theta, 2*theta)

    # UK on [imp, up_bath, down_bath] = [N, N+1, N-1]
    qc_UK = QuantumCircuit(L)
    qc_UK.append(UnitaryGate(build_UK(theta_K, theta_z), label="U_K"), [N, N+1, N-1])
    qc = qc.compose(qc_UK)

    # inv half: even layer only with theta
    qc_inv = QuantumCircuit(L)
    gate_e = UnitaryGate(fsim_matrix(theta), label="fSim_inv")
    for j in range(0, L-1, 2):
        if j == N-1 or j == N: continue
        qc_inv.append(gate_e, [j, j+1])
    qc = qc.compose(qc_inv)

    return qc

def build_kinetic_energy_observable(N):

    L = 2 * N + 1

    pauli_strings = []
    coeffs = []

    t_hop = 1.0

    for j in range(L - 1):

        if j == N - 1 or j == N:
            continue

        # XX term
        x_str = ['I'] * L
        x_str[j] = 'X'
        x_str[j+1] = 'X'

        pauli_strings.append("".join(x_str))
        coeffs.append(-0.5 * t_hop)

        # YY term
        y_str = ['I'] * L
        y_str[j] = 'Y'
        y_str[j+1] = 'Y'

        pauli_strings.append("".join(y_str))
        coeffs.append(-0.5 * t_hop)

    return SparsePauliOp(pauli_strings, coeffs)

def run_simulation(N, steps, theta, theta_K, theta_z, empty_adjacent=False):
    L = 2 * N + 1
    print(f"N={N}, L={L}, steps={steps}, theta_z={theta_z:.6f}, empty_adjacent={empty_adjacent}")

    qc_F = build_floquet_operator(N, theta, theta_K, theta_z)

    sv = Statevector(get_initial_state(N, empty_adjacent=empty_adjacent)).reverse_qargs()

    z_string = ['I']*L; z_string[N] = 'Z'
    Z_imp = SparsePauliOp("".join(z_string))
    H_kin_obs = build_kinetic_energy_observable(N)

    trajectory_mz, trajectory_heat = [], []

    for step in range(steps):
        trajectory_mz.append(sv.expectation_value(Z_imp).real)
        trajectory_heat.append(sv.expectation_value(H_kin_obs).real)
        if True:
            print(f"Step {step}/{steps} | mz = {trajectory_mz[-1]:.5f}")
        sv = sv.evolve(qc_F)

    return trajectory_mz, trajectory_heat

if __name__ == "__main__":

    N_system = 6
    theta = np.pi / 3
    theta_k = np.pi / 4
    t_steps = 400

    if len(sys.argv) > 1:
        N_system = int(sys.argv[1])
        theta = float(sys.argv[2])
        theta_k = float(sys.argv[3])
        t_steps = int(sys.argv[4])

    theta_z = 0.5 * np.sqrt(2) * (np.sqrt(2) - 1) * np.sin(theta)

    print(f"\nFloquet 1CK | N={N_system}, theta={theta:.4f}, theta_k={theta_k:.4f}, theta_z={theta_z:.6f}\n")

    mz_data, heat_data = run_simulation(
        N_system, t_steps, theta, theta_k, theta_z,
        empty_adjacent=False  # set True to start with site adjacent to impurity empty
    )

    heat_array = np.array(heat_data)
    delta_heat = heat_array - heat_array[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(range(t_steps), mz_data, marker='o', markersize=2,
             linestyle='-', color='darkblue', label=r'$\langle Z_{imp} \rangle$')
    ax1.set_title(f"Floquet 1CK (N={N_system})", fontsize=14)
    ax1.set_ylabel(r"Magnetization $\langle Z_{imp} \rangle$", fontsize=12)
    ax1.axhline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(range(t_steps), delta_heat, marker='s', markersize=2,
             linestyle='-', color='darkred', label=r'$\Delta E(t)$')
    ax2.set_xlabel(r"Floquet Step ($N_s$)", fontsize=12)
    ax2.set_ylabel(r"Absorbed Energy $\Delta E$", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()