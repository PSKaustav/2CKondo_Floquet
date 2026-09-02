import numpy as np
from helper_functions_gem import *

def run_occupancy_verification():
    # =========================================================================
    # 1. Construct the 5-Qubit Entangled Core State
    # =========================================================================
    # Layout: [Active_Up_0, Active_Up_1, Impurity, Active_Down_0, Active_Down_1]
    # State: sqrt(0.5)|10>_u |0>_imp |10>_d 
    #      + sqrt(0.1)|01>_u |0>_imp |10>_d 
    #      + sqrt(0.4)|01>_u |1>_imp |01>_d
    #
    # Binary Mapping: 
    # |10010> -> 16 + 2 = 18
    # |01010> -> 8 + 2 = 10
    # |01101> -> 8 + 4 + 1 = 13
    
    psi = np.zeros(32, dtype=complex)
    psi[18] = np.sqrt(0.5) 
    psi[10] = np.sqrt(0.1) 
    psi[13] = np.sqrt(0.4) 
    
    # Natively encode into the qlimb MPS class with a maximum bond dimension
    psi_mps = state_mps(psi, N=2, bond_dim=256)
    
    # =========================================================================
    # 2. Extract Occupancies via build_1rdm_bath
    # =========================================================================
    imp_idx = 2
    up_qubits = [0, 1]
    down_qubits = [3, 4]
    
    # Use the helper function to contract the MPS with the required Jordan-Wigner 
    # strings and Number operators to evaluate the 1-body Reduced Density Matrix.
    C_up = build_1rdm_bath(psi_mps, bath_qubits=up_qubits, imp_qubit=imp_idx)
    C_down = build_1rdm_bath(psi_mps, bath_qubits=down_qubits, imp_qubit=imp_idx)
    
    # The diagonal elements of the 1RDM perfectly yield the occupation numbers
    occupancies_up = np.real(np.diag(C_up))
    occupancies_down = np.real(np.diag(C_down))
    
    print("=== MPS Occupancy Verification ===")
    print(f"Target Up Occupancies:   [0.5, 0.5]")
    print(f"Measured Up Occupancies: {np.round(occupancies_up, 3)}")
    
    print(f"\nTarget Down Occupancies:   [0.6, 0.4]")
    print(f"Measured Down Occupancies: {np.round(occupancies_down, 3)}")
    
    # Assertions to mathematically verify correctness
    assert np.allclose(occupancies_up, [0.5, 0.5]), "Up chain occupancies failed!"
    assert np.allclose(occupancies_down, [0.6, 0.4]), "Down chain occupancies failed!"
    print("\n[Success]: The build_1rdm_bath helper accurately evaluates the 1RDM.")


def run_native_qlimb_9qubit_example():
    # =========================================================================
    # 1. Constructing the 5-Qubit Entangled Core (N=2)
    # =========================================================================
    # Target Active Core Layout: [Active_Up_0, Active_Up_1, Impurity, Active_Down_0, Active_Down_1]
    # State: sqrt(0.5)|10>_u |0>_imp |10>_d 
    #      + sqrt(0.1)|01>_u |0>_imp |10>_d 
    #      + sqrt(0.4)|01>_u |1>_imp |01>_d
    #
    # Binary Mapping: 
    # |10010> -> 16 + 2 = 18
    # |01010> -> 8 + 2 = 10
    # |01101> -> 8 + 4 + 1 = 13
    
    psi = np.zeros(32, dtype=complex)
    psi[18] = np.sqrt(0.5) 
    psi[10] = np.sqrt(0.1) 
    psi[13] = np.sqrt(0.4) 
    
    # Natively encode the 5-qubit core into the qlimb MPS class
    psi_mps = state_mps(psi, N=2, bond_dim=256)
    
    # Manually re-tag the active core for visual clarity
    psi_mps.tags = ["Active_Up_0", "Active_Up_1", "Impurity", "Active_Down_0", "Active_Down_1"]
    
    print("=== 1. Active 5-Qubit Core MPS Constructed ===")
    for i, T in enumerate(psi_mps.tensors):
        print(f"Tensor {i:<15} '{psi_mps.tags[i]:<15}' Shape (chi_L, phys_dim, chi_R): {T.shape}")
        
    print("\n[Observation]: The internal virtual bonds connecting the up and down active ")
    print("orbitals to the impurity all exhibit a non-trivial bond dimension of chi=2.")

    # =========================================================================
    # 2. Augmenting to 9 Qubits (Injecting Trivial Edges)
    # =========================================================================
    # The rank-1 coupling isolates exactly one empty and one full orbital per step.
    # We append them to the far boundaries so we do not sever the active entanglement.
    
    # A. Up-spin inactive modes appended to the FAR LEFT
    # Target: [Filled_Up] - [Empty_Up] - [Active Core...]
    psi_mps = augment_mps_safe(psi_mps, insert_idx=0, state_type=1) 
    psi_mps.tags[0] = "Filled_Up"
    
    psi_mps = augment_mps_safe(psi_mps, insert_idx=1, state_type=0) 
    psi_mps.tags[1] = "Empty_Up"
    
    # B. Down-spin inactive modes appended to the FAR RIGHT
    # Target: [...Active Core] - [Empty_Down] - [Filled_Down]
    end_idx = psi_mps.nqbits
    psi_mps = augment_mps_safe(psi_mps, insert_idx=end_idx, state_type=0) 
    psi_mps.tags[end_idx] = "Empty_Down"
    
    psi_mps = augment_mps_safe(psi_mps, insert_idx=end_idx + 1, state_type=1) 
    psi_mps.tags[end_idx + 1] = "Filled_Down"
    
    print("\n=== 2. Augmented 9-Qubit MPS ===")
    for i, T in enumerate(psi_mps.tensors):
        print(f"Tensor {i:<15} '{psi_mps.tags[i]:<15}' Shape (chi_L, phys_dim, chi_R): {T.shape}")
        
    print("\n[Observation]: The newly augmented tensors at the boundaries strictly possess")
    print("a trivial bond dimension of chi=1. The active bulk's chi=2 entanglement remains perfectly intact.")

if __name__ == "__main__":
    run_occupancy_verification()
    run_native_qlimb_9qubit_example()