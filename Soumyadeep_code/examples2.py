import numpy as np
from helper_functions_gem import truncate_mps_safe

class MockMPS:
    """
    A minimal mock MPS class replicating qlimb functionality to 
    demonstrate the exact array manipulations.
    """
    def __init__(self, tensors, tags):
        self.tensors = tensors
        self.nqbits = len(tensors)
        self.tags = tags
        self.center_idx = 0

    def move_center(self, idx):
        self.center_idx = idx

    def norm(self):
        return np.linalg.norm(self.tensors[self.center_idx])

def vector_to_mps(psi_vec, N):
    """
    Converts a 2^N statevector into an MPS via sequential SVDs.
    This demonstrates exactly how amplitudes are encoded into the 
    (chi_left, phys_dim, chi_right) tensor shapes[cite: 8].
    """
    tensors = []
    chi_left = 1
    T = psi_vec.reshape([2] * N)
    #print(f"Initial statevector {T} reshaped to tensor of shape: {T.shape}")
    
    for i in range(N - 1):
        # Reshape for SVD: (chi_left * phys_dim, remaining_dims)
        T_mat = T.reshape(chi_left * 2, -1)
        U, S, Vh = np.linalg.svd(T_mat, full_matrices=False)
        
        # Filter non-zero singular values to dynamically compress the bond dimension
        tol = 1e-10
        keep = S > tol
        U, S, Vh = U[:, keep], S[keep], Vh[keep, :]
        
        chi_right = len(S)
        
        # The MPS tensor for site i[cite: 8]
        tensors.append(U.reshape(chi_left, 2, chi_right))
        
        # Contract the singular values into the remaining subsystem
        T = (np.diag(S) @ Vh).reshape([chi_right] + [2] * (N - i - 1))
        chi_left = chi_right
        
    # Append the final site tensor
    tensors.append(T.reshape(chi_left, 2, 1))
    
    return MockMPS(tensors, [f"Orb_{i}" for i in range(N)])

def run_nontrivial_example():
    # =========================================================================
    # 1. State Construction & MPS Encoding
    # =========================================================================
    # Target Occupancies: (1.0, 0.5, 0.5, 0.0)
    # State: 1/sqrt(2) * ( |1100> + |1010> )
    psi = np.zeros(16, dtype=complex)
    psi[12] = 1 / np.sqrt(2) # |1100> in binary is 12
    psi[10] = 1 / np.sqrt(2) # |1010> in binary is 10
    
    # Encode into MPS via sequential SVD[cite: 8]
    psi_mps = vector_to_mps(psi, N=4)
    print(' Entangled state MPS:' , psi_mps.tensors)
    
    print("=== 1. Initial MPS Representation ===")
    for i, T in enumerate(psi_mps.tensors):
        print(f"Tensor {i} '{psi_mps.tags[i]}' shape (chi_L, phys, chi_R): {T.shape}")
        
    # Notice the shapes:
    # Orb_0: (1, 2, 1) -> Bond dimension 1 confirms it's unentangled[cite: 8].
    # Orb_1: (1, 2, 2) -> Bond dimension jumps to 2 (entangled Bell pair).
    # Orb_2: (2, 2, 1) -> Bond dimension drops back to 1.
    # Orb_3: (1, 2, 1) -> Unentangled empty state.

    # =========================================================================
    # 2. Verify Occupancies
    # =========================================================================
    print("\n=== 2. Natural Orbital Occupancies ===")
    print("By construction, the diagonal elements of the 1RDM are:")
    print("Orbital 0: 1.0 (Filled)")
    print("Orbital 1: 0.5 (Active)")
    print("Orbital 2: 0.5 (Active)")
    print("Orbital 3: 0.0 (Empty)")

    # =========================================================================
    # 3. Truncation & Amplitude Absorption
    # =========================================================================
    # We remove site 0 (state |1>) and site 3 (state |0>)[cite: 5].
    print("\n=== 3. Truncation (Absorbing Trivial Amplitudes) ===")
    
    # Format: (MPS_index, physical_state_to_project)
    indices_to_remove = [(0, 1), (3, 0)] 
    
    # truncate_mps_safe sorts indices in reverse to safely pop from right to left[cite: 5].
    psi_mps = truncate_mps_safe(psi_mps, indices_to_remove)
    
    print("After truncation:")
    for i, T in enumerate(psi_mps.tensors):
        print(f"Remaining Tensor {i} '{psi_mps.tags[i]}' shape: {T.shape}")
        
    # =========================================================================
    # 4. Exposing the Math of the Absorption
    # =========================================================================
    print("\n=== 4. How Amplitudes were Absorbed ===")
    print("When Orb_0 was truncated, the code extracted the slice M = T[:, 1, :].")
    print("Because Orb_0 was perfectly unentangled, M was exactly [[1.0]].")
    print("This 1x1 scalar matrix was contracted via np.tensordot into Orb_1[cite: 5].")
    print("\nAmplitudes left inside the new first tensor (formerly Orb_1):")
    print(f"Physical |0> slice:\n{psi_mps.tensors[0][:, 0, :]}")
    print(f"Physical |1> slice:\n{psi_mps.tensors[0][:, 1, :]}")

if __name__ == "__main__":
    run_nontrivial_example()