import numpy as np
from helper_functions_gem import *

class MockMPS:
    """
    A minimal mock MPS class simulating the qlimb MPS interface 
    to demonstrate the array manipulations independently.
    """
    def __init__(self):
        # Initialize a simple 2-site MPS with bond dimension 1
        self.nqbits = 2
        self.tensors = [
            np.ones((1, 2, 1), dtype=complex) / np.sqrt(2),
            np.ones((1, 2, 1), dtype=complex) / np.sqrt(2)
        ]
        self.tags = ["Active_0", "Active_1"]
        self.center_idx = 0

    def move_center(self, idx):
        self.center_idx = idx

    def norm(self):
        # Mock normalization
        return 1.0

def run_examples():
    # Setup
    psi = MockMPS()
    print(f"--- Initial State ---")
    print(f"Qubits: {psi.nqbits} | Tags: {psi.tags}\n")

    # =========================================================================
    # Example 1: augment_mps_safe
    # =========================================================================
    # We insert a "filled" orbital (state |1>) at the very beginning (index 0).
    # This prepares the MPS to include the localized f_{0} orbital.
    
    insert_index = 0
    state_type = 1 
    psi = augment_mps_safe(psi, insert_index, state_type)
    
    print(f"--- After Augmentation (Inserting |1> at index 0) ---")
    print(f"Qubits: {psi.nqbits} | Tags: {psi.tags}")
    print(f"New Tensor Shape: {psi.tensors[0].shape}")
    print(f"Amplitude for |0>: {psi.tensors[0][0, 0, 0].real}")
    print(f"Amplitude for |1>: {psi.tensors[0][0, 1, 0].real}\n")

    # =========================================================================
    # Example 2: truncate_mps_safe
    # =========================================================================
    # After diagonalizing the 1RDM, suppose orbital 0 is classified as "filled".
    # We slice out the |1> amplitude and absorb it into the adjacent tensor to 
    # compress the active MPS window back down.
    
    # Format: [(index_to_remove, known_state)]
    indices_to_remove = [(0, 1)] 
    psi = truncate_mps_safe(psi, indices_to_remove)
    
    print(f"--- After Truncation (Removing orbital 0, projecting |1>) ---")
    print(f"Qubits: {psi.nqbits} | Tags: {psi.tags}")
    print(f"First Tensor Shape (Absorbed): {psi.tensors[0].shape}")

if __name__ == "__main__":
    run_examples()