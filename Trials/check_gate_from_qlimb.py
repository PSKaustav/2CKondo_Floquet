import numpy as np 
from qlimb.classical.gates import Gate
from qlimb.classical.mps import MPS

phys_dim = 2
nqbits = 6

# 1. Create a RANDOM quantum state (normalized)
# This tests the gate against all possible states simultaneously.
rng = np.random.default_rng(42)
vec_in = rng.random(phys_dim**nqbits) + 1j * rng.random(phys_dim**nqbits)
vec_in /= np.linalg.norm(vec_in)

mps_in = MPS.from_vec(vec_in, bond_dim=phys_dim**nqbits, phys_dim=phys_dim, trunc_tol=1e-12, nqbits=nqbits)

# 2. Define the properly padded gate matrix
gate_matrix = np.zeros((4, 4), dtype=complex)
gate_matrix[0,0] = 1.0  
gate_matrix[1,1] = 1.0  
gate_matrix[2,2] = 1.0
gate_matrix[3,3] = 1.0
gate_matrix[2,3] = 1.0
gate_matrix[3,2] = 1.0

# 3. Apply the gates with different orderings
gate_34 = Gate(gate_matrix, [3, 4])
mps_out_34 = gate_34 @ mps_in

gate_43 = Gate(gate_matrix, [4, 3])
mps_out_43 = gate_43 @ mps_in

# 4. Compare the resulting vectors
a = mps_out_34.to_vec()
b = mps_out_43.to_vec()

# This will print False, proving that order matters!
print("Are [3,4] and [4,3] the same operation? :", np.allclose(a, b))
