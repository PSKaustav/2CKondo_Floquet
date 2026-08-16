
from openfermion.linalg.givens_rotations import givens_decomposition
from openfermion.linalg.givens_rotations import givens_matrix_elements
import numpy as np

 
'''A_hand = np.array([[1/2, np.sqrt(3)/2, 0, 0], 
                   [-np.sqrt(3)/2, 1/2, 0, 0], 
                   [0, 0, 1, 0], 
                   [0, 0, 0, 1]], dtype=complex)

B_hand = np.array([[1,0,0,0],
                   [0,1/np.sqrt(3),np.sqrt(2/3),0],
                   [0,-np.sqrt(2/3),1/np.sqrt(3),0],
                   [0,0,0,1]], dtype=complex)

C_hand = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1/np.sqrt(2),1/np.sqrt(2)],
                   [0,0,-1/np.sqrt(2),1/np.sqrt(2)]], dtype=complex)'''



'''A_hand = np.array([

    [2/np.sqrt(6),  1/np.sqrt(3), 0, 0],

    [-1/np.sqrt(3), 2/np.sqrt(6), 0, 0],

    [0, 0, 1, 0],

    [0, 0, 0, 1]

])

B_hand = np.array([

    [1, 0, 0, 0],

    [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],

    [0, -1/np.sqrt(2), 1/np.sqrt(2), 0],

    [0, 0, 0, 1]

])

C_hand = np.array([

    [1,0,0,0],

    [0,1,0,0],

    [0,0,0,1],

    [0,0,-1,0]

])'''

C_hand = np.array([

    [1/np.sqrt(5), -2/np.sqrt(5), 0, 0],

    [2/np.sqrt(5),  1/np.sqrt(5), 0, 0],

    [0,0,1,0],

    [0,0,0,1]

])

B_hand = np.array([

    [1,0,0,0],

    [0,0,-1,0],

    [0,1,0,0],

    [0,0,0,1]

])

A_hand = np.array([

    [1,0,0,0],

    [0,1,0,0],

    [0,0,1/np.sqrt(6),-np.sqrt(5)/np.sqrt(6)],

    [0,0,np.sqrt(5)/np.sqrt(6),1/np.sqrt(6)]

])

Mat_hand = A_hand @ B_hand @ C_hand


vector = np.array([2,1,0,1], dtype=complex)
original_vector = vector.copy()
vector_test = vector.copy()

G_total = np.eye(len(vector), dtype=complex)

'''for i in range(len(vector_test)-1, 0, -1):

    g_small = givens_matrix_elements(vector_test[i-1], vector_test[i], which = 'right')

    G = np.eye(len(vector), dtype=complex)

    G[np.ix_([i - 1, i], [i - 1, i])] = g_small

    G_total = G @ G_total
    vector_test = G @ vector_test'''

for i in range(len(vector_test)-1):

    g_small = givens_matrix_elements(vector_test[i], vector_test[i+1], which = 'left')

    G = np.eye(len(vector), dtype=complex)

    G[np.ix_([i, i+1], [i, i+1])] = g_small

    G_total = G @ G_total
    vector_test = G @ vector_test

max_diff = np.max(np.abs(G_total - Mat_hand))
frob = np.linalg.norm(G_total - Mat_hand)

print(f"Maximum difference: {max_diff:.3e}")
print(f"Frobenius norm: {frob:.3e}")

assert np.allclose(G_total, Mat_hand, atol=1e-12), (
    f"G_total and Mat_hand differ.\n"
    f"Maximum difference = {max_diff:.3e}\n"
    f"Frobenius norm     = {frob:.3e}"
)

print("All tests passed.")


def _reduce_vector_and_unitary(vector, reduce_position="first"):
    v = vector[::-1].copy() if reduce_position == "last" else vector.copy()
    L = len(v)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return vector.copy().astype(complex), np.eye(L, dtype=complex), []
    decomposition, _, _ = givens_decomposition((v / norm).reshape(1, -1))
    Q = np.eye(L, dtype=complex)
    for i in range(L - 1, 0, -1):

        g_small = givens_matrix_elements(v[i - 1], v[i], which='right')

        G = np.eye(L, dtype=complex)

        G[np.ix_([i - 1, i], [i - 1, i])] = g_small

        Q = G @ Q
        v = G @ v
    if reduce_position == "last":
        v = v[::-1]
        P = np.eye(L)[::-1]
        Q = P @ Q @ P
    return v, Q, decomposition

vec_transformed, Q, _ = _reduce_vector_and_unitary(original_vector.copy(), reduce_position="last")

'''assert not np.allclose(vec_transformed, vector, atol=1e-12), (
    f"G_total and Mat_hand differ.\n"
    f"Maximum difference = {max_diff:.3e}\n"
    f"Frobenius norm     = {frob:.3e}"
)

print("All tests passed.")'''

assert np.allclose(Q, Mat_hand, atol=1e-12), (
    f"G_total and Mat_hand differ.\n"
    f"Maximum difference = {max_diff:.3e}\n"
    f"Frobenius norm     = {frob:.3e}"
)

print("All tests passed.")


#------------------------------------------------------------------------------------------------------------
# Takeaways
#------------------------------------------------------------------------------------------------------------

#1. if reduce position = "first", then it should be i-1 -> i in givens_matrix_elements. Range of i is 
#   from len(vector_test)-1 to 0. This is because we are reducing the vector from the last element to the first element.
#2. if reduce position = "last", then it should be i -> i+1 in givens_matrix_elements. Range of i is
#   from 0 to len(vector_test)-1. This is because we are reducing the vector from the first element to the last element.

