import numpy as np

def householder_unitary(v, target_idx=0):
    n = len(v)
    v = np.asarray(v, dtype=complex)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.eye(n, dtype=complex)
    vn = v / norm
    e = np.zeros(n, dtype=complex)
    e[target_idx] = 1.0
    phase = vn[target_idx] / abs(vn[target_idx]) if abs(vn[target_idx]) > 1e-14 else 1.0
    w = vn - phase * e
    wnorm = np.linalg.norm(w)
    if wnorm < 1e-12:
        return np.eye(n, dtype=complex)
    w = w / wnorm
    H = np.eye(n, dtype=complex) - 2.0 * np.outer(w, w.conj())
    return H

H = householder_unitary(np.array([3,4,3,4,3,4,3,4]), target_idx=1)
v_givens = H@np.array([3,4,3,4,3,4,3,4])
print(v_givens)  