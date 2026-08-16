import numpy as np
from itertools import combinations, cycle

# =====================================================================
# METHOD A: The Optimized Slater Determinant Approach
# =====================================================================
def get_fermi_sea_slater(M, N_f):
    H_sp = np.zeros((M, M))
    for i in range(M - 1):
        H_sp[i, i+1] = -1.0
        H_sp[i+1, i] = -1.0
    energies, vecs = np.linalg.eigh(H_sp)
    P = vecs[:, :N_f]
    
    amplitudes = {}
    for occupied_sites in combinations(range(M), N_f):
        amp = np.linalg.det(P[list(occupied_sites), :])
        if np.abs(amp) > 1e-12:
            bit_integer = sum(2**s for s in occupied_sites)
            amplitudes[bit_integer] = amp + 0.0j
            
    norm = np.sqrt(sum(np.abs(v)**2 for v in amplitudes.values()))
    for k in amplitudes:
        amplitudes[k] /= norm
    return amplitudes

# =====================================================================
# METHOD B: The User's Recursive Permutation Approach
# =====================================================================
coeff_dict = {} # Global dictionary for the recursive function

def sort_bitstr(bitstr):
    bit_array = [int(i) for i in bitstr]
    bit_array.sort()
    return ''.join(map(str, bit_array))

def perm_str2(cmpr, word):
    swaps = 0
    chars = {c: [] for c in word}
    [chars[c].append(i) for i, c in enumerate(word)]
    for k in chars.keys():
        chars[k] = cycle(chars[k])
    idxs = [next(chars[c]) for c in cmpr]
    for cmb in combinations(idxs, 2):
        if cmb[0] > cmb[1]:
            swaps += 1
    return 1 if swaps % 2 == 0 else -1

def recursive_nested(l, M, N_f, coeff_array, coeff=1, bitstr=''):
    global coeff_dict
    if l == N_f - 1:
        for i in range(M):
            if str(i) not in bitstr:
                current_coeff = coeff * coeff_array[l, i]
                current_bitstr = bitstr + str(i)
                bitstr_sorted = sort_bitstr(current_bitstr)
                perm = perm_str2(current_bitstr, bitstr_sorted)
                if bitstr_sorted in coeff_dict.keys():
                    coeff_dict[bitstr_sorted] += current_coeff * perm
                else:
                    coeff_dict[bitstr_sorted] = current_coeff * perm
    else:
        for i in range(M):
            if str(i) not in bitstr:
                current_coeff = coeff * coeff_array[l, i]
                current_bitstr = bitstr + str(i)
                recursive_nested(l + 1, M, N_f, coeff_array, current_coeff, current_bitstr)
    return coeff_dict

def get_fermi_sea_recursive(M, N_f):
    global coeff_dict
    coeff_dict.clear() # Clear state for clean run
    
    H_sp = np.zeros((M, M))
    for i in range(M - 1):
        H_sp[i, i+1] = -1.0
        H_sp[i+1, i] = -1.0
    energies, vecs = np.linalg.eigh(H_sp)
    
    P = vecs[:, :N_f]
    coeff_array = P.T  # Transpose to match the recursive logic indexing
    
    raw_dict = recursive_nested(0, M, N_f, coeff_array)
    
    amplitudes = {}
    for bstr, amp in raw_dict.items():
        if np.abs(amp) > 1e-12:
            bit_integer = sum(2**int(s) for s in bstr)
            amplitudes[bit_integer] = amp + 0.0j
            
    norm = np.sqrt(sum(np.abs(v)**2 for v in amplitudes.values()))
    for k in amplitudes:
        amplitudes[k] /= norm
    return amplitudes

# =====================================================================
# UNIT TEST EXECUTION & COMPARISON
# =====================================================================
if __name__ == "__main__":
    # Parameters for N=6 (meaning Fermi Sea has M=5 sites, filled with 2 fermions)
    M_sites = 4
    N_fermions = 2

    print(f"Running Amplitude Verification for M={M_sites} sites, N_f={N_fermions} fermions...\n")

    slater_amps = get_fermi_sea_slater(M_sites, N_fermions)
    recursive_amps = get_fermi_sea_recursive(M_sites, N_fermions)

    # Format a beautiful comparison table
    print(f"{'Basis State Integer':<20} | {'Slater Determinant Amp':<25} | {'Recursive Amp':<25} | {'Absolute Diff'}")
    print("-" * 90)

    max_diff = 0.0
    all_keys = set(slater_amps.keys()).union(set(recursive_amps.keys()))

    for key in sorted(all_keys):
        val_slater = slater_amps.get(key, 0.0j)
        val_recur  = recursive_amps.get(key, 0.0j)
        
        diff = np.abs(val_slater - val_recur)
        if diff > max_diff:
            max_diff = diff
            
        print(f"{key:<20} | {val_slater.real:>8.5f} + {val_slater.imag:>8.5f}j | {val_recur.real:>8.5f} + {val_recur.imag:>8.5f}j | {diff:.2e}")

    print("-" * 90)
    print(f"\nMAXIMUM ABSOLUTE DIFFERENCE: {max_diff:.3e}")
    
    if max_diff < 1e-10:
        print("\n[VERDICT]: SUCCESS! Both methods yield mathematically identical quantum amplitudes.")
    else:
        print("\n[VERDICT]: FAILURE. The amplitudes differ.")