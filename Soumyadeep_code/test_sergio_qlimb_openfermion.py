import unittest
import warnings

import numpy as np

import sergio_full_mps as dense
import sergio_mock_backend as mock
import sergio_qlimb_openfermion as sergio


def direct_floquet_with_supplied_convention(N, steps, Jk, Jz, boundary):
    occupied, _, _ = dense.build_fermi_sea_orbitals(N, boundary=boundary)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        state = dense._slater_state(
            N, occupied[::-1, :], occupied, impurity_bit=0
        )
    _, _, free = sergio.diagonalize_h1(
        sergio.build_h1(N, boundary=boundary)
    )
    reverse = np.eye(N)[::-1]
    free_up = reverse @ free @ reverse
    kondo = mock.build_kondo_gate(Jk, Jz, 0.0, 1.0, 0)
    values = []
    for n in range(steps + 1):
        zket = dense._apply_dense_gate(
            state, mock.sigma_z, [N], 2 * N + 1
        )
        values.append(float(np.vdot(state, zket).real))
        if n == steps:
            break
        state = dense._apply_dense_gate(
            state, kondo, [N - 1, N, N + 1], 2 * N + 1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            state = dense._apply_gaussian_dense(
                state, free_up, list(range(N)), 2 * N + 1
            )
            state = dense._apply_gaussian_dense(
                state, free, list(range(N + 1, 2 * N + 1)), 2 * N + 1
            )
    return np.asarray(values)


class SergioQlimbStructureTests(unittest.TestCase):
    def test_fixed_full_mps_and_mirrored_initialization(self):
        data = sergio.initialize_sergio_mps(6, 0.8, 0.8, boundary="obc")
        self.assertEqual(data["psi_mps"].nqbits, 13)
        self.assertEqual(data["occupations_up"].tolist(), [1, 0, 1, 0, 1, 0])
        self.assertEqual(data["occupations_down"].tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(data["psi_mps"].tags[6], "impurity")

    def test_only_qfew_is_applied_to_mps(self):
        data = sergio.initialize_sergio_mps(4, 0.8, 0.8)
        calls = []
        original = sergio.hf.apply_qfew_quantum_gates

        def recording_apply(state, gates):
            calls.append(gates)
            return original(state, gates)

        sergio.hf.apply_qfew_quantum_gates = recording_apply
        try:
            sergio.sergio_step(
                data["U_up"],
                data["V_up"],
                data["U_down"],
                data["V_down"],
                data["Jk"],
                data["Jz"],
                data["T"],
                data["M_up"],
                data["M_down"],
                data["psi_mps"],
                n=0,
                free_step=data["free_step"],
            )
        finally:
            sergio.hf.apply_qfew_quantum_gates = original

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(gates[0][0] == "MOCK_QFEW" for gates in calls))

    def test_obc_and_pbc_hamiltonians_differ_only_by_wrap_bond(self):
        obc = sergio.build_h1(6, boundary="obc")
        pbc = sergio.build_h1(6, boundary="pbc")
        difference = pbc - obc
        expected = np.zeros((6, 6), dtype=complex)
        expected[0, 5] = expected[5, 0] = -0.5
        self.assertTrue(np.allclose(difference, expected))

    def test_corrected_1rdm_is_diagonal_for_initial_fermi_sea(self):
        data = sergio.initialize_sergio_mps(6, 0.8, 0.8, boundary="obc")
        occupations, C = sergio._natural_occupations(
            data["psi_mps"], range(6), 6, 1.0e-9
        )
        self.assertTrue(np.allclose(C, np.diag(data["occupations_up"])))
        self.assertTrue(np.allclose(occupations, data["occupations_up"]))

    def test_n6_matches_direct_floquet_for_obc_and_pbc(self):
        for boundary in ("obc", "pbc"):
            with self.subTest(boundary=boundary):
                result = sergio.sergio_step_floquet(
                    N=6,
                    no_floquet_steps=2,
                    Jk=0.8,
                    Jz=0.8,
                    boundary=boundary,
                )
                direct_values = direct_floquet_with_supplied_convention(
                    6, 2, 0.8, 0.8, boundary
                )
                self.assertTrue(
                    np.allclose(
                        result["magnetization"], direct_values, atol=2.0e-10
                    )
                )
                self.assertEqual(result["psi_mps"].nqbits, 13)


if __name__ == "__main__":
    unittest.main()
