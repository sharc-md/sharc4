import logging
import os

import numpy as np
import pytest
from SHARC_NWCHEM import SHARC_NWCHEM
from utils import expand_path

PATH = expand_path("$SHARC/../tests/interface")
LOGGER = logging.getLogger("NWCHEM")


def test_civecs(caplog):
    tests = [
        (
            "TDA False, restr 1, roots 3, nocc [9, 0], nmo [29, 0], nov [180, 0]",
            np.array([0.1725486, 0.2029915, 0.3185422]),
            np.zeros((3), dtype=float),
            "nwchem/civecs/1.civecs",
            False,
        ),
        (
            "TDA False, restr 2, roots 3, nocc [9, 8], nmo [50, 50], nov [369, 336]",
            np.array([0.0647623, 0.1533046, 0.1819447]),
            np.array([0.75575, 0.75567, 0.75816]),
            "nwchem/civecs/2.civecs",
            False,
        ),
        (
            "TDA True, restr 1, roots 5, nocc [9, 0], nmo [50, 0], nov [369, 0]",
            np.array([0.2706171, 0.3279408, 0.3407144, 0.3659487, 0.3728658]),
            np.zeros((5), dtype=float),
            "nwchem/civecs/3.civecs",
            True,
        ),
        (
            "TDA True, restr 2, roots 4, nocc [1, 0], nmo [5, 5], nov [4, 0]",
            np.array([0.6907609, 1.5316617, 1.5316617, 1.5316617]),
            np.array([0.75, 0.75, 0.75, 0.75]),
            "nwchem/civecs/4.civecs",
            True,
        ),
        (
            "TDA True, restr 1, roots 12, nocc [9, 0], nmo [96, 0], nov [783, 0]",
            np.array(
                [
                    0.1736810,
                    0.1947931,
                    0.2753085,
                    0.2843663,
                    0.2954071,
                    0.2983479,
                    0.3098631,
                    0.3111910,
                    0.3323932,
                    0.3345910,
                    0.3363023,
                    0.3475876,
                ]
            ),
            np.zeros((12), dtype=float),
            "nwchem/civecs/5.civecs",
            True,
        ),
    ]

    for log, eref, s2ref, civec, tda in tests:
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            test_interface = SHARC_NWCHEM(loglevel=10, logname="NWCHEM")
            test_interface.log.propagate = True
            test_interface.QMin.resources["wfthres"] = 10.0
            en, s2, cis = test_interface._dets_from_civec(os.path.join(PATH, civec))
        assert log in caplog.text  # Test for header data
        assert np.allclose(np.array(en), eref)  # Test energies
        assert np.allclose(np.array(s2), s2ref)  # Test S**2
        for idx, ci in enumerate(cis[1:], 1):  # CI vecs should be orthogonal
            for j in range(idx + 1, len(cis)):
                assert np.dot(list(ci.values()), list(cis[j].values())) == pytest.approx(0.0, abs=5e-3 if not tda else None)


def test_movecs():
    # Transform MO to AO overlaps and check if diagonal is 1
    tests = [
        ("nwchem/movecs/1.in", "def2-svp", True, -1, "nwchem/movecs/1.movecs"),
        ("nwchem/movecs/2.in", "cc-pvtz", True, 0, "nwchem/movecs/2.movecs"),
        ("nwchem/movecs/2.in", "cc-pvtz", True, 0, "nwchem/movecs/3.movecs"),
        ("nwchem/movecs/2.in", "cc-pvtz", True, 0, "nwchem/movecs/4.movecs"),
        ("nwchem/movecs/2.in", "6-31g", False, 0, "nwchem/movecs/5.movecs"),
        ("nwchem/movecs/2.in", "6-31g", False, 0, "nwchem/movecs/6.movecs"),
        ("nwchem/movecs/2.in", "6-31g", False, 0, "nwchem/movecs/7.movecs"),
    ]

    for qmin, basis, spherical, charge, mo in tests:
        test_interface = SHARC_NWCHEM()
        test_interface.setup_mol(os.path.join(PATH, qmin))
        test_interface.set_coords(os.path.join(PATH, qmin))
        test_interface.QMin.template["spherical"] = spherical
        test_interface.QMin.molecule["charge"] = [charge]
        test_interface._basis = basis
        _, coeff = test_interface._mo_from_movec(os.path.join(PATH, mo))
        for i in np.diagonal(np.linalg.inv(coeff[: coeff.shape[1], :].T).T @ np.linalg.inv(coeff[: coeff.shape[1], :].T)):
            assert i == pytest.approx(1.0)
        if coeff.shape[0] != coeff.shape[1]:
            for i in np.diagonal(np.linalg.inv(coeff[coeff.shape[1] :, :].T).T @ np.linalg.inv(coeff[coeff.shape[1] :, :].T)):
                assert i == pytest.approx(1.0)


def test_energy():
    tests = [
        (  # H
            "nwchem/energy/1.log",
            "nwchem/energy/1.civecs",
            np.array([0.0, 0.2787822, 0.2787822, 0.3134959, 0.3135053, 0.3325833, 0.3339757, 0.3479437]) - 157.060105432223,
        ),
        (
            "nwchem/energy/2.log",
            "nwchem/energy/2.civecs",
            np.array([0.0, 0.0730633, 0.0996732]) - 156.712310493553,
        ),
        (  # H
            "nwchem/energy/3.log",
            None,
            np.array([-157.060105432223]),
        ),
        (  # H GRAD DM
            "nwchem/energy/4.log",
            "nwchem/energy/4.civecs",
            np.array([0.0, 0.2787822, 0.2787822, 0.3325833]) - 157.060105432386,
        ),
    ]

    for log, cis, ref in tests:
        test_interface = SHARC_NWCHEM()
        with open(os.path.join(PATH, log), "r", encoding="utf-8") as f:
            energy, _ = test_interface._get_energies(f.read(), os.path.join(PATH, cis) if cis else None, 1)
        assert np.allclose(np.array(energy), ref)


def test_gradients():
    tests = [
        (
            "nwchem/gradients/1.log",
            (
                (
                    1,
                    np.array(
                        [
                            [0.017501, -0.014751, 0.000022],
                            [-0.017501, 0.014751, 0.000022],
                            [-0.004677, 0.004163, -0.000010],
                            [-0.004895, 0.003905, -0.000010],
                            [0.004895, -0.003905, -0.000010],
                            [0.004677, -0.004163, -0.000010],
                            [0.017501, -0.014751, -0.000022],
                            [-0.017501, 0.014751, -0.000022],
                            [-0.004677, 0.004163, 0.000010],
                            [-0.004895, 0.003905, 0.000010],
                            [0.004895, -0.003905, 0.000010],
                            [0.004677, -0.004163, 0.000010],
                        ]
                    ),
                ),
                (
                    2,
                    np.array(
                        [
                            [0.097625, -0.082284, -0.000595],
                            [-0.097625, 0.082284, -0.000595],
                            [-0.003845, 0.005856, 0.000102],
                            [-0.006422, 0.002798, 0.000102],
                            [0.006422, -0.002798, 0.000102],
                            [0.003845, -0.005856, 0.000102],
                            [0.097597, -0.082260, 0.000592],
                            [-0.097597, 0.082260, 0.000592],
                            [-0.003841, 0.005852, -0.000101],
                            [-0.006418, 0.002795, -0.000101],
                            [0.006418, -0.002795, -0.000101],
                            [0.003841, -0.005852, -0.000101],
                        ]
                    ),
                ),
                (
                    3,
                    np.array(
                        [
                            [0.097597, -0.082260, -0.000593],
                            [-0.097597, 0.082260, -0.000593],
                            [-0.003841, 0.005852, 0.000101],
                            [-0.006418, 0.002795, 0.000101],
                            [0.006418, -0.002795, 0.000101],
                            [0.003841, -0.005852, 0.000101],
                            [0.097624, -0.082283, 0.000596],
                            [-0.097624, 0.082283, 0.000596],
                            [-0.003845, 0.005856, -0.000103],
                            [-0.006422, 0.002798, -0.000103],
                            [0.006422, -0.002798, -0.000103],
                            [0.003845, -0.005856, -0.000103],
                        ]
                    ),
                ),
                (
                    4,
                    np.array(
                        [
                            [0.081241, -0.068474, -0.000132],
                            [-0.081241, 0.068474, -0.000132],
                            [-0.005737, 0.004261, 0.000033],
                            [-0.005170, 0.004933, 0.000033],
                            [0.005170, -0.004933, 0.000033],
                            [0.005737, -0.004261, 0.000033],
                            [0.081241, -0.068474, 0.000132],
                            [-0.081241, 0.068474, 0.000132],
                            [-0.005737, 0.004261, -0.000033],
                            [-0.005170, 0.004933, -0.000033],
                            [0.005170, -0.004933, -0.000033],
                            [0.005737, -0.004261, -0.000033],
                        ]
                    ),
                ),
            ),
        ),
        (
            "nwchem/gradients/2.log",
            (
                (
                    1,
                    np.array(
                        [
                            [0.017501, -0.014751, 0.000022],
                            [-0.017501, 0.014751, 0.000022],
                            [-0.004677, 0.004163, -0.000010],
                            [-0.004895, 0.003905, -0.000010],
                            [0.004895, -0.003905, -0.000010],
                            [0.004677, -0.004163, -0.000010],
                            [0.017501, -0.014751, -0.000022],
                            [-0.017501, 0.014751, -0.000022],
                            [-0.004677, 0.004163, 0.000010],
                            [-0.004895, 0.003905, 0.000010],
                            [0.004895, -0.003905, 0.000010],
                            [0.004677, -0.004163, 0.000010],
                        ]
                    ),
                ),
            ),
        ),
    ]

    for log, ref in tests:
        test_interface = SHARC_NWCHEM()
        with open(os.path.join(PATH, log), "r", encoding="utf-8") as f:
            grads = test_interface._get_gradients(f.read())

            for g1, g2 in zip(grads, ref):
                assert g1[0] == g2[0]
                assert np.allclose(g1[1], g2[1])


def test_dm():
    tests = [
        ("nwchem/dm/1.log", 2, np.array([[0.0000000187, -0.0000000197, 0.0000000000], [0.0, 0.0, 0.0]])),
        ("nwchem/dm/2.log", 2, np.array([[0.0000000031, -0.0000000022, 0.0000584712], [0.0, 0.0, -7.83850]])),
        (
            "nwchem/dm/3.log",
            4,
            np.array(
                [
                    [-0.0191609263, 0.0010705329, 0.4542973898],
                    [-0.00086, 0.00004, -0.00004],
                    [-0.06268, 0.00322, -0.00265],
                    [0.01044, 0.01971, -0.22284],
                ]
            ),
        ),
    ]

    for log, states, ref in tests:
        test_interface = SHARC_NWCHEM()
        with open(os.path.join(PATH, log), "r", encoding="utf-8") as f:
            assert np.allclose(test_interface._get_dipoles(f.read(), states), ref)
