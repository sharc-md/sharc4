import pytest
from SHARC_ABINITIO import SHARC_ABINITIO

SHARC_ABINITIO.__abstractmethods__ = set()


def setup_interface(qmin: str, template: str, maps: dict):
    test = SHARC_ABINITIO()
    test.setup_mol(qmin)
    test.read_template(template)
    test._read_resources = True
    test.read_requests(qmin)
    test.setup_interface()

    for k, v in maps.items():
        assert test.QMin.maps[k] == v, ValueError(f"{test.QMin.maps[k]} {k}")


def test_setupinterface1():
    tests = [
        (
            "inputs/QM1.in",
            "inputs/abinitio_template1",
            {
                "gradmap": {(1, 2), (2, 1), (3, 4), (3, 1), (1, 1), (3, 3), (3, 2), (1, 3), (3, 5)},
                "chargemap": {1: 0, 2: 1, 3: 0},
                "nacmap": None,
            },
        ),
        (
            "inputs/QM1.in",
            "inputs/abinitio_template2",
            {
                "gradmap": {(1, 2), (2, 1), (3, 4), (3, 1), (1, 1), (3, 3), (3, 2), (1, 3), (3, 5)},
                "chargemap": {1: 0, 2: -1, 3: 0},
                "nacmap": None,
            },
        ),
        (
            "inputs/QM3.in",
            "inputs/abinitio_template3",
            {
                "gradmap": {
                    (6, 1),
                    (5, 4),
                    (5, 1),
                    (5, 7),
                    (6, 4),
                    (6, 7),
                    (5, 2),
                    (6, 2),
                    (5, 5),
                    (6, 5),
                    (5, 8),
                    (6, 8),
                    (5, 3),
                    (5, 6),
                    (6, 6),
                    (5, 9),
                    (6, 3),
                    (6, 9),
                },
                "chargemap": {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1},
                "nacmap": {
                    (5, 1, 5, 4),
                    (6, 3, 6, 4),
                    (5, 6, 5, 8),
                    (6, 7, 6, 9),
                    (6, 3, 6, 7),
                    (6, 2, 6, 5),
                    (5, 3, 5, 4),
                    (5, 5, 5, 6),
                    (5, 3, 5, 7),
                    (6, 2, 6, 8),
                    (6, 6, 6, 7),
                    (5, 5, 5, 9),
                    (5, 2, 5, 5),
                    (5, 4, 5, 7),
                    (5, 2, 5, 8),
                    (6, 1, 6, 5),
                    (6, 1, 6, 2),
                    (6, 1, 6, 8),
                    (6, 5, 6, 7),
                    (6, 4, 6, 5),
                    (6, 4, 6, 8),
                    (6, 8, 6, 9),
                    (5, 1, 5, 3),
                    (5, 1, 5, 9),
                    (5, 7, 5, 9),
                    (5, 1, 5, 6),
                    (5, 6, 5, 7),
                    (6, 3, 6, 6),
                    (6, 3, 6, 9),
                    (6, 7, 6, 8),
                    (5, 3, 5, 6),
                    (6, 2, 6, 4),
                    (5, 5, 5, 8),
                    (5, 3, 5, 9),
                    (6, 6, 6, 9),
                    (6, 2, 6, 7),
                    (5, 2, 5, 4),
                    (6, 5, 6, 6),
                    (5, 4, 5, 6),
                    (5, 2, 5, 7),
                    (6, 1, 6, 4),
                    (6, 5, 6, 9),
                    (5, 4, 5, 9),
                    (6, 1, 6, 7),
                    (6, 4, 6, 7),
                    (5, 1, 5, 2),
                    (5, 7, 5, 8),
                    (5, 1, 5, 5),
                    (5, 1, 5, 8),
                    (5, 6, 5, 9),
                    (6, 3, 6, 5),
                    (6, 2, 6, 3),
                    (6, 3, 6, 8),
                    (6, 6, 6, 8),
                    (6, 2, 6, 6),
                    (5, 3, 5, 5),
                    (5, 5, 5, 7),
                    (5, 2, 5, 3),
                    (5, 3, 5, 8),
                    (6, 2, 6, 9),
                    (5, 4, 5, 5),
                    (5, 2, 5, 6),
                    (6, 1, 6, 3),
                    (6, 5, 6, 8),
                    (5, 4, 5, 8),
                    (5, 2, 5, 9),
                    (5, 8, 5, 9),
                    (6, 1, 6, 6),
                    (6, 1, 6, 9),
                    (6, 4, 6, 9),
                    (6, 4, 6, 6),
                    (5, 1, 5, 7),
                },
            },
        ),
        (
            "inputs/QM5.in",
            "inputs/abinitio_template1",
            {
                "gradmap": {(1, 2), (3, 3), (3, 1), (1, 1), (1, 3), (3, 2)},
                "chargemap": {1: 0, 2: 1, 3: 0},
                "nacmap": {(1, 1, 1, 3), (3, 2, 3, 3), (3, 1, 3, 3), (1, 1, 1, 2), (1, 2, 1, 3), (3, 1, 3, 2)},
            },
        ),
        (
            "inputs/QM6.in",
            "inputs/abinitio_template1",
            {
                "gradmap": {(1, 2), (2, 1), (3, 4), (3, 1), (1, 1), (3, 3), (3, 2), (1, 3), (3, 5)},
                "chargemap": {1: 0, 2: 1, 3: 0},
                "nacmap": set(),
            },
        ),
    ]

    for qmin, template, maps in tests:
        setup_interface(qmin, template, maps)


def test_setupinterface2():
    tests = [
        ("inputs/QM3.in", "inputs/abinitio_template1", {}),
        ("inputs/QM2.in", "inputs/abinitio_template1", {"gradmap": None, "chargemap": {}, "nacmap": set()}),
    ]

    for qmin, template, maps in tests:
        with pytest.raises(ValueError):
            setup_interface(qmin, template, maps)
