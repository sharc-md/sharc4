import pytest
from SHARC_ORCA_new import SHARC_ORCA


def setup_interface(path: str, maps: dict):
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface._read_template = True
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.maps[k] == v, test_interface.QMin.maps[k]

@pytest.mark.dependency()
def test_orcaversion():
    test_interface = SHARC_ORCA()
    test_interface._setup_mol = True
    test_interface.read_resources("inputs/orcapath")
    assert isinstance(SHARC_ORCA.get_orca_version(test_interface.QMin.resources["orcadir"]), tuple)


def test_requests1():
    tests = ["inputs/QM2.in", "inputs/QM3.in", "inputs/QM4.in", "inputs/orca_requests_fail"]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_ORCA()
            test_interface.setup_mol(i)
            test_interface._read_template = True
            test_interface._read_resources = True
            test_interface.read_requests(i)


def test_requests2():
    tests = ["inputs/orca_requests"]
    for i in tests:
        test_interface = SHARC_ORCA()
        test_interface.setup_mol(i)
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.read_requests(i)


def test_maps():
    tests = [
        (
            "inputs/QM1.in",
            {
                "multmap": {1: 1, 3: 1, -1: [1, 3], 2: 2, -2: [2]},
                "ionmap": [(1, 1, 2, 2), (2, 2, 3, 1)],
                "gsmap": {
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 4,
                    5: 5,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 1,
                    13: 1,
                    14: 1,
                    15: 1,
                    16: 1,
                    17: 1,
                    18: 1,
                    19: 1,
                    20: 1,
                },
            },
        )
    ]

    for path, maps in tests:
        setup_interface(path, maps)

@pytest.mark.dependency(depends=["test_orcaversion"])
def test_resources():
    test_pass = ["inputs/orcapath"]
    test_fail = ["inputs/orcapath_fail"]

    for i in test_pass:
        test_interface = SHARC_ORCA()
        test_interface._setup_mol = True
        test_interface.read_resources(i)

    for i in test_fail:
        with pytest.raises(ValueError):
            test_interface = SHARC_ORCA()
            test_interface._setup_mol = True
            test_interface.read_resources(i)