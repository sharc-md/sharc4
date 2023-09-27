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


def build_jobs(path: str, maps: dict):
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface._read_template = True
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.control[k] == v, test_interface.QMin.control[k]


def get_energy(outfile: str, template: str, qmin: str, mults: list, energies: dict):
    test_interface = SHARC_ORCA()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    with open(outfile, "r", encoding="utf-8") as file:
        parsed = test_interface._get_energy(file.read(), mults)
        for k, v in parsed.items():
            assert v == pytest.approx(energies[k])


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
    test_fail = ["inputs/orcapath_fail", "inputs/orcapath_fail2"]

    for i in test_pass:
        test_interface = SHARC_ORCA()
        test_interface._setup_mol = True
        test_interface.read_resources(i)

    for i in test_fail:
        with pytest.raises(ValueError):
            test_interface = SHARC_ORCA()
            test_interface._setup_mol = True
            test_interface.read_resources(i)


def test_energies():
    tests = [
        (
            "inputs/orca1.out",
            "inputs/orca_template",
            "inputs/QM1.in",
            [1, 3],
            {
                (1, 1): -550.164846079,
                (1, 2): -550.065349079,
                (1, 3): -550.051038079,
                (1, 4): -549.960953079,
                (1, 5): -549.902495079,
                (3, 1): -550.096449079,
                (3, 2): -550.090568079,
                (3, 3): -550.074080079,
                (3, 4): -549.942447079,
                (3, 5): -549.936124079,
            },
        ),
        (
            "inputs/orca1-2.out",
            "inputs/orca_template",
            "inputs/QM1.in",
            [2],
            {(2, 1): -549.725632289},
        ),
        (
            "inputs/orca3.out",
            "inputs/orca_template",
            "inputs/orca3.in",
            [2],
            {
                (2, 1): -549.725632289,
                (2, 2): -549.691766289,
                (2, 3): -549.690712289,
                (2, 4): -549.639773289,
                (2, 5): -549.631470289,
            },
        ),
        ("inputs/orca4.out", "inputs/orca_template", "inputs/orca4.in", [4], {(4, 1): -549.649784479, (4, 2): -549.641911479}),
    ]
    for outfile, template, qmin, mults, energies in tests:
        get_energy(outfile, template, qmin, mults, energies)


def test_buildjobs():
    tests = [
        (
            "inputs/QM1.in",
            {
                "joblist": [1, 2],
                "states_to_do": [6, 1, 5],
                "jobs": {1: {"mults": [1, 3], "restr": True}, 2: {"mults": [2], "restr": False}},
            },
        ),
        (
            "inputs/orca3.in",
            {"joblist": [2], "states_to_do": [0, 5], "jobs": {2: {"mults": [2], "restr": False}}},
        ),
        (
            "inputs/orca4.in",
            {
                "joblist": [2, 4],
                "states_to_do": [0, 2, 0, 2],
                "jobs": {2: {"mults": [2], "restr": False}, 4: {"mults": [4], "restr": False}},
            },
        )
    ]

    for path, maps in tests:
        build_jobs(path, maps)
