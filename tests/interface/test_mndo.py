import pytest
from SHARC_MNDO import SHARC_MNDO

def setup_interface(path: str, maps: dict):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface._read_template = True
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.maps[k] == v, test_interface.QMin.maps[k]


def get_energy(outfile: str, template: str, qmin: str, energies: dict):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    with open(outfile, "r", encoding="utf-8") as file:
        parsed = test_interface._get_energy(file.read())
        for k, v in parsed.items():
            assert v == pytest.approx(energies[k])

def test_requests1():
    tests = ["inputs/mndo/QM2.in"]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_MNDO()
            test_interface.setup_mol(i)
            test_interface._read_template = True
            test_interface._read_resources = True
            test_interface.read_requests(i)

def test_requests2():
    tests = ["inputs/mndo/QM1.in"]
    for i in tests:
        test_interface = SHARC_MNDO()
        test_interface.setup_mol(i)
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.read_requests(i)

def test_energies():
    tests = [
        (
            "inputs/mndo/MNDO1.out",
            "inputs/mndo/MNDO1.template",
            "inputs/mndo/QM1.in",
            {
                (1, 1): -14.455173216195167,
                (1, 2): -14.174554972348169,
                (1, 3): -14.14113048726381,
                (1, 4): -13.90633425352368
            },
        ),
        (
            "inputs/mndo/MNDO3.out",
            "inputs/mndo/MNDO3.template",
            "inputs/mndo/QM3.in",
            {
                (1, 1): -14.455209046767495,
                (1, 2): -14.174552951136395,
                (1, 3): -14.141157534752255,
                (1, 4): -13.906299966422159}
        )
    ]
    for outfile, template, qmin, energies in tests:
        get_energy(outfile, template, qmin, energies)