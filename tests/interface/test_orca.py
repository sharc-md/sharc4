import pytest
from SHARC_ORCA_new import SHARC_ORCA


@pytest.mark.dependency()
def test_orcaversion():
    test_interface = SHARC_ORCA()
    test_interface._setup_mol = True
    test_interface.read_resources("inputs/orcapath")
    assert SHARC_ORCA.get_orca_version(test_interface.QMin.resources["orcadir"])

@pytest.mark.dependency(depends=["test_orcaversion"])
def test_requests1():
    tests = ["inputs/QM2.in", "inputs/QM3.in", "inputs/QM4.in", "inputs/orca_requests_fail"]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_ORCA()
            test_interface.setup_mol(i)
            test_interface._read_template = True
            test_interface._read_resources = True
            test_interface.read_requests(i)


@pytest.mark.dependency(depends=["test_orcaversion"])
def test_requests2():
    tests = ["inputs/orca_requests"]
    for i in tests:
        test_interface = SHARC_ORCA()
        test_interface.setup_mol(i)
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.read_requests(i)