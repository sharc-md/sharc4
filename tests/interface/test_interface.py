import pytest
from SHARC_INTERFACE import SHARC_INTERFACE

SHARC_INTERFACE.__abstractmethods__ = set()


def get_states(path: str, states: list):
    test_interface = SHARC_INTERFACE()

    test_interface.setup_mol(path)
    assert test_interface.QMin.molecule["states"] == states


def set_requests(path: str, requests: dict):
    test_interface = SHARC_INTERFACE()
    test_interface.setup_mol(path)
    test_interface._read_template = True
    test_interface._read_resources = True
    test_interface.read_requests(path)
    for k, v in requests.items():
        assert test_interface.QMin.requests[k] == v, ValueError(f"{k}")


def test_states1():
    tests = [("inputs/QM1.in", [3, 1, 5]), ("inputs/QM2.in", []), ("inputs/QM3.in", [0, 0, 0, 0, 9, 9])]
    for path, state in tests:
        get_states(path, state)


def test_states2():
    tests = [("inputs/QM_failstate1.in", []), ("inputs/QM_failstate2.in", [])]
    for path, state in tests:
        with pytest.raises(ValueError):
            get_states(path, state)


def test_requests1():
    tests = [
        (
            "inputs/QM1.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "nacdr": None,
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM2.in",
            {
                "h": True,
                "soc": False,
                "dm": True,
                "grad": [],
                "nacdr": ["all"],
                "overlap": False,
                "phases": False,
                "ion": False,
                "socdr": False,
                "dmdr": True,
                "multipolar_fit": None,
                "theodore": True,
            },
        ),
        (
            "inputs/QM3.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": list(range(1,100)),
                "nacdr": ["all"],
                "overlap": False,
                "phases": False,
                "ion": False,
                "socdr": True,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": False,
            },
        ),
        (
            "inputs/QM4.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": list(range(1,21)),
                "nacdr": [["1", "2"]],
                "overlap": False,
                "phases": False,
                "ion": True,
                "socdr": False,
                "dmdr": False,
                "multipolar_fit": None,
                "theodore": True,
            },
        )
    ]
    for path, req in tests:
        set_requests(path, req)

def test_reqests2():
    tests = [("inputs/QM_failreq1.in", []), ("inputs/QM_failreq2.in", []),("inputs/QM_failreq3.in", [])]

    for path, req in tests:
        with pytest.raises(AssertionError):
            set_requests(path, req)