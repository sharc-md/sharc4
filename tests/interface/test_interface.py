import pytest
import os
from utils import expand_path
from SHARC_INTERFACE import SHARC_INTERFACE

SHARC_INTERFACE.__abstractmethods__ = set()

PATH = "$SHARC/../tests/interface"

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


def read_resources(path: str, params: dict, whitelist: list):
    test_interface = SHARC_INTERFACE()
    test_interface.QMin.resources.types = {"int_key": int, "float_key": float, "key1": str, "key2": list, "key4": bool}
    test_interface.QMin.resources.data = {"int_key": None, "float_key": None, "key1": None, "key2": None, "key4": None}
    test_interface._setup_mol = True
    test_interface.read_resources(path, whitelist)
    for k, v in params.items():
        assert test_interface.QMin.resources[k] == v


def test_states1():
    tests = [("inputs/QM1.in", [3, 1, 5]), ("inputs/QM3.in", [0, 0, 0, 0, 9, 9])]
    for path, state in tests:
        get_states(os.path.join(expand_path(PATH), path), state)


def test_states2():
    tests = [("inputs/QM_failstate1.in", []), ("inputs/QM_failstate2.in", []), ("inputs/QM2.in", [])]
    for path, state in tests:
        with pytest.raises(ValueError):
            get_states(os.path.join(expand_path(PATH), path), state)


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
            "inputs/QM3.in",
            {
                "h": True,
                "soc": False,
                "dm": True,
                "grad": list(range(1, 100)),
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
                "grad": list(range(1, 21)),
                "nacdr": [[1, 2]],
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
            "inputs/QM0.in",
            {
                "h": True,
                "soc": True,
                "dm": True,
                "grad": [1, 2],
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
    ]
    for path, req in tests:
        set_requests(os.path.join(expand_path(PATH), path), req)


def test_reqests2():
    tests = [
        ("inputs/QM_failreq1.in", {}),
        ("inputs/QM_failreq2.in", {}),
        ("inputs/QM_failreq3.in", {}),
        ("inputs/QM2.in", {}),
        ("inputs/QM_failreq4.in", {}),
        ("inputs/QM_failreq5.in", {}),
        ("inputs/QM_failreq6.in", {}),
    ]

    for path, req in tests:
        with pytest.raises((AssertionError, ValueError)):
            set_requests(os.path.join(expand_path(PATH), path), req)


def test_resources1():
    tests = [
        ("inputs/interface_resources1", {"key1": "test", "key2": ["test1", "test2"], "key4": True}, []),
        ("inputs/interface_resources2", {"key1": "test2", "key2": [["test1", "test2"], ["test3", "test4"]]}, ["key2"]),
        ("inputs/interface_resources3", {"int_key": 13123, "float_key": -3.0}, []),
        ("inputs/interface_resources4", {"key1": "test2", "key2": ["test4"]}, []),
        ("inputs/interface_resources5", {"key1": "test1", "key2": ["test3", "test4"]}, []),
        ("inputs/interface_resources6", {"key1": "test2", "key2": ["test1", "test2", "test3", "test4"]}, ["key2"]),
        ("inputs/interface_resources7", {"key1": "test2", "key2": [["test1", "test2"], ["test3", "test4"]]}, []),
        ("inputs/interface_resources8", {"key1": "test2", "key2": ["test1", "test2", "test3", "test4"]}, []),
    ]
    for path, params, whitelist in tests:
        read_resources(os.path.join(expand_path(PATH), path), params, whitelist)


def test_resources2():
    tests = [
        ("inputs/interface_resources9", {}, []),
    ]
    for path, params, whitelist in tests:
        with pytest.raises(ValueError):
            read_resources(os.path.join(expand_path(PATH), path), params, whitelist)
