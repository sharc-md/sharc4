import pytest
from SHARC_INTERFACE import SHARC_INTERFACE

SHARC_INTERFACE.__abstractmethods__ = set()


def get_states(path: str, states: list):
    test_interface = SHARC_INTERFACE()

    test_interface.setup_mol(path)
    assert test_interface.QMin.molecule["states"] == states


def test_states1():
    tests = [("inputs/QM1.in", [3, 1, 5]), ("inputs/QM2.in", []), ("inputs/QM3.in", [0, 0, 0, 0, 9, 9])]
    for path, state in tests:
        get_states(path, state)


def test_states2():
    tests = [("inputs/QM_failstate1.in", []), ("inputs/QM_failstate2.in", [])]
    for path, state in tests:
        with pytest.raises(ValueError):
            get_states(path, state)


def test_nmstates1():
    pass
