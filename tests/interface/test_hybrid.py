import pytest
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_ORCA import SHARC_ORCA
from SHARC_GAUSSIAN import SHARC_GAUSSIAN
from SHARC_LVC import SHARC_LVC
from SHARC_DO_NOTHING import SHARC_DO_NOTHING
from SHARC_ABINITIO import SHARC_ABINITIO

SHARC_HYBRID.__abstractmethods__ = set()


def test_instanciate_childs():
    tests = [
        ({"child1": "ORCA", "child2": "GAUSSIAN"}, {"child1": SHARC_ORCA, "child2": SHARC_GAUSSIAN}),
        ({"child1": "SHARC_ORCA", "child2": "SHARC_GAUSSIAN"}, {"child1": SHARC_ORCA, "child2": SHARC_GAUSSIAN}),
        ({"child1": "ORCA", "child2": "ORCA"}, {"child1": SHARC_ORCA, "child2": SHARC_ORCA}),
        ({"child1": "DO_NOTHING", "child2": "LVC"}, {"child1": SHARC_DO_NOTHING, "child2": SHARC_LVC}),
        ({"DO_NOTHING": "DO_NOTHING", "LVC": "LVC"}, {"DO_NOTHING": SHARC_DO_NOTHING, "LVC": SHARC_LVC}),
    ]

    for childs, ref in tests:
        test_interface = SHARC_HYBRID()
        test_interface.instantiate_children(childs)
        for name, interface in ref.items():
            assert isinstance(test_interface._kindergarden[name], interface)


def test_instanciate_childs2():
    tests = [
        {"child1": "ASDFSDF", "child2": "ORCA"},
        {"child1": "numpy", "child2": "ORCA"},
    ]

    for childs in tests:
        test_interface = SHARC_HYBRID()
        with pytest.raises(ModuleNotFoundError):
            test_interface.instantiate_children(childs)


def test_instanciate_childs3():
    tests = [
        [{"child1": "ORCA"}, {"child1": "ORCA"}],
        [{"child1": "GAUSSIAN"}, {"child1": "ORCA"}],
    ]

    for childs in tests:
        test_interface = SHARC_HYBRID()
        test_interface.instantiate_children(childs[0])
        with pytest.raises(ValueError):
            test_interface.instantiate_children(childs[1])
