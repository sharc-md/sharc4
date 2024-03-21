import pytest
from SHARC_DO_NOTHING import SHARC_DO_NOTHING
from SHARC_GAUSSIAN import SHARC_GAUSSIAN
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_LVC import SHARC_LVC
from SHARC_ORCA import SHARC_ORCA

SHARC_HYBRID.__abstractmethods__ = set()


def test_instantiate_childs():
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


def test_instantiate_childs2():
    tests = [
        {"child1": "ASDFSDF", "child2": "ORCA"},
        {"child1": "numpy", "child2": "ORCA"},
    ]

    for childs in tests:
        test_interface = SHARC_HYBRID()
        with pytest.raises(ModuleNotFoundError):
            test_interface.instantiate_children(childs)


def test_instantiate_childs3():
    tests = [
        [{"child1": "ORCA"}, {"child1": "ORCA"}],
        [{"child1": "GAUSSIAN"}, {"child1": "ORCA"}],
    ]

    for childs in tests:
        test_interface = SHARC_HYBRID()
        test_interface.instantiate_children(childs[0])
        with pytest.raises(ValueError):
            test_interface.instantiate_children(childs[1])


def test_instantiate_args():
    tests = [
        (
            {"child1": ("ORCA", [], {"loglevel": 19}), "child2": ("GAUSSIAN", [], {"loglevel": 14})},
            {"child1": (SHARC_ORCA, 19), "child2": (SHARC_GAUSSIAN, 14)},
        ),
        ({"child1": ("SHARC_ORCA", [False, "1", "2", 99], {})}, {"child1": (SHARC_ORCA, 99)}),
    ]
    for childs, ref in tests:
        test_interface = SHARC_HYBRID()
        test_interface.instantiate_children(childs)
        for name, interface in ref.items():
            assert isinstance(test_interface._kindergarden[name], interface[0])
            assert test_interface._kindergarden[name].log.level == interface[1]

def test_instantiate_args2():
    tests = [
        ({"child1": ("LVC", {})}),
        ({"child1": ("ORCA", [],None)}),
        ({"child1": ("LVC", "vs",{})}),
        ({"child1": ("GAUSSIAN", {}, [])}),
        ({"child1": ("LVC", None)}),
    ]
    for childs in tests:
        test_interface = SHARC_HYBRID()
        with pytest.raises(ValueError):
            test_interface.instantiate_children(childs)