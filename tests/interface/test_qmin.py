from copy import deepcopy
from qmin import QMin


def test_copy():
    qmin = QMin()

    qmin.template.update({"a": None, "b": None})
    qmin.template.types.update({"b": str})
    try:
        qmin2 = deepcopy(qmin)
    except Exception as exc:
        assert False, f"Copy failed {exc}"

def test_copy_recursive():
    qmin = QMin()
    schedule = [{}]

    for _ in range(200):
        try:
            qmin2 = deepcopy(qmin)
            schedule[-1][0] = qmin2
            qmin.scheduling["schedule"] = schedule
        except Exception as exc:
            assert False, f"Recursion test failed {exc}"
