import pytest
import os
import numpy as np
from SHARC_MNDO import SHARC_MNDO
from SHARC_MNDO_OLD import SHARC_MNDO_OLD
from utils import expand_path

PATH = expand_path("$SHARC/../tests/interface")

def setup_interface(path: str, maps: dict):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface._read_template = True
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.maps[k] == v, test_interface.QMin.maps[k]

# def test_setup_interface_old():
#     test_interface = SHARC_MNDO()
#     test_interface.setup_mol("inputs/mndo/QMout_test.in")
#     test_interface._read_resources = True
#     test_interface._read_template = True
#     test_interface.read_resources("inputs/mndo/MNDO1.resources")
#     test_interface.read_template("inputs/mndo/MNDO_new.template")
#     test_interface.setup_interface()
#     test_interface.read_requests("inputs/mndo/QMout_test.in")
#     test_interface.set_coords("inputs/mndo/QMout_test.in")
#     test_interface.run()
#     test_interface.getQMout()

#     test_interface_old = SHARC_MNDO_OLD()
#     test_interface_old.setup_mol("inputs/mndo/QMout_test.in")
#     test_interface_old._read_resources = True
#     test_interface_old._read_template = True
#     test_interface_old.read_resources("inputs/mndo/MNDO1.resources")
#     test_interface_old.read_template("inputs/mndo/MNDO_old.template")
#     test_interface_old.setup_interface()
#     test_interface_old.read_requests("inputs/mndo/QMout_test.in")
#     test_interface_old.set_coords("inputs/mndo/QMout_test.in")
#     test_interface_old.run()
#     test_interface_old.getQMout()

#     #breakpoint()

#     assert same(test_interface.QMout, test_interface_old.QMout)

    

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
            assert np.allclose(v ,energies[k], rtol=1e-5)


def get_tdm(outfile: str, template: str, qmin: str, tdms: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_transition_dipoles(outfile)
    assert np.allclose(parsed, tdms, rtol=1e-4)
    

def get_grads(outfile: str, template: str, qmin: str, grads: list, grads_pc: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_grad(outfile)
    parsed_pc = test_interface._get_grad_pc(outfile)
    assert np.allclose(parsed, grads, rtol=1e-5)
    assert np.allclose(parsed_pc, grads_pc, rtol=1e-5)

def get_grads_no_pc(outfile: str, template: str, qmin: str, grads: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_grad(outfile)
    assert np.allclose(parsed, grads, rtol=1e-5)

def get_nacs(logfile:str, fortfile: str, template: str, qmin: str, nacs: list, nacs_pc: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    states, interstates = test_interface._get_states_interstates(logfile)
    file = open(logfile, "r")
    output = file.read()
    energies = test_interface._get_energy(output)
    test_interface.QMout["h"] = np.zeros((max(states)+1,max(states)+1))
    for i in range(len(energies)):
        test_interface.QMout["h"][i][i] = energies[(1, i + 1)]
    parsed = test_interface._get_nacs(fortfile, interstates)
    parsed_pc = test_interface._get_nacs_pc(fortfile, interstates)
    assert np.allclose(parsed, nacs, rtol=1e-5)
    assert np.allclose(parsed_pc, nacs_pc, rtol=1e-5)

def get_nacs_no_pc(logfile:str, fortfile: str, template: str, qmin: str, nacs: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    states, interstates = test_interface._get_states_interstates(logfile)
    file = open(logfile, "r")
    output = file.read()
    energies = test_interface._get_energy(output)
    test_interface.QMout["h"] = np.zeros((max(states)+1,max(states)+1))
    for i in range(len(energies)):
        test_interface.QMout["h"][i][i] = energies[(1, i + 1)]
    parsed = test_interface._get_nacs(fortfile, interstates)
    assert np.allclose(parsed, nacs, rtol=1e-5)
                

def test_requests1():
    tests = [os.path.join(PATH, "inputs/mndo/QM2.in")]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_MNDO()
            test_interface.setup_mol(i)
            test_interface._read_template = True
            test_interface._read_resources = True
            test_interface.read_requests(i)

def test_requests2():
    tests = [os.path.join(PATH, "inputs/mndo/QM1.in")]
    for i in tests:
        test_interface = SHARC_MNDO()
        test_interface.setup_mol(i)
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.read_requests(i)

def test_requests3():
    tests = [os.path.join(PATH, "inputs/mndo/QM4.in"), os.path.join(PATH, "inputs/mndo/QM5.in")]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_MNDO()
            test_interface.setup_mol(i)
            test_interface.setup_interface()

def test_energies():
    tests = [
        (
            os.path.join(PATH, "inputs/mndo/MNDO1.out"),
            os.path.join(PATH, "inputs/mndo/MNDO1.template"),
            os.path.join(PATH, "inputs/mndo/QM1.in"),
            {
                (1, 1): -14.455173216195169,
                (1, 2): -14.174554972348169,
                (1, 3): -14.14113048726381,
                (1, 4): -13.90633425352368
            },
        ),
        (
            os.path.join(PATH, "inputs/mndo/MNDO3.out"),
            os.path.join(PATH, "inputs/mndo/MNDO3.template"),
            os.path.join(PATH, "inputs/mndo/QM3.in"),
            {
                (1, 1): -14.455209046767495,
                (1, 2): -14.174552951136395,
                (1, 3): -14.141157534752255,
                (1, 4): -13.906299966422159}
        )
    ]
    for outfile, template, qmin, energies in tests:
        get_energy(outfile, template, qmin, energies)

def test_tdms():
    tests = [
        (
            os.path.join(PATH, "inputs/mndo/MNDO1.out"),
            os.path.join(PATH, "inputs/mndo/MNDO1.template"),
            os.path.join(PATH, "inputs/mndo/QM1.in"),
            np.array([[[0.0, 0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0], [0.0, -0.0, 0.0, -0.0], [-0.0, -0.0, -0.0, 0.0]], [[0.0, 0.0, 0.12200519721599998, 0.0], [0.0, 0.0, 0.07015635244799999, -0.0], [0.12200519721599998, 0.07015635244799999, -0.0, 0.013038738384], [0.0, -0.0, 0.013038738384, 0.0]], [[1.475570561136, -1.4476048890239999, 0.0, -0.26226557246399995], [-1.4476048890239999, 1.210667259648, 0.0, 1.1842998058079999], [0.0, 0.0, 1.5170195769119998, 0.0], [-0.26226557246399995, 1.1842998058079999, 0.0, 4.330066692672]]]),
        )
    ]
    for outfile, template, qmin, tdms in tests:
        get_tdm(outfile, template, qmin, tdms)

def test_grads():
    tests = [
        (
            os.path.join(PATH, "inputs/mndo/fort1.15"),
            os.path.join(PATH, "inputs/mndo/MNDO1.template"),
            os.path.join(PATH, "inputs/mndo/QM1_pc.in"),
            np.array([[[0.0, -0.0, -0.09494103284884164], [-0.0, -0.0, 0.06534109398831474], [-0.001668924263116929, 0.0, 0.0038429067260890206], [0.03751611538637951, -0.0, 0.010957058487696165], [-0.03751611538637951, -0.0, 0.010957058487696165], [0.001668924263116929, -0.0, 0.0038429067260890206]], [[-0.0, -0.0, 0.14141621162988025], [-0.0, -0.0, -0.1747717086527036], [-0.004582898662060879, 0.0, 0.004437826510784253], [0.040076504350755215, 0.0, 0.012239922000627434], [-0.040076504350755215, 0.0, 0.012239922000627434], [0.004582898662060879, -0.0, 0.004437826510784253]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-0.0, -0.0, 0.21335615313775658], [0.0, 0.0, -0.20293514396518275], [0.020592529326051, 0.0, -0.006759098992601753], [0.00980264576871046, -0.0, 0.00154859440631485], [-0.00980264576871046, -0.0, 0.00154859440631485], [-0.020592529326051, 0.0, -0.006759098992601753]]]),
            np.array([[[0., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]]]),
        )
    ]
    for outfile, template, qmin, grads, grads_pc in tests:
        get_grads(outfile, template, qmin, grads, grads_pc)

def test_grads_no_pc():
    tests = [
        (
            os.path.join(PATH, "inputs/mndo/fort1_no_pc.15"),
            os.path.join(PATH, "inputs/mndo/MNDO1_no_pc.template"),
            os.path.join(PATH, "inputs/mndo/QM1.in"),
            np.array([[[0.0, -0.0, -0.09494103284884164], [-0.0, -0.0, 0.06534109398831474], [-0.001668924263116929, 0.0, 0.0038429067260890206], [0.03751611538637951, -0.0, 0.010957058487696165], [-0.03751611538637951, -0.0, 0.010957058487696165], [0.001668924263116929, -0.0, 0.0038429067260890206]], [[-0.0, -0.0, 0.14141621162988025], [-0.0, -0.0, -0.1747717086527036], [-0.004582898662060879, 0.0, 0.004437826510784253], [0.040076504350755215, 0.0, 0.012239922000627434], [-0.040076504350755215, 0.0, 0.012239922000627434], [0.004582898662060879, -0.0, 0.004437826510784253]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[-0.0, -0.0, 0.21335615313775658], [0.0, 0.0, -0.20293514396518275], [0.020592529326051, 0.0, -0.006759098992601753], [0.00980264576871046, -0.0, 0.00154859440631485], [-0.00980264576871046, -0.0, 0.00154859440631485], [-0.020592529326051, 0.0, -0.006759098992601753]]]),
        )
    ]
    for outfile, template, qmin, grads in tests:
        get_grads_no_pc(outfile, template, qmin, grads)

def test_nacs():
    tests = [
        (
            os.path.join(PATH, "inputs/mndo/MNDO1.out"),
            os.path.join(PATH, "inputs/mndo/fort1.15"),
            os.path.join(PATH, "inputs/mndo/MNDO1.template"),
            os.path.join(PATH, "inputs/mndo/QM1_pc.in"),
            np.array([[[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[-1.88386700e-08, -0.00000000e+00, -1.17839654e-01],
                        [ 4.86842035e-09, -0.00000000e+00, -1.87422803e-01],
                        [-4.25931889e-02, -0.00000000e+00,  1.73931890e-02],
                        [ 4.93296831e-02,  0.00000000e+00,  1.86082400e-02],
                        [-4.93296832e-02,  0.00000000e+00,  1.86082403e-02],
                        [ 4.25932030e-02, 0.00000000e+00, 1.73931901e-02]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[-6.82637201e-09,  0.00000000e+00,  1.09861742e-01],
                        [ 1.79919882e-09,  0.00000000e+00, -2.01801675e-01],
                        [-3.90783910e-03, -0.00000000e+00,  1.73022336e-03],
                        [ 4.40323437e-03, -0.00000000e+00,  1.83997529e-03],
                        [-4.40323442e-03,  0.00000000e+00,  1.83997539e-03],
                        [ 3.90784424e-03, -0.00000000e+00,  1.73022379e-03]]],


                        [[[ 1.88386700e-08,  0.00000000e+00,  1.17839654e-01],
                        [-4.86842035e-09,  0.00000000e+00,  1.87422803e-01],
                        [ 4.25931889e-02,  0.00000000e+00, -1.73931890e-02],
                        [-4.93296831e-02, -0.00000000e+00, -1.86082400e-02],
                        [ 4.93296832e-02, -0.00000000e+00, -1.86082403e-02],
                        [-4.25932030e-02, -0.00000000e+00, -1.73931901e-02]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 1.50815196e-08, -0.00000000e+00, -3.40839503e-01],
                        [-3.91590333e-09, -0.00000000e+00,  5.87940853e-01],
                        [ 3.80662475e-02, -0.00000000e+00, -1.40419491e-02],
                        [-4.26752470e-02,  0.00000000e+00, -1.63936630e-02],
                        [ 4.26752471e-02, -0.00000000e+00, -1.63936632e-02],
                        [-3.80662587e-02,  0.00000000e+00, -1.40419500e-02]]],


                        [[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]],


                        [[[ 6.82637201e-09, -0.00000000e+00, -1.09861742e-01],
                        [-1.79919882e-09, -0.00000000e+00,  2.01801675e-01],
                        [ 3.90783910e-03,  0.00000000e+00, -1.73022336e-03],
                        [-4.40323437e-03,  0.00000000e+00, -1.83997529e-03],
                        [ 4.40323442e-03, -0.00000000e+00, -1.83997539e-03],
                        [-3.90784424e-03,  0.00000000e+00, -1.73022379e-03]],

                        [[-1.50815196e-08,  0.00000000e+00,  3.40839503e-01],
                        [ 3.91590333e-09,  0.00000000e+00, -5.87940853e-01],
                        [-3.80662475e-02,  0.00000000e+00,  1.40419491e-02],
                        [ 4.26752470e-02, -0.00000000e+00,  1.63936630e-02],
                        [-4.26752471e-02,  0.00000000e+00,  1.63936632e-02],
                        [ 3.80662587e-02, -0.00000000e+00,  1.40419500e-02]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]]),
            np.array([[[[ 0. , 0. , 0.]],
                        [[ 0. , 0. , 0.]],
                        [[ 0. , 0. , 0.]],
                        [[ 0. , 0. , 0.]]],

                        [[[-0. ,-0. ,-0.]],
                        [[ 0. , 0. , 0.]],
                        [[ 0.  ,0.  ,0.]],
                        [[ 0. , 0. , 0.]]],

                        [[[ 0. , 0. , 0.]],
                        [[ 0.  ,0.  ,0.]],
                        [[ 0. , 0. , 0.]],
                        [[ 0. , 0. , 0.]]],

                        [[[-0., -0. ,-0.]],
                        [[-0. ,-0. , -0.]],
                        [[ 0. , 0.  ,0.]],
                        [[ 0.  ,0. , 0.]]]]),
        )
    ]
    for logfile, fortfile, template, qmin, nacs, nacs_pc in tests:
        get_nacs(logfile, fortfile, template, qmin, nacs, nacs_pc)
    
def test_nacs_no_pc():
    tests = [
        (
            os.path.join(PATH, "inputs/mndo/MNDO1_no_pc.out"),
            os.path.join(PATH, "inputs/mndo/fort1_no_pc.15"),
            os.path.join(PATH, "inputs/mndo/MNDO1_no_pc.template"),
            os.path.join(PATH, "inputs/mndo/QM1.in"),
            np.array([[[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[-1.88386700e-08, -0.00000000e+00, -1.17839654e-01],
                        [ 4.86842035e-09, -0.00000000e+00, -1.87422803e-01],
                        [-4.25931889e-02, -0.00000000e+00,  1.73931890e-02],
                        [ 4.93296831e-02,  0.00000000e+00,  1.86082400e-02],
                        [-4.93296832e-02,  0.00000000e+00,  1.86082403e-02],
                        [ 4.25932030e-02, 0.00000000e+00, 1.73931901e-02]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[-6.82637201e-09,  0.00000000e+00,  1.09861742e-01],
                        [ 1.79919882e-09,  0.00000000e+00, -2.01801675e-01],
                        [-3.90783910e-03, -0.00000000e+00,  1.73022336e-03],
                        [ 4.40323437e-03, -0.00000000e+00,  1.83997529e-03],
                        [-4.40323442e-03,  0.00000000e+00,  1.83997539e-03],
                        [ 3.90784424e-03, -0.00000000e+00,  1.73022379e-03]]],


                        [[[ 1.88386700e-08,  0.00000000e+00,  1.17839654e-01],
                        [-4.86842035e-09,  0.00000000e+00,  1.87422803e-01],
                        [ 4.25931889e-02,  0.00000000e+00, -1.73931890e-02],
                        [-4.93296831e-02, -0.00000000e+00, -1.86082400e-02],
                        [ 4.93296832e-02, -0.00000000e+00, -1.86082403e-02],
                        [-4.25932030e-02, -0.00000000e+00, -1.73931901e-02]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 1.50815196e-08, -0.00000000e+00, -3.40839503e-01],
                        [-3.91590333e-09, -0.00000000e+00,  5.87940853e-01],
                        [ 3.80662475e-02, -0.00000000e+00, -1.40419491e-02],
                        [-4.26752470e-02,  0.00000000e+00, -1.63936630e-02],
                        [ 4.26752471e-02, -0.00000000e+00, -1.63936632e-02],
                        [-3.80662587e-02,  0.00000000e+00, -1.40419500e-02]]],


                        [[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]],


                        [[[ 6.82637201e-09, -0.00000000e+00, -1.09861742e-01],
                        [-1.79919882e-09, -0.00000000e+00,  2.01801675e-01],
                        [ 3.90783910e-03,  0.00000000e+00, -1.73022336e-03],
                        [-4.40323437e-03,  0.00000000e+00, -1.83997529e-03],
                        [ 4.40323442e-03, -0.00000000e+00, -1.83997539e-03],
                        [-3.90784424e-03,  0.00000000e+00, -1.73022379e-03]],

                        [[-1.50815196e-08,  0.00000000e+00,  3.40839503e-01],
                        [ 3.91590333e-09,  0.00000000e+00, -5.87940853e-01],
                        [-3.80662475e-02,  0.00000000e+00,  1.40419491e-02],
                        [ 4.26752470e-02, -0.00000000e+00,  1.63936630e-02],
                        [-4.26752471e-02,  0.00000000e+00,  1.63936632e-02],
                        [ 3.80662587e-02, -0.00000000e+00,  1.40419500e-02]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]]),
        )
    ]
    for logfile, fortfile, template, qmin, nacs in tests:
        get_nacs_no_pc(logfile, fortfile, template, qmin, nacs) 

def test_template():
    tests = [
        (
            "inputs/mndo/templatetest1.dat",
            {
                "disp": 0,
                "nciref": 3,
                "kitscf": 5000,
                "ici1": 2,
                "ici2": 1,
                "act_orbs": [4,6,7],
                "movo": 1,
                "imomap": 3,
                "iop": -6,
                "rohf": 0,
                "levexc": 2,
            },
        ),
        (
            "inputs/mndo/templatetest2.dat",
            {
                "disp" : 1,
                "nciref": 6,
                "kitscf": 9999,
                "ici1": 3,
                "ici2": 2,
                "act_orbs": [4, 5, 6, 7, 8],
                "movo": 1,
                "imomap": 0,
                "iop": -22,
                "rohf": 0,
                "levexc": 2,
            },
        ),
    ]
    for template, ref in tests:
        test_interface = SHARC_MNDO()
        test_interface.setup_mol(os.path.join(PATH, "inputs/mndo/QM1.in"))
        test_interface.read_template(os.path.join(PATH, template))
        for k, v in ref.items():
            assert test_interface.QMin["template"][k] == v

def test_template_error():

    tests = ["inputs/mndo/templatetest3.dat", "inputs/mndo/templatetest4.dat"]

    for template in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_MNDO()
            test_interface.setup_mol(os.path.join(PATH, "inputs/mndo/QM1.in"))
            test_interface.read_template(os.path.join(PATH, template))

def same(self, other):
    if isinstance(self, other.__class__):
        A = self.nmstates == other.nmstates and self.natom == other.natom and self.npc == other.npc and self.point_charges == other.point_charges
        B = self.states == other.states  and  np.allclose(self.h, other.h, rtol=1e-5) and  np.allclose(self.dm, other.dm, rtol=1e-5) and np.allclose(self.grad, other.grad, rtol=1e-5) and np.allclose(self.grad_pc, other.grad_pc, rtol=1e-5) and np.allclose(self.nacdr, other.nacdr, rtol=1e-5) and np.allclose(self.nacdr_pc, other.nacdr_pc, rtol=1e-5)
        return A and B
    return False
