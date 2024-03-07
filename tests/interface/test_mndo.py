import pytest
import os
import numpy as np
from SHARC_MNDO import SHARC_MNDO
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

def get_tdm(outfile: str, template: str, qmin: str, tdms: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_transition_dipoles(outfile)
    print(parsed)
    np.allclose(parsed, tdms, rtol=1e-5)
    

def get_grads(outfile: str, template: str, qmin: str, grads: list, grads_pc: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_grad(outfile)
    parsed_pc = test_interface._get_grad_pc(outfile)
    np.allclose(parsed, grads, rtol=1e-8)
    np.allclose(parsed_pc, grads_pc, rtol=1e-8)

def get_nacs(logfile:str, fortfile: str, template: str, qmin: str, nacs: list, nacs_pc: list):
    test_interface = SHARC_MNDO()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    states, interstates = test_interface._get_states_interstates(logfile)
    energies = test_interface._get_energy(logfile)
    test_interface.QMout["h"] = np.zeros((max(states)+1,max(states)+1))
    for i in range(len(energies)):
        test_interface.QMout["h"][i][i] = energies[(1, i + 1)]
    parsed = test_interface._get_nacs(fortfile, interstates)
    parsed_pc = test_interface._get_nacs_pc(fortfile, interstates)
    np.allclose(parsed, nacs, rtol=1e-8)
    np.allclose(parsed_pc, nacs_pc, rtol=1e-8)
                

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
            np.array([[[0.0, 0.0, 0.0, -0.0], [0.0, 0.0, -0.0, -0.0], [0.0, -0.0, 0.0, -0.0], [-0.0, -0.0, -0.0, 0.0]], [[0.0, 0.0, 0.7881084543125534, 0.0], [0.0, 0.0, 0.45318409174088087, -0.0], [0.7881084543125534, 0.45318409174088087, -0.0, 0.0842254280021146], [0.0, -0.0, 0.0842254280021146, 0.0]], [[9.531640132568828, -9.350992232930748, 0.0, -1.6941386076206744], [-9.350992232930748, 7.820462770932457, 0.0, 7.650138770281812], [0.0, 0.0, 9.799385445894842, 0.0], [-1.6941386076206744, 7.650138770281812, 0.0, 27.970629498597052]]]),
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

                        [[ 1.06982799e-10,  0.00000000e+00,  6.69198835e-04],
                        [-2.76472403e-11,  0.00000000e+00,  1.06435412e-03],
                        [ 2.41882180e-04,  0.00000000e+00, -9.87740665e-05],
                        [-2.80138012e-04, -0.00000000e+00, -1.05674212e-04],
                        [ 2.80138013e-04, -0.00000000e+00, -1.05674213e-04],
                        [-2.41882260e-04, -0.00000000e+00, -9.87740728e-05]],

                        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

                        [[ 1.98209578e-11, -0.00000000e+00, -3.18993011e-04],
                        [-5.22412842e-12, -0.00000000e+00,  5.85948506e-04],
                        [ 1.13467467e-05,  0.00000000e+00, -5.02385224e-06],
                        [-1.27851694e-05,  0.00000000e+00, -5.34252637e-06],
                        [ 1.27851696e-05, -0.00000000e+00, -5.34252668e-06],
                        [-1.13467616e-05,  0.00000000e+00, -5.02385347e-06]]],


                        [[[-1.06982799e-10, -0.00000000e+00, -6.69198835e-04],
                        [ 2.76472403e-11, -0.00000000e+00, -1.06435412e-03],
                        [-2.41882180e-04, -0.00000000e+00,  9.87740665e-05],
                        [ 2.80138012e-04,  0.00000000e+00,  1.05674212e-04],
                        [-2.80138013e-04,  0.00000000e+00,  1.05674213e-04],
                        [ 2.41882260e-04,  0.00000000e+00,  9.87740728e-05]],

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

                        [[-8.96050321e-11,  0.00000000e+00,  2.02505686e-03],
                        [ 2.32658680e-11,  0.00000000e+00, -3.49317977e-03],
                        [-2.26166025e-04,  0.00000000e+00,  8.34285497e-05],
                        [ 2.53549841e-04, -0.00000000e+00,  9.74009746e-05],
                        [-2.53549841e-04,  0.00000000e+00,  9.74009755e-05],
                        [ 2.26166091e-04, -0.00000000e+00,  8.34285551e-05]]],


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


                        [[[-1.98209578e-11,  0.00000000e+00,  3.18993011e-04],
                        [ 5.22412842e-12,  0.00000000e+00, -5.85948506e-04],
                        [-1.13467467e-05, -0.00000000e+00,  5.02385224e-06],
                        [ 1.27851694e-05, -0.00000000e+00,  5.34252637e-06],
                        [-1.27851696e-05,  0.00000000e+00,  5.34252668e-06],
                        [ 1.13467616e-05, -0.00000000e+00,  5.02385347e-06]],

                        [[ 8.96050321e-11, -0.00000000e+00, -2.02505686e-03],
                        [-2.32658680e-11, -0.00000000e+00,  3.49317977e-03],
                        [ 2.26166025e-04, -0.00000000e+00, -8.34285497e-05],
                        [-2.53549841e-04,  0.00000000e+00, -9.74009746e-05],
                        [ 2.53549841e-04, -0.00000000e+00, -9.74009755e-05],
                        [-2.26166091e-04,  0.00000000e+00, -8.34285551e-05]],

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
    
def test_template():
    tests = [
        (
            "inputs/mndo/templatetest1.dat",
            {
                "nciref": 3,
                "kitscf": 5000,
                "ici1": 2,
                "ici2": 1,
                "ncigrd": 3,
                "dstep": 1e-5,
                "act_orbs": [4,6,7],
                "iroot": 4,
                "mminp": 2,
                "numatm": 1,
                "movo": 1,
                "grads": [1,2,4],
                "kharge": 1,
                "imomap": 3,
            },
        ),
        (
            "inputs/mndo/templatetest2.dat",
            {
                "nciref": 6,
                "kitscf": 9999,
                "ici1": 3,
                "ici2": 2,
                "ncigrd": 4,
                "dstep": 1e-6,
                "act_orbs": [4, 5, 6, 7, 8],
                "iroot": 4,
                "mminp": 0,
                "numatm": 1,
                "movo": 1,
                "grads": [1, 2, 3, 4],
                "kharge": 0,
                "imomap": 0,
            },
        ),
    ]
    for template, ref in tests:
        test_interface = SHARC_MNDO()
        test_interface.setup_mol(os.path.join(PATH, "inputs/mndo/QM1.in"))
        test_interface.read_template(os.path.join(PATH, template))
        for k, v in ref.items():
            assert test_interface.QMin["template"][k] == v