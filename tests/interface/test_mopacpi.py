import pytest
import os
import numpy as np
from SHARC_MOPACPI import SHARC_MOPACPI
from utils import expand_path

PATH = expand_path("$SHARC/../tests/interface")

def setup_interface(path: str, maps: dict):
    test_interface = SHARC_MOPACPI()
    test_interface.setup_mol(path)
    test_interface._read_resources = True
    test_interface._read_template = True
    test_interface.read_requests(path)
    test_interface.setup_interface()
    for k, v in maps.items():
        assert test_interface.QMin.maps[k] == v, test_interface.QMin.maps[k]


    

def get_energy(outfile: str, template: str, qmin: str, energies: list):
    test_interface = SHARC_MOPACPI()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_energy(outfile)
    for i,k in enumerate(parsed):
        assert np.allclose(energies[i] ,k, rtol=1e-5)


def get_tdm(outfile: str, template: str, qmin: str, tdms: list):
    test_interface = SHARC_MOPACPI()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_dip(outfile)
    assert np.allclose(parsed, tdms, rtol=1e-4)
    

def get_grads(outfile: str, template: str, qmin: str, grads: list):
    test_interface = SHARC_MOPACPI()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_grad(outfile)
    assert np.allclose(parsed, grads, rtol=1e-5)

def get_nacs(logfile:str , template: str, qmin: str, nacs: list):
    test_interface = SHARC_MOPACPI()
    test_interface.setup_mol(qmin)
    test_interface._read_resources = True
    test_interface.read_template(template)
    test_interface.setup_interface()
    test_interface.read_requests(qmin)
    parsed = test_interface._get_nac(logfile)
    assert np.allclose(parsed, nacs, rtol=1e-5)


def test_requests1():
    tests = [os.path.join(PATH, "inputs/mopacpi/QM1.in")]
    for i in tests:
        test_interface = SHARC_MOPACPI()
        test_interface.setup_mol(i)
        test_interface._read_template = True
        test_interface._read_resources = True
        test_interface.read_requests(i)                

def test_requests2():
    tests = [os.path.join(PATH, "inputs/mopacpi/QM2.in")]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_MOPACPI()
            test_interface.setup_mol(i)
            test_interface._read_template = True
            test_interface._read_resources = True
            test_interface.read_requests(i)

def test_requests3():
    tests = [os.path.join(PATH, "inputs/mopacpi/QM4.in"), os.path.join(PATH, "inputs/mopacpi/QM5.in")]
    for i in tests:
        with pytest.raises(ValueError):
            test_interface = SHARC_MOPACPI()
            test_interface.setup_mol(i)
            test_interface.setup_interface()

def test_energies():
    tests = [
        (
            os.path.join(PATH, "inputs/mopacpi/MOPACPI_nx.epot"),
            os.path.join(PATH, "inputs/mopacpi/MOPACPI.template"),
            os.path.join(PATH, "inputs/mopacpi/QM1.in"),
            np.array([-13.938534338, -13.549479661, -13.843932398,-13.843932398])
        ),
        (
            os.path.join(PATH, "inputs/mopacpi/MOPACPI_nx_3.epot"),
            os.path.join(PATH, "inputs/mopacpi/MOPACPI.template"),
            os.path.join(PATH, "inputs/mopacpi/QM3.in"), 
            np.array([ -13.938523330494123, -13.549449276196198, -13.843915301216297,-13.843915301216297])
        )
    ]
    for outfile, template, qmin, energies in tests:
        get_energy(outfile, template, qmin, energies)

def test_tdms():
    tests = [
        (
            os.path.join(PATH, "inputs/mopacpi/MOPACPI_nx.dipoles"),
            os.path.join(PATH, "inputs/mopacpi/MOPACPI.template"),
            os.path.join(PATH, "inputs/mopacpi/QM1.in"),
            np.array([[[ -0.0000000000,       0.4792711246,       0.0000000000,       0.0000000000],
            [  0.4792711246,      -0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000]],
            [[ 0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000]],
            [[-6.2982634199,       0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,      -7.9397282407,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000],
            [  0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000]]])
        
        
        )
    ]
    for outfile, template, qmin, tdms in tests:
        get_tdm(outfile, template, qmin, tdms)


def test_grads():
    tests = [
        (
            os.path.join(PATH, "inputs/mopacpi/MOPACPI_nx.grad.all"),
            os.path.join(PATH, "inputs/mopacpi/MOPACPI.template"),
            os.path.join(PATH, "inputs/mopacpi/QM1.in"),
            np.array([[[2.4445109612154476E-016,   4.9740018077808390E-018, -0.10968883448668214],     
            [-2.2733378030257359E-014,   2.7233601303131871E-016,  0.12245301068996960],     
            [-7.0356435068263772E-003,   3.5026107237178580E-016,   1.0632154004534405E-002],
            [-2.2900718499025949E-002,  -3.2375012131425658E-016,  -1.7014242106157131E-002],
            [2.2900718499086835E-002,  8.7294357267350548E-017,  -1.7014242106157040E-002],
            [7.0356435068263997E-003,  -3.9111532316397928E-016,   1.0632154004534572E-002]],
            [[1.6066932898744060E-016,  -8.2486803626030848E-018, -0.25936940363034983],     
            [-2.5853124712802705E-014,   8.0957564129642854E-016,  0.28924089486714449],     
            [-1.9151407292171217E-002,   4.6310294118487925E-015,   2.8050556253268157E-002],
            [-6.9105633605104272E-002,  -4.5983702643904690E-015,  -4.2986301871621488E-002],
            [6.9105633605177935E-002,   3.9453750237750595E-015,  -4.2986301871621446E-002],
            [1.9151407292171265E-002,  -4.7793611321672078E-015,   2.8050556253268313E-002]],
            [[2.4873203635793405E-016,   1.3146835719513180E-016, -0.72606482324058819],     
            [-2.2773979897847025E-014,   2.9195503316289468E-016,  0.53337499941247235],     
            [-0.10691379384726632,        4.0036928560222781E-017,  0.16791886226866143],     
            [-9.1468133469815249E-002,  -5.4706949171999517E-016,  -7.1573950354517038E-002],
            [9.1468133469837232E-002,   1.4878010613192303E-016,  -7.1573950354517024E-002],
            [0.10691379384726642,       -6.5170933330176890E-017,  0.16791886226866165]],     
            [[2.4873203635793405E-016,   1.3146835719513180E-016, -0.72606482324058819],     
            [-2.2773979897847025E-014,   2.9195503316289468E-016,  0.53337499941247235],     
            [-0.10691379384726632,        4.0036928560222781E-017,  0.16791886226866143],     
            [-9.1468133469815249E-002,  -5.4706949171999517E-016,  -7.1573950354517038E-002],
            [9.1468133469837232E-002,   1.4878010613192303E-016,  -7.1573950354517024E-002],
            [0.10691379384726642,       -6.5170933330176890E-017,  0.16791886226866165 ]]])        
        )
    ]
    for outfile, template, qmin, grads in tests:
        get_grads(outfile, template, qmin, grads)

def test_nacs():
    tests = [
        (
            os.path.join(PATH, "inputs/mopacpi/MOPACPI_nx.nad_vectors"),
            os.path.join(PATH, "inputs/mopacpi/MOPACPI.template"),
            os.path.join(PATH, "inputs/mopacpi/QM1.in"),
            np.array([[[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
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
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]]),
        )
    ]
    for logfile, template, qmin, nacs in tests:
        get_nacs(logfile, template, qmin, nacs)
    

def test_template():
    tests = [
        (
            "inputs/mopacpi/MOPACPI.template",
            {
                "ham"           : "AM1",
                "external_par"  : 54,
                "numb_elec"     : 3,
                "numb_orb"      : 2,
                "flocc"         : 0.10,
                "memory"        : 1100,
                "meci"          : 20,
                "mxroot"        : 20,
                "singlet"       : True,
                "micros"        : None,
                "add_pot"       : False,
                "qmmm"          : None,
                "link_atoms"    : None,
                "link_atom_pos" : None,
                "force_field"   : None,
            },
        ),
        (
            "inputs/mopacpi/MOPACPI.template_test",
            {
                "ham"           : "AM1",
                "external_par"  : None,
                "numb_elec"     : 6,
                "numb_orb"      : 3,
                "flocc"         : 0.10,
                "memory"        : 1100,
                "meci"          : 20,
                "mxroot"        : 20,
                "singlet"       : True,
                "micros"        : None,
                "add_pot"       : False,
                "qmmm"          : None,
                "link_atoms"    : None,
                "link_atom_pos" : None,
                "force_field"   : None,
            },
        ),
    ]
    for template, ref in tests:
        test_interface = SHARC_MOPACPI()
        test_interface.setup_mol(os.path.join(PATH, "inputs/mopacpi/QM1.in"))
        test_interface.read_template(os.path.join(PATH, template))
        for k, v in ref.items():
            assert test_interface.QMin["template"][k] == v

# def test_template_error():

#     tests = ["inputs/mopacpi/MOPACPI.template_test_error"]

#     for template in tests:
#         with pytest.raises(ValueError):
#             test_interface = SHARC_MOPACPI()
#             test_interface.setup_mol(os.path.join(PATH, "inputs/mopacpi/QM1.in"))
#             test_interface.read_template(os.path.join(PATH, template))

# def same(self, other):
#     if isinstance(self, other.__class__):
#         A = self.nmstates == other.nmstates and self.natom == other.natom and self.npc == other.npc and self.point_charges == other.point_charges
#         B = self.states == other.states  and  np.allclose(self.h, other.h, rtol=1e-5) and  np.allclose(self.dm, other.dm, rtol=1e-5) and np.allclose(self.grad, other.grad, rtol=1e-5) and np.allclose(self.grad_pc, other.grad_pc, rtol=1e-5) and np.allclose(self.nacdr, other.nacdr, rtol=1e-5) and np.allclose(self.nacdr_pc, other.nacdr_pc, rtol=1e-5)
#         return A and B
#     return False
