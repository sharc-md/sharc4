#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

# IMPORTS
# external
import os
import datetime
import numpy as np

# internal
from SHARC_HYBRID import SHARC_HYBRID
from factory import factory
from utils import ATOM, mkdir, readfile, InDir, itnmstates, question
from error import Error
from globals import DEBUG, PRINT
from constants import ATOMCHARGE, FROZENS
from copy import deepcopy

version = '3.0'
versiondate = datetime.datetime(2023, 8, 24)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)


class QMMM(SHARC_HYBRID):

    _version = version
    _versiondate = versiondate
    _changelogstring = changelogstring
    _step = 0
    _qm_s = 0.3
    _mm_s = 1 - _qm_s

    def description(self):
        pass

    def version(self):
        return self._version

    def get_infos(self):

        pass

    def name(self) -> str:
        return "QM/MM"

    def prepare(self, INFOS, iconddir):

        QMin = self._QMin
        # obtain the statemap
        QMin.maps.statemap = {
            i + 1: [*v]
            for i, v in enumerate(itnmstates(QMin.molecules.states))
        }
        if 'savedir' not in QMin:
            print(
                'savedir not specified in QM.in, setting savedir to current directory!'
            )
            QMin.save.savedir = os.getcwd()
        # dynamic import of both interfaces
        self.qm_interface = factory(QMin.template['qm-program'])(
            self._DEBUG, self._PRINT, self._persistent)

        self.mml_interface = factory(QMin.template['mm-program'])(
            self._DEBUG, self._PRINT, self._persistent)
        qm_name = self.qm_interface.__class__.__name__
        mml_name = self.mml_interface.__class__.__name__
        # folder setup and savedir
        qm_savedir = os.path.join(QMin.save.savedir,
                                  'QM_' + QMin.template['qm-program'].upper())
        if not os.path.isdir(qm_savedir):
            mkdir(qm_savedir)
        self.qm_interface._QMin.save.savedir = qm_savedir
        self.qm_interface._QMin.resources.scratchdir = os.path.join(
            QMin.resources.scratchdir,
            'QM_' + QMin.template['qm-program'].upper())
        self.qm_interface.prepare()

        mml_savedir = os.path.join(
            QMin.save.savedir, 'MML_' + QMin.template['mm-program'].upper())
        if not os.path.isdir(mml_savedir):
            mkdir(mml_savedir)
        self.mml_interface._QMin.save.savedir = mml_savedir
        self.mml_interface._QMin.resources.scratchdir = os.path.join(
            QMin.resources.scratchdir,
            'MML_' + QMin.template['mm-program'].upper())
        self.mml_interface.prepare()
        if QMin.template['embedding'] == 'subtractive':
            self.mms_interface = factory(
                QMin.template['mm-program'])(self._DEBUG, self._PRINT,
                                             self._persistent)
            mms_name = self.mms_interface.__class__.__name__
            mms_savedir = os.path.join(
                QMin.save.savedir,
                'MMS_' + QMin.template['mm-program'].upper())
            if not os.path.isdir(mms_savedir):
                mkdir(mms_savedir)
            self.mms_interface._QMin.save.savedir = mms_savedir
            self.mms_interface._QMin.resources.scratchdir = os.path.join(
                QMin.resources.scratchdir,
                'MMS_' + QMin.template['mm-program'].upper())
            self.mms_interface.prepare

    def printQMout(self):
        pass

    def print_qmin(self):
        pass

    def write_step_file(self):
        pass

    @staticmethod
    def about():
        pass

    def versiondate(self):
        return self._versiondate

    def changelogstring(self):
        return self._changelogstring

    def authors(self) -> str:
        return 'Sebastian Mai, Maximilian Xaver Tiefenbacher and Severin Polonius'

    # TODO: update for other embeddings
    def get_features(self):
        tmp_file = question(
            "Please specify the path to your QMMM.template file",
            str,
            default="QMMM.template")
        self.read_template(tmp_file)
        qm_features = self.qm_interface.get_features()
        mm_features = self.mml_interface.get_features()
        if "point_charges" in qm_features:
            qm_features.remove("point_charges")
        else:
            raise Exception(
                "Your QM interface needs to be able to include point charges in its calculations"
            )
        if "grad" in qm_features and "grad" not in mm_features:
            qm_features.remove("grad")

        if "h" in qm_features and "h" not in mm_features:
            qm_features.remove("h")

        return qm_features

    def _step_logic(self):
        super()._step_logic()

    def request_logics(self):
        super().request_logic()

    def read_requests(self):
        super().read_requests()

    def read_template(self, template_filename="QMMM.template"):
        QMin = self._QMin

        special = {'qmmm_table': ''}
        strings = {
            'qm-program': '',
            'mm-program': '',
            'embedding': 'subtractive'
        }
        paths = {
            'mms-dir': '',  # paths to prepared calculations
            'mml-dir': '',
            'qm-dir': ''
        }
        bools = {'mm_dipole': False}
        super().read_template(template_filename)

        # check
        allowed_embeddings = ['additive', 'subtractive']
        if QMin.template['embedding'] not in allowed_embeddings:
            raise Error(
                'Chosen embedding "{}" is not available (available: {})'.
                format(QMin.template['embedding'],
                       ', '.join(allowed_embeddings)))

        required: set = {
            'qm-program', 'mm-program', 'qmmm_table', 'mml-dir', 'qm-dir'
        }
        if QMin.template['embedding'] == 'subtractive':
            required.add('mms-dir')

        if not required.issubset(QMin.template.keys()):
            raise Error(
                '"{}" not specified in {}'.format(
                    '", "'.join(
                        filter(lambda x: x not in QMin.template, required)),
                    template_filename), 78)
        # QMin.template['qmmm'] = True  # this is a qmmm interface

        self.atoms = [
            ATOM(i, v[0].lower() == 'qm', v[1], [0., 0., 0.], set(v[2:]))
            for i, v in enumerate(QMin.template['qmmm_table'])
        ]

        # sanitize mmatoms
        # set links
        self.qm_ids = []
        self.mm_ids = []
        self._linkatoms: set = {}  # set to hold tuple
        for i in self.atoms:
            for jd in i.bonds:
                # jd = j - 1
                if i.id == jd:
                    raise Error(f'Atom bound to itself:\n{i}')
                j = self.atoms[jd]
                if i.id not in j.bonds:
                    j.bonds.add(i.id)
                if i.qm != j.qm:
                    self._linkatoms.add((i.id, j.id) if i.qm else (j.id, i.id))
        # sort out qm atoms
            self.qm_ids.append(i.id) if i.qm else self.mm_ids.append(i.id)
        self._num_qm = len(self.qm_ids)
        self._num_mm = len(self.mm_ids)
        self.mm_links = set(
            mm for _, mm in self._linkatoms
        )  # set of all mm_ids in link bonds (deleted in point charges!)

        # check of linkatoms: map linkatoms to sets of unique qm and mm ids: decreased number -> Error
        if len(self._linkatoms) > len(set(map(lambda x: x[0],
                                              self._linkatoms))):
            raise Error('Some QM atom is involved in more than one link bond!',
                        23)
        if len(self._linkatoms) > len(set(map(lambda x: x[1],
                                              self._linkatoms))):
            raise Error('Some MM atom is involved in more than one link bond!',
                        23)
        self._linkatoms = list(self._linkatoms)
        self._read_template = True

    def read_resources(self, resources_filename='QMMM.resources'):
        super().read_resources(resources_filename)
        self._read_resources = True

    def setup_interface(self):
        QMin = self._QMin
        # obtain the statemap
        QMin.maps.statemap = {
            i + 1: [*v]
            for i, v in enumerate(itnmstates(QMin.molecules.states))
        }
        # prepare info for both interfaces
        el = QMin.molecule.elements
        n_link = len(self._linkatoms)
        qm_el = [a.symbol for a in self.atoms if a.qm] + ['H'] * n_link
        # setup mol for qm
        qm_QMin = self.qm_interface._QMin
        qm_QMin.molecule.elements = qm_el
        qm_QMin.molecule.Atomcharge = sum(map(lambda x: ATOMCHARGE[x], qm_el))
        qm_QMin.molecule.frozcore = sum(map(lambda x: FROZENS[x], qm_el))
        qm_QMin.molecule.natom = self._num_qm + n_link
        qm_QMin.molecules.states = QMin.molecule.states
        qm_QMin.maps.statemap = QMin.maps.statemap
        qm_QMin.molecule.nmstates = QMin.molecule.nmstates
        qm_QMin.molecule.unit = QMin.molecule.unit
        self.qm_interface._setup_mol = True

        # setup mol for mml
        mml_QMin = self.mml_interface._QMin
        mml_QMin.molecule.elements = el
        mml_QMin.moleculeAtomcharge = QMin.molecule.Atomcharge
        mml_QMin.molecule.frozcore = QMin.molecule.frozcore
        mml_QMin.molecule.natom = QMin.molecule.natom
        mml_QMin.molecules.states = [1]
        mml_QMin.moleculenm.states = 1
        mml_QMin.molecule.unit = QMin.molecule.unit
        self.mml_interface._setup_mol = True

        # read template and resources
        # print('-' * 80, f'{"preparing QM INTERFACE (" + qm_name + ")":^80}', '-' * 80, sep='\n')
        with InDir(QMin.template['qm-dir']) as _:
            self.qm_interface.read_resources()
            qm_QMin.save.savedir = qm_savedir  # overwrite savedir
            self.qm_interface.read_template()
            self.qm_interface.setup_interface()

        # print('-' * 80, f'{"preparing MM INTERFACE (large system) (" + mml_name + ")":^80}', '-' * 80, sep='\n')
        with InDir(QMin.template['mml-dir']) as _:
            self.mml_interface.read_resources()
            mml_QMin.save.savedir = mml_savedir  # overwrite savedir
            self.mml_interface.read_template()
            self.mml_interface.setup_interface()
        # switch for subtractive
        if QMin.template['embedding'] == 'subtractive':

            # setup mol for mms
            ########## needs revision ###############
            mms_el = [a.symbol for a in QMin.molecule['atoms'] if a.qm]
            mms_el += [
                QMin.molecule.molecule['atoms'][x[1]].symbol
                for x in self._linkatoms
            ]  # add symbols of link atoms (original element -> proper bonded terms in MM calc)
            mms_QMin = self.mms_interface._QMin
            mms_QMin.molecule.elements = mms_el
            mms_QMin.molecule.Atomcharge = sum((ATOMCHARGE[x] for x in mms_el))
            mms_QMin.molecule['frozcore'] = sum((FROZENS[x] for x in mms_el))
            mms_QMin.molecule['natom'] = self._num_qm + n_link
            mms_QMin.molecules.states = [1]
            mms_QMin.molecule['nmstates'] = 1
            mms_QMin.molecule['unit'] = QMin.molecule['unit']
            self.mms_interface._setup_mol = True

            # print('-' * 80, f'{"preparing MM INTERFACE (small system) (" + mms_name + ")":^80}', '-' * 80, sep='\n')
            # read template and resources
            with InDir(QMin.template['mms-dir']) as _:
                self.mms_interface.read_resources()
                mms_QMin.save.savedir = mms_savedir  # overwrite savedir
                self.mms_interface.read_template()
                self.mms_interface.setup_interface()

            self._qm_interface_QMin_backup = deepcopy(self.qm_interface._QMin)
        return

    def run(self):
        QMin = self._QMin

        # reset qm_interface_ QMin
        self.qm_interface._QMin = deepcopy(self._qm_interface_QMin_backup)
        # set coords
        qm_coords = np.array([
            QMin.coords.data["coords"][self.qm_ids[i]].copy()
            for i in range(self._num_qm)
        ])
        if len(self._linkatoms) > 0:
            # get linkatom coords
            def get_link_coord(link: tuple) -> np.ndarray[float]:
                qm_id, mm_id = link
                return QMin.coords.data["coords"][
                    qm_id] * self._qm_s + QMin.coords.data["coords"][
                        mm_id] * self._mm_s

            link_coords = np.fromiter(map(get_link_coord, self._linkatoms),
                                      dtype=float,
                                      count=len(self._linkatoms))
            self.qm_interface._QMin.coords.data['coords'] = np.vstack(
                (qm_coords, link_coords))
        else:
            self.qm_interface._QMin.coords.data['coords'] = qm_coords

        self.mml_interface._QMin.coords.data['coords'] = QMin.coords.data[
            'coords'].copy()
        # setting requests for qm and mm regions based on the QMMM requests

        all_requests = QMin.requests
        qm_requests = []
        mm_requests = []

        for key, value in all_requests.items():
            match key:
                case "h":
                    qm_requests[key] = value
                    mm_requests[key] = value
                case "grad":
                    qm_requests[key] = value
                    mm_requests[key] = [1]
                case _:
                    qm_requests[key] = value
                # TODO: also to MMs: dm, multipolar_fit

        self.qm_interface.set_requests(qm_requests)
        self.mml_interface.set_requests(mm_requests)
        if QMin.template['embedding'] == 'subtractive':
            self.mms_interface.set_requests(mm_requests)
        #  for i in filter(lambda x: x in QMin, possible):
        #  self.qm_interface._QMin[i] = QMin[i]
        #  self.qm_interface._request_logic()
        #  self.qm_interface._step_logic()
        ############# update request logic ################

        # set mm requests
        #  possible = ['cleanup', 'backup', 'h', 'dm', 'step']
        #  for i in filter(lambda x: x in QMin, possible):
        #  if 'dm' in QMin:
        #  self.mml_interface._QMin['dm'] = [1]
        #  self.mml_interface._QMin['multipolar_fit'] = True

        #  self.mml_interface._request_logic()

        # calc mm
        # print('-' * 80, f'{"running MM INTERFACE (large system)":^80}', '-' * 80, sep='\n')
        with InDir(QMin.template['mml-dir']) as _:
            self.mml_interface.run()
            self.mml_interface.getQMout()
            # is analogous to the density fit from QM interfaces -> generated upon same request
            raw_pc = self.mml_interface._QMout['multipolar_fit']

        # redistribution of mm pc of link atom (charge is not the same in qm calc but pc would be too close)
        for _, mmid in self._linkatoms:
            atom: ATOM = self.atoms[mmid]
            # -> redistribute charge to neighboring atoms (look in old ORCA line 1300)
            neighbor_ids = [
                x.id for x in map(lambda y: self.atoms[y], atom.bonds)
                if not x.qm
            ]
            chrg = raw_pc[mmid] / len(neighbor_ids)
            for nb in neighbor_ids:
                raw_pc[nb] += chrg
        self._pc_mm = [[*QMin['coords'][i, :].tolist(), raw_pc[i]]
                       for i in self.mm_ids
                       if i not in self.mm_links]  # shallow copy

        if QMin.template['embedding'] == 'subtractive':
            # print('-' * 80, f'{"running MM INTERFACE (small system)":^80}', '-' * 80, sep='\n')
            self.mms_interface._QMin['coords'] = qm_coords
            with InDir(QMin.template['mms-dir']) as _:
                self.mms_interface.run()
                self.mms_interface.getQMout()

        # calc qm
        # print('-' * 80, f'{"running QM INTERFACE":^80}', '-' * 80, sep='\n')
        # pc: list[list[float]] = each pc is x, y, z, qpc[p[mmid][1]][3] = 0.  # set the charge of the mm atom to zero
        self.qm_interface._QMin['point_charges'] = self._pc_mm
        # TODO indicator to include pointcharges?

        with InDir(QMin.template['qm-dir']) as _:
            self.qm_interface.run()
            self.qm_interface.getQMout()
            self.qm_interface.write_step_file()

        # print(datetime.datetime.now())
        # print('#================ END ================#')
        # s2 = time.perf_counter_ns()
        # print('Timing: QMMM', (s2 - s1) * 1e-6, 'ms')

    def getQMout(self):
        # s1 = time.perf_counter_ns()
        qm_QMout = self.qm_interface._QMout
        QMin = self._QMin
        QMout = self._QMout

        def add_to_xyz(xyz1, xyz2, fac=1.):
            xyz1[0] += xyz2[0] * fac
            xyz1[1] += xyz2[1] * fac
            xyz1[2] += xyz2[2] * fac

        mm_e = float(self.mml_interface._QMout['h'][0][0])
        if QMin.template['embedding'] == 'subtractive':
            mm_e -= float(self.mms_interface._QMout['h'][0][0])

        QMout['qmmm'] = {'MMEnergy_terms': {'MM Energy': mm_e}}
        # Hamiltonian
        if 'h' in qm_QMout:
            QMout['h'] = [[j for j in i] for i in qm_QMout['h']]
            for i in range(QMin['nmstates']):
                QMout['h'][i][i] += mm_e
        # print('     getQMout ene', (time.perf_counter_ns() - s1) * 1e-6, 'ms')
        # gen output
        if 'grad' in QMin.requests.grad:
            qm_grad = qm_QMout['grad']
            mm_grad = self.mml_interface._QMout['grad'][0]

            if QMin.template['embedding'] == 'subtractive':
                mms_grad = self.mms_interface.QMout['grad'][0]

                for n, qm_id in enumerate(self.qm_ids):  # loop over qm atoms
                    # add to qm_id in big mm list
                    add_to_xyz(mm_grad[qm_id], mms_grad[n], fac=-1.)

            grad = {}
            # print('     getQMout grad1', (time.perf_counter_ns() - s1) * 1e-6, 'ms')

            for i, qm_grad_i in enumerate(qm_grad):
                # init gradient as mm_gradient
                grad[i] = [[x[0], x[1], x[2]] for x in mm_grad]

                # add gradient of all qm atoms for each state
                for n, qm_id in enumerate(self.qm_ids):
                    add_to_xyz(grad[i][qm_id], qm_grad_i[n])

                # linkatoms come after qm atoms
                for n, link_id in enumerate(self._linkatoms):
                    qm_id, mm_id = self._linkatoms[n]
                    qm_grad_in = qm_grad_i[n + self._num_qm]
                    add_to_xyz(grad[i][mm_id], qm_grad_in, self._mm_s)
                    add_to_xyz(grad[i][qm_id], qm_grad_in, self._qm_s)
            # print('     getQMout grad2', (time.perf_counter_ns() - s1) * 1e-6, 'ms')

            if 'pc_grad' in qm_QMout:  # apply pc grad
                for i, grad_i in enumerate(qm_QMout['pc_grad']):
                    # mm_ids stay in order even after grouping qm_ids at the fron and deleting link mm atoms
                    # -> get all residual mm ids in order for correct order in pcgrad
                    for grad_in, mm_id in zip(
                            grad_i,
                            filter(lambda i: i not in self.mm_links,
                                   self.mm_ids)):
                        add_to_xyz(grad[i][mm_id], grad_in)
            else:
                print("Warning: No 'pc_grad' in QMout of QM interface!")

            self._QMout['grad'] = grad
        # print('     getQMout pcgrad', (time.perf_counter_ns() - s1) * 1e-6, 'ms')

        if QMin.requests.nacdr:
            # nacs would have to inserted in the whole system matrix only for qm atoms
            nacdr = [[[[0., 0., 0.] for _ in range(QMin.molecule.natom)]
                      for _ in range(QMin.molecule.nmstates)]
                     for _ in range(QMin.moleculenmstates)]
            for i, s_i in enumerate(self.qm_interface._QMout['nacdr']):
                for j, s_j in enumerate(s_i):
                    for n, qm_id in enumerate(self.qm_ids):
                        add_to_xyz(nacdr[i][j][qm_id], s_j[n])
                    for n, link_id in enumerate(
                            self._linkatoms):  # linkatoms come after qm atoms
                        qm_id, mm_id = self._linkatoms[n]
                        nac = nacdr[i][j]
                        add_to_xyz(nac[mm_id], s_j[n + self._num_qm],
                                   self._mm_s)
                        add_to_xyz(nac[qm_id], s_j[n + self._num_qm],
                                   self._qm_s)
            QMout['nacdr'] = nacdr
        # print('     getQMout nac', (time.perf_counter_ns() - s1) * 1e-6, 'ms')

        if QMin.requests.dm:
            QMout['dm'] = self.qm_interface._QMout['dm']
            if QMin.template['mm_dipole']:
                for i, dm_i in enumerate(QMout['dm']):
                    mm_dm_i: float = self.mml_interface._QMout['dm'][i]
                    for dm_in in dm_i:
                        for dm_inm in dm_in:
                            dm_inm += mm_dm_i[0][
                                0]  # add mm dipole moment to all states
        if 'overlap' in QMin:
            QMout['overlap'] = self.qm_interface._QMout['overlap']

        # potentially print out other contributions and properties...
        for i in ['ion', 'prop', 'theodore']:
            if i in qm_QMout:
                QMout[i] = qm_QMout[i]

    def create_restart_files(self):
        self.qm_interface.create_restart_files()
        self.mml_interface.create_restart_files()
        if self._QMin.template['embedding'] == 'subtractive':
            self.mml_interface.create_restart_files()


if __name__ == "__main__":
    try:
        #qmmm = QMMM(DEBUG, PRINT)
        qmmm = QMMM()
        qmmm.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
    except Error:
        raise
