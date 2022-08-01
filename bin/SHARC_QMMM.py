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
import datetime
import numpy as np
from copy import deepcopy

# internal
from SHARC_INTERFACE import INTERFACE
from factory import factory
from utils import *
from constants import ATOMCHARGE, FROZENS

authors = 'Sebastian Mai and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 9, 3)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)


class QMMM(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring
    _step = 0
    _qm_s = 0.3
    _mm_s = 1 - _qm_s

    @property
    def version(self):
        return self._version

    @property
    def versiondate(self):
        return self._versiondate

    @property
    def changelogstring(self):
        return self._changelogstring

    @property
    def authors(self):
        return self._authors

    def read_template(self, template_filename="QMMM.template"):
        QMin = self._QMin

        special = {'qmmm_table': ''}
        strings = {'qm-program': '', 'mm-program': '', 'embedding': 'subtractive'}
        paths = {
            'mms-dir': '',    # paths to prepared calculations
            'mml-dir': '',
            'qm-dir': ''
        }
        bools = {'mm_dipole': False}

        lines = readfile(template_filename)
        QMin['template'] = {
            **bools,
            **self.parse_keywords(lines, paths=paths, bools=bools, special=special, strings=strings)
        }

        # check
        allowed_embeddings = ['additive', 'subtractive']
        if QMin['template']['embedding'] not in allowed_embeddings:
            raise Error(
                'Chosen embedding "{}" is not available (available: {}}'.format(
                    QMin['template']['embedding'], ', '.join(allowed_embeddings)
                )
            )

        required: set = {'qm-program', 'mm-program', 'qmmm_table', 'mml-dir', 'qm-dir'}
        if QMin['template']['embedding'] == 'subtractive':
            required.add('mms-dir')

        if not required.issubset(QMin['template'].keys()):
            raise Error(
                '"{}" not specified in {}'.format(
                    '", "'.join(filter(lambda x: x not in QMin['template'], required)), template_filename
                ), 78
            )

        QMin['atoms'] = [
            ATOM(i, v[0].lower() == 'qm', v[1], [0., 0., 0.], v[2], set(v[3:]))
            for i, v in enumerate(QMin['template']['qmmm_table'])
        ]

        # sanitize mmatoms
        # set links
        self.qm_ids = []
        self.mm_ids = []
        self._linkatoms: set = {}    # set to hold tuple
        for i in QMin['atoms']:
            for jd in i.bonds:
                # jd = j - 1
                if i.id == jd:
                    raise Error(f'Atom bound to itself:\n{i}')
                j = QMin['atoms'][jd]
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
        )    # set of all mm_ids in link bonds (deleted in point charges!)

        # check of linkatoms: map linkatoms to sets of unique qm and mm ids: decreased number -> Error
        if len(self._linkatoms) > len(set(map(lambda x: x[0], self._linkatoms))):
            raise Error('Some QM atom is involved in more than one link bond!', 23)
        if len(self._linkatoms) > len(set(map(lambda x: x[1], self._linkatoms))):
            raise Error('Some MM atom is involved in more than one link bond!', 23)
        self._linkatoms = list(self._linkatoms)
        self._read_template = True

    def read_resources(self, resources_filename='QMMM.resources'):
        super().read_resources(resources_filename)
        self._read_resources = True

    def setup_run(self):
        QMin = self._QMin
        # obtain the statemap
        QMin['statemap'] = {i + 1: [*v] for i, v in enumerate(itnmstates(QMin['states']))}
        if 'savedir' not in QMin:
            print('savedir not specified in QM.in, setting savedir to current directory!')
            QMin['savedir'] = os.getcwd()
        # dynamic import of both interfaces
        self.qm_interface: INTERFACE = factory(QMin['template']['qm-program']
                                               )(self._DEBUG, self._PRINT, self._persistent)

        self.mml_interface: INTERFACE = factory(QMin['template']['mm-program']
                                                )(self._DEBUG, self._PRINT, self._persistent)
        qm_name = self.qm_interface.__class__.__name__
        mml_name = self.mml_interface.__class__.__name__
        # folder setup and savedir
        qm_savedir = os.path.join(QMin['savedir'], 'QM_' + QMin['template']['qm-program'].upper())
        if not os.path.isdir(qm_savedir):
            mkdir(qm_savedir)
        self.qm_interface._QMin['savedir'] = qm_savedir
        self.qm_interface._QMin['scratchdir'] = os.path.join(
            QMin['scratchdir'], 'QM_' + QMin['template']['qm-program'].upper()
        )

        mml_savedir = os.path.join(QMin['savedir'], 'MML_' + QMin['template']['mm-program'].upper())
        if not os.path.isdir(mml_savedir):
            mkdir(mml_savedir)
        self.mml_interface._QMin['savedir'] = mml_savedir
        self.mml_interface._QMin['scratchdir'] = os.path.join(
            QMin['scratchdir'], 'MML_' + QMin['template']['mm-program'].upper()
        )

        # prepare info for both interfaces
        el = QMin['elements']
        n_link = len(self._linkatoms)
        qm_el = [a.symbol for a in QMin['atoms'] if a.qm] + ['H'] * n_link
        # setup mol for qm
        qm_QMin = self.qm_interface._QMin
        qm_QMin['elements'] = qm_el
        qm_QMin['Atomcharge'] = sum(map(lambda x: ATOMCHARGE[x], qm_el))
        qm_QMin['frozcore'] = sum(map(lambda x: FROZENS[x], qm_el))
        qm_QMin['natom'] = self._num_qm + n_link
        qm_QMin['states'] = QMin['states']
        qm_QMin['statemap'] = QMin['statemap']
        qm_QMin['nmstates'] = QMin['nmstates']
        qm_QMin['unit'] = QMin['unit']
        self.qm_interface._setup_mol = True

        # setup mol for mml
        mml_QMin = self.mml_interface._QMin
        mml_QMin['elements'] = el
        mml_QMin['Atomcharge'] = QMin['Atomcharge']
        mml_QMin['frozcore'] = QMin['frozcore']
        mml_QMin['natom'] = QMin['natom']
        mml_QMin['states'] = [1]
        mml_QMin['nmstates'] = 1
        mml_QMin['unit'] = QMin['unit']
        self.mml_interface._setup_mol = True

        # read template and resources
        # print('-' * 80, f'{"preparing QM INTERFACE (" + qm_name + ")":^80}', '-' * 80, sep='\n')
        with InDir(QMin['template']['qm-dir']) as _:
            self.qm_interface.read_resources()
            qm_QMin['savedir'] = qm_savedir    # overwrite savedir
            self.qm_interface.read_template()
            self.qm_interface.setup_run()

        # print('-' * 80, f'{"preparing MM INTERFACE (large system) (" + mml_name + ")":^80}', '-' * 80, sep='\n')
        with InDir(QMin['template']['mml-dir']) as _:
            self.mml_interface.read_resources()
            mml_QMin['savedir'] = mml_savedir    # overwrite savedir
            self.mml_interface.read_template()
            self.mml_interface.setup_run()
        # switch for subtractive
        if QMin['template']['embedding'] == 'subtractive':

            self.mms_interface: INTERFACE = factory(QMin['template']['mm-program']
                                                    )(self._DEBUG, self._PRINT, self._persistent)
            mms_name = self.mms_interface.__class__.__name__
            mms_savedir = os.path.join(QMin['savedir'], 'MMS_' + QMin['template']['mm-program'].upper())
            if not os.path.isdir(mms_savedir):
                mkdir(mms_savedir)
            self.mms_interface._QMin['savedir'] = mms_savedir
            self.mms_interface._QMin['scratchdir'] = os.path.join(
                QMin['scratchdir'], 'MMS_' + QMin['template']['mm-program'].upper()
            )
            # setup mol for mms

            mms_el = [a.symbol for a in QMin['atoms'] if a.qm]
            mms_el += [
                QMin['atoms'][x[1]].symbol for x in self._linkatoms
            ]    # add symbols of link atoms (original element -> proper bonded terms in MM calc)
            mms_QMin = self.mms_interface._QMin
            mms_QMin['elements'] = mms_el
            mms_QMin['Atomcharge'] = sum((ATOMCHARGE[x] for x in mms_el))
            mms_QMin['frozcore'] = sum((FROZENS[x] for x in mms_el))
            mms_QMin['natom'] = self._num_qm + n_link
            mms_QMin['states'] = [1]
            mms_QMin['nmstates'] = 1
            mms_QMin['unit'] = QMin['unit']
            self.mms_interface._setup_mol = True

            # print('-' * 80, f'{"preparing MM INTERFACE (small system) (" + mms_name + ")":^80}', '-' * 80, sep='\n')
            # read template and resources
            with InDir(QMin['template']['mms-dir']) as _:
                self.mms_interface.read_resources()
                mms_QMin['savedir'] = mms_savedir    # overwrite savedir
                self.mms_interface.read_template()
                self.mms_interface.setup_run()
        return

    def run(self):
        QMin = self._QMin
        # set coords
        qm_coords = np.array([QMin['coords'][self.qm_ids[i]].copy() for i in range(self._num_qm)])
        if len(self._linkatoms) > 0:
            # get linkatom coords
            def get_link_coord(link: tuple) -> np.ndarray[float]:
                qm_id, mm_id = link
                return QMin['coords'][qm_id] * self._qm_s + QMin['coords'][mm_id] * self._mm_s

            link_coords = np.fromiter(map(get_link_coord, self._linkatoms), dtype=float, count=len(self._linkatoms))
            self.qm_interface._QMin['coords'] = np.vstack((qm_coords, link_coords))
        else:
            self.qm_interface._QMin['coords'] = qm_coords

        self.mml_interface._QMin['coords'] = QMin['coords'].copy()

        # set qm requests: grad, nac, soc,
        possible = [
            'cleanup', 'backup', 'h', 'soc', 'dm', 'grad', 'overlap', 'dmdr', 'nac', 'nacdr', 'socdr', 'ion',
            'theodore', 'phases', 'step', 'restart'
        ]
        for i in filter(lambda x: x in QMin, possible):
            self.qm_interface._QMin[i] = QMin[i]
        self.qm_interface._request_logic()
        self.qm_interface._step_logic()

        # set mm requests
        possible = ['cleanup', 'backup', 'h', 'dm', 'step']
        for i in filter(lambda x: x in QMin, possible):
            self.mml_interface._QMin[i] = QMin[i]
        if 'grad' in QMin:
            self.mml_interface._QMin['grad'] = [1]
        if 'dm' in QMin:
            self.mml_interface._QMin['dm'] = [1]
        self.mml_interface._QMin['multipolar_fit'] = True

        self.mml_interface._request_logic()

        # calc mm
        # print('-' * 80, f'{"running MM INTERFACE (large system)":^80}', '-' * 80, sep='\n')
        with InDir(QMin['template']['mml-dir']) as _:
            self.mml_interface.run()
            self.mml_interface.getQMout()
            # is analogous to the density fit from QM interfaces -> generated upon same request
            raw_pc = self.mml_interface._QMout['multipolar_fit']

        # redistribution of mm pc of link atom (charge is not the same in qm calc but pc would be too close)
        for _, mmid in self._linkatoms:
            atom: ATOM = QMin['atoms'][mmid]
            # -> redistribute charge to neighboring atoms (look in old ORCA line 1300)
            neighbor_ids = [x.id for x in map(lambda y: QMin['atoms'][y], atom.bonds) if not x.qm]
            chrg = raw_pc[mmid] / len(neighbor_ids)
            for nb in neighbor_ids:
                raw_pc[nb] += chrg
        self._pc_mm = [
            [*QMin['coords'][i, :].tolist(), raw_pc[i]] for i in self.mm_ids if i not in self.mm_links
        ]    # shallow copy

        if QMin['template']['embedding'] == 'subtractive':
            # print('-' * 80, f'{"running MM INTERFACE (small system)":^80}', '-' * 80, sep='\n')
            self.mms_interface._QMin['coords'] = qm_coords
            with InDir(QMin['template']['mms-dir']) as _:
                self.mms_interface.run()
                self.mms_interface.getQMout()

        # calc qm
        # print('-' * 80, f'{"running QM INTERFACE":^80}', '-' * 80, sep='\n')
        # pc: list[list[float]] = each pc is x, y, z, qpc[p[mmid][1]][3] = 0.  # set the charge of the mm atom to zero
        self.qm_interface._QMin['point_charges'] = self._pc_mm
        # TODO indicator to include pointcharges?

        with InDir(QMin['template']['qm-dir']) as _:
            self.qm_interface.run()
            self.qm_interface.getQMout()
            self.qm_interface.write_step_file()

        # print(datetime.datetime.now())
        # print('#================ END ================#')

    def getQMout(self):
        qm_QMout = self.qm_interface._QMout
        QMin = self._QMin
        QMout = self._QMout

        def add_to_xyz(xyz1, xyz2, fac=1.):
            xyz1[0] += xyz2[0] * fac
            xyz1[1] += xyz2[1] * fac
            xyz1[2] += xyz2[2] * fac

        mm_e = float(self.mml_interface._QMout['h'][0][0])
        if QMin['template']['embedding'] == 'subtractive':
            mm_e -= float(self.mms_interface._QMout['h'][0][0])
        # Hamiltonian
        if 'h' in qm_QMout:
            QMout['h'] = deepcopy(qm_QMout['h'])
            for i in range(QMin['nmstates']):
                QMout['h'][i][i] += mm_e
        # gen output
        if 'grad' in QMin:
            qm_grad = qm_QMout['grad']
            mm_grad = self.mml_interface._QMout['grad'][0]

            if QMin['template']['embedding'] == 'subtractive':
                mms_grad = self.mms_interface.QMout['grad'][0]
                for atom in range(len(mms_grad)):    # loop over atoms
                    for qm_grad_i in qm_grad:
                        add_to_xyz(qm_grad_i[atom], mms_grad[atom], fac=-1.)    # check if id is the same in both calcs

            grad = {}

            for i, qm_grad_i in enumerate(qm_grad):
                grad[i] = deepcopy(mm_grad)
                for n, qm_grad_in in enumerate(qm_grad_i):
                    if n < self._num_qm:    # pure qm atoms
                        add_to_xyz(grad[i][self.qm_ids[n]], qm_grad_in)
                    else:    # linkatoms come after qm atoms
                        qm_id, mm_id = self._linkatoms[n - self._num_qm]
                        add_to_xyz(grad[i][mm_id], qm_grad_in, self._mm_s)
                        add_to_xyz(grad[i][qm_id], qm_grad_in, self._qm_s)

            if 'pc_grad' in qm_QMout:    # apply pc grad
                for i, grad_i in enumerate(qm_QMout['pc_grad']):
                    # mm_ids stay in order even after grouping qm_ids at the fron and deleting link mm atoms
                    # -> get all residual mm ids in order for correct order in pcgrad
                    for grad_in, mm_id in zip(grad_i, filter(lambda i: i not in self.mm_links, self.mm_ids)):
                        add_to_xyz(grad[i][mm_id], grad_in)

            self._QMout['grad'] = grad

        if 'nacdr' in QMin:
            # nacs would have to inserted in the whole system matrix only for qm atoms
            nacdr = [
                [[[0., 0., 0.] for _ in range(QMin['natom'])] for _ in range(QMin['nmstates'])]
                for _ in range(QMin['nmstates'])
            ]
            for i, s_i in enumerate(self.qm_interface._QMout['nacdr']):
                for s_j in s_i:
                    for n in range(self._num_qm):
                        add_to_xyz(s_j[n], nacdr[i][n][self.qm_ids[n]])
                    for n in range(self._num_qm, len(s_j)):    # linkatoms come after qm atoms
                        qm_id, mm_id = self._linkatoms[n - self._num_qm]
                        n = nacdr[i][n]
                        add_to_xyz(n[mm_id], s_j[n], self._mm_s)
                        add_to_xyz(n[qm_id], s_j[n], self._qm_s)
            QMout['nacdr'] = nacdr

        if 'dm' in QMin:
            QMout['dm'] = self.qm_interface._QMout['dm']
            if QMin['template']['mm_dipole']:
                for i, dm_i in enumerate(QMout['dm']):
                    mm_dm_i: float = self.mml_interface._QMout['dm'][i]
                    for dm_in in dm_i:
                        for dm_inm in dm_in:
                            dm_inm += mm_dm_i[0][0]    # add mm dipole moment to all states
        if 'overlap' in QMin:
            QMout['overlap'] = self.qm_interface._QMout['overlap']

        # potentially print out other contributions and properties...
        for i in ['ion', 'prop', 'theodore']:
            if i in qm_QMout:
                QMout[i] = qm_QMout[i]

    def create_restart_files(self):
        self.qm_interface.create_restart_files()
        self.mml_interface.create_restart_files()
        if self._QMin['template']['embedding'] == 'subtractive':
            self.mml_interface.create_restart_files()


if __name__ == "__main__":
    try:
        qmmm = QMMM(DEBUG, PRINT)
        qmmm.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
    except Error:
        raise
