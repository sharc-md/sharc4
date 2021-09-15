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
import sys
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
        strings = {'qm-program': '',
                   'mm-program': '',
                   }
        bools = {'mm_dipole': False}

        lines = readfile(template_filename)
        QMin['template'] = {**bools, **self.parse_keywords(lines, bools=bools, special=special, strings=strings)}

        # check
        required: set = {'qm-program', 'mm-program', 'qmmm_table'}
        if not required.issubset(QMin['template'].keys()):
            raise Error(
                '"{}" not specified in {}'.format(
                    '", "'.join(filter(lambda x: x not in QMin['template'], required)), template_filename
                ), 78
            )

        QMin['atoms'] = [
            MMATOM(i, v[0].lower() == 'qm', v[1], [0., 0., 0.], v[2], set(v[3:]))
            for i, v in enumerate(QMin['template']['qmmm_table'])
        ]

        # sanitize mmatoms
        # set links
        qm = []
        mm = []
        self._linkatoms: set = {}    # set with number as qm_id * 1000000 + mm_id
        # map storing the permutations (map[x][0] is old->new, map[x][1] is new->old)
        self._perm = [[0, 0] for _ in range(len(QMin['atoms']))]
        for i in QMin['atoms']:
            for jd in i.bonds:
                # jd = j - 1
                if i.id == jd:
                    raise Error(f'Atom bound to itself:\n{i}')
                j = QMin['atoms'][jd]
                if i.id not in j.bonds:
                    j.bonds.add(i.id)
                if i.qm != j.qm:
                    self._linkatoms.add(i.id * 1000000 + j.id if i.qm else j.id * 1000000 + i.id)
        # sort out qm atoms
            qm.append(i.id) if i.qm else mm.append(i.id)
        self._num_qm = len(qm)
        self._num_mm = len(mm)

        # check of linkatoms: map linkatoms to sets of unique qm and mm ids: decreased number -> Error
        if len(self._linkatoms) > len(set(map(lambda x: x // 1000000, self._linkatoms))):
            raise Error('Some QM atom is involved in more than one link bond!', 23)
        if len(self._linkatoms) > len(set(map(lambda x: x % 1000000, self._linkatoms))):
            raise Error('Some MM atom is involved in more than one link bond!', 23)
        self._linkatoms = list(self._linkatoms)

        for i, id in enumerate(qm + mm):
            self._perm[i][1] = id
            self._perm[id][0] = i
        self._read_template = True

    def read_resources(self, resources_filename):
        pass

    def setup_run(self):
        QMin = self._QMin
        if 'savedir' not in QMin:
            print('savedir not specified in QM.in, setting savedir to current directory!')
            QMin['savedir'] = os.getcwd()

        self.read_template()

        # dynamic import of both interfaces
        self.qm_interface: INTERFACE = factory(QMin['template']['qm-program']
                                               )(self._DEBUG, self._PRINT, self._persistent)

        self.mm_interface: INTERFACE = factory(QMin['template']['mm-program']
                                               )(self._DEBUG, self._PRINT, self._persistent)

        # folder setup and savedir
        qmdir = os.path.join(QMin['savedir'], 'QM_' + QMin['template']['qm-program'].upper())
        mkdir(qmdir)
        mmdir = os.path.join(QMin['savedir'], 'MM_' + QMin['template']['mm-program'].upper())
        mkdir(mmdir)

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
        qm_QMin['grad'] = QMin['grad']
        qm_QMin['nmstates'] = QMin['nmstates']
        qm_QMin['unit'] = QMin['unit']
        self.qm_interface._setup_mol = True

        # setup mol for mm
        mm_QMin = self.mm_interface._QMin
        mm_QMin['elements'] = el
        mm_QMin['Atomcharge'] = QMin['Atomcharge']
        mm_QMin['frozcore'] = QMin['frozcore']
        mm_QMin['natom'] = QMin['natom']
        mm_QMin['states'] = [1]
        mm_QMin['nmstates'] = 1
        mm_QMin['unit'] = QMin['unit']
        self.mm_interface._setup_mol = True

        # read template and resources
        self.qm_interface.read_resources()
        qm_QMin['savedir'] = qmdir    # overwrite savedir
        self.qm_interface.read_template()
        self.qm_interface._PRINT = False
        self.qm_interface.setup_run()
        self.qm_interface._PRINT = self._PRINT

        self.mm_interface.read_resources()
        mm_QMin['savedir'] = mmdir    # overwrite savedir
        self.mm_interface.read_template()
        self.mm_interface.setup_run()
        return

    def run(self):
        QMin = self._QMin
        # set coords
        qm_coords = np.array([QMin['coords'][self._perm[i][1]].copy() for i in range(self._num_qm)])
        if len(self._linkatoms) > 0:
            # get linkatom coords
            def get_link_coord(hash: int) -> np.ndarray[float]:
                qm_id, mm_id = divmod(hash, 1000000)    # combination of '//' and '%'
                return QMin['coords'][qm_id] * self._qm_s + QMin['coords'][mm_id] * self._mm_s

            link_coords = np.fromiter(map(get_link_coord, self._linkatoms), dtype=float)
            self.qm_interface._QMin['coords'] = np.vstack((qm_coords, link_coords))
        else:
            self.qm_interface._QMin['coords'] = qm_coords

        self.mm_interface._QMin['coords'] = QMin['coords'].copy()

        # set qm requests: grad, nac, soc,
        possible = [
            'cleanup', 'backup', 'h', 'soc', 'dm', 'grad', 'overlap', 'dmdr',
            'nac', 'nacdr', 'socdr', 'ion', 'theodore', 'phases', 'step'
        ]
        for i in filter(lambda x: x in QMin, possible):
            self.qm_interface._QMin[i] = QMin[i]

        # set mm requests
        possible = ['cleanup', 'backup', 'h', 'dm', 'step']
        for i in filter(lambda x: x in QMin, possible):
            self.mm_interface._QMin[i] = QMin[i]
        if 'grad' in QMin:
            self.mm_interface._QMin['grad'] = [1]
        if 'dm' in QMin:
            self.mm_interface._QMin['dm'] = [1]

        # calc mm
        self.mm_interface.run()
        raw_pc = self.mm_interface._QMout['raw_pc']
        mm_e = self.mm_interface._QMout['h'][0]
        # prepare PC for qm interface
        sorted_pc_ids = sorted(raw_pc.keys())
        # pc: list[list[float]] = each pc is x, y, z, q
        pc = [[*QMin['coords'][k].tolist(), raw_pc[k]] for k in sorted_pc_ids]
        # calc qm
        self.qm_interface._QMin['pointcharges'] = pc
        # TODO indicator to include pointcharges?
        self.qm_interface.run()
        qm_QMout = self.qm_interface._QMout
        QMout = self._QMout

        # Hamiltonian
        if 'h' in qm_QMout:
            QMout['h'] = deepcopy(qm_QMout['h'])
            for i in range(QMin['nmstates']):
                print(i, QMout['h'][i][i], mm_e)
                QMout['h'][i][i] += mm_e[0]
        # gen output
        if 'grad' in QMin:
            qm_grad = self.qm_interface._QMout['grad']
            mm_grad = self.mm_interface._QMout['grad'][0]

            grad = {}

            for i, qm_grad_i in enumerate(qm_grad):
                grad[i] = deepcopy(mm_grad)
                for n, qm_grad_in in enumerate(qm_grad_i):
                    if n < self._num_qm:    # pure qm atoms
                        x = grad[i][self._perm[n][1]]
                        x[0] += qm_grad_in[0]
                        x[1] += qm_grad_in[1]
                        x[2] += qm_grad_in[2]
                    else:    # linkatoms come after qm atoms
                        qm_id, mm_id = divmod(self._linkatoms[n - self._num_qm], 1000000)
                        grad[i][mm_id][0] += qm_grad_in[0] * self._mm_s
                        grad[i][mm_id][1] += qm_grad_in[1] * self._mm_s
                        grad[i][mm_id][2] += qm_grad_in[2] * self._mm_s
                        grad[i][qm_id][0] += qm_grad_in[0] * self._qm_s
                        grad[i][qm_id][1] += qm_grad_in[1] * self._qm_s
                        grad[i][qm_id][2] += qm_grad_in[2] * self._qm_s

            if 'pcgrad' in qm_QMout:  # apply pc grad
                for i, grad_i in enumerate(qm_QMout['pcgrad']):
                    for n, grad_in in enumerate(grad_i):
                        atom_id = sorted_pc_ids[n]
                        grad[i][atom_id][0] += grad_in[0]
                        grad[i][atom_id][1] += grad_in[1]
                        grad[i][atom_id][2] += grad_in[2]
            
            self._QMout['grad'] = grad
                
        if 'nacdr' in QMin:
            # nacs would have to inserted in the whole system matrix only for qm atoms
            nacdr = [
                [[[0., 0., 0.] for _ in range(QMin['natom'])] for _ in range(QMin['nmstates'])]
                for _ in range(QMin['nmstates'])
            ]
            for i, s_i in enumerate(self.qm_interface._QMout):
                for n, s_j in enumerate(s_i):
                    for n, atom in enumerate(s_j):
                        if n < self._num_qm:    # pure qm atoms
                            x = nacdr[i][n][self._perm[1][n]]
                            x[0] += atom[0]
                            x[1] += atom[1]
                            x[2] += atom[2]
                        else:    # linkatoms come after qm atoms
                            qm_id, mm_id = divmod(self._linkatoms[n - self._num_qm], 1000000)
                            n = nacdr[i][n]
                            n[mm_id][0] += atom[0] * self._mm_s
                            n[mm_id][1] += atom[1] * self._mm_s
                            n[mm_id][2] += atom[2] * self._mm_s
                            n[qm_id][0] += atom[0] * self._qm_s
                            n[qm_id][1] += atom[1] * self._qm_s
                            n[qm_id][2] += atom[2] * self._qm_s
            QMout['nacdr'] = nacdr

        if QMin['dm']:
            QMout['dm'] = self.qm_interface._QMout['dm']
            if QMin['template']['mm_dipole']:
                for i, dm_i in enumerate(QMout['dm']):
                    mm_dm_i: float = self.mm_interface._QMout['dm'][i]
                    for dm_in in dm_i:
                        for dm_inm in dm_in:
                            dm_inm += mm_dm_i  # add mm dipole moment to all states
        if 'overlap' in QMin:
            QMout['overlap'] = self.qm_interface._QMout['overlap']
        
        # potentially print out other contributions and properties...
        for i in ['ion', 'prop', 'theodore']:
            if i in qm_QMout:
                QMout[i] = qm_QMout[i]
        
        for i in ['ea', 'ev', 'eb', 'ed', 'ec']:
            if i in self.mm_interface._QMout:
                QMout[i] = self.mm_interface._QMout[i]


if __name__ == "__main__":
    try:
        qmmm = QMMM(DEBUG, PRINT)
        qmmm.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
    except Error:
        raise
