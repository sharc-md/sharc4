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

# This script calculates QC results for a system described by the LVC model
#
# Reads QM.in
# Calculates SOC matrix, dipole moments, gradients, nacs and overlaps
# Writes these back to QM.out

# IMPORTS
# external
from io import TextIOWrapper
import datetime
import numpy as np
from functools import reduce

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *

authors = 'Sebastian Mai and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 8, 17)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)


class TINKER(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring
    _read_resources = True

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

    # TODO: get txyz from coords in QMin and others stuff from template (setup possible with awk from initial txyz)
    # TODO: distinguish MM from QM (special version of OpenMOLCAS)
    #       only vdW-Terms are considered
    #       potentially abuse resources, template file...
    # TODO: warnings for unsupported settings, i.e. nacs, socs etc
    # TODO: write .key, .in, xyz
    # TODO: QM.in only for coords
    '''
    3

    O       0.00000000      -0.50988030      -0.80750292
    H       0.00000000       0.05602171      -1.54130310
    H       0.00000000       0.00000000       0.00000000

    '''
    # TODO: template: not sure how to indicate qm and mm either with new atom type (-1) or directly
    '''
    qmmm start
    qm  6 2 3
    mm 21 1
    mm 21 1
    end
    '''

    def setup_mol(self, QMinfilename: str):
        super().setup_mol(QMinfilename)
        QMin = self._QMin
        self._n_linkbonds = reduce(
            lambda x, y: x + 1 if y else x + 0, map(lambda x: re.match(r'.LA', x), QMin['elements']), 0
        )

    def read_template(self, template_filename='TINKER.template'):
        QMin = self._QMin

        special = {
            'qmmm_table': [],
        }
        path = {'ff_file': ''}

        lines = readfile(template_filename)
        QMin['template'] = {**special, **self.parse_keywords(lines, paths=path, special=special)}

        QMin['mmatoms'] = [
            MMATOM(i, v[0].lower() == 'qm', QMin['elements'][i], [0., 0., 0.], v[1], set(v[1:]))
            for i, v in enumerate(QMin['template']['qmmm_table'])
        ]

        # sanitize mmatoms
        # set links
        qm = []
        mm = []
        # map storing the permutations (map[x][0] is old->new, map[x][1] is new->old)
        self._perm = [(0, 0)] * len(QMin['mmatoms'])
        for i in QMin['mmatoms']:
            for jd in i.bonds:
                if i.id == jd:
                    raise Error(f'Atom bound to itself:\n{i}')
                j = QMin['mmatoms'][jd]
                if i.id not in j.bonds:
                    j.bonds.add(i.id)

        # sort out qm atoms
            qm.append(i.id) if i.qm else mm.append(i.id)
        for i, id in qm + mm:
            self._perm[i][1] = id
            self._perm[id][0] = i

    def read_resources(self, resources_filename='TINKER.resources'):

        if not self._setup_mol:
            raise Error('Interface is not set up for this template. Call setup_mol with the QM.in file first!', 23)
        QMin = self._QMin

        pwd = os.getcwd()
        QMin['pwd'] = pwd

        paths = {
            'tinker': '',
            'scratchdir': '',
            'savedir': '',    # NOTE: savedir from QMin
        }
        bools = {
            'debug': False,
            'save_stuff': False,
        }
        integers = {
            'ncpu': 1,
            'memory': 100,
        }
        floats = {
            'delay': 0.0,
            'schedule_scaling': 0.9,
        }
        lines = readfile(resources_filename)

        QMin['resources'] = {
            **bools,
            **paths,
            **integers,
            **floats,
            **self.parse_keywords(lines, bools=bools, paths=paths, integers=integers, floats=floats)
        }

    def run(self):
        QMin = self._QMin

        WORKDIR = os.path.join(QMin['scratchdir'], 'TINKER')
        mkdir(WORKDIR)
        # Writing TINKER inputs
        ## TINKER.key
        input_str = f'parameters {QMin["ff_file"]}\n'

        # TODO: add link and QM atoms if QMMM

        if QMin['resources']['ncpu'] > 1:
            input_str += f'\nOPENMP-THREADS {QMin["resources"]["ncpu"]}\n'
            os.environ['OMP_NUM_THREADS'] = str(QMin["resources"]["ncpu"])
        input_str += '\n'
        writefile(os.path.join(WORKDIR), 'TINKER.key', input_str)

        ## TINKER.xyz
        input_str = f'{len(QMin["mmatoms"])} # generated by SHARC {self.__class__.__name__} Interface (v{version})\n'
        for otn, _ in self._perm:
            atom: MMATOM = QMin['mmatoms'][otn]
            atom.xyz = QMin['coords'][otn].tolist()
            input_str += f'{atom}\n'

        writefile(os.path.join(WORKDIR), 'TINKER.xyz', input_str)

        ## TINKER.xyz
        input_str = 'SHARC 0 -1\n'
        input_str += '\n'.join(map(lambda x: '{:12.12} {:12.12} {:12.12}'.format(x.xyz), QMin['mmatoms']))
        input_str += '\n'

        writefile(os.path.join(WORKDIR), 'TINKER.qmmm', input_str)

        ## TINKER.xyz
        writefile(os.path.join(WORKDIR), 'TINKER.in', 'TINKER.xyz')

        ## execute tkr2qm_s
        exec_str = os.path.join(QMin["resources"]["tinkerdir"], 'bin/tkr2qm_s') + ' < TINKER.in'

        exit_code = self.runProgram(exec_str, WORKDIR, 'TINKER.out', 'TINKER.err')

        output: TextIOWrapper = open(os.path.join(WORKDIR), 'TINKER.qmmm', 'r')

        if 'MMisOK' not in output.readline():
            raise Error('Tinker run not successfull!', 20)

        QMout = self._QMout
        QMout['h'] = [[float(output.readline().split()[-1])]]

        grad = {0: [None] * len(QMin['mmatoms'])}    # grad only for ground state
        line_lst = output.readline().split()
        while line_lst[0] == 'MMGradient':
            idx = int(line_lst[1]) - 1
            grad[0][self._perm[idx][1]] = [float(x) for x in line_lst[2:]]
        line_lst = output.readline().split()

        if grad[0][0]:
            QMout['grad'] = grad

        if line_lst[0] == 'MMq':
            #XXX: pc in new order !!!!
            QMout['raw_pc'] = [
                [float(y) for y in x.split()] for x in map(lambda x: output.readline(), range(int(line_lst[1])))
            ]
        output.close()
