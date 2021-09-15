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
from constants import kcal_to_Eh, au2a

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

    def setup_mol(self, QMinfilename: str):
        super().setup_mol(QMinfilename)
        QMin = self._QMin
        self._n_linkbonds = reduce(
            lambda x, y: x + 1 if y else x + 0, map(lambda x: re.match(r'.LA', x), QMin['elements']), 0
        )
        if QMin['states'] != [1]:
            print('WARNING: calculation only conducted for ground state!')
        QMin['states'] = [1]
        QMin['nmstates'] = 1

    def _request_logic(self):
        QMin = self._QMin
        possibletasks = {'h', 'grad'}
        tasks = possibletasks & QMin.keys()

        if len(tasks) == 0:
            raise Error(f'No tasks found! Tasks are {possibletasks}.', 39)

        not_allowed = {'soc', 'dm', 'overlap', 'dmdr', 'socdr', 'ion', 'nacdr', 'theodore', 'phases'}
        if not QMin.keys().isdisjoint(not_allowed):
            raise Error('Cannot perform tasks: {}'.format(' '.join(QMin.keys() & not_allowed)), 13)

        if 'grad' in QMin:
            print('Warning: gradient only calculated for first singlet!')

    def read_template(self, template_filename='TINKER.template'):
        QMin = self._QMin

        special = {
            'qmmm_table': 'TINKER.table',
        }
        paths = {'ff_file': ''}

        lines = readfile(template_filename)
        QMin['template'] = {**special, **self.parse_keywords(lines, paths=paths, special=special)}
        QMin['template']['paddingstates'] = [0]

        QMin['mmatoms'] = [
            MMATOM(i, v[0].lower() == 'qm', v[1], [0., 0., 0.], v[2], set(v[3:]))
            for i, v in enumerate(QMin['template']['qmmm_table'])
        ]
        # sanitize mmatoms
        # set links
        qm = []
        mm = []
        # map storing the permutations (map[x][0] is old->new, map[x][1] is new->old)
        self._perm = [[0, 0] for _ in range(len(QMin['mmatoms']))]
        for i in QMin['mmatoms']:
            for jd in i.bonds:
                # jd = j - 1
                if i.id == jd:
                    raise Error(f'Atom bound to itself:\n{i}')
                j = QMin['mmatoms'][jd]
                if i.id not in j.bonds:
                    j.bonds.add(i.id)
        # sort out qm atoms
            qm.append(i.id) if i.qm else mm.append(i.id)
        self._num_qm = len(qm)
        self._num_mm = len(mm)

        for i, id in enumerate(qm + mm):
            self._perm[i][1] = id
            self._perm[id][0] = i
        self._read_template = True

    def read_resources(self, resources_filename='TINKER.resources'):

        if not self._setup_mol:
            raise Error('Interface is not set up for this template. Call setup_mol with the QM.in file first!', 23)
        QMin = self._QMin

        pwd = os.getcwd()
        QMin['pwd'] = pwd

        paths = {
            'tinkerdir': '',
            'scratchdir': '',
            'savedir': pwd,    # NOTE: savedir from QMin
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

        empty_paths = list(
            filter(lambda x: not (bool(QMin['resources'][x]) or QMin['resources'][x].isspace()), paths.keys())
        )

        if len(empty_paths) != 0:
            raise Error('Resource paths are empty!: {}'.format(' '.join(empty_paths)), 14)

        if 'scratchdir' not in QMin:
            QMin['scratchdir'] = QMin['resources']['scratchdir']

        if 'savedir' not in QMin:
            QMin['savedir'] = QMin['resources']['savedir']
        self._read_resources = True

    def run(self):
        QMin = self._QMin
        self.clock.starttime = datetime.datetime.now()
        WORKDIR = os.path.join(QMin['scratchdir'], 'TINKER')
        mkdir(WORKDIR)
        # Writing TINKER inputs
        ## TINKER.key
        input_str = f'parameters {QMin["template"]["ff_file"]}\n'
        input_str += f'QMMM {len(QMin["mmatoms"])}\n'
        input_str += f'QM -1 {self._num_qm}\n'
        input_str += f'MM -{self._num_qm + 1} {self._num_qm + self._num_mm}\n'

        if QMin['resources']['ncpu'] > 1:
            input_str += f'\nOPENMP-THREADS {QMin["resources"]["ncpu"]}\n'
            os.environ['OMP_NUM_THREADS'] = str(QMin["resources"]["ncpu"])
        input_str += '\n'
        writefile(os.path.join(WORKDIR, 'TINKER.key'), input_str)

        ## TINKER.xyz
        input_str = f'{len(QMin["mmatoms"])} # generated by SHARC {self.__class__.__name__} Interface (v{version})\n'
        for i in range(QMin['natom']):
            otn = self._perm[i][1]
            atom: MMATOM = QMin['mmatoms'][otn]
            atom.xyz = (QMin['coords'][otn] * au2a).tolist()
            input_str += '{: >4}  {: <4}  {: 14.12f} {: 14.12f} {: 14.12f}  {: >3} {}\n'.format(
                i + 1, atom.symbol, *atom.xyz, atom.type,
                ' '.join(map(lambda x: str(self._perm[x][0] + 1), atom.bonds))
            )
        input_str += '\n'
        writefile(os.path.join(WORKDIR, 'TINKER.xyz'), input_str)

        ## TINKER.xyz
        input_str = 'SHARC 0 -1\n'
        input_str += '\n'.join(
            map(
                lambda x: '{: 14.12f} {: 14.12f} {: 14.12f}'.format(*QMin['mmatoms'][self._perm[x][1]].xyz),
                range(QMin['natom'])
            )
        )
        input_str += '\n'

        writefile(os.path.join(WORKDIR, 'TINKER.qmmm'), input_str)
        writefile(os.path.join(WORKDIR, 'orig.qmmm'), input_str)

        ## TINKER.xyz
        writefile(os.path.join(WORKDIR, 'TINKER.in'), 'TINKER.xyz')

        ## execute tkr2qm_s
        exec_str = os.path.join(QMin["resources"]["tinkerdir"], 'bin/tkr2qm_s') + ' < TINKER.in'

        exit_code = self.runProgram(exec_str, WORKDIR, 'TINKER.out', 'TINKER.err')

        output: TextIOWrapper = open(os.path.join(WORKDIR, 'TINKER.qmmm'), 'r')

        if 'MMisOK' not in output.readline():
            raise Error('Tinker run not successfull!', 20)

        QMout = self._QMout
        QMout['h'] = [[float(output.readline().split()[-1]) * kcal_to_Eh]]

        grad = {0: [None] * len(QMin['mmatoms'])}    # grad only for ground state
        line_lst = output.readline().split()
        while line_lst[0] == 'MMGradient':
            idx = int(line_lst[1]) - 1
            grad[0][self._perm[idx][1]] = [float(x) * kcal_to_Eh * au2a for x in line_lst[2:]]
            line_lst = output.readline().split()

        if grad[0][0]:
            QMout['grad'] = grad

        if line_lst[0] == 'MMq':
            QMout['raw_pc'] = {
                i: float(v[-1])
                for i, v in map(
                    lambda i: (self._perm[i][1], output.readline().split()),
                    range(self._num_qm,
                          int(line_lst[1]) + self._num_qm)
                )
            }
        output.close()
        if 'dm' in QMin and QMin['dm']:
            dm = np.zeros((3), float)
            for i, q in enumerate(QMout['raw_pc']):
                dm += q * QMin['coords'][i]
            QMout['dm'] = dm.tolist()
        QMout['runtime'] = self.clock.measuretime()

        # read additional free energy components
        with open(os.path.join(WORKDIR, 'TINKER.qmmm'), 'r') as f:
            parse = False
            for line in f:
                if parse:
                    try:
                        llist = f.readline().split()
                        QMout[llist[0]] = float(llist[2])
                    except (IndexError, ValueError):
                        break
                if 'and the MM contributions to the gradient' not in line:
                    parse = True


    def setup_run(self):
        QMin = self._QMin
        # make name for backup directory
        if 'backup' in QMin:
            backupdir = QMin['savedir'] + '/backup'
            backupdir1 = backupdir
            i = 0
            while os.path.isdir(backupdir1):
                i += 1
                if 'step' in QMin:
                    backupdir1 = backupdir + '/step%s_%i' % (QMin['step'][0], i)
                else:
                    backupdir1 = backupdir + '/calc_%i' % (i)
            QMin['backup'] = backupdir


if __name__ == '__main__':
    try:
        tinker = TINKER(DEBUG, PRINT)
        tinker.main()
    except KeyboardInterrupt:
        print('Aborting... You might over-TINK calling me again...')
    except Error:
        raise
