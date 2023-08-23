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
# *****************************************

# IMPORTS
# external
from itertools import chain
import pprint
import sys
import math
import time
import datetime
from multiprocessing import Pool
from copy import deepcopy
from socket import gethostname
import traceback
import numpy as np

# internal
from resp import Resp
from SHARC_INTERFACE_old import INTERFACE
from utils import *
from globals import DEBUG, PRINT
from constants import IToMult, au2eV, IAn2AName
from error import Error, exception_hook

sys.excepthook = exception_hook

authors = 'Sebastian Mai, David Lehrner, and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2022, 8, 1)
changelogstring = '''
01.08.2022: Port to new interface style
- includes RESP fitting
'''


class MOLCAS(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring

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


    def read_resources(self, resources_filename="MOLCAS.resources"):
        super().read_resources(resources_filename)
        QMin = self._QMin
        QMin['version'] = MOLCAS.getVersion(QMin['molcasdir'])
        print('Detected MOLCAS version %3.1f\n' % (QMin['version']))

        os.environ['MOLCAS'] = QMin['molcasdir']

        # check driver
        if QMin['molcas_driver'] == '':
            driver = os.path.join(QMin['molcasdir'], 'bin', 'pymolcas')
            if not os.path.isfile(driver):
                driver = os.path.join(QMin['molcasdir'], 'bin', 'molcas.exe')
                if not os.path.isfile(driver):
                    raise Error('No driver (pymolcas or molcas.exe) found in $MOLCAS/bin. Please add the path to the driver via the "driver" keyword.', 52)
        else:
            if os.path.isfile(QMin['molcas_driver']):
                driver = QMin['molcas_driver']
            else:
                raise Error('Driver not found in %s' % QMin['molcas_driver'], 52)
        QMin['molcas_driver'] = driver

        self._read_resources = True







    def read_template(self, template_filename='MOLCAS.template'):
        '''reads the template file
        has to be called after setup_mol!'''

        if not self._read_resources:
            raise Error('Interface is not set up correctly. Call read_resources with the .resources file first!', 23)
        QMin = self._QMin
        # define classes and defaults
        bools = {
            'cholesky': False,
            'cholesky_analytical': False,
            'no-douglas-kroll': False,
            'diab_num_grad': False,
            'qmmm': False,      # TODO: qmmm will require some work
            'cobramm': False,   # TODO: cobramm will require some work
        }
        strings = {
            'basis': 'ANO-S-MB',
            'method': 'casscf',
            'baslib': '',
            'pdft-functional': 'tpbe'
        }
        integers = {
            'nactel': 0,        # requires input
            'inactive': 0,      # requires input
            'ras2': 0,          # requires input
            'frozen': -1
        }
        floats = {
            'ipea': 0.25,
            'imaginary': 0.0,
            'gradaccumax': 1.e-2,
            'gradaccudefault': 1.e-4,
            'displ': 0.005,     # better in base interface where num grad will be
            'rasscf_thrs_e': 1e-8,
            'rasscf_thrs_rot': 1e-4,
            'rasscf_thrs_egrd': 1e-4,
            'cholesky_accu': 1e-4
        }
        special = {
            'roots': [0 for i in QMin['states']],
            'rootpad': [0 for i in QMin['states']],
            # 'charge': [i % 2 for i in range(len(QMin['states']))],
            'pcmset': {'solvent': 'water', 'aare': 0.4, 'r-min': 1.0, 'on': False},
            'pcmstate': [QMin['statemap'][1][0], QMin['statemap'][1][1]],
            'iterations': [200, 100]
        }

        lines = readfile(template_filename)
        QMin['template'] = {
            **bools,
            **strings,
            **integers,
            **floats,
            **special,
            **self.parse_keywords(
                lines, bools=bools, strings=strings, integers=integers, floats=floats, special=special
            )
        }

        # get path to basis set library
        if QMin['template']['baslib']:
            QMin['template']['baslib'] = os.path.abspath(QMin['template']['baslib'])

        # do logic checks

        # 1: states, roots, rootpad
        QMin['template']['roots'] = QMin['template']['roots'][:len(QMin['states'])]
        for i, n in enumerate(QMin['template']['roots']):
            if not n >= QMin['states'][i]:
                raise Error('Too few states in state-averaging in multiplicity %i! %i requested, but only %i given' % (i + 1, QMin['states'][i], n), 58)
        QMin['template']['rootpad'] = QMin['template']['rootpad'][:len(QMin['states'])]
        for i, n in enumerate(QMin['template']['rootpad']):
            if n < 0:
                raise Error('Rootpad must not be negative!', 60)

        # 2: nactel, ras2, and inactive
        necessary = ['nactel', 'ras2', 'inactive']
        for i in necessary:
            if QMin['template'][i] == 0:
                raise Error('Key %s missing in template file!' % (i), 62)
        nelec = 2 * QMin['template']['inactive'] + QMin['template']['nactel']
        if nelec % 2 == 0:
            nelec = [nelec - (i + 0) % 2 for i in range(len(QMin['states']))]
        else:
            nelec = [nelec - (i + 1) % 2 for i in range(len(QMin['states']))]
        charge = [QMin['Atomcharge'] - i for i in nelec]
        QMin['template']['charge'] = charge

        # 3: pcm and qmmm/cobramm
        if QMin['template']['pcmset']['on']:
            if QMin['template']['qmmm']:
                raise Error('PCM and QM/MM cannot be used together!', 63)
            if QMin['template']['cobramm']:
                raise Error('PCM and COBRAMM cannot be used together!', 63)

        # 4: allowed methods
        allowed_methods = ['casscf', 'caspt2', 'ms-caspt2', 'mc-pdft', 'xms-pdft', 'cms-pdft']
        # 0: casscf
        # 1: caspt2 (single state)
        # 2: ms-caspt2
        # 3: mc-pdft (single state)
        # 4: xms-pdft
        # 5: cms-pdft
        for i, m in enumerate(allowed_methods):
            if QMin['template']['method'] == m:
                QMin['method'] = i
                break
            else:
                raise Error('Unknown method "%s" given in MOLCAS.template' % (QMin['template']['method']), 64)

        # 5: gradient mode
        # TODO!

        # 6: in progress disabled features
        if QMin['template']['qmmm']:
            raise Error('qmmm not implemented')
        if QMin['template']['cobramm']:
            raise Error('cobramm not implemented')


        # 7: TODO add keywords to be removed later
        QMin['template']['unrestricted_triplets'] = False

        self._read_template = True
        return


    # ======================================================================= #


    def getVersion(path):
        # get version from version file
        molcasversion = os.path.join(path, '.molcasversion')
        if os.path.isfile(molcasversion):
            with open(molcasversion) as vf:
                string = vf.readline()
        # extract
        a = re.search('[0-9]+\\.[0-9]+', string)
        # check
        if a is None:
            raise Error('No MOLCAS version found.\nCheck whether MOLCAS path is set correctly in MOLCAS.resources\nand whether $MOLCAS/.molcasversion exists.', 17)
        v = float(a.group())
        allowedrange = [(18.0, 22.999)]
        if not any([i[0] <= v <= i[1] for i in allowedrange]):
            raise Error('MOLCAS version %3.1f not supported! ' % (v), 18)
        return v




















    def create_restart_files(self):
        pass

    def getQMout(self):
        pass

    def run(self):
        self.printQMin()
        sys.exit(1)























if __name__ == '__main__':
    molcas = MOLCAS(get_bool_from_env('DEBUG', False), get_bool_from_env('PRINT'))
    molcas.main()
