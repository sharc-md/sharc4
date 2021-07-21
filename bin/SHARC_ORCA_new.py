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
import re
import datetime
from functools import reduce

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import au2a, kcal_to_Eh, NUMBERS, BASISSETS, IToMult, rcm_to_Eh

authors = 'Sebastian Mai, Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 7, 15)

changelogstring = '''
12.02.2016:     Initial version 0.1
- CC2 and ADC(2) from Turbomole
- SOC from Orca call
- only singlets and triplets
  => Doublets and Dyson could be added later

15.03.2016:     0.1.1
- ridft can be used for the SCF calculation, but dscf has to be run afterwards anyways (for SOC and overlaps).
- Laplace-transformed SOS-CC2/ADC(2) can be used ("spin-scaling lt-sos"), but does not work for soc or trans-dm
- if ricc2 does not converge, it is rerun with different combinations of npre and nstart (until convergence or too many tries)

07.03.2017:
- wfthres is now interpreted as in other interfaces (default is 0.99 now)

25.04.2017:
- can use external basis set libraries

23.08.2017
- Resource file is now called "RICC2.resources" instead of "SH2CC2.inp" (old filename still works)

24.08.2017:
- numfrozcore in resources file can now be used to override number of frozen cores for overlaps
- added Theodore capabilities (compute descriptors, OmFrag, and NTOs (also activate MOLDEN key for that))

11.11.2020:
- COBRAMM can be used for QM/MM calculations
'''


class ORCA(INTERFACE):

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

    def read_template(self, template_filename='ORCA.template'):
        '''reads the template file
        has to be called after setup_mol!'''

        if not self._read_resources:
            raise Error('Interface is nor set up correctly. Call read_resources with the .resources file first!', 23)
        QMin = self._QMin
        # define keywords and defaults
        bools = {'no_tda': False,
                 'unrestricted_triplets': False,
                 'qmmm': False,
                 'cobramm': False,
                 'picture_change': False
                 }
        strings = {'basis': '6-31G',
                   'auxbasis': '',
                   'functional': 'PBE',
                   'dispersion': '',
                   'grid': '2',
                   'gridx': '',
                   'gridxc': '',
                   'ri': '',
                   'scf': '',
                   'qmmm_table': 'ORCA.qmmm.table',
                   'qmmm_ff_file': 'ORCA.ff',
                   'keys': '',
                   'paste_input_file': ''
                   }
        integers = {
            'frozen': -1,
            'maxiter': 700
        }
        floats = {
            'hfexchange': -1.,
            'intacc': -1.
        }
        special = {'paddingstates': [0 for i in QMin['states']],
                   'charge': [i % 2 for i in range(len(QMin['states']))],

                   'basis_per_element': {},
                   'ecp_per_element': {},
                   'basis_per_atom': {},
                   'range_sep_settings': {'do': False, 'mu': 0.14, 'scal': 1.0, 'ACM1': 0.0, 'ACM2': 0.0, 'ACM3': 1.0},
                   'paste_input_file': ''
                   }
        lines = readfile(template_filename)
        QMin['template'] = {**bools, **strings, **integers, **floats, **special,
                            **self.parse_keywords(bools, strings, integers, floats, special, lines)}

        # do logic checks
        if not QMin['template']['unrestricted_triplets']:
            if QMin['OrcaVersion'] < (4, 1):
                if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                    raise Error('With Orca v<4.1, triplets can only be computed with the unrestricted_triplets option!', 62)
            if len(QMin['template']['charge']) >= 3 and QMin['template']['charge'][0] != QMin['template']['charge'][2]:
                raise Error('Charges of singlets and triplets differ. Please enable the "unrestricted_triplets" option!', 63)
        if QMin['template']['unrestricted_triplets'] and 'soc' in QMin:
            if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                raise Error('Request "SOC" is not compatible with "unrestricted_triplets"!', 64)
        if QMin['template']['ecp_per_element'] and 'soc' in QMin:
            if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                raise Error('Request "SOC" is not compatible with using ECPs!', 64)

        return


    def read_resources(self, resources_filename="ORCA.resources"):

        if not self._setup_mol:
            raise Error('Interface is nor set up for this template. Call setup_mol with the QM.in file first!', 23)
        QMin = self._QMin

        pwd = os.getcwd()
        QMin['pwd'] = pwd

        strings = {'orcadir': '',
                   'tinker': '',
                   'scratchdir': '',
                   'savedir': '',  # NOTE: savedir from QMin
                   'theodir': '',
                   'wfoverlap': '$SHARC/wfoverlap.x',
                   'qmmm_table': '',
                   'qmmm_ff_file': ''
                   }
        bools = {'debug': False,
                 'save_stuff': False,
                 'no_print': False,
                 'no_overlap': False,
                 'always_orb_init': False,
                 'always_guess': False,
                 }
        integers = {'ncpu': 1,
                    'memory': 100,
                    'numfrozcore': 0,
                    'numocc': 0
                    }
        floats = {'delay': 0.0,
                  'schedule_scaling': 0.9,
                  'wfthres': 0.99
                  }
        special = {
            'neglect_gradient': 'zero',
            'theodore_prop': ['Om', 'PRNTO', 'S_HE', 'Z_HE', 'RMSeh'],
            'theodore_fragment': [],
        }
        lines = readfile(resources_filename)
        # assign defaults first, which get updated by the parsed entries, which are updated by the entries that were already in QMin
        QMin['resources'] = {**bools, **strings, **integers, **floats, **special,
                             **self.parse_keywords(bools, strings, integers, floats, special, lines)}
        # reassign QMin after losing the reference
        QMin['OrcaVersion'] = self._getOrcaVersion(QMin['orcadir'])

        self._read_resources = True
        return

    def readQMin(self, QMinfilename):
        raise NotImplementedError

    def set_requests(self, QMinfilename):
        raise NotImplementedError

    def set_coords(self, xyz):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def get_QMout(self):
        raise NotImplementedError

    def main(self):
        raise NotImplementedError

    @staticmethod
    def _getOrcaVersion(path):
        # run orca with nonexisting file
        string = os.path.join(path, 'orca') + ' nonexisting'
        try:
            proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        except OSError:
            ty, val, tb = sys.exc_info()
            raise Error(f'Orca call has had some serious problems:\n {ty.__name__}: {val}', 25).with_traceback(tb)
        comm = proc.communicate()[0].decode()
        # find version string
        for line in comm.split('\n'):
            if 'Program Version' in line:
                s = line.split('-')[0].split()[2].split('.')
                s = tuple(int(i) for i in s)
                return s
        raise Error('Could not find Orca version!', 26)
