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
from parse_template import TemplateParser

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
        QMin = self._QMin
        if not self._setup_mol:
            raise Error('Interface is nor set up for this template. Call setup_mol with the QM.in file first!', 23)
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
                   'theodore_prop': ['Om', 'PRNTO', 'S_HE', 'Z_HE', 'RMSeh'],
                   'theodore_fragment': [],
                   'basis_per_element': {},
                   'ecp_per_element': {},
                   'basis_per_atom': {},
                   'range_sep_settings': {'do': False, 'mu': 0.14, 'scal': 1.0, 'ACM1': 0.0, 'ACM2': 0.0, 'ACM3': 1.0}
                   }
        template_parser = TemplateParser(QMin['nstates'], QMin['Atomcharge'])
        # prepare dict with parsers for every value type
        bool_parser = {k: lambda x: True for k in bools}
        string_parser = {k: lambda x: x for k in strings}
        integer_parser = {k: lambda x: int(float(x)) for k in integers}
        float_parser = {k: lambda x: float(x) for k in floats}
        special_parser = {k: lambda x: getattr(template_parser, k)(x) for k in special}

        parsers = {**bool_parser, **string_parser, **integer_parser, **float_parser, **special_parser}
        defaults = {**bools, **strings, **integers, **floats, **special}
        # open template file
        lines = readfile(template_filename)
        # replaces all comments with white space. filters all empty lines
        filtered = filter(lambda x: not re.match(r'^\s*$', x), map(lambda x: re.sub(r'#.*$', '', x), lines))

        # split line into key and args, calls parser for args and adds key: parser(args) to dict
        def _parse_to_dict(d: dict, line: str) -> dict:
            llist = line.split(None, 1)
            key = llist[0].lower()
            args = ' '
            if len(llist) == 2:
                args = llist[1]
            try:
                if key in d and isinstance(d[key], dict):
                    d[key].update(parsers[key](args))
                else:
                    d[key] = parsers[key](args)
            except Error:
                raise
            except Exception:
                type, val, tb = sys.exc_info()
                raise Error(f'Something went wrong while parsing the keyword: {key} {args} from {template_filename}:\n\
                    {type}: {val}\nPlease consult the examples folder in the $SHARCDIR for more information!').with_traceback(tb)
            return d
        QMin['template'] = {**defaults, **reduce(_parse_to_dict, filtered)}
        return


    def main(self):
        raise NotImplementedError

    def readQMin(self, QMinfilename):
        raise NotImplementedError

    def read_resources(self, resource_filename):
        raise NotImplementedError

    def set_requests(self, QMinfilename):
        raise NotImplementedError

    def set_coords(self, xyz):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def get_QMout(self):
        raise NotImplementedError
