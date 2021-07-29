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
from os import stat
import sys
import math
import time
import datetime
import numpy as np
from functools import reduce

# internal
from SHARC_INTERFACE import INTERFACE
from globals import DEBUG, PRINT
from utils import *
from constants import IToMult, rcm_to_Eh

authors = 'Sebastian Mai and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 7, 29)

changelogstring = '''
'''


class LVC(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring
    _HCMCH = {0: np.ndarray((10, 10), dtype=float)}

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

    def read_template(self, template_filename='LVC.template'):
        QMin = self._QMin
        lines = readfile(template_filename)
        r3N = 3 * QMin['natom']
        # removes all comments and empty lines
        filtered = filter(lambda x: not re.match(r'^\s*$', x), map(lambda x: re.sub(r'#.*$', '', x).strip(), lines))
        file_str = '\n'.join(filtered)

        def format_match(x: re.Match) -> str:
            return '\n{} {}'.format(x.group(1), re.sub(r'\n+', ",", "{}".format(x.group(4))))
        lines = re.sub(r'\n([a-zA-Z]{3,}(\s[IR])?)(\n\d+)?((\s*(-?\d+(\.\d+(e[-+]\d+)?)?))+)',
                       format_match, file_str).split('\n')

        V0file = lines[0]
        dsp = self.read_V0(V0file)
        states = [int(s) for s in lines[1].split()]
        if states != QMin['states']:
            print(f'states from QM.in and nstates from LVC.template are inconsistent! {QMin["states"]} != {states}')
            sys.exit(25)

        lines = map(lambda x: (' '.join(x[0]), x[1:]), [[y.split() for y in x.split(',')] for x in lines[2:]])

        self._HMCH = {im: np.zeros((n, n), dtype=float) for im, n in enumerate(states) if n != 0}
        self._dHMCH = {im: np.zeros((r3N, n, n), dtype=float) for im, n in enumerate(states) if n != 0}
        for name, val in lines:
            # NOTE: possibly assign whole arry with index accessor (numpy)
            if name == 'epsilon':
                for im, s, v in map(lambda v: [int(v[0]) - 1, int(v[1]) - 1, float(v[2])], val):
                    self._HMCH[im][s, s] += v
            elif name == 'kappa':
                for im, s, i, v in map(lambda v: [int(v[0]) - 1, int(v[1]) - 1, int(v[2]) - 1, float(v[3])], val):
                    self._HMCH[im][s, s] = v
        return

    def read_V0(self, filename):
        with open(filename, 'r') as f:
            pass
        return


    def read_resources(self, resources_filename):
        return

    def run(self):
        return
