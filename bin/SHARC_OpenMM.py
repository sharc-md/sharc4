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
import numpy as np

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *

authors = 'Sebastian Mai and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2022, 4, 8)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)


class OpenMM(INTERFACE):

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

    # TODO: warnings for unsupported settings, i.e. nacs, socs etc
    def run(self):
        pass

    def read_template(self, template_filename):
        pass
    
    def setup_run(self):
        pass
    
    def _request_logic(self):
        pass