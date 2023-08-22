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
import sys
import datetime
import numpy as np

# internal
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import readfile, mkdir, Error
from constants import U_TO_AMU, MASSES
from kabsch import kabsch_w as kabsch




authors = 'Sebastian Mai'
version = '3.0'
versiondate = datetime.datetime(2023, 8, 29)
name = 'SHARC Do Nothing Interface'
description = 'Zero energies/gradients/couplings/etc and unity overlap matrices/phases.'

changelogstring = '''
'''
np.set_printoptions(linewidth=400, formatter={'float': lambda x: f'{x: 9.7}'})


class SHARC_DO_NOTHING(SHARC_INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring
    _name = name
    _description = description
    _step = 0

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

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description









