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
from importlib import import_module
from error import Error
from SHARC_INTERFACE import INTERFACE
def factory(name: str) -> INTERFACE:
    if name.upper() not in ['LVC', 'ORCA', 'MOLCAS', 'BAGEL', 'MOLPRO', 'COLUMBUS', 'AMS-ADF', 'RICC2', 'GAUSSIAN', 'TINKER', 'QMMM']:
        raise Error(f'Interface with name "{name}" does not exist!')
    interface_mod = import_module('SHARC_{}_new'.format(name.upper()))
    interface = getattr(interface_mod, name.upper())
    if issubclass(interface, INTERFACE):
        return interface
    else:
        raise Error(f"factory could not produce an interface:\n {interface}")