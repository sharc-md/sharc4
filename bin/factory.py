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

AVAILABLE_INTERFACES = [
    'LVC', 'ORCA', 'MOLCAS', 'BAGEL', 'MOLPRO', 'COLUMBUS', 'AMS-ADF', 'RICC2', 'GAUSSIAN', 'TINKER', 'QMMM', 'MNDO', 'OpenMM'
]


def factory(name: str) -> INTERFACE:
    try:
        ind = [i.upper() for i in AVAILABLE_INTERFACES].index(name.upper())
    except ValueError:
        raise Error(f'Interface with name "{name}" does not exist!')
    int_name = AVAILABLE_INTERFACES[ind]
    interface_mod = import_module('SHARC_{}'.format(int_name))
    interface = getattr(interface_mod, int_name)
    if issubclass(interface, INTERFACE):
        return interface
    else:
        raise Error(f"factory could not produce an interface:\n {interface}")
