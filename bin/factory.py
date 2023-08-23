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
from utils import expand_path
from typing import Union
from logger import log
import glob
from SHARC_INTERFACE import SHARC_INTERFACE

AVAILABLE_INTERFACES = [
    'LVC', 'ORCA', 'MOLCAS', 'BAGEL', 'MOLPRO', 'COLUMBUS', 'AMS-ADF', 'RICC2', 'GAUSSIAN', 'TINKER', 'QMMM', 'MNDO', 'OpenMM'
]

def get_available_interfaces() -> list[tuple[str, Union[SHARC_INTERFACE, str]]]:
    """
    returns available interfaces classes

    dynamically determines interfaces from set SHARC folder and returns the classes.

    Returns
    -------
    list[SHARC_INTERFACE]
        list of SHARC interface classes
    """
    sharc_bin = expand_path('$SHARC')
    log.debug(f"factory interface collection: {sharc_bin}")
    interfaces = []
    for path in sorted(glob.glob(sharc_bin + '/SHARC_*.py')):
        filename = path.split('/')[-1]
        interface_name = filename.split('.')[0]
        interface = "(Not Available!)"
        try:
            mod = import_module(interface_name)
        except TypeError:
            log.debug(f"{mod} could not be imported (not a package)")
        except (ModuleNotFoundError, ImportError):
            log.debug(f"{mod} could not be imported (missing dependencies)")

        try:
            interface = getattr(mod, interface_name)
        except AttributeError:
            log.debug(f"class {interface_name} not found in {mod}")

        if type(interface) == str or not issubclass(interface, SHARC_INTERFACE):
            log.debug(f"class {interface_name} in {mod} is not derived from 'SHARC_INTERFACE'")

        interfaces.append((interface_name, interface))
    log.debug(interfaces)
    return interfaces


def factory(name: str) -> SHARC_INTERFACE:
    try:
        ind = [i.upper() for i in AVAILABLE_INTERFACES].index(name.upper())
    except ValueError as e:
        raise e(f'Interface with name "{name}" does not exist!')
    int_name = AVAILABLE_INTERFACES[ind]
    interface_mod = import_module('SHARC_{}'.format(int_name))
    interface = getattr(interface_mod, int_name)
    if issubclass(interface, SHARC_INTERFACE):
        return interface
    else:
        raise ValueError(f"factory could not produce an interface:\n {interface}")
