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
import time
from SHARC_INTERFACE import SHARC_INTERFACE
from SHARC_OLD import SHARC_OLD

global AVAILABLE_INTERFACES
AVAILABLE_INTERFACES = None
def get_available_interfaces() -> list[tuple[str, Union[SHARC_INTERFACE, str]]]:
    """
    returns available interfaces classes

    dynamically determines interfaces from set SHARC folder and returns the classes.

    Returns
    -------
    list[SHARC_INTERFACE]
        list of SHARC interface classes
    """
    global AVAILABLE_INTERFACES
    if AVAILABLE_INTERFACES is not None:
        return AVAILABLE_INTERFACES

    sharc_bin = expand_path('$SHARC')
    log.debug(f"factory interface collection: {sharc_bin}")
    interfaces = []
    start = time.time_ns()
    for path in sorted(glob.glob(sharc_bin + '/SHARC_*.py')):
        filename = path.split('/')[-1]
        interface_name = filename.split('.')[0]
        try:
            mod = import_module(interface_name)
        except TypeError as e:
            log.debug(f"{interface_name} could not be imported (not a package)\n\t{e}")
            interfaces.append((interface_name, "(Not Available!)", False))
            continue
        except (ModuleNotFoundError, ImportError) as e:
            log.debug(f"{interface_name} could not be imported (missing dependencies)\n\t{e}")
            interfaces.append((interface_name, "(Not Available!)", False))
            continue

        try:
            interface = getattr(mod, interface_name)
        except AttributeError as e:
            log.debug(f"class {interface_name} not found in {mod}\n\t{e}")
            interfaces.append((interface_name, "(Not Available!)", False))
            continue

        if issubclass(interface, SHARC_OLD):
            log.debug(f"class {interface_name} in {mod} is a legacy class")
            interfaces.append((interface_name, "(Not Available! Use SHARC_LEGACY to work with this interface)", False))
            continue

        if type(interface) == str or not issubclass(interface, SHARC_INTERFACE):
            log.debug(f"class {interface_name} in {mod} is not derived from 'SHARC_INTERFACE'")
            interfaces.append((interface_name, "(Not Available!)", False))
            continue

        interfaces.append((interface_name, interface, True))
    stop = time.time_ns()
    log.debug("Timing for finding interfaces: %.1f sec" % ((stop - start) * 1e-9) )
    log.debug(interfaces)
    AVAILABLE_INTERFACES = interfaces[:]
    return interfaces


def factory(name: str) -> SHARC_INTERFACE:
    available_interfaces = [i[1] for i in get_available_interfaces() if i[2] ]
    names = [i.__name__.split("_", maxsplit=1)[1] for i in available_interfaces]
    log.debug(f"{available_interfaces}\n{names}")
    try:
        ind = [i.upper() for i in names].index(name.upper())
    except ValueError as e:
        raise e(f'Interface with name "{name}" does not exist!')
    return available_interfaces[ind]
