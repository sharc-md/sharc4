#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
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

import argparse
import inspect
import os
import time
from importlib import import_module
from typing import Any

import numpy as np

try:
    import sharc.sharc as sharc
except:
    print("ERROR: sharc.sharc import failed. Do you have the correct Python environment loaded?")
    raise

from error import Error
from logger import log
from logger import loglevel as loglevel_env
from qmout import QMout
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import InDir


class QMOUT:
    """Wrapper for C-object used in sharc QMout"""

    def __init__(self, interface: str, natoms: int, nmstates: int):
        self._QMout = sharc.QMout(interface, natoms, nmstates)

    def printInfos(self):
        self._QMout.printInfos()

    def printAll(self):
        self._QMout.printAll()

    def set_props(self, data: QMout, icall):
        """set QMout"""
        # set hamiltonian, dm only in first call
        if icall == 1:
            log.debug("setting h and dm")
            if "h" in data:
                self._QMout.set_hamiltonian(np.asfortranarray(data["h"]))
            if "dm" in data:
                self._QMout.set_dipolemoment(np.asfortranarray(data["dm"]))
                log.debug("setting dm")
            if "mdm" in data:
                self._QMout.set_mag_dipolemoment(np.asfortranarray(data["mdm"]))
                log.debug("setting mdm")
            if "eqm" in data:
                self._QMout.set_el_quadrupolemoment(np.asfortranarray(data["eqm"]))
                log.debug("setting eqm")
        if "overlap" in data:
            # assumes type is numpy array
            self._QMout.set_overlap(np.asfortranarray(data["overlap"]))
        if "phases" in data:
            # assumes type is numpy array
            self._QMout.set_phases(np.asfortranarray(data["phases"]))
        if "grad" in data:
            self._QMout.set_gradient_full_array(np.asfortranarray(data["grad"]))
        if "nacdr" in data:
            self._QMout.set_nacdr_full_array(np.asfortranarray(data["nacdr"]))


def setup_sharc(inp_file: str) -> int:
    """parses input file and returns restart flag as int"""
    return sharc.setup_sharc(inp_file)


def set_qmout(qmout: QMOUT, icall: int):
    return sharc.set_qmout(qmout, icall)


def get_basic_info() -> dict[str, Any]:
    """returns dict {states: str, dt: str, savedir: str, NAtoms: int, NSteps: int, istep: int, IAn: list[int]}"""
    return sharc.get_basic_info()


def get_all_tasks(icall: int, nstates: int) -> dict:
    """returns {tasks: str, grad: str, nacdr: str}"""
    return sharc.get_all_tasks(icall, nstates)


def get_crd(unit: int = 0) -> list[list[float]]:
    """returns coordinates in specified unit (0 = Bohr, 1 = Angstrom)"""
    return sharc.get_crd(unit)


def get_vel() -> np.ndarray:
    """returns velocities"""
    return sharc.get_vel()


def initial_qm_pre():
    return sharc.initial_qm_pre()


def initial_qm_post():
    return sharc.initial_qm_post()


def initial_step(IRestart: int):
    return sharc.initial_step(IRestart)


def verlet_xstep(istep: int):
    return sharc.verlet_xstep(istep)


def verlet_vstep():
    return sharc.verlet_vstep(1)


def verlet_finalize(iskip=1):
    return sharc.verlet_finalize(iskip)


def finalize_sharc():
    return sharc.finalize_sharc()


def safe(func: callable):
    try:
        func()
    except Exception as e:
        sharc.error_finalize_sharc(str(e))
        raise


def do_qm_calc(i: SHARC_INTERFACE, qmout: QMOUT, nstates: int):
    icall = 1
    log.debug("\tset_requ")

    i.read_requests(get_all_tasks(icall, nstates))

    log.debug("\tcoords")
    i.set_coords(get_crd())
    i.set_veloc(get_vel())
    with InDir("QM"):
        log.debug("\trun")
        safe(i.run)
        log.debug("\twrite Stepfile")
        i.write_step_file()
    log.debug("\tset_props")
    qmdata = i.getQMout()
    qmout.set_props(qmdata, icall)
    i.clean_savedir()

    isecond = set_qmout(qmout._QMout, icall)
    if isecond == 1:
        icall = 2
        i.read_requests(get_all_tasks(icall, nstates))
        with InDir("QM"):
            safe(i.run)
        qmdata = i.getQMout()
        qmout.set_props(qmdata, icall)
        isecond = set_qmout(qmout._QMout, icall)
    return icall


def main():
    start = time.time_ns()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--interface", dest="name", help="Name of the Interface you want to use.")
    parser.add_argument(
        "-P", "--nonpersistent", dest="persistent", action="store_false", default=True, help="to turn off interface persistency"
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="sets verbosity, i.e. print and debug option")
    parser.add_argument("-s", "--silent", action="store_true", default=False, help="only error and critical output")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="debug flag for printing")
    parser.add_argument("-p", "--print", dest="print", action="store_true", default=False, help="flag for printing")
    parser.add_argument(
        "-f",
        "--fast_queue",
        dest="fast",
        action="store_true",
        default=False,
        help="Enable fast queue for hybrids with fast children.",
    )
    parser.add_argument("input_file", nargs="?", help="Path to input file for SHARC")

    options = parser.parse_args()

    loglevel = loglevel_env
    if options.silent:
        loglevel = log.ERROR
    if options.verbose:
        loglevel = log.SHARCPRINT
    if options.debug:
        loglevel = log.DEBUG
    if not options.name:
        raise Error('please specifiy the interface with "-i <name>"')
    if not options.input_file:
        print("call with path to input file for SHARC")
        exit(0)

    inp_file = options.input_file

    # load interface without factory
    interface_name = options.name.upper()
    interface_name = interface_name if interface_name.split("_")[0] == "SHARC" else f"SHARC_{interface_name}"
    try:
        module = import_module(interface_name)
    except (ModuleNotFoundError, ImportError, TypeError):
        log.error(f"{interface_name} could not be imported!")
        raise
    try:
        interface = getattr(module, interface_name)
        if not issubclass(interface, SHARC_INTERFACE):
            log.error(f"Class {interface_name} is not derived from SHARC_INTERFACE")
            raise ImportError()
        if inspect.isabstract(interface):
            log.error(f"{interface_name} is an abstract base class!")
            raise ImportError()
    except AttributeError as exc:
        log.error(f"Class {interface_name} not found in {module}")
        raise AttributeError from exc

    with InDir("QM"):
        derived_int: SHARC_INTERFACE = interface(persistent=options.persistent, loglevel=loglevel, fast_queue=options.fast)
    derived_int.QMin.molecule["unit"] = "bohr"
    derived_int.QMin.molecule["factor"] = 1.0
    if options.print:
        derived_int.printheader()
    IRestart = setup_sharc(inp_file)

    basic_info = get_basic_info()
    basic_info.update(derived_int.parseStates(basic_info["states"]))
    basic_info["savedir"] = os.path.join(os.getcwd(), "restart")
    QMout = QMOUT(derived_int.__class__.__name__, basic_info["NAtoms"], basic_info["nmstates"])

    derived_int.setup_mol(basic_info)
    nstates = derived_int.QMin.molecule["nmstates"]

    with InDir("QM"):
        derived_int.read_resources()
        derived_int.read_template()
        derived_int.setup_interface()
    if IRestart == 0:
        initial_qm_pre()
        do_qm_calc(derived_int, QMout, nstates)
        initial_qm_post()
        initial_step(IRestart)
    lvc_time = 0.0
    all_time = 0.0
    for istep in range(basic_info["istep"] + 1, basic_info["NSteps"] + 1):
        log.debug(f"{istep} starting step")
        all_s1 = time.perf_counter_ns()
        log.debug(f"{istep} verlet_xstep")
        verlet_xstep(istep)
        log.debug(f"{istep} done")
        s1 = time.perf_counter_ns()
        log.debug(f"{istep} do_qm_calc")
        count = do_qm_calc(derived_int, QMout, nstates)
        log.debug(f"{istep} done")
        s2 = time.perf_counter_ns()
        lvc_time += s2 - s1
        log.debug(f"{istep} done")
        log.debug(f"{istep} verlet_vstep")
        IRedo = verlet_vstep()
        log.debug(f"{istep} done")

        if IRedo == 2:
            with InDir("QM"):
                derived_int.read_requests(get_all_tasks(count, nstates))
                safe(derived_int.run)
                QMout.set_props(derived_int.getQMout(), 3)
        iexit = verlet_finalize(1)
        all_s2 = time.perf_counter_ns()
        all_time += all_s2 - all_s1
        if iexit == 1:
            break

    derived_int.create_restart_files()
    finalize_sharc()
    stop = time.time_ns()
    print(f"Timing per step ({derived_int.__class__.__name__}):", lvc_time / basic_info["NSteps"] * 1e-6, "ms")
    print("Timing per step full", all_time / basic_info["NSteps"] * 1e-6, "ms")
    print("Timing:", (all_time) * 1e-6, "ms")
    print("Timing:", (stop - start) * 1e-6, "ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C makes me a sad SHARC ;-(\n")
        exit(1)
