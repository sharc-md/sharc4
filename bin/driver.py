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
# EXTERNAL
import time
import numpy as np
from typing import Any, Union
from optparse import OptionParser
from constants import IAn2AName, ATOMCHARGE, FROZENS

# INTERNAL
import sharc.sharc as sharc

# import sharc
from factory import factory
from SHARC_INTERFACE import SHARC_INTERFACE
from qmout import QMout
from error import Error
from utils import list2dict, InDir, convert_list
from logger import log, loglevel as loglevel_env


class QMOUT:
    """Wrapper for C-object used in sharc QMout"""

    def __init__(self, interface: str, natoms: int, nmstates: int):
        self._QMout = sharc.QMout(interface, natoms, nmstates)

    def set_hamiltonian(self, h: list[list[Union[float, complex]]]):
        log.debug(f"{type(h)}")
        self._QMout.set_hamiltonian(h)

    def set_gradient(self, grad: dict[list[list[float], list[float], list[float]]], icall: int):
        log.debug(f"{type(grad)}")
        self._QMout.set_gradient(grad, icall)

    def set_dipolemoment(self, dip: list[list[list[Union[complex, float]]]]):
        log.debug(f"{type(dip)}")
        self._QMout.set_dipolemoment(dip)

    def set_overlap(self, ovl: list[list[float]]):
        log.debug(f"{type(ovl)}")
        self._QMout.set_overlap(ovl)

    def set_nacdr(self, nac: dict[int, dict[int, list[float, float, float]]], icall: int):
        log.debug(f"{type(nac)}")
        self._QMout.set_nacdr(nac, icall)

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
                self._QMout.set_hamiltonian(data["h"])
            if "dm" in data:
                self._QMout.set_dipolemoment(data["dm"])

        if "overlap" in data:
            # assumes type is numpy array
            self._QMout.set_overlap(data["overlap"])
        if "grad" in data:
            if isinstance(data["grad"], list):
                self._QMout.set_gradient(list2dict(data["grad"]), icall)
            elif data["grad"] is None:
                self._QMout.set_gradient({}, icall)
            elif isinstance(data["grad"], np.ndarray):
                self._QMout.set_gradient_full_array(data["grad"])
            else:
                raise RuntimeError
        if "nacdr" in data:
            if isinstance(data["nacdr"], dict):
                self._QMout.set_nacdr(data["nacdr"], icall)
            elif isinstance(data["nacdr"], list):
                nacdr = {}
                for i, ele in enumerate(data["nacdr"]):
                    nacdr[i] = list2dict(ele)
                self._QMout.set_nacdr(nacdr, icall)
            elif isinstance(data["nacdr"], np.ndarray):
                self._QMout.set_nacdr_full_array(data["nacdr"])
            else:
                raise RuntimeError

        return


def setup_sharc(inp_file: str) -> int:
    """parses input file and returns restart flag as int"""
    return sharc.setup_sharc(inp_file)


def set_qmout(qmout: QMOUT, icall: int):
    return sharc.set_qmout(qmout, icall)


def get_constants() -> dict:
    """returns dict with conversion constants"""
    return sharc.get_constants()


def get_tasks() -> str:
    """returns tasks string"""
    return sharc.get_tasks()


def get_basic_info() -> dict[str, Any]:
    """returns dict {states: str, dt: str, savedir: str, NAtoms: int, NSteps: int, istep: int, IAn: list[int]}"""
    return sharc.get_basic_info()


def get_all_tasks(icall: int) -> dict:
    """returns {tasks: str, grad: str, nacdr: str}"""
    return sharc.get_all_tasks(icall)


def get_crd(unit: int = 0) -> list[list[float]]:
    """returns coordinates in specified unit (0 = Bohr, 1 = Angstrom)"""
    return sharc.get_crd(unit)


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


def do_qm_calc(i: SHARC_INTERFACE, qmout: QMOUT):
    icall = 1
    log.debug(f"\tset_requ")
    i.read_requests(get_all_tasks(icall))
    log.debug(f"\tcoords")
    i.set_coords(get_crd())
    with InDir("QM"):
        log.debug(f"\trun")
        safe(i.run)
        log.debug(f"\twrite Stepfile")
        i.write_step_file()
    log.debug(f"\tset_props")
    qmout.set_props(i.getQMout(), icall)
    i.clean_savedir()

    isecond = set_qmout(qmout._QMout, icall)
    if isecond == 1:
        icall = 2
        i.read_requests(get_all_tasks(icall))
        with InDir("QM"):
            safe(i.run)
        qmout.set_props(i.getQMout(), icall)
        isecond = set_qmout(qmout._QMout, icall)
    return icall


def main():
    start = time.time_ns()
    parser = OptionParser()

    parser.add_option("-i", "--interface", dest="name", help="Name of the Interface you want to use.")
    parser.add_option("-P", "--nonpersistent", dest="persistent", action="store_false", default=True, help="to turn off interface persistency")
    parser.add_option(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="sets verbosity, i.e. print and debug option"
    )
    parser.add_option("-s", "--silent", dest="silent", action="store_true", default=False, help="only error and critical output")
    parser.add_option("-d", "--debug", dest="debug", action="store_true", default=False, help="debug flag for printing")
    parser.add_option("-p", "--print", dest="print", action="store_true", default=False, help="flag for printing")

    (options, args) = parser.parse_args()

    loglevel = loglevel_env
    if options.silent:
        loglevel = log.ERROR
    if options.verbose:
        loglevel = log.SHARCPRINT
    if options.debug:
        loglevel = log.DEBUG
    if not options.name:
        raise Error('please specifiy the interface with "-i <name>"')
    if len(args) == 0:
        print("call with path to input file for SHARC")
        exit(0)
    inp_file = args[0]
    # param = args[0:-1]
    interface = factory(options.name)

    derived_int: SHARC_INTERFACE = interface(persistent=options.persistent, loglevel=loglevel)
    derived_int.QMin.molecule["unit"] = "bohr"
    derived_int.QMin.molecule["factor"] = 1.0
    if options.print:
        derived_int.printheader()
    IRestart = setup_sharc(inp_file)

    basic_info = get_basic_info()
    basic_info.update(derived_int.parseStates(basic_info["states"]))
    QMout = QMOUT(derived_int.__class__.__name__, basic_info["NAtoms"], basic_info["nmstates"])

    derived_int.setup_mol(basic_info)

    with InDir("QM"):
        derived_int.read_resources()
        derived_int.read_template()
        # derived_int.QMin.save['savedir'] = basic_info['savedir']
        # derived_int.update_step(basic_info["step"])
        derived_int.setup_interface()
    if IRestart == 0:
        initial_qm_pre()
        do_qm_calc(derived_int, QMout)
        initial_qm_post()
        initial_step(IRestart)
        derived_int.update_step()
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
        count = do_qm_calc(derived_int, QMout)
        log.debug(f"{istep} done")
        s2 = time.perf_counter_ns()
        # print(" do_qm_calc: ", (s2 - s1) * 1e-6)
        lvc_time += s2 - s1
        log.debug(f"{istep} done")
        log.debug(f"{istep} verlet_vstep")
        IRedo = verlet_vstep()
        log.debug(f"{istep} done")

        if IRedo == 2:
            with(InDir("QM")):
                derived_int.read_requests(get_all_tasks(count))
                safe(derived_int.run)
                QMout.set_props(derived_int.getQMout(), 3)
        iexit = verlet_finalize(1)
        all_s2 = time.perf_counter_ns()
        all_time += all_s2 - all_s1
        if iexit == 1:
            break
        derived_int.update_step()

    derived_int.create_restart_files()
    finalize_sharc()
    stop = time.time_ns()
    print(f"Timing per step ({derived_int.__class__.__name__}):", lvc_time / basic_info["NSteps"] * 1e-6, "ms")
    print(f"Timing per step full", all_time / basic_info["NSteps"] * 1e-6, "ms")
    print("Timing:", (all_time) * 1e-6, "ms")
    print("Timing:", (stop - start) * 1e-6, "ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C makes me a sad SHARC ;-(\n")
        exit(1)
