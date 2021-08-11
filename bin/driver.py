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
import os
import sys
import numpy as np
from typing import Any, Union

# INTERNAL
import sharc
from factory import factory
from SHARC_INTERFACE import INTERFACE
from error import Error
from utils import list2dict


class QMOUT():
    '''Wrapper for C-object used in sharc QMout'''

    def __init__(self, interface: str, natoms: int, nmstates: int):
        self._QMout = sharc.QMout(interface, natoms, nmstates)

    def set_hamiltonian(self, h: list[list[Union[float, complex]]]):
        self._QMout.set_hamiltonian(h)

    def set_gradient(self, grad: dict[list[list[float], list[float], list[float]]], icall: int):
        self._QMout.set_gradient(grad, icall)

    def set_dipolemoment(self, dip: list[list[list[Union[complex, float]]]]):
        self._QMout.set_dipolemoment(dip)

    def set_overlap(self, ovl: list[list[float]]):
        self._QMout.set_overlap(ovl)

    def set_nacdr(self, nac: dict[int, dict[int, list[float, float, float]]], icall: int):
        self._QMout.set_nacdr(nac, icall)

    def printInfos(self):
        self._QMout.printInfos()

    def printAll(self):
        self._QMout.printAll()
    

    def set_props(self, data: dict, icall):
            """ set QMout """
            # set hamiltonian, dm only in first call
            if icall == 1:
                if 'h' in data:
                    self._QMout.set_hamiltonian(data['h'])
                if 'dm' in data:
                    self._QMout.set_dipolemoment(data['dm'])

            if 'overlap' in data:
                if not isinstance(data['overlap'], type([])):
                    # assumes type is numpy array
                    data['overlap'] = [list(ele) for ele in data['overlap']]
                self._QMout.set_overlap(data['overlap'])

            if 'grad' in data:
                if isinstance(data['grad'], type([])):
                    self._QMout.set_gradient(list2dict(data['grad']), icall)
                else:
                    if data['grad'] is None:
                        data['grad'] = {}
                    self._QMout.set_gradient(data['grad'], icall)
            if 'nacdr' in data:
                if isinstance(data['nacdr'], type([])):
                    nacdr = {}
                    for i, ele in enumerate(data['nacdr']):
                        nacdr[i] = list2dict(ele)
                    self._QMout.set_nacdr(nacdr, icall)

                else:
                    self._QMout.set_nacdr(data['nacdr'], icall)

            return


def setup_sharc(inp_file: str) -> int:
    '''parses input file and returns restart flag as int'''
    return sharc.setup_sharc(inp_file)


def set_qmout(qmout: QMOUT):
    return sharc.set_qmout(qmout)


def get_constants() -> dict:
    '''returns dict with conversion constants'''
    return sharc.get_constants()


def get_tasks() -> str:
    '''returns tasks string'''
    return sharc.get_tasks()


def get_basic_info() -> dict[str, Any]:
    '''returns dict {states: str, dt: str, savedir: str, NAtoms: int, NSteps: int, istep: int, IAn: list[int]} '''
    return sharc.get_basic_info()


def get_all_tasks(icall: int) -> dict:
    '''returns {tasks: str, grad: str, nacdr: str}'''
    return sharc.get_all_tasks(icall)


def get_crd(unit: int) -> list[list[float]]:
    '''returns coordinates in specified unit (0 = Bohr, 1 = Angstrom)'''
    return sharc.get_crd(unit)


def initial_qm_pre():
    return sharc.initial_qm_pre()


def initial_qm_post():
    return sharc.initial_qm_post()


def initial_step():
    return sharc.initial_step()


def verlet_xstep():
    return sharc.verlet_xstep()


def verlet_ystep():
    return sharc.verlet_ystep()


def verlet_finalize():
    return sharc.verlet_finalize()


def finalize_sharc():
    return sharc.finalize_sharc()


name = "lvc"


def safe(func: callable):
    try:
        func
    except Error:
        finalize_sharc()
        raise


def do_qm_calc(i: INTERFACE, qmout: QMOUT):
    icall = 1
    i.set_requests(get_all_tasks(icall))
    i.set_coords(get_crd())
    safe(i.run())
    qmout.set_props(i._QMout, icall)

    isecond = set_qmout(qmout)
    if isecond == 1:
        icall = 2
        i.set_requests(get_all_tasks(icall))
        i.set_coords(get_crd())
        sharc.set_qmout(i._QMout, icall)
    return


def main():
    args = sys.argv
    if len(args) == 0:
        print("call with path to input file for SHARC")
        exit(0)
    inp_file = args[0]
    param = args[0:-1]
    interface = factory(name)

    i: INTERFACE = interface()

    i.printheader()

    IRestart = setup_sharc(inp_file)
    basic_info = get_basic_info()
    basic_info.update(i.parseStates(basic_info['states']))
    QMout = QMOUT(i.__class__.__name__, basic_info['NAtoms'], basic_info['nmstates'])


    if IRestart == 0:
        initial_qm_pre()
        i.run()
        initial_qm_post()
