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
from typing import Any, Union
from optparse import OptionParser
from constants import IAn2AName, ATOMCHARGE, FROZENS

# INTERNAL
import sharc.sharc as sharc
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
            if isinstance(data['grad'], list):
                self._QMout.set_gradient(list2dict(data['grad']), icall)
            else:
                if data['grad'] is None:
                    data['grad'] = {}
                self._QMout.set_gradient(data['grad'], icall)
        if 'nacdr' in data:
            if isinstance(data['nacdr'], list):
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


def set_qmout(qmout: QMOUT, icall: int):
    return sharc.set_qmout(qmout, icall)


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


def get_crd(unit: int = 0) -> list[list[float]]:
    '''returns coordinates in specified unit (0 = Bohr, 1 = Angstrom)'''
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
    return sharc.verlet_vstep()


def verlet_finalize(iskip=1):
    return sharc.verlet_finalize(iskip)


def finalize_sharc():
    return sharc.finalize_sharc()


def safe(func: callable):
    try:
        func()
    except Error:
        finalize_sharc()
        raise


def do_qm_calc(i: INTERFACE, qmout: QMOUT):
    icall = 1
    i.set_requests(get_all_tasks(icall))
    i.set_coords(get_crd())
    safe(i.run)
    i.getQMout()
    i.write_step_file()
    qmout.set_props(i._QMout, icall)

    isecond = set_qmout(qmout._QMout, icall)
    if isecond == 1:
        icall = 2
        i.set_requests(get_all_tasks(icall))
        i.set_coords(get_crd())
        qmout.set_props(i._QMout, icall)
        set_qmout(qmout._QMout, icall)
    return


def main():
    start = time.time_ns()
    parser = OptionParser()

    parser.add_option('-i', '--interface', dest='name', help='Name of the Interface you want to use.')
    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        default=False,
        help='sets verbosity, i.e. print and debug option'
    )
    parser.add_option('-d', '--debug', dest='debug', action='store_true', default=False, help='debug flag for printing')
    parser.add_option('-p', '--print', dest='print', action='store_true', default=False, help='flag for printing')

    (options, args) = parser.parse_args()

    if options.verbose:
        options.print = True
        options.debug = True
    if not options.name:
        raise Error('please specifiy the interface with "-i <name>"')
    if len(args) == 0:
        print("call with path to input file for SHARC")
        exit(0)
    inp_file = args[0]
    param = args[0:-1]
    interface = factory(options.name)

    derived_int: INTERFACE = interface(options.debug, options.print, persistent=True)
    derived_int.set_unit('bohr')
    if options.print:
        derived_int.printheader()
    IRestart = setup_sharc(inp_file)

    basic_info = get_basic_info()
    basic_info.update(derived_int.parseStates(basic_info['states']))
    QMout = QMOUT(derived_int.__class__.__name__, basic_info['NAtoms'], basic_info['nmstates'])

    basic_info['step'] = basic_info['istep']

    derived_int._QMin.update({k.lower(): v for k, v in basic_info.items()})
    derived_int._QMin['natom'] = basic_info['NAtoms']
    derived_int._QMin['elements'] = [IAn2AName[x] for x in basic_info['IAn']]
    derived_int._QMin['Atomcharge'] = sum(map(lambda x: ATOMCHARGE[x], derived_int._QMin['elements']))
    derived_int._QMin['frozcore'] = sum(map(lambda x: FROZENS[x], derived_int._QMin['elements']))
    derived_int._setup_mol = True
    derived_int.read_template()
    derived_int.read_resources()
    derived_int._step_logic()
    derived_int.setup_run()
    if IRestart == 0:
        initial_qm_pre()
        do_qm_calc(derived_int, QMout)
        initial_qm_post()
        initial_step(IRestart)
    lvc_time = 0.
    for istep in range(basic_info['istep'] + 1, basic_info['NSteps'] + 1):
        verlet_xstep(istep)
        s1 = time.perf_counter_ns()
        do_qm_calc(derived_int, QMout)
        s2 = time.perf_counter_ns()
        lvc_time += s2 - s1
        crd = get_crd()
        IRedo = verlet_vstep()

        if False:  # IRedo == 1:
            # calculate gradients numerically by setting up 6N calculations
            # TODO what if I want to get gradients only ? i.e. samestep
            # possibly skip whole Hamiltonian build in LVC -> major timesave
            i.set_requests(get_all_tasks(3))
            i.set_coords(crd)
            safe(i.run)
            QMout.set_gradient(list2dict(i._QMout['grad']), 3)
            set_qmout(QMout._QMout, 3)
        derived_int._QMin['step'] += 1
        iexit = verlet_finalize(1)
        if iexit == 1:
            break
    
    derived_int.create_restart_files()
    finalize_sharc()
    stop = time.time_ns()
    print(f'Timing per step ({derived_int.__class__.__name__}):', lvc_time / basic_info['NSteps'] * 1e-6, 'ms')
    print('Timing:', (stop - start) * 1e-6, 'ms')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCtrl+C makes me a sad SHARC ;-(\n')
        exit(1)
