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
from copy import deepcopy
import math
import sys
from abc import ABC, abstractmethod

# internal
from printing import printcomplexmatrix, printgrad, printtheodore
from utils import itnmstates, eformat
from constants import IToMult, IToPol


class INTERFACE(ABC):

    def __init__(self, PRINT=False, DEBUG=False):
        self.PRINT = PRINT
        self.DEBUG = DEBUG

    @abstractmethod
    def printheader():
        pass

    @abstractmethod
    def readQMin():
        pass
    # ======================================================================= #

    # ============================PRINTING ROUTINES========================== #
    def printQMout(self, QMin, QMout):
        '''If PRINT, prints a summary of all requested QM output values. Matrices are formatted using printcomplexmatrix, vectors using printgrad.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout'''

        # if DEBUG:
        # pprint.pprint(QMout)
        if not self.PRINT:
            return
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        print('===> Results:\n')
        # Hamiltonian matrix, real or complex
        if 'h' in QMin or 'soc' in QMin:
            eshift = math.ceil(QMout['h'][0][0].real)
            print('=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f' % (eshift))
            matrix = deepcopy(QMout['h'])
            for i in range(nmstates):
                matrix[i][i] -= eshift
            printcomplexmatrix(matrix, states)
        # Dipole moment matrices
        if 'dm' in QMin:
            print('=> Dipole Moment Matrices:\n')
            for xyz in range(3):
                print('Polarisation %s:' % (IToPol[xyz]))
                matrix = QMout['dm'][xyz]
                printcomplexmatrix(matrix, states)
        # Gradients
        if 'grad' in QMin:
            print('=> Gradient Vectors:\n')
            istate = 0
            for imult, i, ms in itnmstates(states):
                print('%s\t%i\tMs= % .1f:' % (IToMult[imult], i, ms))
                printgrad(QMout['grad'][istate], natom, QMin['geo'])
                istate += 1
        # Overlaps
        if 'overlap' in QMin:
            print('=> Overlap matrix:\n')
            matrix = QMout['overlap']
            printcomplexmatrix(matrix, states)
            if 'phases' in QMout:
                print('=> Wavefunction Phases:\n')
                for i in range(nmstates):
                    print('% 3.1f % 3.1f' % (QMout['phases'][i].real, QMout['phases'][i].imag))
                print('\n')
        # Spin-orbit coupling derivatives
        if 'socdr' in QMin:
            print('=> Spin-Orbit Gradient Vectors:\n')
            istate = 0
            for imult, i, ims in itnmstates(states):
                jstate = 0
                for jmult, j, jms in itnmstates(states):
                    print('%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:' % (IToMult[imult], i, ims, IToMult[jmult], j, jms))
                    printgrad(QMout['socdr'][istate][jstate], natom, QMin['geo'])
                    jstate += 1
                istate += 1
        # Dipole moment derivatives
        if 'dmdr' in QMin:
            print('=> Dipole moment derivative vectors:\n')
            istate = 0
            for imult, i, msi in itnmstates(states):
                jstate = 0
                for jmult, j, msj in itnmstates(states):
                    if imult == jmult and msi == msj:
                        for ipol in range(3):
                            print('%s\tStates %i - %i\tMs= % .1f\tPolarization %s:' % (IToMult[imult], i, j, msi, IToPol[ipol]))
                            printgrad(QMout['dmdr'][ipol][istate][jstate], natom, QMin['geo'])
                    jstate += 1
                istate += 1
        # Property matrix (dyson norms)
        if 'ion' in QMin and 'prop' in QMout:
            print('=> Property matrix:\n')
            matrix = QMout['prop']
            printcomplexmatrix(matrix, states)
        # TheoDORE
        if 'theodore' in QMin:
            print('=> TheoDORE results:\n')
            matrix = QMout['theodore']
            printtheodore(matrix, QMin)
        sys.stdout.flush()

    # ======================================================================= #


    def writeQMoutsoc(QMin, QMout):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the SOC matrix'''

        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (eformat(QMout['h'][i][j].real, 12, 3), eformat(QMout['h'][i][j].imag, 12, 3))
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #


    def writeQMoutdm(QMin, QMout):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The string contains three such matrices.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the DM matrices'''

        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (eformat(QMout['dm'][xyz][i][j].real, 12, 3), eformat(QMout['dm'][xyz][i][j].imag, 12, 3))
                string += '\n'
            string += ''
        return string
