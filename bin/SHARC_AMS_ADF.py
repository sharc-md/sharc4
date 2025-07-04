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

from SHARC_OLD import SHARC_OLD
class SHARC_AMS_ADF(SHARC_OLD):
    pass



# Modules:
# Operating system, isfile and related routines, move files, create directories
import os
import shutil
# External Calls to MOLCAS
import subprocess as sp
# Command line arguments
import sys
# Regular expressions
import re
# debug print for dicts and arrays
import pprint
# sqrt and other math
import math
import cmath
# runtime measurement
import datetime
# copy of arrays of arrays
from copy import deepcopy
# parallel calculations
from multiprocessing import Pool
import time
# hostname
from socket import gethostname
# write debug traces when in pool threads
import traceback
# parse Python literals from input
import ast
from constants import ATOMCHARGE, au2a, D2au, FROZENS, IToMult, IToPol
# =========================================================0
# compatibility stuff

if sys.version_info[0] < 3:
    print('This is a script for Python 3.5 or higher!')
    sys.exit(0)

if sys.version_info[1] < 5:
    print('This is a script for Python 3.5 or higher!')
    sys.exit(0)
# ======================================================================= #

version = '4.0'
versiondate = datetime.date(2025, 4, 1)

changelogstring = '''
12.04.21 copied from SHARC_ADF.py
'''

# ======================================================================= #
# holds the system time when the script was started
starttime = datetime.datetime.now()

# global variables for printing (PRINT gives formatted output, DEBUG gives raw output)
DEBUG = False
PRINT = True


# =============================================================================================== #
# =============================================================================================== #
# =========================================== general routines ================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #

def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(14)
    return out

# ======================================================================= #


def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
            sys.exit(15)
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(16)

# ======================================================================= #


def isbinary(path):
    return (re.search(r':.* text', sp.Popen(["file", '-L', path], stdout=sp.PIPE).stdout.read()) is None)

# ======================================================================= #


def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))

# ======================================================================= #


def measuretime():
    '''Calculates the time difference between global variable starttime and the time of the call of measuretime.

    Prints the Runtime, if PRINT or DEBUG are enabled.

    Arguments:
    none

    Returns:
    1 float: runtime in seconds'''

    endtime = datetime.datetime.now()
    runtime = endtime - starttime
    hours = runtime.seconds // 3600
    minutes = runtime.seconds // 60 - hours * 60
    seconds = runtime.seconds % 60
    print('==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days, hours, minutes, seconds))
    total_seconds = runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds / 1.e6
    return total_seconds

# ======================================================================= #


def removekey(d, key):
    '''Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary'''

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d

# =============================================================================================== #
# =============================================================================================== #
# ============================= iterator routines  ============================================== #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #
def itmult(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return

# ======================================================================= #


def itnmstates(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return

# =============================================================================================== #
# =============================================================================================== #
# =========================================== print routines ==================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def printheader():
    '''Prints the formatted header of the log file. Prints version number and version date

    Takes nothing, returns nothing.'''

    print(starttime, gethostname(), os.getcwd())
    if not PRINT:
        return
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    input = [' ',
             'SHARC - AMS - Interface',
             ' ',
             'Authors: Severin Polonius, Andrew Atkins, and Sebastian Mai ',
             ' ',
             'Version: %s' % (version),
             'Date: %s' % (versiondate.strftime("%d.%m.%y")),
             ' ']
    for inp in input:
        string += '||{:^80}||\n'.format(inp)
    string += '  ' + '=' * 80 + '\n\n'
    print(string)
    if DEBUG:
        print(changelogstring)

# ======================================================================= #


def printQMin(QMin):

    # if not PRINT:
    #     return
    print('==> QMin Job description for:\n%s' % (QMin['comment']))

    string = 'Mode:   '
    if 'init' in QMin:
        string += '\tINIT'
    if 'restart' in QMin:
        string += '\tRESTART'
    if 'samestep' in QMin:
        string += '\tSAMESTEP'
    if 'newstep' in QMin:
        string += '\tNEWSTEP'

    string += '\nTasks:  '
    if 'h' in QMin:
        string += '\tH'
    if 'soc' in QMin:
        string += '\tSOC'
    if 'dm' in QMin:
        string += '\tDM'
    if 'grad' in QMin:
        string += '\tGrad'
    if 'nacdr' in QMin:
        string += '\tNac(ddr)'
    if 'nacdt' in QMin:
        string += '\tNac(ddt)'
    if 'overlap' in QMin:
        string += '\tOverlap'
    if 'angular' in QMin:
        string += '\tAngular'
    if 'ion' in QMin:
        string += '\tDyson'
    if 'dmdr' in QMin:
        string += '\tDM-Grad'
    if 'socdr' in QMin:
        string += '\tSOC-Grad'
    if 'theodore' in QMin:
        string += '\tTheoDORE'
    if 'phases' in QMin:
        string += '\tPhases'
    print(string)

    string = 'States:        '
    for i in itmult(QMin['states']):
        string += '% 2i %7s  ' % (QMin['states'][i - 1], IToMult[i])
    print(string)

    string = 'Charges:       '
    for i in itmult(QMin['states']):
        string += '%+2i %7s  ' % (QMin['chargemap'][i], '')
    print(string)

    string = 'Restricted:    '
    for i in itmult(QMin['states']):
        string += '%5s       ' % (QMin['jobs'][QMin['multmap'][i]]['restr'])
    print(string)

    string = 'Method: \t'
    if QMin['template']['no_tda']:
        string += 'TD-'
    else:
        string += 'TDA-'
    string += QMin['template']['functional'].split()[-1].upper()
    string += '/%s' % (QMin['template']['basis'])
    parts = []
    if QMin['template']['relativistic']:
        parts.append(QMin['template']['relativistic'].split()[1].upper())
    if QMin['template']['dispersion']:
        parts.append(QMin['template']['dispersion'].split()[0].upper())
    if QMin['template']['modifyexcitations']:
        parts.append('modifyexcitations(%i)' % QMin['template']['modifyexcitations'])
    if QMin['template']['totalenergy']:
        parts.append('Total Energy')
    if QMin['template']['cosmo']:
        parts.append('COSMO')
    if QMin['template']['fullkernel']:
        parts.append('fullKernel')
    if len(parts) > 0:
        string += '\t('
        string += ','.join(parts)
        string += ')'
    print(string)

    string = 'Found Geo'
    if 'veloc' in QMin:
        string += ' and Veloc! '
    else:
        string += '! '
    string += 'NAtom is %i.\n' % (QMin['natom'])
    print(string)

    string = 'Geometry in Bohrs (%i atoms):\n' % QMin['natom']
    if DEBUG:
        for i in range(QMin['natom']):
            string += '%2s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['geo'][i][j + 1])
            string += '\n'
    else:
        for i in range(min(QMin['natom'], 5)):
            string += '%2s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['geo'][i][j + 1])
            string += '\n'
        if QMin['natom'] > 5:
            string += '..     ...     ...     ...\n'
            string += '%2s ' % (QMin['geo'][-1][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['geo'][-1][j + 1])
            string += '\n'
    print(string)

    if 'veloc' in QMin and DEBUG:
        string = ''
        for i in range(QMin['natom']):
            string += '%s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['veloc'][i][j])
            string += '\n'
        print(string)

    if 'grad' in QMin:
        string = 'Gradients requested:   '
        for i in range(1, QMin['nmstates'] + 1):
            if i in QMin['grad']:
                string += 'X '
            else:
                string += '. '
        string += '\n'
        print(string)

    print('State map:')
    pprint.pprint(QMin['statemap'])
    print()

    for i in sorted(QMin):
        if not any([i == j for j in ['h', 'dm', 'soc', 'dmdr', 'socdr', 'theodore', 'geo', 'veloc', 'states', 'comment', 'grad', 'nacdr', 'ion', 'overlap', 'template', 'statemap']]):
            if not any([i == j for j in ['ionlist']]) or DEBUG:
                string = i + ': '
                string += str(QMin[i])
                print(string)
    print('\n')
    sys.stdout.flush()


# ======================================================================= #
def printcomplexmatrix(matrix, states):
    '''Prints a formatted matrix. Zero elements are not printed, blocks of different mult and MS are delimited by dashes. Also prints a matrix with the imaginary parts, of any one element has non-zero imaginary part.

    Arguments:
    1 list of list of complex: the matrix
    2 list of integers: states specs'''

    nmstates = 0
    for i in range(len(states)):
        nmstates += states[i] * (i + 1)
    string = 'Real Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].real == 0.:
                string += ' ' * 11
            else:
                string += '% .3e ' % (matrix[istate][jstate].real)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    print(string)
    imag = False
    string = 'Imaginary Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].imag == 0.:
                string += ' ' * 11
            else:
                imag = True
                string += '% .3e ' % (matrix[istate][jstate].imag)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    string += '\n'
    if imag:
        print(string)

# ======================================================================= #


def printgrad(grad, natom, geo):
    '''Prints a gradient or nac vector. Also prints the atom elements. If the gradient is identical zero, just prints one line.

    Arguments:
    1 list of list of float: gradient
    2 integer: natom
    3 list of list: geometry specs'''

    string = ''
    iszero = True
    for atom in range(natom):
        if not DEBUG:
            if atom == 5:
                string += '...\t...\t     ...\t     ...\t     ...\n'
            if 5 <= atom < natom - 1:
                continue
        string += '%i\t%s\t' % (atom + 1, geo[atom][0])
        for xyz in range(3):
            if grad[atom][xyz] != 0:
                iszero = False
            g = grad[atom][xyz]
            if isinstance(g, float):
                string += '% .5f\t' % (g)
            elif isinstance(g, complex):
                string += '% .5f\t% .5f\t\t' % (g.real, g.imag)
        string += '\n'
    if iszero:
        print('\t\t...is identical zero...\n')
    else:
        print(string)

# ======================================================================= #


def printtheodore(matrix, QMin):
    string = '%6s ' % 'State'
    for i in QMin['template']['theodore_prop']:
        string += '%6s ' % i
    for i in range(len(QMin['template']['theodore_fragment'])):
        for j in range(len(QMin['template']['theodore_fragment'])):
            string += '  Om_%1i_%1i ' % (i + 1, j + 1)
    string += '\n' + '-------' * (1 + QMin['template']['theodore_n']) + '\n'
    istate = 0
    for imult, i, ms in itnmstates(QMin['states']):
        istate += 1
        string += '%6i ' % istate
        for i in matrix[istate - 1]:
            string += '%6.4f ' % i.real
        string += '\n'
    print(string)

# ======================================================================= #


def printQMout(QMin, QMout):
    '''If PRINT, prints a summary of all requested QM output values. Matrices are formatted using printcomplexmatrix, vectors using printgrad.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout'''

    # if DEBUG:
    # pprint.pprint(QMout)
    if not PRINT:
        return
    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    print('\n\n>>>>>>>>>>>>> Results\n')
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
                        print('%s\tStates %i - %i\tMs= % .1f\tPolarization %s:' %
                              (IToMult[imult], i, j, msi, IToPol[ipol]))
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


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Matrix initialization ================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #         OK
def makecmatrix(a, b):
    '''Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex'''

    mat = [[complex(0., 0.) for i in range(a)] for j in range(b)]
    return mat

# ======================================================================= #         OK


def makermatrix(a, b):
    '''Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real'''

    mat = [[0. for i in range(a)] for j in range(b)]
    return mat


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #
def writeQMout(QMin, QMout, QMinfilename):
    '''Writes the requested quantities to the file which SHARC reads in. The filename is QMinfilename with everything after the first dot replaced by "out".

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout
    3 string: QMinfilename'''

    k = QMinfilename.find('.')
    if k == -1:
        outfilename = QMinfilename + '.out'
    else:
        outfilename = QMinfilename[:k] + '.out'
    if PRINT:
        print('===> Writing output to file %s in SHARC Format\n' % (outfilename))
    string = ''

    # add header info
    string += '! 0 Basic information\nstates '
    for i in QMin['states']:
        string += '%i ' % i
    string += '\nnmstates %i\n' % QMin['nmstates']
    string += 'natom %i\n' % QMin['natom']
    string += 'npc 0\n'
    string += 'charges '
    for i in QMin['states']:
        string += '%i ' % 0
    string += '\n\n'

    if 'h' in QMin or 'soc' in QMin:
        string += writeQMoutsoc(QMin, QMout)
    if 'dm' in QMin:
        string += writeQMoutdm(QMin, QMout)
    if 'grad' in QMin:
        string += writeQMoutgrad(QMin, QMout)
    if 'overlap' in QMin:
        string += writeQMoutnacsmat(QMin, QMout)
    if 'socdr' in QMin:
        string += writeQMoutsocdr(QMin, QMout)
    if 'dmdr' in QMin:
        string += writeQMoutdmdr(QMin, QMout)
    if 'ion' in QMin:
        string += writeQMoutprop(QMin, QMout)
    if 'theodore' in QMin:
        string += writeQMoutTHEODORE(QMin, QMout)
    if 'phases' in QMin:
        string += writeQmoutPhases(QMin, QMout)
    string += writeQMouttime(QMin, QMout)
    outfile = os.path.join(QMin['pwd'], outfilename)
    writefile(outfile, string)
    return

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
            string += '%s %s ' % (eformat(QMout['h'][i][j].real, 9, 3), eformat(QMout['h'][i][j].imag, 9, 3))
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
                string += '%s %s ' % (eformat(QMout['dm'][xyz][i][j].real, 9, 3),
                                      eformat(QMout['dm'][xyz][i][j].imag, 9, 3))
            string += '\n'
        # string+='\n'
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutgrad(QMin, QMout):
    '''Generates a string with the Gradient vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the Gradient vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
        for atom in range(natom):
            for xyz in range(3):
                string += '%s ' % (eformat(QMout['grad'][i][atom][xyz], 9, 3))
            string += '\n'
        # string+='\n'
        i += 1
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutnacsmat(QMin, QMout):
    '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the transformation matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for j in range(nmstates):
        for i in range(nmstates):
            string += '%s %s ' % (eformat(QMout['overlap'][j][i].real, 9, 3),
                                  eformat(QMout['overlap'][j][i].imag, 9, 3))
        string += '\n'
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutdmdr(QMin, QMout):

    states = QMin['states']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n' % (12, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            for ipol in range(3):
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n' % (
                    natom, 3, imult, istate, ims, jmult, jstate, jms, ipol)
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s ' % (eformat(QMout['dmdr'][ipol][i][j][atom][xyz], 12, 3))
                    string += '\n'
                string += ''
            j += 1
        i += 1
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutsocdr(QMin, QMout):

    states = QMin['states']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n' % (13, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (
                natom, 3, imult, istate, ims, jmult, jstate, jms)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s %s ' % (eformat(QMout['socdr'][i][j][atom][xyz].real, 12, 3),
                                          eformat(QMout['socdr'][i][j][atom][xyz].imag, 12, 3))
            string += '\n'
            string += ''
            j += 1
        i += 1
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutprop(QMin, QMout):

    nmstates = QMin['nmstates']

    # print property matrix (flag 11) for backwards compatibility
    string = ''
    string += '! %i Property Matrix (%ix%i, complex)\n' % (11, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
        string += '\n'
    string += '\n'

    # print property matrices (flag 20) in new format
    string += '! %i Property Matrices\n' % (20)
    string += '%i    ! number of property matrices\n' % (1)

    string += '! Property Matrix Labels (%i strings)\n' % (1)
    string += 'Dyson norms\n'

    string += '! Property Matrices (%ix%ix%i, complex)\n' % (1, nmstates, nmstates)
    string += '%i %i   ! Dyson norms\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
        string += '\n'
    string += '\n'

    return string

# ======================================================================= #


def writeQMoutTHEODORE(QMin, QMout):

    nmstates = QMin['nmstates']
    nprop = QMin['template']['theodore_n']
    if nprop <= 0:
        return '\n'

    string = ''

    string += '! %i Property Vectors\n' % (21)
    string += '%i    ! number of property vectors\n' % (nprop)

    string += '! Property Vector Labels (%i strings)\n' % (nprop)
    descriptors = []
    if 'theodore' in QMin:
        for i in QMin['template']['theodore_prop']:
            descriptors.append('%s' % i)
            string += descriptors[-1] + '\n'
        for i in range(len(QMin['template']['theodore_fragment'])):
            for j in range(len(QMin['template']['theodore_fragment'])):
                descriptors.append('Om_{%i,%i}' % (i + 1, j + 1))
                string += descriptors[-1] + '\n'

    string += '! Property Vectors (%ix%i, real)\n' % (nprop, nmstates)
    if 'theodore' in QMin:
        for i in range(QMin['template']['theodore_n']):
            string += '%i ! TheoDORE descriptor %i (%s)\n' % (nmstates, i + 1, descriptors[i])
            for j in range(nmstates):
                string += '%s\n' % (eformat(QMout['theodore'][j][i].real, 12, 3))
    string += '\n'

    return string

# ======================================================================= #


def writeQmoutPhases(QMin, QMout):
    string = "! 7 Wave function phases (%ix1, complex)\n%i\n" % (QMin['nmstates'], QMin['nmstates'])
    for i in range(QMin['nmstates']):
        string += "%s %s\n" % (
            eformat(QMout['phases'][i].real, 9, 3),
            eformat(QMout['phases'][i].imag, 9, 3),
        )
    return string

# ======================================================================= #


def writeQMouttime(QMin, QMout):
    '''Generates a string with the quantum mechanics total runtime in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the runtime is given

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the runtime'''

    string = '! 8 Runtime\n%s\n' % (eformat(QMout['runtime'], 9, 3))
    return string


# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO readQMin =========================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def checkscratch(SCRATCHDIR):
    '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

    Arguments:
    1 string: path to SCRATCHDIR'''

    exist = os.path.exists(SCRATCHDIR)
    if exist:
        isfile = os.path.isfile(SCRATCHDIR)
        if isfile:
            print('$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR))
            sys.exit(17)
    else:
        try:
            os.makedirs(SCRATCHDIR)
        except OSError:
            print('Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR))
            sys.exit(18)

# ======================================================================= #


def removequotes(string):
    if string.startswith("'") and string.endswith("'"):
        return string[1:-1]
    elif string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    else:
        return string

# ======================================================================= #


def getAMSversion(AMSBIN):
    data = readfile(os.path.join(AMSBIN, 'version'))
    for line in data:
        if 'release=' in line:
            s = line.split('=')
            x = s[-1].split('.')
            return (int(x[0]), int(x[1]))
    else:
        print('WARNING: Could not detect AMS version!')
        return (1900, 0)

# ======================================================================= #


def getsh2AMSkey(sh2AMS, key):
    i = -1
    while True:
        i += 1
        try:
            line = re.sub('#.*$', '', sh2AMS[i])
        except IndexError:
            break
        line = line.split(None, 1)
        if line == []:
            continue
        if key.lower() in line[0].lower():
            return line
    return ['', '']

# ======================================================================= #


def get_sh2AMS_environ(sh2AMS, key, environ=True, crucial=True):
    line = getsh2AMSkey(sh2AMS, key)
    if line[0]:
        LINE = line[1]
        LINE = removequotes(LINE).strip()
    else:
        if environ:
            LINE = os.getenv(key.upper())
            if not LINE:
                if crucial:
                    print('Either set $%s or give path to %s in AMS_ADF.resources!' % (key.upper(), key.upper()))
                    sys.exit(19)
                else:
                    return None
        else:
            if crucial:
                print('Give path to %s in AMS_ADF.resources!' % (key.upper()))
                sys.exit(20)
            else:
                return None
    LINE = os.path.expandvars(LINE)
    LINE = os.path.expanduser(LINE)
    if ';' in LINE:
        print("$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (
            key.upper(), key.upper()))
        sys.exit(21)
    return LINE

# ======================================================================= #


def get_pairs(QMinlines, i):
    nacpairs = []
    while True:
        i += 1
        try:
            line = QMinlines[i].lower()
        except IndexError:
            print('"keyword select" has to be completed with an "end" on another line!')
            sys.exit(22)
        if 'end' in line:
            break
        fields = line.split()
        try:
            nacpairs.append([int(fields[0]), int(fields[1])])
        except ValueError:
            print('"nacdr select" is followed by pairs of state indices, each pair on a new line!')
            sys.exit(23)
    return nacpairs, i

# ======================================================================= #         OK


def readQMin(QMinfilename):
    '''Reads the time-step dependent information from QMinfilename.

    Arguments:
    1 string: name of the QMin file

    Returns:
    1 dictionary: QMin'''


# --------------------------------------------- QM.in ----------------------------------

    QMinlines = readfile(QMinfilename)
    QMin = {}
    # Get natom
    try:
        natom = int(QMinlines[0])
    except ValueError:
        print('first line must contain the number of atoms!')
        sys.exit(24)
    QMin['natom'] = natom
    if len(QMinlines) < natom + 4:
        print('Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task')
        sys.exit(25)

    # Save Comment line
    QMin['comment'] = QMinlines[1]

    # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
    QMin['geo'] = []
    QMin['veloc'] = []
    hasveloc = True
    QMin['frozcore'] = 0
    QMin['Atomcharge'] = 0
    for i in range(2, natom + 2):
        if re.search(r'[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]) is None:
            print('Input file does not comply to xyz file format! Maybe natom is just wrong.')
            sys.exit(26)
        fields = QMinlines[i].split()
        fields[0] = fields[0].title()
        symb = fields[0]
        QMin['frozcore'] += FROZENS[symb]
        QMin['Atomcharge'] += ATOMCHARGE[symb]
        for j in range(1, 4):
            fields[j] = float(fields[j])
        QMin['geo'].append(fields[0:4])
        if len(fields) >= 7:
            for j in range(4, 7):
                fields[j] = float(fields[j])
            QMin['veloc'].append(fields[4:7])
        else:
            hasveloc = False
    if not hasveloc:
        QMin = removekey(QMin, 'veloc')

    # Parse remaining file
    i = natom + 1
    while i + 1 < len(QMinlines):
        i += 1
        line = QMinlines[i]
        line = re.sub('#.*$', '', line)
        if len(line.split()) == 0:
            continue
        key = line.lower().split()[0]
        if 'savedir' in key:
            args = line.split()[1:]
        else:
            args = line.lower().split()[1:]
        if key in QMin:
            print('Repeated keyword %s in line %i in input file! Check your input!' % (key, i + 1))
            continue  # only first instance of key in QM.in takes effect
        if len(args) >= 1 and args[0] == 'select':
            pairs, i = get_pairs(QMinlines, i)
            QMin[key] = pairs
        else:
            QMin[key] = args

    if 'unit' in QMin:
        if QMin['unit'][0] == 'angstrom':
            factor = 1. / au2a
        elif QMin['unit'][0] == 'bohr':
            factor = 1.
        else:
            print('Dont know input unit %s!' % (QMin['unit'][0]))
            sys.exit(27)
    else:
        factor = 1. / au2a

    for iatom in range(len(QMin['geo'])):
        for ixyz in range(3):
            QMin['geo'][iatom][ixyz + 1] *= factor

    if 'states' not in QMin:
        print('Keyword "states" not given!')
        sys.exit(28)
    # Calculate states, nstates, nmstates
    for i in range(len(QMin['states'])):
        QMin['states'][i] = int(QMin['states'][i])
    reduc = 0
    for i in reversed(QMin['states']):
        if i == 0:
            reduc += 1
        else:
            break
    for i in range(reduc):
        del QMin['states'][-1]
    nstates = 0
    nmstates = 0
    for i in range(len(QMin['states'])):
        nstates += QMin['states'][i]
        nmstates += QMin['states'][i] * (i + 1)
    QMin['nstates'] = nstates
    QMin['nmstates'] = nmstates

    # Various logical checks
    if 'states' not in QMin:
        print('Number of states not given in QM input file %s!' % (QMinfilename))
        sys.exit(29)

    possibletasks = ['h', 'soc', 'dm', 'grad', 'overlap', 'dmdr', 'socdr', 'ion', 'theodore', 'phases']
    if not any([i in QMin for i in possibletasks]):
        print('No tasks found! Tasks are %s.' % possibletasks)
        sys.exit(30)

    if 'h' not in QMin and 'soc' not in QMin:
        QMin['h'] = []

    if 'h' in QMin and 'soc' in QMin:
        QMin = removekey(QMin, 'h')

    if 'soc' in QMin and (len(QMin['states']) < 3 or QMin['states'][2] <= 0):
        QMin = removekey(QMin, 'soc')
        QMin['h'] = []
        print('HINT: No triplet states requested, turning off SOC request.')

    if 'samestep' in QMin and 'init' in QMin:
        print('"Init" and "Samestep" cannot be both present in QM.in!')
        sys.exit(31)

    if 'restart' in QMin and 'init' in QMin:
        print('"Init" and "Samestep" cannot be both present in QM.in!')
        sys.exit(32)

    if 'phases' in QMin:
        QMin['overlap'] = []

    if 'overlap' in QMin and 'init' in QMin:
        print('"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"')
        sys.exit(33)

    if 'init' not in QMin and 'samestep' not in QMin and 'restart' not in QMin:
        QMin['newstep'] = []

    if not any([i in QMin for i in ['h', 'soc', 'dm', 'grad']]) and 'overlap' in QMin:
        QMin['h'] = []

    if 'nacdt' in QMin or 'nacdr' in QMin:
        print('Within the SHARC-AMS interface couplings can only be calculated via the overlap method. "nacdr" and "nacdt" are not supported.')
        sys.exit(34)

    if 'dmdr' in QMin:
        print('Dipole derivatives ("dmdr") not currently supported')
        sys.exit(35)

    if 'socdr' in QMin:
        print('Spin-orbit coupling derivatives ("socdr") are not implemented')
        sys.exit(36)

    # Check for correct gradient list
    if 'grad' in QMin:
        if len(QMin['grad']) == 0 or QMin['grad'][0] == 'all':
            QMin['grad'] = [i + 1 for i in range(nmstates)]
            # pass
        else:
            for i in range(len(QMin['grad'])):
                try:
                    QMin['grad'][i] = int(QMin['grad'][i])
                except ValueError:
                    print('Arguments to keyword "grad" must be "all" or a list of integers!')
                    sys.exit(37)
                if QMin['grad'][i] > nmstates:
                    print('State for requested gradient does not correspond to any state in QM input file state list!')
                    sys.exit(38)


# --------------------------------------------- AMS_ADF.resources ----------------------------------

    QMin['pwd'] = os.getcwd()

    # open AMS_ADF.resources
    filename = 'AMS_ADF.resources'
    if os.path.isfile(filename):
        sh2AMS = readfile(filename)
    else:
        print('HINT: reading resources from SH2AMS.inp')
        sh2AMS = readfile('SH2AMS.inp')

    # Set up scratchdir
    line = get_sh2AMS_environ(sh2AMS, 'scratchdir', False, False)
    if line is None:
        line = QMin['pwd'] + '/SCRATCHDIR/'
    line = line.replace("$$", str(os.getpid()))
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)
    # checkscratch(line)
    QMin['scratchdir'] = line
    link(QMin['scratchdir'], os.path.join(QMin['pwd'], 'SCRATCH'), False, False)

    # Set up savedir
    if 'savedir' in QMin:
        # savedir may be read from QM.in file
        line = QMin['savedir'][0]
    else:
        line = get_sh2AMS_environ(sh2AMS, 'savedir', False, False)
        if line is None:
            line = QMin['pwd'] + '/SAVEDIR/'
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)
    if 'init' in QMin:
        checkscratch(line)
    QMin['savedir'] = line
    link(QMin['savedir'], os.path.join(QMin['pwd'], 'SAVE'), False, False)

    # setup environment for AMS
    QMin['AMSHOME'] = get_sh2AMS_environ(sh2AMS, 'amshome')
    if not os.path.isfile(os.path.join(QMin['AMSHOME'], 'bin', 'ams')):
        print('AMS executable at "%s" not found!' % os.path.join(QMin['AMSHOME'], 'bin', 'ams'))
        sys.exit(39)
    os.environ['AMSHOME'] = QMin['AMSHOME']
    os.environ['AMSBIN'] = QMin['AMSHOME'] + '/bin'
    os.environ['AMSRESOURCES'] = QMin['AMSHOME'] + '/atomicdata'
    os.environ['PATH'] = '$AMSBIN:' + os.environ['PATH']
    sys.path.append(os.path.join(QMin['AMSHOME'], 'scripting'))
    sys.path.append(os.path.join(QMin['AMSHOME'], 'scripting', 'plams', 'src'))

    # AMS version
    version = getAMSversion(QMin['AMSHOME'] + '/bin')
    QMin['AMSversion'] = version
    print('Detected AMS version %i.%i' % version)

    # globally import KFFile
    global KFFile
    if QMin['AMSversion'][0] >= 2018:
        try:
            from scm.plams import KFFile
        except ImportError:
            print("scm.plams module not found")
            sys.exit(39)
    else:
        print('AMS 2018 and older are not supported by this script!')
        sys.exit(40)

    # setup license
    QMin['scmlicense'] = get_sh2AMS_environ(sh2AMS, 'scmlicense')
    os.environ['SCMLICENSE'] = QMin['scmlicense']

    # setup SCM-own scratch
    # TODO: make abspath of SCMTEMPDIR
    SCMTEMPDIR = get_sh2AMS_environ(sh2AMS, 'scm_tmpdir', True, False)
    if SCMTEMPDIR is None:
        if 'SCM_TEMPDIR' in os.environ:
            SCMTEMPDIR = os.environ['SCM_TMPDIR']
        else:
            SCMTEMPDIR = os.path.join(QMin['scratchdir'], 'SCM_TEMPDIR')
        mkdir(SCMTEMPDIR)
    # if SCMTEMPDIR==QMin['scratchdir']:
    try:
        if os.path.samefile(SCMTEMPDIR, QMin['scratchdir']):
            SCMTEMPDIR = os.path.join(QMin['scratchdir'], 'SCM_TEMPDIR')
    except OSError:
        pass
    if not os.path.isdir(SCMTEMPDIR):
        mkdir(SCMTEMPDIR)
    os.environ['SCM_TMPDIR'] = SCMTEMPDIR
    QMin['scm_tmpdir'] = SCMTEMPDIR

    # unset all queueing system variables so that we have full control over AMS
    if os.environ.get('PBS_JOBID') is not None:
        os.unsetenv('PBS_JOBID')
        if os.environ.get('PBS_ENVIRONMENT') is not None:
            os.unsetenv('PBS_ENVIRONMENT')
    if os.environ.get('SLURM_JOB_ID') is not None:
        os.unsetenv('SLURM_JOB_ID')
        if os.environ.get('SLURM_JOBID') is not None:
            os.unsetenv('SLURM_JOBID')
    if os.environ.get('LSB_JOBID') is not None:
        os.unsetenv.get('LSB_JOBID')
    if os.environ.get('SGE_JOB_SPOOL_DIR') is not None:
        os.unsetenv('SGE_JOB_SPOOL_DIR')
        if os.environ.get('PE_HOSTFILE') is not None:
            os.unsetenv('PE_HOSTFILE')
    if os.environ.get('SCM_MACHINEFILE') is not None:
        os.unsetenv('SCM_MACHINEFILE')

    # debug option
    line = getsh2AMSkey(sh2AMS, 'debug')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            global DEBUG
            DEBUG = True

    # debug option
    line = getsh2AMSkey(sh2AMS, 'no_print')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            global PRINT
            PRINT = False

    # resources
    QMin['ncpu'] = 1
    line = getsh2AMSkey(sh2AMS, 'ncpu')
    if line[0]:
        try:
            QMin['ncpu'] = int(line[1])
        except ValueError:
            print('Number of CPUs does not evaluate to numerical value!')
            sys.exit(41)
    if os.environ.get('NSLOTS') is not None:
        QMin['ncpu'] = int(os.environ.get('NSLOTS'))
        print('Detected $NSLOTS variable. Will use ncpu=%i' % (QMin['ncpu']))
    elif os.environ.get('SLURM_NTASKS_PER_NODE') is not None:
        QMin['ncpu'] = int(os.environ.get('SLURM_NTASKS_PER_NODE'))
        print('Detected $SLURM_NTASKS_PER_NODE variable. Will use ncpu=%i' % (QMin['ncpu']))
    QMin['ncpu'] = max(1, QMin['ncpu'])

    QMin['delay'] = 0.0
    line = getsh2AMSkey(sh2AMS, 'delay')
    if line[0]:
        try:
            QMin['delay'] = float(line[1])
        except ValueError:
            print('Submit delay does not evaluate to numerical value!')
            sys.exit(42)

    QMin['schedule_scaling'] = 0.9
    line = getsh2AMSkey(sh2AMS, 'schedule_scaling')
    if line[0]:
        try:
            x = float(line[1])
            if 0 < x <= 2.:
                QMin['schedule_scaling'] = x
        except ValueError:
            print('"schedule_scaling" does not evaluate to numerical value!')
            sys.exit(43)

    # initial MO guess settings
    # if neither keyword is present, the interface will reuse MOs from savedir, or let AMS generate a guess
    line = getsh2AMSkey(sh2AMS, 'always_orb_init')
    if line[0]:
        QMin['always_orb_init'] = []
    line = getsh2AMSkey(sh2AMS, 'always_guess')
    if line[0]:
        QMin['always_guess'] = []
    if 'always_orb_init' in QMin and 'always_guess' in QMin:
        print('Keywords "always_orb_init" and "always_guess" cannot be used together!')
        sys.exit(44)

    # wfoverlap settings
    if 'overlap' in QMin or 'ion' in QMin:
        QMin['wfoverlap'] = get_sh2AMS_environ(sh2AMS, 'wfoverlap', False, False)
        if QMin['wfoverlap'] is None:
            ciopath = os.path.join(os.path.expandvars(os.path.expanduser('$SHARC')), 'wfoverlap.x')
            if os.path.isfile(ciopath):
                QMin['wfoverlap'] = ciopath
            else:
                print('Give path to wfoverlap.x in AMS_ADF.resources!')
                sys.exit(45)

    # memory
    QMin['memory'] = 100
    line = getsh2AMSkey(sh2AMS, 'memory')
    if line[0]:
        QMin['memory'] = float(line[1])

    # truncation threshold
    QMin['wfthres'] = 0.99
    line = getsh2AMSkey(sh2AMS, 'wfthres')
    if line[0]:
        QMin['wfthres'] = float(line[1])

    # get the nooverlap keyword: no dets will be extracted if present
    line = getsh2AMSkey(sh2AMS, 'nooverlap')
    if line[0]:
        QMin['nooverlap'] = []

    # TheoDORE settings
    if 'theodore' in QMin:
        QMin['theodir'] = get_sh2AMS_environ(sh2AMS, 'theodir', False, False)
        if QMin['theodir'] is None or not os.path.isdir(QMin['theodir']):
            print('Give path to the TheoDORE installation directory in AMS_ADF.resources!')
            sys.exit(46)
        os.environ['THEODIR'] = QMin['theodir']
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = QMin['theodir'] + os.pathsep + os.path.join(
                QMin['theodir'], 'lib') + os.pathsep + os.path.join(QMin['AMSHOME'], 'scripting') + os.pathsep + os.environ['PYTHONPATH']
            # os.environ['PYTHONPATH']+=os.pathsep + os.path.join(QMin['theodir'],'lib')
            # os.environ['PYTHONPATH']+=os.pathsep + os.path.join(QMin['AMSHOME'],'scripting')
        else:
            os.environ['PYTHONPATH'] = QMin['theodir'] + os.pathsep + \
                os.path.join(QMin['theodir'], 'lib') + os.pathsep + os.path.join(QMin['AMSHOME'], 'scripting')

    # neglected gradients
    QMin['neglected_gradient'] = 'zero'
    if 'grad' in QMin:
        line = getsh2AMSkey(sh2AMS, 'neglected_gradient')
        if line[0]:
            arg = line[1].lower().strip()
            if arg in ['zero', 'gs', 'closest']:
                QMin['neglected_gradient'] = arg
            else:
                print('Unknown argument to "neglected_gradient"!')
                sys.exit(47)


# --------------------------------------------- AMS.template ----------------------------------

    # define classes and defaults
    bools = {'totalenergy': False,
             'exactdensity': False,
             'no_tda': False,
             'unrestricted_triplets': False,
            #  'dvd_mblocksmall': False,
             'functional_xcfun': False,
             'fullkernel': False
             }
    strings = {'relativistic': '',
               'basis': 'SZ',
               'basis_path': '',
               'functional': 'GGA PBE',
               'dispersion': '',
               'grid': 'beckegrid normal',
               'fit': 'zlmfit normal',
               'rihartreefock': '',
               'occupations': '',
               'cosmo': ''
               }
    integers = {'dvd_vectors': -1,
                'modifyexcitations': 0,
                'scf_iterations': 100,
                'linearscaling': 0,
                }
    floats = {'dvd_tolerance': 1e-6,
            #   'dvd_residu': -1.0,
              'cosmo_neql': -1.0,
              'cpks_eps': 0.0001,
              'grid_qpnear': 4.0 / au2a
              }
    special = {'basis_per_element': {},
               'define_fragment': {},
               'paddingstates': [0] * len(QMin['states']),
               'charge': [i % 2 for i in range(len(QMin['states']))],
               'theodore_prop': ['Om', 'PRNTO', 'S_HE', 'Z_HE', 'RMSeh'],
               'theodore_fragment': [],
               'grid_per_atom': {},
               'fit_per_atom': {},
               'rihf_per_atom': {}
               }

    # create QMin subdictionary
    QMin['template'] = {**bools, **strings, **integers, **floats, **special}

    # open template
    template = readfile('AMS_ADF.template')

    # go through template
    for line in template:
        orig = re.sub('#.*$', '', line).strip()
        line = orig.lower().split()
        if len(line) == 0:
            continue
        elif line[0] in bools:
            QMin['template'][line[0]] = True
        elif line[0] in strings:
            QMin['template'][line[0]] = orig.split(None, 1)[1]
        elif line[0] in integers:
            QMin['template'][line[0]] = int(float(line[1]))
        elif line[0] in floats:
            QMin['template'][line[0]] = float(line[1])
        elif line[0] in special:

            # basis_per_element can occur several times
            if line[0] == 'basis_per_element':
                line2 = orig.split(None, 2)
                QMin['template']['basis_per_element'][line2[1]] = os.path.expandvars(os.path.expanduser(line2[2]))

            # define_fragment can occur several times
            if line[0] == 'define_fragment':
                label = line[1].title()
                if '.' not in label:
                    print('ERROR: fragments for "define_fragment" have to be formatted like "El.i".')
                    sys.exit(48)
                s = label.split('.')
                if not len(s) == 2:
                    print('ERROR: fragments for "define_fragment" have to be formatted like "El.i".')
                    sys.exit(49)
                try:
                    x = int(s[1])
                except ValueError:
                    print('ERROR: fragments for "define_fragment" have to be formatted like "El.i".')
                    sys.exit(50)
                element = label.split('.')[0]
                atoms = [int(i) - 1 for i in line[2:]]
                ok = True
                for i in atoms:
                    if element != QMin['geo'][i][0]:
                        ok = False
                        print('ERROR: for "define_fragment" key, atom %i is not of element %s!' % (i + 1, element))
                    if any([i in j for j in list(QMin['template']['define_fragment'].values())]):
                        print('ERROR: atom %i used in two different in "define_fragment" lines!' % (i + 1))
                        ok = False
                if not ok:
                    print('"define_fragment" key misuse.')
                    sys.exit(51)
                QMin['template']['define_fragment'][label] = atoms

            # paddingstates needs to be autoexpanded and checked
            elif line[0] == 'paddingstates':
                if len(line) == 2:
                    QMin['template']['paddingstates'] = [int(line[1]) for i in range(len(QMin['states']))]
                elif len(line) - 1 >= len(QMin['states']):
                    QMin['template']['paddingstates'] = [int(line[1 + i]) for i in range(len(QMin['states']))]
                else:
                    print('Length of "paddingstates" does not match length of "states"!')
                    sys.exit(52)
                for i in range(len(QMin['template']['paddingstates'])):
                    if QMin['template']['paddingstates'][i] < 0:
                        QMin['template']['paddingstates'][i] = 0

            # charge needs to be autoexpanded, checked and assigned to the multiplicities
            elif line[0] == 'charge':
                if len(line) == 2:
                    charge = int(float(line[1]))
                    if (QMin['Atomcharge'] + charge) % 2 == 1 and len(QMin['states']) > 1:
                        print('HINT: Charge shifted by -1 to be compatible with multiplicities.')
                        charge -= 1
                    QMin['template']['charge'] = [i % 2 + charge for i in range(len(QMin['states']))]
                    print('HINT: total charge per multiplicity automatically assigned, please check (%s).' %
                          QMin['template']['charge'])
                    print('You can set the charge in the template manually for each multiplicity ("charge 0 +1 0 ...")\n')
                elif len(line) - 1 >= len(QMin['states']):
                    QMin['template']['charge'] = [int(float(line[1 + i])) for i in range(len(QMin['states']))]
                    compatible = True
                    for imult, cha in enumerate(QMin['template']['charge']):
                        if not (QMin['Atomcharge'] + cha + imult) % 2 == 0:
                            compatible = False
                    if not compatible:
                        print('WARNING: Charges from template not compatible with multiplicities!  (this is probably OK if you use QM/MM)')
                else:
                    print('Length of "charge" does not match length of "states"!')
                    sys.exit(54)

            # those can occur several times
            elif line[0] == 'grid_per_atom' or line[0] == 'fit_per_atom' or line[0] == 'rihf_per_atom':
                quality = line[1]
                if quality not in QMin['template'][line[0]]:
                    QMin['template'][line[0]][quality] = []
                for i in line[2:]:
                    n = int(i)
                    if 1 <= n <= QMin['natom']:
                        QMin['template'][line[0]][quality].append(n)

    # go through sh2AMS for the theodore settings and QM/MM file names
    for line in sh2AMS:
        orig = re.sub('#.*$', '', line).strip()
        line = orig.lower().split()
        if len(line) == 0:
            continue
        elif line[0] in special:

            # TheoDORE properties need to be parsed in a special way
            if line[0] == 'theodore_prop':
                if '[' in orig:
                    string = orig.split(None, 1)[1]
                    QMin['template']['theodore_prop'] = ast.literal_eval(string)
                else:
                    QMin['template']['theodore_prop'] = []
                    s = orig.split(None)[1:]
                    for i in s:
                        QMin['template']['theodore_prop'].append(i)
                theodore_spelling = ['Om',
                                     'PRNTO',
                                     'Z_HE', 'S_HE', 'RMSeh',
                                     'POSi', 'POSf', 'POS',
                                     'PRi', 'PRf', 'PR', 'PRh',
                                     'CT', 'CT2', 'CTnt',
                                     'MC', 'LC', 'MLCT', 'LMCT', 'LLCT',
                                     'DEL', 'COH', 'COHh']
                for i in range(len(QMin['template']['theodore_prop'])):
                    for j in theodore_spelling:
                        if QMin['template']['theodore_prop'][i].lower() == j.lower():
                            QMin['template']['theodore_prop'][i] = j

            # TheoDORE fragments need to be parsed in a special way
            elif line[0] == 'theodore_fragment':
                if '[' in orig:
                    string = orig.split(None, 1)[1]
                    QMin['template']['theodore_fragment'] = ast.literal_eval(string)
                else:
                    s = orig.split(None)[1:]
                    frag_list = []
                    for i in s:
                        frag_list.append(int(i))
                    QMin['template']['theodore_fragment'].append(frag_list)

    # do logic checks
    if QMin['template']['unrestricted_triplets'] and 'soc' in QMin:
        if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
            print('Request "SOC" is not compatible with "unrestricted_triplets"!')
            sys.exit(57)
    if 'soc' in QMin and not QMin['template']['relativistic']:
        print('You have to use a relativistic Hamiltonian (e.g., ZORA) for spin-orbit couplings!')
        sys.exit(61)
    if QMin['template']['totalenergy'] and QMin['template']['relativistic']:
        print('Key "totalenergy" is not compatible with "relativistic"!')
        sys.exit(62)
    if not QMin['template']['unrestricted_triplets']:
        if len(QMin['template']['charge']) >= 3 and QMin['template']['charge'][0] != QMin['template']['charge'][2]:
            print('Charges of singlets and triplets differ. Please enable the "unrestricted_triplets" option!')
            sys.exit(63)
    if QMin['template']['cosmo'] and 'grad' in QMin:
        print('COSMO is not compatible with gradient calculations!')
        sys.exit(64)
    if 'grad' in QMin:
        allowed_functionals = ['lda', 'gga', 'hybrid', 'hartreefock']
        if not QMin['template']['functional'].split()[0].lower() in allowed_functionals:
            print('Gradients can only be calculated with these functional classes: %s' % allowed_functionals)
            sys.exit(65)
        if QMin['template']['fullkernel']:
            print('Gradients are not possible with full adiabatic XC kernel')
            sys.exit(66)
    atom_frags = set()
    for iatom, atom in enumerate(QMin['geo']):
        label = atom[0]
        if QMin['template']['define_fragment']:
            for frag_list in QMin['template']['define_fragment']:
                if iatom in QMin['template']['define_fragment'][frag_list]:
                    label = frag_list
        atom_frags.add(label)
    QMin['atom_frags'] = atom_frags
    # number of frozen core orbitals for wfoverlap (no frozen orbitals in AMS!)
    # this call is down here because we need to check for QM/MM in template first
    line = getsh2AMSkey(sh2AMS, 'numfrozcore')
    if line[0]:
        numfroz = int(line[1])
        if numfroz == 0:
            QMin['frozcore'] = 0
        elif numfroz > 0:
            QMin['frozcore'] = numfroz
        elif numfroz < 0:
            pass        # here we take frozcore from above
    if QMin['template']['modifyexcitations'] and QMin['frozcore'] > 0:
        if 'nooverlap' not in QMin or 'ion' in QMin:
            print('Calculations with "modifyexcitations" should only be done with numfrozcore=0!')
            sys.exit(76)

    # check and process the grid_per_atom and fit_per_atom keys
    nmax = QMin['natom']
    grid_map = {}
    for quality in QMin['template']['grid_per_atom']:
        for i in QMin['template']['grid_per_atom'][quality]:
            if i <= nmax:
                grid_map[i] = quality
    QMin['template']['grid_per_atom'] = grid_map
    fit_map = {}
    for quality in QMin['template']['fit_per_atom']:
        for i in QMin['template']['fit_per_atom'][quality]:
            if i <= nmax:
                fit_map[i] = quality
    QMin['template']['fit_per_atom'] = fit_map
    rihf_map = {}
    for quality in QMin['template']['rihf_per_atom']:
        for i in QMin['template']['rihf_per_atom'][quality]:
            if i <= nmax:
                rihf_map[i] = quality
    QMin['template']['rihf_per_atom'] = rihf_map

    # number of doubly occupied orbitals for Dyson
    line = getsh2AMSkey(sh2AMS, 'numocc')
    if line[0]:
        numfroz = int(line[1])
        if numfroz <= 0:
            QMin['ndocc'] = 0
        elif numfroz > 0:
            QMin['ndocc'] = max(0, numfroz - QMin['frozcore'])
    else:
        QMin['ndocc'] = 0

# --------------------------------------------- Logic ----------------------------------

    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(QMin['states']):
        statemap[i] = [imult, istate, ims]
        i += 1
    QMin['statemap'] = statemap

    # obtain the states to actually compute
    states_to_do = deepcopy(QMin['states'])
    for i in range(len(QMin['states'])):
        if states_to_do[i] > 0:
            states_to_do[i] += QMin['template']['paddingstates'][i]
    if not QMin['template']['unrestricted_triplets']:
        if len(QMin['states']) >= 3 and QMin['states'][2] > 0 and QMin['states'][0] <= 1:
            if 'soc' in QMin:
                states_to_do[0] = 2
            else:
                states_to_do[0] = 1
    QMin['states_to_do'] = states_to_do

    # make the jobs
    jobs = {}
    if QMin['states_to_do'][0] > 0:
        jobs[1] = {'mults': [1], 'restr': True}
    if len(QMin['states_to_do']) >= 2 and QMin['states_to_do'][1] > 0:
        jobs[2] = {'mults': [2], 'restr': False}
    if len(QMin['states_to_do']) >= 3 and QMin['states_to_do'][2] > 0:
        if not QMin['template']['unrestricted_triplets'] and QMin['states_to_do'][0] > 0:
            jobs[1]['mults'].append(3)
        else:
            jobs[3] = {'mults': [3], 'restr': False}
    if len(QMin['states_to_do']) >= 4:
        for imult, nstate in enumerate(QMin['states_to_do'][3:]):
            if nstate > 0:
                jobs[len(jobs) + 1] = {'mults': [imult + 4], 'restr': False}
    QMin['jobs'] = jobs

    # make the multmap (mapping between multiplicity and job)
    # multmap[imult]=ijob
    # multmap[-ijob]=[imults]
    multmap = {}
    for ijob in jobs:
        job = jobs[ijob]
        for imult in job['mults']:
            multmap[imult] = ijob
        multmap[-(ijob)] = job['mults']
    QMin['multmap'] = multmap

    # get the joblist
    joblist = set()
    for i in jobs:
        joblist.add(i)
    joblist = sorted(joblist)
    QMin['joblist'] = joblist
    njobs = len(joblist)
    QMin['njobs'] = njobs

    # make the gsmap
    gsmap = {}
    for i in range(QMin['nmstates']):
        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
        gs = (m1, 1, ms1)
        job = QMin['multmap'][m1]
        if m1 == 3 and QMin['jobs'][job]['restr']:
            gs = (1, 1, 0.0)
        for j in range(QMin['nmstates']):
            m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
            if (m2, s2, ms2) == gs:
                break
        gsmap[i + 1] = j + 1
    QMin['gsmap'] = gsmap

    # get the set of states for which gradients actually need to be calculated
    gradmap = set()
    if 'grad' in QMin:
        for i in QMin['grad']:
            gradmap.add(tuple(statemap[i][0:2]))
            gradmap.add((statemap[i][0], 1))
    gradmap = list(gradmap)
    gradmap.sort()
    QMin['gradmap'] = gradmap

    # make the chargemap
    QMin['chargemap'] = {}
    for i, c in enumerate(QMin['template']['charge']):
        QMin['chargemap'][i + 1] = c

    # make the ionmap
    if 'ion' in QMin:
        ionmap = []
        for m1 in itmult(QMin['states']):
            job1 = QMin['multmap'][m1]
            el1 = QMin['chargemap'][m1]
            for m2 in itmult(QMin['states']):
                if m1 >= m2:
                    continue
                job2 = QMin['multmap'][m2]
                el2 = QMin['chargemap'][m2]
                # print m1,job1,el1,m2,job2,el2
                if abs(m1 - m2) == 1 and abs(el1 - el2) == 1:
                    ionmap.append((m1, job1, m2, job2))
        QMin['ionmap'] = ionmap

    # number of properties/entries calculated by TheoDORE
    if 'theodore' in QMin:
        QMin['template']['theodore_n'] = len(QMin['template']['theodore_prop']) + \
            len(QMin['template']['theodore_fragment'])**2
    else:
        QMin['template']['theodore_n'] = 0

# --------------------------------------------- File setup ----------------------------------

    # check for initial orbitals
    initorbs = {}
    if 'always_guess' in QMin:
        QMin['initorbs'] = {}
    elif 'init' in QMin or 'always_orb_init' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['pwd'], 'AMS_ADF.t21.%i.init' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename
        if 'always_orb_init' in QMin and len(initorbs) < njobs:
            print('Initial orbitals missing for some jobs!')
            sys.exit(77)
        QMin['initorbs'] = initorbs
    elif 'newstep' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['savedir'], 'TAPE21.%i' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename + '.old'     # file will be moved to .old
            else:
                print('File %s missing in savedir!' % (filename))
                sys.exit(78)
        QMin['initorbs'] = initorbs
    elif 'samestep' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['savedir'], 'TAPE21.%i' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename
            else:
                print('File %s missing in savedir!' % (filename))
                sys.exit(79)
        QMin['initorbs'] = initorbs
    elif 'restart' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['savedir'], 'TAPE21.%i.old' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename
            else:
                print('File %s missing in savedir!' % (filename))
                sys.exit(80)
        QMin['initorbs'] = initorbs

    # check for atomic fragment files
    frags_there = ('init' not in QMin)
    for i in QMin['atom_frags']:
        filename = os.path.join(QMin['savedir'], 'frag.t21.%s' % (i))
        if not os.path.isfile(filename):
            frags_there = False
    QMin['frags_there'] = frags_there

    # make name for backup directory
    if 'backup' in QMin:
        backupdir = QMin['savedir'] + '/backup'
        backupdir1 = backupdir
        i = 0
        while os.path.isdir(backupdir1):
            i += 1
            if 'step' in QMin:
                backupdir1 = backupdir + '/step%s_%i' % (QMin['step'][0], i)
            else:
                backupdir1 = backupdir + '/calc_%i' % (i)
        QMin['backup'] = backupdir

    if DEBUG:
        print('======= DEBUG print for QMin =======')
        pprint.pprint(QMin)
        print('====================================')
    return QMin

# =============================================================================================== #
# =============================================================================================== #
# =========================================== Job Scheduling ==================================== #
# =============================================================================================== #
# =============================================================================================== #


def parallel_speedup(N, scaling):
    # computes the parallel speedup from Amdahls law
    # with scaling being the fraction of parallelizable work and (1-scaling) being the serial part
    return 1. / ((1 - scaling) + scaling / N)


def divide_slots(ncpu, ntasks, scaling):
    # this routine figures out the optimal distribution of the tasks over the CPU cores
    #   returns the number of rounds (how many jobs each CPU core will contribute to),
    #   the number of slots which should be set in the Pool,
    #   and the number of cores for each job.
    minpar = 1
    ntasks_per_round = ncpu // minpar
    if ncpu == 1:
        ntasks_per_round = 1
    ntasks_per_round = min(ntasks_per_round, ntasks)
    optimal = {}
    for i in range(1, 1 + ntasks_per_round):
        nrounds = int(math.ceil(float(ntasks) / i))
        ncores = ncpu // i
        optimal[i] = nrounds / parallel_speedup(ncores, scaling)
    # print optimal
    best = min(optimal, key=optimal.get)
    nrounds = int(math.ceil(float(ntasks) / best))
    ncores = ncpu // best

    cpu_per_run = [0 for i in range(ntasks)]
    if nrounds == 1:
        itask = 0
        for icpu in range(ncpu):
            cpu_per_run[itask] += 1
            itask += 1
            if itask >= ntasks:
                itask = 0
        nslots = ntasks
    else:
        for itask in range(ntasks):
            cpu_per_run[itask] = ncores
        nslots = ncpu // ncores
    # print nrounds,nslots,cpu_per_run
    return nrounds, nslots, cpu_per_run

# =============================================================================================== #


def generate_joblist(QMin):

    # pprint.pprint(QMin)

    # sort the gradients into the different jobs
    gradjob = {}
    for ijob in QMin['joblist']:
        gradjob['master_%i' % ijob] = {}
    for grad in QMin['gradmap']:
        ijob = QMin['multmap'][grad[0]]
        isgs = False
        if not QMin['jobs'][ijob]['restr']:
            if grad[1] == 1:
                isgs = True
        else:
            if grad == (1, 1):
                isgs = True
        if isgs:
            gradjob['master_%i' % ijob][grad] = {'gs': True}
        else:
            # in QM/MM, one cannot combine gs and es gradient in master
            n = 0
            for gradx in gradjob['master_%i' % ijob]:
                if gradjob['master_%i' % ijob][gradx]['gs'] is False:
                    n += 1
            gradjob['master_%i' % ijob][grad] = {'gs': False}

    # make map for states onto gradjobs
    jobgrad = {}
    for job in gradjob:
        for state in gradjob[job]:
            jobgrad[state] = (job, gradjob[job][state]['gs'])
    QMin['jobgrad'] = jobgrad

    schedule = []
    QMin['nslots_pool'] = []

    # add the master calculations
    ntasks = 0
    for i in gradjob:
        if 'master' in i:
            ntasks += 1
    nrounds, nslots, cpu_per_run = divide_slots(QMin['ncpu'], ntasks, QMin['schedule_scaling'])
    QMin['nslots_pool'].append(nslots)
    schedule.append({})
    icount = 0
    for i in sorted(gradjob):
        if 'master' in i:
            QMin1 = deepcopy(QMin)
            QMin1['master'] = True
            QMin1['IJOB'] = int(i.split('_')[1])
            remove = ['gradmap', 'ncpu']
            for r in remove:
                QMin1 = removekey(QMin1, r)
            QMin1['gradmap'] = list(gradjob[i])
            QMin1['ncpu'] = cpu_per_run[icount]
            icount += 1
            schedule[-1][i] = QMin1

    # add the gradient calculations
    # TODO: this code is not executed
    ntasks = 0
    for i in gradjob:
        if 'grad' in i:
            ntasks += 1
    if ntasks > 0:
        nrounds, nslots, cpu_per_run = divide_slots(QMin['ncpu'], ntasks, QMin['schedule_scaling'])
        QMin['nslots_pool'].append(nslots)
        schedule.append({})
        icount = 0
        for i in gradjob:
            if 'grad' in i:
                QMin1 = deepcopy(QMin)
                mult = list(gradjob[i])[0][0]
                QMin1['IJOB'] = QMin['multmap'][mult]
                remove = ['gradmap', 'ncpu', 'h', 'soc', 'dm', 'overlap',
                          'ion', 'always_guess', 'always_orb_init', 'init']
                for r in remove:
                    QMin1 = removekey(QMin1, r)
                QMin1['gradmap'] = list(gradjob[i])
                QMin1['ncpu'] = cpu_per_run[icount]
                QMin1['gradonly'] = []
                icount += 1
                schedule[-1][i] = QMin1

    return QMin, schedule


# =============================================================================================== #
# =============================================================================================== #
# ======================================= AMS Job Execution ===================================== #
# =============================================================================================== #
# =============================================================================================== #

def runjobs(schedule, QMin):

    if 'newstep' in QMin:
        moveOldFiles(QMin)

    print('>>>>>>>>>>>>> Starting the AMS job execution')

    errorcodes = {}
    for ijobset, jobset in enumerate(schedule):
        if not jobset:
            continue
        pool = Pool(processes=QMin['nslots_pool'][ijobset])
        for job in jobset:
            QMin1 = jobset[job]
            WORKDIR = os.path.join(QMin['scratchdir'], job)

            errorcodes[job] = pool.apply_async(run_calc, [WORKDIR, QMin1])
            time.sleep(QMin['delay'])
        pool.close()
        pool.join()

    for i in errorcodes:
        errorcodes[i] = errorcodes[i].get()
    j = 0
    string = 'Error Codes:\n'
    for i in errorcodes:
        string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
        j += 1
        if j == 4:
            j = 0
            string += '\n'
    print(string)
    if any((i != 0 for i in list(errorcodes.values()))):
        print('Some subprocesses did not finish successfully!')
        print('See %s:%s for error messages in AMS output.' % (gethostname(), QMin['scratchdir']))
        sys.exit(81)
    print()

    if PRINT:
        print('>>>>>>>>>>>>> Saving files')
        starttime = datetime.datetime.now()
    for ijobset, jobset in enumerate(schedule):
        if not jobset:
            continue
        for job in jobset:
            if 'master' in job:
                WORKDIR = os.path.join(QMin['scratchdir'], job)
                if 'samestep' not in QMin:
                    saveFiles(WORKDIR, jobset[job])
                if 'ion' in QMin and ijobset == 0:
                    saveAOmatrix(WORKDIR, QMin)
    if PRINT:
        endtime = datetime.datetime.now()
        print('Saving Runtime: %s' % (endtime - starttime))
    print()

    return errorcodes

# ======================================================================= #


def run_calc(WORKDIR, QMin):
    try:
        setupWORKDIR(WORKDIR, QMin)
        strip = True
        err = runAMS(WORKDIR, QMin['AMSHOME'], QMin['ncpu'], QMin['savedir'], strip)
    except Exception as problem:
        print('*' * 50 + '\nException in run_calc(%s)!' % (WORKDIR))
        traceback.print_exc()
        print('*' * 50 + '\n')
        raise problem

    return err

# ======================================================================= #


def setupWORKDIR(WORKDIR, QMin):
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
    # then put the AMS.input file

    # setup the directory
    mkdir(WORKDIR)

    # write AMS.input
    inputstring = writeAMSinput(QMin, WORKDIR)
    filename = os.path.join(WORKDIR, 'AMS.run')
    writefile(filename, inputstring)
    if DEBUG:
        print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        print(inputstring)
        print('AMS input written to: %s' % (filename))
        print('====================================================================')

    # wf file copying
    if 'master' in QMin:
        job = QMin['IJOB']
        if job in QMin['initorbs']:
            fromfile = QMin['initorbs'][job]
            tofile = os.path.join(WORKDIR, 'TAPE21.guess')
            shutil.copy(fromfile, tofile)
    elif 'grad' in QMin:
        job = QMin['IJOB']
        fromfile = os.path.join(QMin['scratchdir'], 'master_%i' % job, 'TAPE21')
        tofile = os.path.join(WORKDIR, 'TAPE21.guess')
        shutil.copy(fromfile, tofile)

    # atomic fragment files
    if QMin['frags_there']:
        for i in QMin['atom_frags']:
            fromfile = os.path.join(QMin['savedir'], 'frag.t21.%s' % i)
            tofile = os.path.join(WORKDIR, 't21.%s' % i)
            shutil.copy(fromfile, tofile)

    return

# ======================================================================= #


def writeAMSinput(QMin, WORKDIR):

    # general setup
    job = QMin['IJOB']
    gsmult = QMin['multmap'][-job][0]
    restr = QMin['jobs'][job]['restr']
    charge = QMin['chargemap'][gsmult]

    # excited states to calculate
    states_to_do = QMin['states_to_do']
    for imult in range(len(states_to_do)):
        if imult + 1 not in QMin['multmap'][-job]:
            states_to_do[imult] = 0
    states_to_do[gsmult - 1] -= 1

    # do minimum number of states for gradient jobs
    if 'gradonly' in QMin:
        gradmult = QMin['gradmap'][0][0]
        gradstat = QMin['gradmap'][0][1]
        for imult in range(len(states_to_do)):
            if imult + 1 == gradmult:
                states_to_do[imult] = gradstat - (gradmult == gsmult)
            else:
                states_to_do[imult] = 0

    # number of states to calculate
    if restr:
        ncalc = max(states_to_do)
        sing = states_to_do[0] > 0
        trip = (len(states_to_do) >= 3 and states_to_do[2] > 0)
        if sing and not trip:
            onlysing = True
            onlytrip = False
        elif not sing and trip:
            onlysing = False
            onlytrip = True
        elif sing and trip:
            onlysing = False
            onlytrip = False
    else:
        ncalc = max(states_to_do)
        onlysing = False
        onlytrip = False

    # whether to do SOC
    sopert = False
    gscorr = False
    if 'soc' in QMin:
        if restr:
            nsing = QMin['states'][0]
            if len(QMin['states']) >= 3:
                ntrip = QMin['states'][2]
            else:
                ntrip = 0
            if nsing + ntrip >= 2 and ntrip >= 1:
                sopert = True
                if nsing >= 1:
                    gscorr = True

    # gradients
    if QMin['gradmap'] and 'AOoverlap' not in QMin:
        dograd = True
        egrad = ()
        for grad in QMin['gradmap']:
            if not (gsmult, 1) == grad:
                egrad = grad
        singgrad = []
        tripgrad = []
        for grad in QMin['gradmap']:
            if not (gsmult, 1) == grad:
                if grad[0] == gsmult:
                    singgrad.append(grad[1] - 1)
                if grad[0] == 3 and restr:
                    tripgrad.append(grad[1])
    else:
        dograd = False

    # construct the input string
    string = 'Task SinglePoint\n'  # TODO: do we need more task?

    # ---------------- SYSTEM BLOCK ------------------------
    string += 'SYSTEM\n'
    # geometry data
    string += '\tATOMS [Bohr]\n'  # Units in header
    for iatom, atom in enumerate(QMin['geo']):
        fragment = ''
        if 'AOoverlap' in QMin:
            if iatom >= QMin['natom']:
                fragment = 'f=f2'
            else:
                fragment = 'f=f1'
        label = atom[0]
        if QMin['template']['define_fragment']:
            for frag in QMin['template']['define_fragment']:
                if iatom in QMin['template']['define_fragment'][frag]:
                    label = frag
        string += '\t  %4s %16.9f %16.9f %16.9f  %s\n' % (label, atom[1], atom[2], atom[3], fragment)
    string += '\tEND\n'
    # charge, multiplicity, restricted
    string += '\tCHARGE %i\n' % charge
    if 'AOoverlap' in QMin:
        string += 'ALLOWCLOSEATOMS True\n'
    string += 'END\n\n'
    # ---------------- SYSTEM BLOCK ------------------------

    string += 'USESYMMETRY False\n'  # do not use symmetry

    if not restr:
        string += 'UNRESTRICTED Yes\n'
    string += '\n'

    # restart orbitals
    restart = False
    if 'master' in QMin:
        if job in QMin['initorbs']:
            restart = True
    elif 'grad' in QMin:
        restart = True
    if restart:
        string += 'EngineRestart %s\n\n' % os.path.join(WORKDIR, 'TAPE21.guess')

    if dograd:
        string += 'PROPERTIES\n  GRADIENTS  yes\nEND\n'

    # ---------------- ENGINE BLOCK ----------------------------
    string += 'ENGINE ADF\n\n'  # TODO variable engine -> DFTB, COSMO etc.

    string += 'SYMMETRY NoSym\n'  # do not use symmetry
    if QMin['template']['relativistic']:
        string += 'RELATIVITY\nLEVEL %s\nFORMALISM %s\nEND\n\n' % tuple(QMin['template']['relativistic'].split())
    if 'AOoverlap' in QMin:
        string += 'FRAGMENTS\n  f1 %s\n  f2 %s\nEND\n\n' % tuple(QMin['AOoverlap'])
    elif QMin['frags_there']:
        string += 'FRAGMENTS\n'
        for i in QMin['atom_frags']:
            string += '  %s t21.%s\n' % (i, i)
        string += 'END\n\n'
    else:
        string += 'BASIS\n  type %s\n  core None\n  createoutput False\n' % (QMin['template']['basis'])
        if QMin['template']['basis_path']:
            string += '  path %s\n' % (QMin['template']['basis_path'])
        if QMin['template']['basis_per_element']:
            string += 'peratomtype\n'
            for i in QMin['template']['basis_per_element']:
                string += '  symbol %s\nfile %s\n' % (i, QMin['template']['basis_per_element'][i])
            string += 'end\n'
        string += 'END\n\n'

    if 'AOoverlap' not in QMin:
        # chemistry
        string += 'XC\n  %s\n' % QMin['template']['functional']
        if QMin['template']['dispersion']:
            string += '  dispersion %s\n' % QMin['template']['dispersion']
        if QMin['template']['functional_xcfun'] or QMin['template']['fullkernel']:
            string += '  xcfun\n'
        string += 'END\n'
        if QMin['template']['totalenergy']:
            string += 'TOTALENERGY\n'
        string += '\n'

        # accuracy
        if 'beckegrid' in QMin['template']['grid']:  # TODO: beckegrid gives warning
            string += 'BECKEGRID\n  quality %s\n' % (QMin['template']['grid'].split()[1])
            if QMin['template']['grid_per_atom']:
                string += '  atomdepquality\n'
                for i in QMin['template']['grid_per_atom']:
                    string += '    %i %s\n' % (i, QMin['template']['grid_per_atom'][i])
                string += '  subend\n'
            string += 'END\n\n'
        elif 'integration' in QMin['template']['grid']:
            string += 'INTEGRATION\n  accint %s\n' % (QMin['template']['grid'].split()[1])
            string += 'END\n'
        if 'zlmfit' in QMin['template']['fit']:
            string += 'ZLMFIT\n  quality %s\n' % QMin['template']['fit'].split()[1]
            if QMin['template']['fit_per_atom']:
                string += '  atomdepquality\n'
                for i in QMin['template']['fit_per_atom']:
                    string += '    %i %s\n' % (i, QMin['template']['fit_per_atom'][i])
                string += '  subend\n'
            string += 'END\n'
        elif 'stofit' in QMin['template']['fit']:
            string += 'STOFIT\n'
        if QMin['template']['exactdensity']:
            string += 'EXACTDENSITY\n'
        if QMin['template']['rihartreefock']:
            string += 'RIHARTREEFOCK\n  useme True\n  quality %s\n' % QMin['template']['rihartreefock']
            if QMin['template']['rihf_per_atom']:
                string += '  atomdepquality\n'
                for i in QMin['template']['rihf_per_atom']:
                    string += '    %i %s\n' % (i, QMin['template']['rihf_per_atom'][i])
                string += '  subend\n'
            string += 'END\n'
        else:
            string += 'RIHARTREEFOCK\n  useme False\nEND\n'
        string += '\n'

        # excitations
        if ncalc > 0:
            string += 'EXCITATIONS\n'
            if onlysing:
                string += '  onlysing\n'
            if onlytrip:
                string += '  onlytrip\n'
            if not onlysing and not onlytrip and restr:
                string += '  lowest %i %i\n' % (states_to_do[0], states_to_do[2])
            else:
                string += '  lowest %i\n' % (ncalc)
            if QMin['template']['dvd_vectors'] > 0:
                string += '  vectors %i\n' % (QMin['template']['dvd_vectors'])
            string += '  tolerance %16.12f\n' % (QMin['template']['dvd_tolerance'])
            # string += '  residu %16.12f\n' % (QMin['template']['dvd_residu'])
            # if QMin['template']['dvd_mblocksmall']:
            #     string += '  iterations %i\n' % (max(200, 20 * ncalc))
            if DEBUG:
                string += '  nto\n'
            if QMin['template']['fullkernel']:
                string += 'FULLKERNEL\n'
            string += 'END\n'
            # if QMin['template']['dvd_mblocksmall']:
            #     string += 'MBLOCKSMALL\n'
            if not QMin['template']['no_tda']:
                string += '\nTDA\n'
            if QMin['template']['modifyexcitations'] > 0:
                string += 'MODIFYEXCITATION\n  useoccupied\n    A'
                for i in range(QMin['template']['modifyexcitations']):
                    string += ' %i' % (i + 1)
                string += '\n  subend\nEND\n'
            string += '\n'

        # spin-orbit coupling
        if sopert:
            string += 'SOPERT\n'
            if gscorr:
                string += 'GSCORR\n'
            string += 'END\n'
            string += 'PRINT SOMATRIX\n\n'

        # gradients
        if dograd and egrad:
            string += 'EXCITEDGO\n'
            if singgrad:
                string += '  SING_GRADS\n    A '
                string += ' '.join([str(i) for i in singgrad])
                string += '\n  END\n'
            if tripgrad:
                string += '  TRIP_GRADS\n    A '
                string += ' '.join([str(i) for i in tripgrad])
                string += '\n  END\n'
            if not (tripgrad or singgrad):
                string += '  state A %i\n' % (egrad[1] - (gsmult == egrad[0]))
                if restr:
                    if egrad[0] == 1:
                        string += '  singlet\n'
                    elif egrad[0] == 3:
                        string += '  triplet\n'
            string += '  output 4\n\n  cpks eps={0:.9f}\n'.format(QMin['template']['cpks_eps'])
            string += 'END\n'
        string += '\n'

        # COSMO
        if QMin['template']['cosmo']:
            string += 'SOLVATION\n  solv name=%s' % QMin['template']['cosmo']
            if QMin['template']['cosmo_neql'] >= 0.:
                string += ' neql=%6.3f' % QMin['template']['cosmo_neql']
            string += '\nEND\n\n'

    # options which are always used
    # MARK
    savefiles = ['info', 'TAPE15']
    if 'ion' in QMin or 'AOoverlap' in QMin:
        savefiles.append('TAPE15')
    string += 'SAVE %s\n' % (' '.join(savefiles))
    string += 'DEPENDENCY\n'  # checks for linear dependencies in basis set
    string += 'END\n'
    string += 'ALLOW POSHOMO\n'
    if DEBUG:
        string += 'PRINT TIMING\n'
    else:
        string += 'NOPRINT LOGFILE\n'
    string += 'SCF\n  iterations %i\nEND\n' % (QMin['template']['scf_iterations'])
    if QMin['template']['occupations']:
        string += 'OCCUPATIONS %s\n' % QMin['template']['occupations']
    if QMin['template']['linearscaling']:
        string += 'LINEARSCALING %i\n' % QMin['template']['linearscaling']
    string += '\n'

    # SHARCOVERLAP
    if 'AOoverlap' in QMin:
        string += 'SHARCOVERLAP True\n\n'

    string += 'ENDENGINE\n'
# -------------------- ENGINE BLOCK -------------------- #

    # TODO optional:
    # convergence schemes
    # nosharedarrays

    return string

# ======================================================================= #


def shorten_DIR(string):
    maxlen = 40
    front = 12
    if len(string) > maxlen:
        return string[0:front] + '...' + string[-(maxlen - 3 - front):]
    else:
        return string + ' ' * (maxlen - len(string))

# ======================================================================= #


def runAMS(WORKDIR, AMS, ncpu, savedir, strip=False):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = os.path.join(AMS, 'bin', 'ams') + ' '
    string += '-n %i < AMS.run' % (ncpu)
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    stdoutfile = open(os.path.join(WORKDIR, 'AMS.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'AMS.err'), 'w')
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(82)
    stdoutfile.close()
    stderrfile.close()
    if os.path.isfile(os.path.join(WORKDIR, 'ams.results/TAPE13')):
        runerror = 1
    stderr = readfile(os.path.join(WORKDIR, 'AMS.err'))
    for line in stderr:
        if 'error' in line.lower():
            sys.stdout.write('ERROR: \t%s\t"%s"\n' % (shorten_DIR(WORKDIR), line.strip()))
            runerror += 1
            break
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' %
                         (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    if DEBUG or runerror != 0:
        copydir = os.path.join(savedir, 'debug_AMS_stdout')
        if not os.path.isdir(copydir):
            mkdir(copydir)
        outfile = os.path.join(WORKDIR, 'AMS.out')
        tofile = os.path.join(copydir, "AMS_problems_%s.out" % (os.path.basename(WORKDIR)))
        shutil.copy(outfile, tofile)
        if runerror != 0:
            print('Error in %s! ' % WORKDIR)
        elif DEBUG:
            print('Debug mode.')
        print('Copied AMS output to %s' % (tofile))
    extractAMSresults(WORKDIR)
    os.chdir(prevdir)
    if strip and not DEBUG and runerror == 0:
        stripWORKDIR(WORKDIR)
    return runerror


# ======================================================================= #
def extractAMSresults(WORKDIR):
    for f in os.listdir(os.path.join(WORKDIR, 'ams.results')):
        frompath = os.path.join(WORKDIR, 'ams.results', f)
        if f == 'adf.rkf':
            f = 'TAPE21'
        topath = os.path.join(WORKDIR, f)
        shutil.move(frompath, topath)
    print("REMOVING", os.path.join(WORKDIR, 'ams.results'))
    os.rmdir(os.path.join(WORKDIR, 'ams.results/'))


# ======================================================================= #
def stripWORKDIR(WORKDIR):
    ls = os.listdir(WORKDIR)
    keep = ['AMS.run$', 'AMS.err$', 'AMS.out$', 'TAPE21', 'TAPE15', 'ams.log', 'ams.rkf']
    for ifile in ls:
        delete = True
        for k in keep:
            if re.search(k, ifile):
                delete = False
        if delete:
            rmfile = os.path.join(WORKDIR, ifile)
            if not DEBUG:
                os.remove(rmfile)

# ======================================================================= #


def moveOldFiles(QMin):
    # moves all relevant files in the savedir to old files (per job)
    if PRINT:
        print('>>>>>>>>>>>>> Moving old files')
    basenames = ['TAPE21']
    if 'nooverlap' not in QMin:
        basenames.append('mos')
    for job in QMin['joblist']:
        for base in basenames:
            fromfile = os.path.join(QMin['savedir'], '%s.%i' % (base, job))
            if not os.path.isfile(fromfile):
                print('File %s not found, cannot move to OLD!' % (fromfile))
                sys.exit(83)
            tofile = os.path.join(QMin['savedir'], '%s.%i.old' % (base, job))
            if PRINT:
                print(shorten_DIR(fromfile) + '   =>   ' + shorten_DIR(tofile))
            shutil.copy(fromfile, tofile)
    # moves all relevant files in the savedir to old files (per mult)
    basenames = []
    if 'nooverlap' not in QMin:
        basenames = ['dets']
    for job in itmult(QMin['states']):
        for base in basenames:
            fromfile = os.path.join(QMin['savedir'], '%s.%i' % (base, job))
            if not os.path.isfile(fromfile):
                print('File %s not found, cannot move to OLD!' % (fromfile))
                sys.exit(84)
            tofile = os.path.join(QMin['savedir'], '%s.%i.old' % (base, job))
            if PRINT:
                print(shorten_DIR(fromfile) + '   =>   ' + shorten_DIR(tofile))
            shutil.copy(fromfile, tofile)
    # also remove aoovl files if present
    delete = ['AO_overl', 'AO_overl.mixed']
    for f in delete:
        rmfile = os.path.join(QMin['savedir'], f)
        if os.path.isfile(rmfile):
            os.remove(rmfile)
            if PRINT:
                print('rm ' + rmfile)
    print()

# ======================================================================= #


def saveFiles(WORKDIR, QMin):

    # copy the TAPE21 from master directories
    job = QMin['IJOB']
    fromfile = os.path.join(WORKDIR, 'TAPE21')
    tofile = os.path.join(QMin['savedir'], 'TAPE21.%i' % (job))
    shutil.copy(fromfile, tofile)
    if PRINT:
        print(shorten_DIR(tofile))

    # copy atomic fragment files
    for i in QMin['atom_frags']:
        fromfile = os.path.join(WORKDIR, 't21.%s' % i)
        tofile = os.path.join(QMin['savedir'], 'frag.t21.%s' % (i))
        if os.path.isfile(tofile):
            continue
        if not os.path.isfile(fromfile):
            continue
        shutil.copy(fromfile, tofile)
        if PRINT:
            print(shorten_DIR(tofile))

    # if necessary, extract the MOs and write them to savedir
    if 'ion' in QMin or 'nooverlap' not in QMin:
        f = os.path.join(WORKDIR, 'TAPE21')
        string = get_MO_from_tape21(f, QMin)
        mofile = os.path.join(QMin['savedir'], 'mos.%i' % job)
        writefile(mofile, string)
        if PRINT:
            print(shorten_DIR(mofile))

    # if necessary, extract the TDDFT coefficients and write them to savedir
    if 'ion' in QMin or 'nooverlap' not in QMin:
        f = os.path.join(WORKDIR, 'TAPE21')
        strings = get_dets_from_tape21(f, QMin)
        for f in strings:
            writefile(f, strings[f])
            if PRINT:
                print(shorten_DIR(f))

# ======================================================================= #


def get_MO_from_tape21(filename, QMin):

    job = QMin['IJOB']
    restr = QMin['jobs'][job]['restr']

    # get all info from TAPE21
    f = KFFile(filename)

    NAO = int(f.read('Basis', 'naos'))
    npart = f.read('A', 'npart')
    NMO_A = int(f.read('A', 'nmo_A'))
    mocoef_A = f.read('A', 'Eigen-Bas_A')
    if not restr:
        NMO_B = int(f.read('A', 'nmo_B'))
        mocoef_B = f.read('A', 'Eigen-Bas_B')

    # prepare npart
    for i in range(len(npart)):
        npart[i] -= 1

    # build MO matrices
    MO_A = [[0. for iao in range(NAO)] for imo in range(NMO_A)]
    iao = 0
    imo = 0
    for i, el in enumerate(mocoef_A):
        iao1 = npart[iao]
        MO_A[imo][iao1] = el
        iao += 1
        if iao >= NAO:
            iao = 0
            imo += 1

    if not restr:
        MO_B = [[0. for iao in range(NAO)] for imo in range(NMO_B)]
        iao = 0
        imo = 0
        for i, el in enumerate(mocoef_B):
            iao1 = npart[iao]
            MO_B[imo][iao1] = el
            iao += 1
            if iao >= NAO:
                iao = 0
                imo += 1

    if restr:
        NMO = NMO_A - QMin['frozcore']
    else:
        NMO = NMO_A + NMO_B - 2 * QMin['frozcore']

    # make string
    string = '''2mocoef
header
 1
MO-coefficients from AMS
 1
 %i   %i
 a
mocoef
(*)
''' % (NAO, NMO)
    x = 0
    for imo, mo in enumerate(MO_A):
        if imo < QMin['frozcore']:
            continue
        for c in mo:
            if x >= 3:
                string += '\n'
                x = 0
            string += '% 6.12e ' % c
            x += 1
        if x > 0:
            string += '\n'
            x = 0
    if not restr:
        x = 0
        for imo, mo in enumerate(MO_B):
            if imo < QMin['frozcore']:
                continue
            for c in mo:
                if x >= 3:
                    string += '\n'
                    x = 0
                string += '% 6.12e ' % c
                x += 1
            if x > 0:
                string += '\n'
                x = 0
    string += 'orbocc\n(*)\n'
    x = 0
    for i in range(NMO):
        if x >= 3:
            string += '\n'
            x = 0
        string += '% 6.12e ' % (0.0)
        x += 1

    return string

# ======================================================================= #


def get_dets_from_tape21(filename, QMin):

    # get general infos
    job = QMin['IJOB']
    restr = QMin['jobs'][job]['restr']
    mults = QMin['jobs'][job]['mults']
    if restr:
        extrmults = ['S', 'T']
    else:
        extrmults = ['S']
    gsmult = QMin['multmap'][-job][0]
    nstates_to_extract = deepcopy(QMin['states'])
    for i in range(len(nstates_to_extract)):
        if not i + 1 in mults:
            nstates_to_extract[i] = 0
        elif i + 1 == gsmult:
            nstates_to_extract[i] -= 1

    # get all info from TAPE21
    f = KFFile(filename)

    lhybrid = f.read('General', 'lhybrid')
    occ_A = f.read('A', 'froc_A')
    occ_A = [int(i) for i in occ_A]
    if restr:
        nocc_A = sum(occ_A) // 2
        nvir_A = len(occ_A) - nocc_A
    else:
        nocc_A = sum(occ_A)
        nvir_A = len(occ_A) - nocc_A
        NMO_B = int(f.read('A', 'nmo_B'))
        occ_B = f.read('A', 'froc_B')
        occ_B = [int(i) for i in occ_B]
        nocc_B = sum(occ_B)
        nvir_B = len(occ_B) - nocc_B

    # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
    if restr:
        m = {0: 0, 2: 3}
        occ_A = [m[i] for i in occ_A]
    else:
        m = {0: 0, 1: 2}
        occ_B = [m[i] for i in occ_B]

    occ_A = tuple(occ_A)
    if not restr:
        occ_B = tuple(occ_B)

    # print occ_A
    # print nocc_A,nvir_A
    # if not restr:
        # print occ_B
        # print nocc_B,nvir_B

    # get eigenvectors
    eigenvectors = {}
    for imult, mult in enumerate(mults):
        eigenvectors[mult] = []
        if mult == gsmult:
            # add ground state
            if restr:
                key = tuple(occ_A[QMin['frozcore']:])
            else:
                key = tuple(occ_A[QMin['frozcore']:] + occ_B[QMin['frozcore']:])
            eigenvectors[mult].append({key: 1.0})
        for istate in range(nstates_to_extract[mult - 1]):
            section = 'Excitations S%s A' % extrmults[imult]
            key = 'eigenvector %i' % (istate + 1)
            try:
                eig = f.read(section, key)
            except AttributeError:
                print('No eigenvectors found in file %s!' % (filename))
                sys.exit(85)
            if lhybrid and QMin['template']['no_tda']:
                key = 'left eigenvector %i' % (istate + 1)
                eigl = f.read(section, key)
                for i in range(len(eig)):
                    eig[i] = (eig[i] + eigl[i]) / 2.
            # make dictionary
            dets = {}
            if restr:
                for iocc in range(nocc_A):
                    for ivirt in range(nvir_A):
                        index = iocc * nvir_A + ivirt
                        dets[(iocc, ivirt, 1)] = eig[index]
            else:
                for iocc in range(nocc_A):
                    for ivirt in range(nvir_A):
                        index = iocc * nvir_A + ivirt
                        dets[(iocc, ivirt, 1)] = eig[index]
                for iocc in range(nocc_B):
                    for ivirt in range(nvir_B):
                        index = iocc * nvir_B + ivirt + max(nvir_A, nvir_B) * max(nocc_A, nocc_B)
                        dets[(iocc, ivirt, 2)] = eig[index]
            # truncate vectors
            norm = 0.
            for k in sorted(dets, key=lambda x: dets[x]**2, reverse=True):
                if norm > QMin['wfthres']:
                    del dets[k]
                    continue
                norm += dets[k]**2
            # create strings and expand singlets
            dets2 = {}
            if restr:
                for iocc, ivirt, dummy in dets:
                    if mult == 1:
                        # alpha excitation
                        key = list(occ_A)
                        key[iocc] = 2
                        key[nocc_A + ivirt] = 1
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] / math.sqrt(2.)
                        # beta excitation
                        key[iocc] = 1
                        key[nocc_A + ivirt] = 2
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] / math.sqrt(2.)
                    elif mult == 3:
                        key = list(occ_A)
                        key[iocc] = 1
                        key[nocc_A + ivirt] = 1
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
            else:
                for iocc, ivirt, dummy in dets:
                    if dummy == 1:
                        key = list(occ_A + occ_B)
                        key[iocc] = 0
                        key[nocc_A + ivirt] = 1
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                    elif dummy == 2:
                        key = list(occ_A + occ_B)
                        key[nocc_A + nvir_A + iocc] = 0
                        key[nocc_A + nvir_A + nocc_B + ivirt] = 2
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
            # remove frozen core
            dets3 = {}
            for key in dets2:
                problem = False
                if restr:
                    if any([key[i] != 3 for i in range(QMin['frozcore'])]):
                        problem = True
                else:
                    if any([key[i] != 1 for i in range(QMin['frozcore'])]):
                        problem = True
                    if any([key[i] != 2 for i in range(nocc_A + nvir_A, nocc_A + nvir_A + QMin['frozcore'])]):
                        problem = True
                if problem:
                    print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                    continue
                    # sys.exit(86)
                if restr:
                    key2 = key[QMin['frozcore']:]
                else:
                    key2 = key[QMin['frozcore']:nocc_A + nvir_A] + key[nocc_A + nvir_A + QMin['frozcore']:]
                dets3[key2] = dets2[key]
            # append
            eigenvectors[mult].append(dets3)

    strings = {}
    for imult, mult in enumerate(mults):
        filename = os.path.join(QMin['savedir'], 'dets.%i' % mult)
        strings[filename] = format_ci_vectors(eigenvectors[mult])

    return strings

# ======================================================================= #


def format_ci_vectors(ci_vectors):

    # get nstates, norb and ndets
    alldets = set()
    for dets in ci_vectors:
        for key in dets:
            alldets.add(key)
    ndets = len(alldets)
    nstates = len(ci_vectors)
    norb = len(next(iter(alldets)))

    string = '%i %i %i\n' % (nstates, norb, ndets)
    for det in sorted(alldets, reverse=True):
        for o in det:
            if o == 0:
                string += 'e'
            elif o == 1:
                string += 'a'
            elif o == 2:
                string += 'b'
            elif o == 3:
                string += 'd'
        for istate in range(len(ci_vectors)):
            if det in ci_vectors[istate]:
                string += ' %11.7f ' % ci_vectors[istate][det]
            else:
                string += ' %11.7f ' % 0.
        string += '\n'
    return string

# ======================================================================= #


def saveAOmatrix(WORKDIR, QMin):
    filename = os.path.join(WORKDIR, 'TAPE15')
    NAO, Smat = get_smat(filename)

    string = '%i %i\n' % (NAO, NAO)
    for irow in range(NAO):
        for icol in range(NAO):
            string += '% .15e ' % (Smat[icol][irow])
        string += '\n'
    filename = os.path.join(QMin['savedir'], 'AO_overl')
    writefile(filename, string)
    if PRINT:
        print(shorten_DIR(filename))

# ======================================================================= #


def get_smat(filename):

    f = KFFile(filename)
    NAO = int(f.read('Basis', 'naos'))
    Smat = f.read('Matrices', 'Smat')

    # Smat is lower triangular matrix, len is NAO*(NAO+1)/2
    ao_ovl = makermatrix(NAO, NAO)
    x = 0
    y = 0
    for el in Smat:
        ao_ovl[x][y] = el
        ao_ovl[y][x] = el
        x += 1
        if x > y:
            x = 0
            y += 1
    return NAO, ao_ovl

# ======================================================================= #


def mkdir(DIR):
    # mkdir the DIR, or clean it if it exists
    if os.path.exists(DIR):
        if os.path.isfile(DIR):
            print('%s exists and is a file!' % (DIR))
            sys.exit(87)
        elif os.path.isdir(DIR):
            if DEBUG:
                print('Remake\t%s' % DIR)
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            if DEBUG:
                print('Make\t%s' % DIR)
            os.makedirs(DIR)
        except OSError:
            print('Can not create %s\n' % (DIR))
            sys.exit(88)

# ======================================================================= #


def link(PATH, NAME, crucial=True, force=True):
    # do not create broken links
    if not os.path.exists(PATH) and crucial:
        print('Source %s does not exist, cannot create link!' % (PATH))
        sys.exit(89)
    if os.path.islink(NAME):
        if not os.path.exists(NAME):
            # NAME is a broken link, remove it so that a new link can be made
            os.remove(NAME)
        else:
            # NAME is a symlink pointing to a valid file
            if force:
                # remove the link if forced to
                os.remove(NAME)
            else:
                print('%s exists, cannot create a link of the same name!' % (NAME))
                if crucial:
                    sys.exit(90)
                else:
                    return
    elif os.path.exists(NAME):
        # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
        print('%s exists, cannot create a link of the same name!' % (NAME))
        if crucial:
            sys.exit(91)
        else:
            return
    os.symlink(PATH, NAME)

# =============================================================================================== #
# =============================================================================================== #
# =======================================  TheoDORE ============================================= #
# =============================================================================================== #
# =============================================================================================== #


def run_theodore(QMin, errorcodes):

    if 'theodore' in QMin:
        print('>>>>>>>>>>>>> Starting the TheoDORE job execution')

        for ijob in QMin['jobs']:
            if not QMin['jobs'][ijob]['restr']:
                if DEBUG:
                    print('Skipping Job %s because it is unrestricted.' % (ijob))
                continue
            WORKDIR = os.path.join(QMin['scratchdir'], 'master_%i' % ijob)
            setupWORKDIR_TH(WORKDIR, QMin)
            os.environ
            errorcodes['theodore_%i' % ijob] = runTHEODORE(WORKDIR, QMin['theodir'])

        # Error code handling
        j = 0
        string = 'Error Codes:\n'
        for i in errorcodes:
            if 'theodore' in i:
                string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
                j += 1
                if j == 4:
                    j = 0
                    string += '\n'
        print(string)
        if any((i != 0 for i in list(errorcodes.values()))):
            print('Some subprocesses did not finish successfully!')
            sys.exit(92)

        print('')

    return errorcodes

# ======================================================================= #


def setupWORKDIR_TH(WORKDIR, QMin):
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

    # write dens_ana.in
    inputstring = '''rtype='ADF'
rfile='TAPE21'
jmol_orbitals=False
molden_orbitals=False
Om_formula=2
eh_pop=1
comp_ntos=True
print_OmFrag=True
output_file='tden_summ.txt'
prop_list=%s
at_lists=%s
''' % (str(QMin['template']['theodore_prop']), str(QMin['template']['theodore_fragment']))

    filename = os.path.join(WORKDIR, 'dens_ana.in')
    writefile(filename, inputstring)
    if DEBUG:
        print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        print(inputstring)
        print('TheoDORE input written to: %s' % (filename))
        print('====================================================================')

    return

# ======================================================================= #


def runTHEODORE(WORKDIR, THEODIR):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = os.path.join(THEODIR, 'bin', 'theodore analyze_tden')
    stdoutfile = open(os.path.join(WORKDIR, 'theodore.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'theodore.err'), 'w')
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile, env=os.environ)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(93)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' %
                         (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)
    return runerror

# =============================================================================================== #
# =============================================================================================== #
# =======================================  Dyson and overlap calcs ============================== #
# =============================================================================================== #
# =============================================================================================== #


def run_wfoverlap(QMin, errorcodes):

    print('>>>>>>>>>>>>> Starting the WFOVERLAP job execution')

    # do Dyson calculations
    if 'ion' in QMin:
        for ionpair in QMin['ionmap']:
            WORKDIR = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i' % ionpair)
            files = {'aoovl': 'AO_overl',
                     'det.a': 'dets.%i' % ionpair[0],
                     'det.b': 'dets.%i' % ionpair[2],
                     'mo.a': 'mos.%i' % ionpair[1],
                     'mo.b': 'mos.%i' % ionpair[3]}
            setupWORKDIR_WF(WORKDIR, QMin, files)
            errorcodes['Dyson_%i_%i_%i_%i' % ionpair] = runWFOVERLAP(
                WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])

    # do overlap calculations
    if 'overlap' in QMin:
        get_Double_AOovl(QMin)
        for m in itmult(QMin['states']):
            job = QMin['multmap'][m]
            WORKDIR = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i' % (m, job))
            files = {'aoovl': 'AO_overl.mixed',
                     'det.a': 'dets.%i.old' % m,
                     'det.b': 'dets.%i' % m,
                     'mo.a': 'mos.%i.old' % job,
                     'mo.b': 'mos.%i' % job}
            setupWORKDIR_WF(WORKDIR, QMin, files)
            errorcodes['WFOVL_%i_%i' % (m, job)] = runWFOVERLAP(
                WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])

    # Error code handling
    j = 0
    string = 'Error Codes:\n'
    for i in errorcodes:
        if 'Dyson' in i or 'WFOVL' in i:
            string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
            j += 1
            if j == 4:
                j = 0
                string += '\n'
    print(string)
    if any((i != 0 for i in list(errorcodes.values()))):
        print('Some subprocesses did not finish successfully!')
        sys.exit(94)

    print('')

    return errorcodes

# ======================================================================= #


def setupWORKDIR_WF(WORKDIR, QMin, files):
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

    # setup the directory
    mkdir(WORKDIR)

    # write wfovl.inp
    inputstring = '''mix_aoovl=aoovl
a_mo=mo.a
b_mo=mo.b
a_det=det.a
b_det=det.b
a_mo_read=0
b_mo_read=0
ao_read=0
'''
    if 'ion' in QMin:
        if QMin['ndocc'] > 0:
            inputstring += 'ndocc=%i\n' % (QMin['ndocc'])
    if QMin['ncpu'] >= 8:
        inputstring += 'force_direct_dets\n'
    filename = os.path.join(WORKDIR, 'wfovl.inp')
    writefile(filename, inputstring)
    if DEBUG:
        print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        print(inputstring)
        print('wfoverlap input written to: %s' % (filename))
        print('====================================================================')

    # link input files from save
    linkfiles = ['aoovl', 'det.a', 'det.b', 'mo.a', 'mo.b']
    for f in linkfiles:
        fromfile = os.path.join(QMin['savedir'], files[f])
        tofile = os.path.join(WORKDIR, f)
        link(fromfile, tofile)

    return

# ======================================================================= #


def runWFOVERLAP(WORKDIR, WFOVERLAP, memory=100, ncpu=1):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = WFOVERLAP + ' -m %i' % (memory) + ' -f wfovl.inp'
    stdoutfile = open(os.path.join(WORKDIR, 'wfovl.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'wfovl.err'), 'w')
    os.environ['OMP_NUM_THREADS'] = str(ncpu)
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(95)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' %
                         (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)
    return runerror


# ======================================================================= #
def get_Double_AOovl(QMin):

    # get old geometry
    job = QMin['joblist'][0]
    filename1 = os.path.join(QMin['savedir'], 'TAPE21.%i.old' % (job))
    oldgeo = get_geometry(filename1)
    filename2 = os.path.join(QMin['savedir'], 'TAPE21.%i' % (job))
    newgeo = get_geometry(filename2)
    if job != 1:
        # make file1 restricted
        fromfile = filename1
        filename1 = os.path.join(QMin['savedir'], 'TAPE21.%i.old.AO' % (job))
        shutil.copy(fromfile, filename1)
        f = KFFile(filename1)
        f.write('General', 'nspin', 1)
        f.close()
        # make file2 restricted
        fromfile = filename2
        filename2 = os.path.join(QMin['savedir'], 'TAPE21.%i.AO' % (job))
        shutil.copy(fromfile, filename2)
        f = KFFile(filename2)
        f.write('General', 'nspin', 1)
        f.close()

    # apply shift
    shift = 1e-5
    for iatom in range(len(oldgeo)):
        for ixyz in range(3):
            oldgeo[iatom][1 + ixyz] += shift

    # build QMin
    QMin1 = deepcopy(QMin)
    QMin1['geo'] = oldgeo + newgeo
    QMin1['AOoverlap'] = [filename1, filename2]
    QMin1['IJOB'] = job
    QMin1['natom'] = len(newgeo)
    remove = ['nacdr', 'grad', 'h', 'soc', 'dm', 'overlap', 'ion']
    for r in remove:
        QMin1 = removekey(QMin1, r)

    # run the calculation
    WORKDIR = os.path.join(QMin['scratchdir'], 'AOoverlap')
    err = run_calc(WORKDIR, QMin1)

    # remove fake TAPE21 files
    if job != 1:
        filename1 = os.path.join(QMin['savedir'], 'TAPE21.%i.old.AO' % (job))
        os.remove(filename1)
        filename2 = os.path.join(QMin['savedir'], 'TAPE21.%i.AO' % (job))
        os.remove(filename2)

    # get output
    filename = os.path.join(WORKDIR, 'TAPE15')
    NAO, Smat = get_smat(filename)

    # Smat is now full matrix NAO*NAO
    # we want the lower left quarter, but transposed
    string = '%i %i\n' % (NAO // 2, NAO // 2)
    for irow in range(NAO // 2, NAO):
        for icol in range(0, NAO // 2):
            string += '% .15e ' % (Smat[icol][irow])          # note the exchanged indices => transposition
        string += '\n'
    filename = os.path.join(QMin['savedir'], 'AO_overl.mixed')
    writefile(filename, string)

    return

# ======================================================================= #


def get_geometry(t21file):

    f = KFFile(t21file)
    geom = f.read('Geometry', 'xyz InputOrder')
    atomtype = f.read('Geometry', 'atomtype').split()
    atomindex = f.read('Geometry', 'fragment and atomtype index')
    atomorder = f.read('Geometry', 'atom order index')
    natom = int(f.read('Geometry', 'nr of atoms'))

    geometry = []
    for iatom in range(natom):
        index = atomorder[iatom] - 1
        atype = atomindex[natom + index] - 1
        el = atomtype[atype].strip()
        geometry.append([el,
                         geom[3 * iatom + 0],
                         geom[3 * iatom + 1],
                         geom[3 * iatom + 2]])
    return geometry


# =============================================================================================== #
# =============================================================================================== #
# ========================================= AMS output parsing ================================== #
# =============================================================================================== #
# =============================================================================================== #

def getQMout(QMin):

    if PRINT:
        print('>>>>>>>>>>>>> Reading output files')
    starttime = datetime.datetime.now()

    QMout = {}
    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    joblist = QMin['joblist']

    # Hamiltonian
    if 'h' in QMin or 'soc' in QMin:
        # make Hamiltonian
        if 'h' not in QMout:
            QMout['h'] = makecmatrix(nmstates, nmstates)
        # go through all jobs
        for job in joblist:
            # first get energies from TAPE21
            t21file = os.path.join(QMin['scratchdir'], 'master_%i/TAPE21' % (job))
            energies = getenergy(t21file, job, QMin)
            # also get SO matrix and mapping
            if 'soc' in QMin and QMin['jobs'][job]['restr']:
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/AMS.out' % (job))
                submatrix, invstatemap = getsocm(outfile, t21file, job, QMin)
            mults = QMin['multmap'][-job]
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if m1 not in mults or m2 not in mults:
                        continue
                    if i == j:
                        QMout['h'][i][j] = energies[(m1, s1)]
                    elif 'soc' in QMin and QMin['jobs'][job]['restr']:
                        if m1 == m2 == 1:
                            continue
                        x = invstatemap[(m1, s1, ms1)]
                        y = invstatemap[(m2, s2, ms2)]
                        QMout['h'][i][j] = submatrix[x - 1][y - 1]

    # Dipole Moments
    if 'dm' in QMin:
        # make matrix
        if 'dm' not in QMout:
            QMout['dm'] = [makecmatrix(nmstates, nmstates) for i in range(3)]
        # go through all jobs
        for job in joblist:
            t21file = os.path.join(QMin['scratchdir'], 'master_%i/TAPE21' % (job))
            dipoles = getdm(t21file, job, QMin)
            mults = QMin['multmap'][-job]
            for i in range(nmstates):
                m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                if m1 not in QMin['jobs'][job]['mults']:
                    continue
                for j in range(nmstates):
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if m2 not in QMin['jobs'][job]['mults']:
                        continue
                    if i == j and (m1, s1) in QMin['gradmap']:
                        path, isgs = QMin['jobgrad'][(m1, s1)]
                        if not isgs:
                            outfile = os.path.join(QMin['scratchdir'], path, 'AMS.out')
                            if QMin['jobs'][job]['restr'] and m1 == 3:
                                multstring = '(Singlet-Triplet)'
                                state = s1
                            else:
                                multstring = ''
                                state = s1 - 1
                            edm = getedm_multi(outfile, state, multstring)
                            for ixyz in range(3):
                                QMout['dm'][ixyz][i][j] = edm[ixyz]
                    if not m1 == m2 == mults[0] or not ms1 == ms2:
                        continue
                    if s1 == 1:
                        for ixyz in range(3):
                            QMout['dm'][ixyz][i][j] = dipoles[(m2, s2)][ixyz]
                    elif s2 == 1:
                        for ixyz in range(3):
                            QMout['dm'][ixyz][i][j] = dipoles[(m1, s1)][ixyz]

    # Gradients
    if 'grad' in QMin:
        if 'grad' not in QMout:
            QMout['grad'] = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
        for grad in QMin['gradmap']:
            path, isgs = QMin['jobgrad'][grad]
            t21file = os.path.join(QMin['scratchdir'], path, 'TAPE21')
            outfile = os.path.join(QMin['scratchdir'], path, 'AMS.out')
            if QMin['AMSversion'] >= (2017, 208):
                g = getgrad_fromTAPE21(t21file, outfile, grad[0], grad[1], QMin)
            else:
                g = getgrad(outfile, isgs, QMin)
            for istate in QMin['statemap']:
                state = QMin['statemap'][istate]
                if (state[0], state[1]) == grad:
                    QMout['grad'][istate - 1] = g
        if QMin['neglected_gradient'] != 'zero':
            for i in range(nmstates):
                m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                if (m1, s1) not in QMin['gradmap']:
                    if QMin['neglected_gradient'] == 'gs':
                        j = QMin['gsmap'][i + 1] - 1
                    elif QMin['neglected_gradient'] == 'closest':
                        e1 = QMout['h'][i][i]
                        de = 999.
                        for grad in QMin['gradmap']:
                            for k in range(nmstates):
                                m2, s2, ms2 = tuple(QMin['statemap'][k + 1])
                                if grad == (m2, s2):
                                    break
                            e2 = QMout['h'][k][k]
                            if de > abs(e1 - e2):
                                de = abs(e1 - e2)
                                j = k
                    QMout['grad'][i] = QMout['grad'][j]

    # Regular Overlaps
    if 'overlap' in QMin:
        if 'overlap' not in QMout:
            QMout['overlap'] = makecmatrix(nmstates, nmstates)
        for mult in itmult(QMin['states']):
            job = QMin['multmap'][mult]
            outfile = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
            out = readfile(outfile)
            if PRINT:
                print('Overlaps: ' + shorten_DIR(outfile))
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if not m1 == m2 == mult:
                        continue
                    if not ms1 == ms2:
                        continue
                    QMout['overlap'][i][j] = getsmate(out, s1, s2)

    # Phases from overlaps
    if 'phases' in QMin:
        if 'phases' not in QMout:
            QMout['phases'] = [complex(1., 0.) for i in range(nmstates)]
        if 'overlap' in QMout:
            for i in range(nmstates):
                if QMout['overlap'][i][i].real < 0.:
                    QMout['phases'][i] = complex(-1., 0.)

    # Dyson norms
    if 'ion' in QMin:
        if 'prop' not in QMout:
            QMout['prop'] = makecmatrix(nmstates, nmstates)
        for ion in QMin['ionmap']:
            outfile = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
            out = readfile(outfile)
            if PRINT:
                print('Dyson:    ' + shorten_DIR(outfile))
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if not (ion[0], ion[2]) == (m1, m2) and not (ion[0], ion[2]) == (m2, m1):
                        continue
                    if not abs(ms1 - ms2) == 0.5:
                        continue
                    # switch multiplicities such that m1 is smaller mult
                    if m1 > m2:
                        s1, s2 = s2, s1
                        m1, m2 = m2, m1
                        ms1, ms2 = ms2, ms1
                    # compute M_S overlap factor
                    if ms1 < ms2:
                        factor = (ms1 + 1. + (m1 - 1.) / 2.) / m1
                    else:
                        factor = (-ms1 + 1. + (m1 - 1.) / 2.) / m1
                    QMout['prop'][i][j] = getDyson(out, s1, s2) * factor

    # TheoDORE
    if 'theodore' in QMin:
        if 'theodore' not in QMout:
            QMout['theodore'] = makecmatrix(QMin['template']['theodore_n'], nmstates)
        for job in joblist:
            if not QMin['jobs'][job]['restr']:
                continue
            sumfile = os.path.join(QMin['scratchdir'], 'master_%i/tden_summ.txt' % job)
            omffile = os.path.join(QMin['scratchdir'], 'master_%i/OmFrag.txt' % job)
            props = get_theodore(sumfile, omffile, QMin)
            for i in range(nmstates):
                m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                if (m1, s1) in props:
                    for j in range(QMin['template']['theodore_n']):
                        QMout['theodore'][i][j] = props[(m1, s1)][j]

    endtime = datetime.datetime.now()
    if PRINT:
        print("Readout Runtime: %s" % (endtime - starttime))

    if DEBUG:
        copydir = os.path.join(QMin['savedir'], 'debug_AMS_stdout')
        if not os.path.isdir(copydir):
            mkdir(copydir)
        for job in joblist:
            outfile = os.path.join(QMin['scratchdir'], 'master_%i/AMS.out' % (job))
            shutil.copy(outfile, os.path.join(copydir, "AMS_%i.out" % job))
            if QMin['jobs'][job]['restr'] and 'theodore' in QMin:
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/tden_summ.txt' % job)
                shutil.copy(outfile, os.path.join(copydir, 'THEO_%i.out' % (job)))
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/OmFrag.txt' % job)
                shutil.copy(outfile, os.path.join(copydir, 'THEO_OMF_%i.out' % (job)))
        if 'grad' in QMin:
            for grad in QMin['gradmap']:
                path, isgs = QMin['jobgrad'][grad]
                outfile = os.path.join(QMin['scratchdir'], path, 'AMS.out')
                shutil.copy(outfile, os.path.join(copydir, "AMS_GRAD_%i_%i.out" % grad))
        if 'overlap' in QMin:
            for mult in itmult(QMin['states']):
                job = QMin['multmap'][mult]
                outfile = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
                shutil.copy(outfile, os.path.join(copydir, 'WFOVL_%i_%i.out' % (mult, job)))
        if 'ion' in QMin:
            for ion in QMin['ionmap']:
                outfile = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
                shutil.copy(outfile, os.path.join(copydir, 'Dyson_%i_%i_%i_%i.out' % ion))

    return QMout

# ======================================================================= #


def getenergy(t21file, ijob, QMin):

    # open TAPE21 file
    f = KFFile(t21file)
    if PRINT:
        print('Energy:   ' + shorten_DIR(t21file))

    # find ground state energy for this job
    if QMin['template']['totalenergy']:
        gsenergy = float(f.read('Total Energy', 'Total energy'))
    else:
        gsenergy = float(f.read('Energy', 'Bond Energy'))

    # figure out the excited state settings
    mults = QMin['jobs'][ijob]['mults']
    restr = QMin['jobs'][ijob]['restr']
    gsmult = mults[0]
    estates_to_extract = deepcopy(QMin['states'])
    estates_to_extract[gsmult - 1] -= 1
    for imult in range(len(estates_to_extract)):
        if not imult + 1 in mults:
            estates_to_extract[imult] = 0

    # extract excitation energies
    # loop also works if no energies should be extracted
    energies = {(gsmult, 1): gsenergy}
    for imult in mults:
        if restr and imult == 3:
            s = 'ST'
        else:
            s = 'SS'
        nstates = estates_to_extract[imult - 1]
        if nstates > 0:
            e = f.read('Excitations %s A' % s, 'excenergies')
            if isinstance(e, float):
                e = [e]
            for istate in range(nstates):
                energies[(imult, istate + 1 + (gsmult == imult))] = float(e[istate]) + gsenergy

    return energies

# ======================================================================= #


def getsocm(outfile, t21file, ijob, QMin):

    # read the AMS standard out into memory
    out = readfile(outfile)
    if PRINT:
        print('SOC:      ' + shorten_DIR(outfile))

    # open TAPE21
    f = KFFile(t21file)
    if PRINT:
        print('SOC:      ' + shorten_DIR(t21file))

    # get number of excitations from TAPE21
    if QMin['AMSversion'] >= (2017, 208):
        nrS = int(f.read('Excitations SS A', 'nr of excenergies'))
        nrT = int(f.read('Excitations ST A', 'nr of excenergies'))
    else:
        nrS = int(f.read('All excitations', 'nr excitations'))
        nrT = nrS
    nrexci = nrS + 3 * nrT

    # get GSCORR variable
    GSCORR = False
    for line in out:
        if 'gscorr' in line.lower():
            GSCORR = True
            break
    if GSCORR:
        nrexci += 1

    # read SOC matrix from TAPE21
    real_tri = f.read('Excitations SO A', 'SOmat-R')
    imag_tri = f.read('Excitations SO A', 'SOmat-I')
    real = [[0 + 0j for i in range(nrexci)] for j in range(nrexci)]
    x = 0
    y = 0
    for i in range(len(real_tri)):
        if abs(real_tri[i]) < 1e-15:
            real_tri[i] = 0.
        if abs(imag_tri[i]) < 1e-15:
            imag_tri[i] = 0.
        real[x][y] = real_tri[i] + (0 + 1j) * imag_tri[i]
        real[y][x] = real_tri[i] + (0 - 1j) * imag_tri[i]
        x += 1
        if x > y:
            y += 1
            x = 0

    # make statemap for the state ordering of the SO matrix
    inv_statemap = {}
    if GSCORR:
        inv_statemap[(1, 1, 0.0)] = nrexci
    i = 0
    for x in range(nrS):
        i += 1
        inv_statemap[(1, x + 2, 0.0)] = i
    for x in range(nrT):
        i += 3
        inv_statemap[(3, x + 1, 0.0)] = i - 2
        inv_statemap[(3, x + 1, +1.0)] = i - 1
        inv_statemap[(3, x + 1, -1.0)] = i - 0

    return real, inv_statemap

# ======================================================================= #


def readSOC(out, string, nrexci):

    # find starting string
    iline = -1
    while True:
        iline += 1
        line = out[iline]
        if string in line:
            break

    # read
    matrix = [[0. for i in range(nrexci)] for j in range(nrexci)]
    for x in range(nrexci):
        for y in range(nrexci):
            if x > y:
                x1, y1 = y, x
            else:
                x1, y1 = x, y
            block = x1 // 4
            xoffset = x1 % 4 + 1
            yoffset = block * (3 + nrexci) + 4 + y1 - block * (block + 1) * 2
            matrix[x][y] = float(out[iline + yoffset].split()[xoffset])
            if abs(matrix[x][y]) < 1e-15:
                matrix[x][y] = 0.

    return matrix

# ======================================================================= #


def getdm(t21file, ijob, QMin):

    # open TAPE21 file
    f = KFFile(t21file)
    if PRINT:
        print('Dipoles:  ' + shorten_DIR(t21file))

    # find ground state energy for this job
    gsdm = f.read('Properties', 'Dipole')

    # figure out the excited state settings
    mults = QMin['jobs'][ijob]['mults']
    gsmult = mults[0]
    estates_to_extract = deepcopy(QMin['states'])
    estates_to_extract[gsmult - 1] -= 1
    for imult in range(len(estates_to_extract)):
        if not imult + 1 in mults:
            estates_to_extract[imult] = 0

    # extract transition dipoles
    dipoles = {(gsmult, 1): gsdm}
    imult = mults[0]
    s = 'SS'
    nstates = estates_to_extract[imult - 1]
    if nstates > 0:
        e = f.read('Excitations %s A' % s, 'transition dipole moments')
        for istate in range(nstates):
            dipoles[(imult, istate + 1 + (gsmult == imult))] = e[3 * istate:3 * istate + 3]

    return dipoles

# ======================================================================= #


def getedm(outfile):

    out = readfile(outfile)
    if PRINT:
        print('Dipoles:  ' + shorten_DIR(outfile))
    for line in out:
        if 'Excited state dipole moment (Debye) =' in line:
            s = line.split()
            dm = [float(i) * D2au for i in s[6:]]
            return dm

# ======================================================================= #


def getedm_multi(outfile, state, multstring):

    out = readfile(outfile)
    if PRINT:
        print('Dipoles:  ' + shorten_DIR(outfile))
    active = False
    for line in out:
        if 'Excited state gradient for:' in line:
            s = line.split()
            if '%iA' % (state) == s[4] and multstring in line:
                active = True
            else:
                active = False
        if 'Excited state dipole moment (Debye) =' in line and active:
            s = line.split()
            dm = [float(i) * D2au for i in s[6:]]
            return dm
    print("Could not find dipole moment for state %i in file %s" % (state, outfile))
    sys.exit(94)

# ======================================================================= #


def getgrad(outfile, isgs, QMin):

    # read file and check if ego is active
    out = readfile(outfile)
    if PRINT:
        print('Gradient: ' + shorten_DIR(outfile))
    ego = False
    for line in out:
        if 'EXCITEDGO' in line:
            ego = True
            break

    # get number of atoms
    natom = QMin['natom']
    nqmatom = natom

    # initialize
    g = [[0. for i in range(3)] for j in range(natom)]

    # get QM gradient
    if isgs and ego:
        string = 'Ground state gradients:'
        shift = 5
    else:
        string = 'Energy gradients wrt nuclear displacements'
        shift = 6
    for iline, line in enumerate(out):
        if string in line:
            for iatom in range(nqmatom):
                s = out[iline + shift + iatom].split()
                for i in range(3):
                    g[iatom][i] = float(s[2 + i]) * au2a

    return g

# ======================================================================= #


def getgrad_fromTAPE21(t21file, outfile, mult, state, QMin):

    # read file and check if ego is active
    out = readfile(outfile)
    if PRINT:
        print('Gradient: ' + shorten_DIR(outfile))

    # open TAPE21 file
    f = KFFile(t21file)
    if PRINT:
        print('Gradient: ' + shorten_DIR(t21file))

    # check if ego is active
    try:
        inp = f.read('General', 'Input')
    except KeyError:
        inp = f.read('General', 'user input')
    ego = False
    for line in inp:
        if 'excitedgo' in line.lower():
            ego = True
            break

    # get number of atoms
    natom = QMin['natom']
    nqmatom = natom

    # get multiplicity info
    job = QMin['multmap'][mult]
    gsmult = QMin['jobs'][job]['mults'][0]
    if (gsmult, 1) == (mult, state):
        string = 'GeoOpt'
        substring = 'Gradients_CART'
    elif gsmult == mult:
        state -= 1
        string = 'Excitations SS A'
        substring = 'Gradients_CART %i' % (state)
    else:
        string = 'Excitations ST A'
        substring = 'Gradients_CART %i' % (state)

    # get from TAPE21
    g1 = f.read(string, substring)
    atomorder = f.read('Geometry', 'atom order index')

    # formatting
    g = [[0. for i in range(3)] for j in range(natom)]
    iatom = 0
    ixyz = 0
    for el in g1:
        iindex = atomorder[nqmatom + iatom] - 1
        g[iindex][ixyz] = el
        ixyz += 1
        if ixyz >= 3:
            iatom += 1
            ixyz = 0

    return g


# ======================================================================= #


def getsmate(out, s1, s2):
    ilines = -1
    while True:
        ilines += 1
        if ilines == len(out):
            print('Overlap of states %i - %i not found!' % (s1, s2))
            sys.exit(96)
        if 'Overlap matrix <PsiA_i|PsiB_j>' in out[ilines]:
            break
    ilines += 1 + s1
    f = out[ilines].split()
    return float(f[s2 + 1])

# ======================================================================= #


def getDyson(out, s1, s2):
    ilines = -1
    while True:
        ilines += 1
        if ilines == len(out):
            print('Dyson norm of states %i - %i not found!' % (s1, s2))
            sys.exit(97)
        if 'Dyson norm matrix <PsiA_i|PsiB_j>' in out[ilines]:
            break
    ilines += 1 + s1
    f = out[ilines].split()
    return float(f[s2 + 1])

# ======================================================================= #


def get_theodore(sumfile, omffile, QMin):
    out = readfile(sumfile)
    if PRINT:
        print('TheoDORE: ' + shorten_DIR(sumfile))
    props = {}
    for line in out[2:]:
        s = line.replace('(', ' ').replace(')', ' ').split()
        if len(s) == 0:
            continue
        n = int(s[0])
        m = int(s[1])
        props[(m, n + (m == 1))] = [theo_float(i) for i in s[5:]]

    out = readfile(omffile)
    if PRINT:
        print('TheoDORE: ' + shorten_DIR(omffile))
    for line in out[1:]:
        s = line.replace('(', ' ').replace(')', ' ').split()
        if len(s) == 0:
            continue
        n = int(s[0])
        m = int(s[1])
        props[(m, n + (m == 1))].extend([theo_float(i) for i in s[4:]])

    return props

# ======================================================================= #


def theo_float(string):
    try:
        s = float(string)
    except ValueError:
        s = 0.
    return s

# =============================================================================================== #
# =============================================================================================== #
# ========================================= Miscellaneous ======================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def cleandir(directory):
    for data in os.listdir(directory):
        path = directory + '/' + data
        if os.path.isfile(path) or os.path.islink(path):
            if DEBUG:
                print('rm %s' % (path))
            try:
                os.remove(path)
            except OSError:
                print('Could not remove file from directory: %s' % (path))
        else:
            if DEBUG:
                print('')
            cleandir(path)
            os.rmdir(path)
            if DEBUG:
                print('rm %s' % (path))
    if PRINT:
        print('===> Cleaning up directory %s' % (directory))

# ======================================================================= #


def backupdata(backupdir, QMin):
    # save all files in savedir, except which have 'old' in their name
    ls = os.listdir(QMin['savedir'])
    for f in ls:
        ff = QMin['savedir'] + '/' + f
        if os.path.isfile(ff) and 'old' not in ff:
            step = int(QMin['step'][0])
            fdest = backupdir + '/' + f + '.stp' + str(step)
            shutil.copy(ff, fdest)

# =============================================================================================== #
# =============================================================================================== #
# ========================================= Main ================================================ #
# =============================================================================================== #
# =============================================================================================== #


def main():

    # Retrieve PRINT and DEBUG
    try:
        envPRINT = os.getenv('SH2AMS_PRINT')
        if envPRINT and envPRINT.lower() == 'false':
            global PRINT
            PRINT = False
        envDEBUG = os.getenv('SH2AMS_DEBUG')
        if envDEBUG and envDEBUG.lower() == 'true':
            global DEBUG
            DEBUG = True
    except ValueError:
        print('PRINT or DEBUG environment variables do not evaluate to numerical values!')
        sys.exit(98)

    # Process Command line arguments
    if len(sys.argv) != 2:
        print('Usage:\n./SHARC_AMS.py <QMin>\n')
        print('version:', version)
        print('date:', versiondate)
        print('changelog:\n', changelogstring)
        sys.exit(99)
    QMinfilename = sys.argv[1]

    # Print header
    printheader()

    # Read QMinfile
    QMin = readQMin(QMinfilename)
    # get the job schedule
    QMin, schedule = generate_joblist(QMin)
    printQMin(QMin)
    if DEBUG:
        pprint.pprint(schedule, depth=1)

    # run all the AMS jobs
    errorcodes = runjobs(schedule, QMin)

    # do all necessary overlap and Dyson calculations
    errorcodes = run_wfoverlap(QMin, errorcodes)

    # do all necessary Theodore calculations
    errorcodes = run_theodore(QMin, errorcodes)

    # read all the output files
    QMout = getQMout(QMin)
    if PRINT or DEBUG:
        printQMout(QMin, QMout)

    # backup data if requested
    if 'backup' in QMin:
        backupdata(QMin['backup'], QMin)

    # Measure time
    runtime = measuretime()
    QMout['runtime'] = runtime

    # Write QMout
    writeQMout(QMin, QMout, QMinfilename)

    # Remove Scratchfiles from SCRATCHDIR
    if not DEBUG:
        cleandir(QMin['scratchdir'])
        if 'cleanup' in QMin:
            cleandir(QMin['savedir'])

    # if needs_finish:
        # finish()

    print()
    print(datetime.datetime.now())
    print('#================ END ================#')


if __name__ == '__main__':
    main()


# kate: indent-width 4
