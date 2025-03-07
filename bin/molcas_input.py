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

# Script for the calculation of Wigner distributions from molden frequency files
#
# usage

import copy
import math
import sys
import re
import os
import stat
import shutil
import datetime
from optparse import OptionParser
import readline
import time
from socket import gethostname
from constants import IToMult, NUMBERS, MASSES, U_TO_AMU
# =========================================================0
# some constants
DEBUG = False

version = '2.1'
versiondate = datetime.date(2019, 9, 1)

# MOLCAS works with g/mol
MASSES = MASSES.update((el, mass/U_TO_AMU) for (el, mass) in MASSES.items())

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def displaywelcome():
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('MOLCAS Input file generator') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Author: Sebastian Mai') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Version:' + version) + '||\n'
    string += '||' + '{:^80}'.format(versiondate.strftime("%d.%m.%y")) + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    string += '''
This script allows to quickly create MOLCAS input files for single-points calculations
on the SA-CASSCF and (MS-)CASPT2 levels of theory.
It also generates MOLCAS.template files to be used with the SHARC-MOLCAS Interface.
  '''
    print(string)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.molcas_input')

# ===================================


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    if typefunc == int or typefunc == float:
        if default is not None and not isinstance(default, list):
            print('Default to int or float question must be list!')
            quit(1)
    if typefunc == str and autocomplete:
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")    # activate autocomplete
    else:
        readline.parse_and_bind("tab: ")            # deactivate autocomplete

    while True:
        s = question
        if default is not None:
            if typefunc == bool or typefunc == str:
                s += ' [%s]' % (str(default))
            elif typefunc == int or typefunc == float:
                s += ' ['
                for i in default:
                    s += str(i) + ' '
                s = s[:-1] + ']'
        if typefunc == str and autocomplete:
            s += ' (autocomplete enabled)'
        if typefunc == int and ranges:
            s += ' (range comprehension enabled)'
        s += ' '

        line = input(s)
        line = re.sub(r'#.*$', '', line).strip()
        if not typefunc == str:
            line = line.lower()

        if line == '' or line == '\n':
            if default is not None:
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return default
            else:
                continue

        if typefunc == bool:
            posresponse = ['y', 'yes', 'true', 't', 'ja', 'si', 'yea', 'yeah', 'aye', 'sure', 'definitely']
            negresponse = ['n', 'no', 'false', 'f', 'nein', 'nope']
            if line in posresponse:
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return True
            elif line in negresponse:
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return False
            else:
                print('I didn''t understand you.')
                continue

        if typefunc == str:
            KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
            return line

        if typefunc == float:
            # float will be returned as a list
            f = line.split()
            try:
                for i in range(len(f)):
                    f[i] = typefunc(f[i])
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return f
            except ValueError:
                print('Please enter floats!')
                continue

        if typefunc == int:
            # int will be returned as a list
            f = line.split()
            out = []
            try:
                for i in f:
                    if ranges and '~' in i:
                        q = i.split('~')
                        for j in range(int(q[0]), int(q[1]) + 1):
                            out.append(j)
                    else:
                        out.append(int(i))
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return out
            except ValueError:
                if ranges:
                    print('Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!')
                else:
                    print('Please enter integers!')
                continue

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def show_massses(masslist):
    s = 'Number\tType\tMass\n'
    for i, atom in enumerate(masslist):
        s += '%i\t%2s\t%12.9f %s\n' % (i + 1, atom[0], atom[1], ['', '*    '][atom[1] != MASSES[atom[0]]])
    print(s)


def ask_for_masses(masslist):
    print('''
Please enter non-default masses:
+ number mass           use non-default mass <mass> for atom <number>
- number                remove non-default mass for atom <number> (default mass will reinstated)
show                    show atom masses
end                     finish input for non-default masses
''')
    show_massses(masslist)
    while True:
        line = question('Change an atoms mass:', str, 'end', False)
        if 'end' in line:
            break
        if 'show' in line:
            show_massses(masslist)
            continue
        if '+' in line:
            f = line.split()
            if len(f) < 3:
                continue
            try:
                num = int(f[1])
                mass = float(f[2])
            except ValueError:
                continue
            if not 0 <= num <= len(masslist):
                print('Atom %i does not exist!' % (num))
                continue
            masslist[num - 1][1] = mass
            continue
        if '-' in line:
            f = line.split()
            if len(f) < 2:
                continue
            try:
                num = int(f[1])
            except ValueError:
                continue
            if not 0 <= num <= len(masslist):
                print('Atom %i does not exist!' % (num))
                continue
            masslist[num - 1][1] = MASSES[masslist[num - 1][0]]
            continue
    return masslist

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_infos():
    '''Asks for the settings of the calculation:
  - type (single point, optimization+freq or MOLCAS.template
  - level of theory
  - basis set
  - douglas kroll
  - memory
  - geometry

  specific:
  - opt: freq?
  - CASSCF: docc, act'''

    INFOS = {}

    # Type of calculation
    print('{:-^60}'.format('Type of calculation'))
    print('''\nThis script generates input for the following types of calculations:
  1       Single point calculations (RASSCF, CASPT2)
  2       Optimizations & Frequency calculations (RASSCF, CASPT2)
  3       MOLCAS.template file for SHARC dynamics (SA-CASSCF)
Please enter the number corresponding to the type of calculation.
''')
    while True:
        ctype = question('Type of calculation:', int)[0]
        if ctype not in [1, 2, 3]:
            print('Enter an integer (1-3)!')
            continue
        break
    INFOS['ctype'] = ctype
    freq = False
    if ctype == 2:
        freq = question('Frequency calculation?', bool, True)
    INFOS['freq'] = freq
    print('')


    guessnact = None
    guessninact = None
    guessnelec = None
    guessnorb = None
    guessbase = None
    guessstates = [0] * 8
    guessspin = None


    # Geometry
    print('{:-^60}'.format('Geometry'))
    if ctype == 3:
        print('\nNo geometry necessary for MOLCAS.template generation\n')
        INFOS['geom'] = None
        # see whether a MOLCAS.input file is there, where we can take the number of electrons from
        nelec = 0
        try:
            molproinput = open('MOLCAS.input', 'r')
            for line in molproinput:
                # print(line.strip())
                if 'basis' in line.lower():
                    guessbase = line.split()[-1]
                if 'nactel' in line.lower():
                    guessnact = int(line.split()[-1].split(',')[0])
                if 'inact' in line.lower():
                    guessninact = int(line.split()[-1])
                if 'ras2' in line.lower():
                    guessnorb = int(line.split()[-1])
                if 'spin' in line.lower():
                    guessspin = int(line.split()[-1])
                if 'ciroot' in line.lower():
                    s = int(line.split()[-1].split(',')[0])
                    guessstates[guessspin - 1] = s
            try:
                for istate in range(len(guessstates) - 1, -1, -1):
                    if guessstates[istate] == 0:
                        guessstates.pop()
                    else:
                        break
                # print(guessnorb,guessnact,guessninact,guessnelec)
                if guessninact is not None and guessnact is not None:
                    guessnelec = [2 * guessninact + guessnact]
                if guessnorb is not None:
                    guessnorb = [guessnorb]
                if guessnact is not None:
                    guessnact = [guessnact]
                if guessninact is not None:
                    guessninact = [guessninact]
                # print(guessnorb,guessnact,guessninact,guessnelec)
            except BaseException:
                pass
        except (IOError, ValueError):
            pass
        # continue with asking for number of electrons
        while True:
            nelec = question('Number of electrons: ', int, guessnelec, False)[0]
            if nelec <= 0:
                print('Enter a positive number!')
                continue
            break
        INFOS['nelec'] = nelec
    else:
        print('\nPlease specify the geometry file (xyz format, Angstroms):')
        while True:
            path = question('Geometry filename:', str, 'geom.xyz')
            try:
                gf = open(path, 'r')
            except IOError:
                print('Could not open: %s' % (path))
                continue
            g = gf.readlines()
            gf.close()
            try:
                natom = int(g[0])
            except ValueError:
                print('Malformatted: %s' % (path))
                continue
            geom = []
            ncharge = 0
            fine = True
            for i in range(natom):
                try:
                    line = g[i + 2].split()
                except IndexError:
                    print('Malformatted: %s' % (path))
                    fine = False
                try:
                    line[0] = re.sub(r'[0-9]', '', line[0])
                    atom = [line[0], float(line[1]), float(line[2]), float(line[3])]
                except (IndexError, ValueError):
                    print('Malformatted: %s' % (path))
                    fine = False
                    continue
                geom.append(atom)
                try:
                    ncharge += NUMBERS[atom[0].title()]
                except KeyError:
                    print('Atom type %s not supported!' % (atom[0]))
                    fine = False
            if not fine:
                continue
            else:
                break
        print('Number of atoms: %i\nNuclear charge: %i\n' % (natom, ncharge))
        INFOS['geom'] = geom
        INFOS['ncharge'] = ncharge
        INFOS['natom'] = natom
        print('Enter the total (net) molecular charge:')
        while True:
            charge = question('Charge:', int, [0])[0]
            break
        INFOS['charge'] = charge
        INFOS['nelec'] = ncharge - charge
        print('Number of electrons: %i\n' % (ncharge - charge))

    # Masses
    if INFOS['freq']:
        # make default mass list
        masslist = []
        for atom in geom:
            masslist.append([atom[0], MASSES[atom[0]]])
        # ask
        # INFOS['nondefmass']=not question('Use standard masses (most common isotope)?',bool,True)
        # if INFOS['nondefmass']:
            # INFOS['masslist']=ask_for_masses(masslist)
        # else:
        INFOS['masslist'] = masslist

    # Level of theory
    print('\n' + '{:-^60}'.format('Level of theory'))
    print('''\nSupported by this script are:
  1       RASSCF
  2       CASPT2 %s
''' % (['', '(Only numerical gradients)'][INFOS['freq']]))
    # if ctype==3:
    # ltype=1
    # print('Choosing RASSCF for MOLCAS.template generation.')
    # else:
    while True:
        ltype = question('Level of theory:', int)[0]
        if ltype not in [1, 2]:
            print('Enter an integer (1-2)!')
            continue
        break
    INFOS['ltype'] = ltype


    # basis set
    print('\nPlease enter the basis set.')
    print('''Common available basis sets:
  Pople:     6-31G**, 6-311G, 6-31+G, 6-31G(d,p), ...    %s
  Dunning:   cc-pVXZ, aug-cc-pVXZ, cc-pVXZ-DK, ...
  ANO:       ANO-S-vdzp, ANO-L, ANO-RCC                   ''' % (['', '(Not available)'][ctype == 3]))
    basis = question('Basis set:', str, guessbase, False)
    INFOS['basis'] = basis
    INFOS['cholesky'] = question('Use Cholesky decomposition?', bool, False)

    # douglas kroll
    dk = question('Douglas-Kroll scalar-relativistic integrals?', bool, True)
    INFOS['DK'] = dk

    # CASSCF
    if ltype >= 1:
        print('\n' + '{:-^60}'.format('CASSCF Settings') + '\n')
        while True:
            nact = question('Number of active electrons:', int, guessnact)[0]
            if nact <= 0:
                print('Enter a positive number larger than zero!')
                continue
            if INFOS['nelec'] < nact:
                print('Number of active electrons cannot be larger than total number of electrons!')
                continue
            if (INFOS['nelec'] - nact) % 2 != 0:
                print('nelec-nact must be even!')
                continue
            break
        INFOS['cas.nact'] = nact
        while True:
            norb = question('Number of active orbitals:', int, guessnorb)[0]
            if norb <= 0:
                print('Enter a positive number!')
                continue
            if 2 * norb <= nact:
                print('norb must be larger than nact/2!')
                continue
            break
        INFOS['cas.norb'] = norb

    if ltype >= 1:
        print('Please enter the number of states for state-averaging as a list of integers\ne.g. 3 0 2 for three singlets, zero doublets and two triplets.')
        while True:
            states = question('Number of states:', int, guessstates)
            maxmult = len(states)
            for i in range(maxmult):
                n = states[i]
                if (not i % 2 == INFOS['nelec'] % 2) and int(n) > 0:
                    print('Nelec is %i. Ignoring states with mult=%i!' % (INFOS['nelec'], i + 1))
                    states[i] = 0
                if n < 0:
                    states[i] = 0
            if sum(states) == 0:
                print('No states!')
                continue
            break
        s = 'Accepted number of states:'
        for i in states:
            s += ' %i' % (i)
        print(s)
        INFOS['maxmult'] = len(states)
        INFOS['cas.nstates'] = states
        if ctype == 2:
            print('\nPlease specify the state to optimize\ne.g. 3 2 for the second triplet state.')
            while True:
                rmult, rstate = tuple(question('Root:', int, [1, 1]))
                if not 1 <= rmult <= INFOS['maxmult']:
                    print('%i must be between 1 and %i!' % (rmult, INFOS['maxmult']))
                    continue
                if not 1 <= rstate <= states[rmult - 1]:
                    print('Only %i states of mult %i' % (states[rmult - 1], rmult))
                    continue
                break
            INFOS['opt.root'] = [rmult, rstate]
            print('Optimization: Only performing one RASSCF for %ss.' % (IToMult[rmult]))
            for imult in range(len(INFOS['cas.nstates'])):
                if INFOS['cas.nstates'][imult] == 0:
                    continue
                if imult + 1 != rmult:
                    INFOS['cas.nstates'][imult] = 0
            s = 'Accepted number of states:'
            for i in INFOS['cas.nstates']:
                s += ' %i' % (i)
            print(s)

    if ltype > 1:
        print('\n' + '{:-^60}'.format('CASPT2 Settings') + '\n')
        if ctype == 1:
            INFOS['pt2.multi'] = question('Multi-state CASPT2?', bool, True)
        else:
            INFOS['pt2.multi'] = True
        INFOS['pt2.ipea'] = not question('Set IPEA shift to zero?', bool, False)
        INFOS['pt2.imag'] = question('Imaginary level shift?', float, [0.0])[0]





    print('\n' + '{:-^60}'.format('Further Settings') + '\n')

    if ctype == 1 and maxmult > 1:
        INFOS['soc'] = question('Do Spin-Orbit RASSI?', bool, False)
    else:
        INFOS['soc'] = False

    print('')

    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def setup_input(INFOS):
    ''''''

    # choose file name
    if INFOS['ctype'] == 3:
        inpf = 'MOLCAS.template'
    else:
        inpf = 'MOLCAS.input'
    print('Writing input to %s' % (inpf))
    try:
        inp = open(inpf, 'w')
    except IOError:
        print('Could not open %s for write!' % (inpf))
        quit(1)


    # template generation
    if INFOS['ctype'] == 3:
        s = 'basis %s\n' % (INFOS['basis'])
        s += 'ras2 %i\n' % (INFOS['cas.norb'])
        s += 'nactel %i\n' % (INFOS['cas.nact'])
        s += 'inactive %i\n' % ((INFOS['nelec'] - INFOS['cas.nact']) // 2)
        s += 'roots'
        for i, n in enumerate(INFOS['cas.nstates']):
            s += ' %i ' % (n)
        s += '\n\n'
        if not INFOS['DK']:
            s += 'no-douglas-kroll\n'
        if INFOS['cholesky']:
            s += 'cholesky\n'
        if INFOS['ltype'] == 1:
            s += 'method CASSCF\n'
        elif INFOS['ltype'] == 2:
            if not INFOS['pt2.ipea']:
                s += 'ipea 0.00\n'
            s += 'imaginary %4.2f\n' % INFOS['pt2.imag']
            if INFOS['pt2.multi']:
                s += 'method MS-CASPT2\n'
            else:
                s += 'method CASPT2\n'
        s += '\n\n'
        s += '#     Infos:\n'
        s += '#     %s@%s\n' % (os.environ['USER'], os.environ['HOSTNAME'])
        s += '#     Date: %s\n' % (datetime.datetime.now())
        s += '#     Current directory: %s\n\n' % (os.getcwd())
        inp.write(s)
        return


    # input file generation
    s = '**     %s generated by molcas_input.py Version %s\n\n' % (inpf, version)
    s += '&GATEWAY\n'
    if INFOS['geom']:
        s += 'COORD\n%i\n\n' % (len(INFOS['geom']))
        for iatom, atom in enumerate(INFOS['geom']):
            s += '%s%i % 16.9f % 16.9f % 16.9f\n' % (atom[0], iatom + 1, atom[1], atom[2], atom[3])
    if INFOS['basis']:
        s += 'GROUP = nosym\nTITLE = Molcas-%s\nBASIS = %s\n' % (['SP', 'Opt', ''][INFOS['ctype'] - 1], INFOS['basis'])


    if INFOS['ctype'] == 2:
        s += '\n\n**     ================ Optimization ================\n\n'
        s += '>> COPY FORCE %sOrbitals.RasOrb INPORB\n' % (IToMult[INFOS['opt.root'][0]])
        s += '>>> DO WHILE\n'


    s += '\n&SEWARD\n'
    if INFOS['DK']:
        s += 'EXPERT\nR02O\n'
    if INFOS['soc']:
        s += '* If using MOLCAS v>=8.1, move the AMFI keyword to the &GATEWAY section.\n'
        s += 'AMFI\n'
    if INFOS['cholesky']:
        s += 'CHOLESKY\n'


    if INFOS['ctype'] == 1:
        if not INFOS['DK'] and INFOS['nelec'] % 2 == 0 and INFOS['charge'] == 0:
            s += '\n\n&SCF\n\n'
        else:
            s += '\n\n** For DKH integrals or with ions, MOLCAS SCF seems to not work properly.\n*&SCF\n\n'


    ijobiph = 0
    for imult, nstate in enumerate(INFOS['cas.nstates']):
        if nstate == 0:
            continue
        mult = imult + 1
        ijobiph += 1

        if INFOS['ctype'] == 1:
            s += '\n\n**     ================ %s states ================\n\n' % (IToMult[mult])
            s += '**     Uncomment the following line in order to restart the orbitals:\n'
            s += '*      >> COPY FORCE %sOrbitals.RasOrb INPORB\n' % (IToMult[mult])

        s += '''
&RASSCF
SPIN   = %i
NACTEL = %i,0,0
INACT  = %i
RAS2   = %i
CIROOT = %i,%i,1
''' % (mult,
            INFOS['cas.nact'],
            (INFOS['nelec'] - INFOS['cas.nact']) // 2,
            INFOS['cas.norb'],
            nstate, nstate)
        if INFOS['ctype'] == 1:
            s += '''**     Uncomment the following line in order to restart the orbitals:
*LUMORB
**     Uncomment the following lines in order to change the orbital order:
*ALTER
*1
*1 1 2
'''
        if INFOS['ctype'] == 2:
            s += 'LUMORB\n'
            if INFOS['ltype'] == 1 and nstate > 1:
                s += 'RLXROOT = %i\n' % (INFOS['opt.root'][1])
        if INFOS['ctype'] == 1:
            s += '''
>> SAVE $Project.rasscf.molden %sOrbitals.molden
>> SAVE $Project.RasOrb %sOrbitals.RasOrb
''' % (IToMult[mult], IToMult[mult])

        if INFOS['ltype'] > 1:
            s += '''
&CASPT2
SHIFT      = 0.0
IMAGINARY  = %4.2f
IPEASHIFT  = %4.2f
MAXITER    = 120
* If using MOLCAS v>=8.1, uncomment the following line to get CASPT2 properties (dipole moments):
*PROP
''' % (INFOS['pt2.imag'], [0., 0.25][INFOS['pt2.ipea']])
            if not INFOS['pt2.multi']:
                s += 'NOMULT\n'
            s += 'MULTISTATE = %i %s\n' % (nstate, ' '.join([str(i + 1) for i in range(nstate)]))
            if INFOS['ctype'] == 2:
                # s+='LUMORB\n'
                if INFOS['ltype'] == 2 and nstate > 1:
                    s += 'RLXROOT = %i\n' % (INFOS['opt.root'][1])

        if INFOS['ctype'] == 1:
            if INFOS['ltype'] == 1:
                s += '\n>> SAVE $Project.JobIph JOB%03i\n\n' % (ijobiph)
            elif INFOS['ltype'] == 2:
                s += '\n>> SAVE $Project.JobMix JOB%03i\n\n' % (ijobiph)



    if INFOS['ctype'] == 2:
        s += '\n&SLAPAF\n>>> ENDDO\n'
    if INFOS['freq']:
        s += '\n**     ================ Frequencies ================\n'
        s += '\n&MCKINLEY\n'
        # s+='\n&MCLR\nMASS\n'
        # for iatom,atom in enumerate(INFOS['geom']):
        # s+='%s%i = %f\n' % (atom[0],iatom+1,INFOS['masslist'][iatom][1])


    if INFOS['ctype'] == 1:
        s += '\n\n**     ================ Final RASSI calculation ================\n\n'

        s += '&RASSI\nNROFJOBIPHS\n'
        njobiph = []
        for nstate in INFOS['cas.nstates']:
            if nstate > 0:
                njobiph.append(nstate)
        s += '%i %s\n' % (len(njobiph), ' '.join([str(i) for i in njobiph]))
        ijobiph = 0
        for imult, nstate in enumerate(INFOS['cas.nstates']):
            if nstate == 0:
                continue
            mult = imult + 1
            ijobiph += 1
            s += '%s\n' % (' '.join([str(i + 1) for i in range(nstate)]))
        s += 'CIPR\n'
        if INFOS['ltype'] == 2:
            s += 'EJOB\n'
        if INFOS['soc']:
            s += 'SPINORBIT\nSOCOUPLING = 0.0\n'

        s += '**If you want to calculate transition densities for TheoDORE:\n'
        s += '*       *Uncomment the corresponding block in the run script.\n'
        if INFOS['ltype'] == 2:
            s += '*       *Delete the EJOB keyword above.\n'
        s += '*       *Uncomment the following line:\n'
        s += '*TRD1\n'

    s += '\n\n'
    s += '*     Infos:\n'
    s += '*     %s at %s\n' % (os.environ['USER'], gethostname())
    s += '*     Date: %s\n' % (datetime.datetime.now())
    s += '*     Current directory: %s\n\n' % (os.getcwd())

    inp.write(s)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def set_runscript(INFOS):

    if INFOS['ctype'] == 3:
        # no run script for template generation
        return

    print('')
    if not question('Runscript?', bool, True):
        return
    print('')

    # MOLCAS executable
    print('{:-^60}'.format('Path to MOLCAS') + '\n')
    path = os.getenv('MOLCAS')
    if path is not None:
        path = os.path.expanduser(os.path.expandvars(path))
        print('Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path))
        if question('Do you want to use this MOLCAS installation?', bool, True):
            INFOS['molcas'] = path
    if 'molcas' not in INFOS:
        print('\nPlease specify path to MOLCAS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        INFOS['molcas'] = question('Path to MOLCAS:', str)
    print('')


    # Scratch directory
    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to run the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculation on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str) + '/WORK'
    print('')
    # Keep scratch directory
    INFOS['delete_scratch'] = question('Delete scratch directory after calculation?', bool, False)

    # Memory
    print('\n' + '{:-^60}'.format('Memory'))
    print('\nRecommendation: for small systems: 100-300 MB, for medium-sized systems: 1000-2000 MB\n')
    mem = abs(question('Memory in MB: ', int, [500])[0])
    # always give at least 50 MB
    mem = max(mem, 50)
    INFOS['mem'] = mem

    # make job name
    cwd = os.path.split(os.getcwd())[-1][0:6]
    if len(cwd) < 6:
        cwd = '_' * (6 - len(cwd)) + cwd
    cwd = 'MCAS' + cwd

    string = '''#!/bin/bash
#$ -N %s
#$ -S /bin/bash
#$ -cwd

PRIMARY_DIR=%s
SCRATCH_DIR=%s

export MOLCAS=%s
export MOLCASMEM=%i
export MOLCASDISK=0
export MOLCASRAMD=0
export MOLCAS_MOLDEN=ON

#export MOLCAS_CPUS=1
#export OMP_NUM_THREADS=1

export Project="MOLCAS"
export HomeDir=$PRIMARY_DIR
export CurrDir=$PRIMARY_DIR
export WorkDir=$SCRATCH_DIR/$Project/
ln -sf $WorkDir $CurrDir/WORK

cd $HomeDir
mkdir -p $WorkDir

''' % (cwd,
       os.getcwd(),
       INFOS['scratchdir'],
       INFOS['molcas'],
       INFOS['mem'])

    for imult, nstate in enumerate(INFOS['cas.nstates']):
        if nstate == 0:
            continue
        mult = imult + 1
        string += 'cp $HomeDir/%sOrbitals.RasOrb $WorkDir\n' % (IToMult[mult])
    string += 'cd $Workdir'

    if os.path.isfile(os.path.join(INFOS['molcas'], 'bin', 'pymolcas')):
        string += '\n$MOLCAS/bin/pymolcas MOLCAS.input &> $CurrDir/MOLCAS.log\n\n'
    elif os.path.isfile(os.path.join(INFOS['molcas'], 'bin', 'molcas.exe')):
        string += '\n$MOLCAS/bin/molcas.exe MOLCAS.input &> $CurrDir/MOLCAS.log\n\n'
    else:
        print('Could not find MOLCAS driver in %s' % os.path.join(INFOS['molcas'], 'bin'))
        sys.exit(1)
    string += 'cd $HomeDir'

    for imult, nstate in enumerate(INFOS['cas.nstates']):
        if nstate == 0:
            continue
        mult = imult + 1
        string += 'cp $WorkDir/%sOrbitals.* $HomeDir\n' % (IToMult[mult])

    string += '#mkdir -p $HomeDir/TRD/\n#cp $WorkDir/TRD2_* $HomeDir/TRD/\n'

    if INFOS['delete_scratch']:
        string += '\nrm -r $SCRATCH_DIR\n'


    runscript = 'run_MOLCAS.sh'
    print('Writing run script %s' % (runscript))
    try:
        runf = open(runscript, 'w')
    except IOError:
        print('Could not write %s' (runscript))
        return
    runf.write(string)
    runf.close()
    os.chmod(runscript, os.stat(runscript).st_mode | stat.S_IXUSR)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def warnings(INFOS):
    print('{:*^60}'.format(' WARNINGS '))
    print('*' + ' ' * 60 + '*')
    if INFOS['ctype'] == 1:
        if INFOS['DK']:
            print('*' + '{: ^60}'.format('Douglas-Kroll: Will not do SCF!') + '*')
            print('*' + ' ' * 60 + '*')
        if INFOS['nelec'] % 2 != 0:
            print('*' + '{: ^60}'.format('Odd number of electrons: Will not do SCF!') + '*')
            print('*' + ' ' * 60 + '*')
        if INFOS['charge'] != 0:
            print('*' + '{: ^60}'.format('Nonzero charge: Will not do SCF!') + '*')
            print('*' + ' ' * 60 + '*')
    print('*' + ' ' * 60 + '*')
    print('*' * 62)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
    '''Main routine'''

    usage = '''
python molcas_input.py

This interactive program prepares a MOLCAS input file for ground state optimizations and frequency calculations with CASSCF. It also generates MOLCAS.template files to be used with the SHARC-MOLCAS interface.
'''

    description = ''
    parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()

    INFOS = get_infos()

    print('{:#^60}'.format('Full input') + '\n')
    for item in INFOS:
        print(item, ' ' * (15 - len(item)), INFOS[item])
    print('')

    setup_input(INFOS)
    set_runscript(INFOS)
    print('\nFinished\n')

    close_keystrokes()

    warnings(INFOS)

# ======================================================================================================================


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
