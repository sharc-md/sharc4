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

# Interactive script for the use of the ORCA external optimizer with SHARC
#
# usage: python setup_orca_opt.py #change

import math
import sys
import re
import os
import stat
import shutil
import datetime
import random
from optparse import OptionParser
import readline
import time
from socket import gethostname
import ast
import pprint

# =========================================================0
# compatibility stuff

if sys.version_info[0] != 3:
    print('This is a script for Python 3!')
    sys.exit(0)

# some constants
DEBUG = False
CM_TO_HARTREE = 1. / 219474.6  # 4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132    # conversion factor from Hartree to eV
U_TO_AMU = 1. / 5.4857990943e-4            # conversion from g/mol to amu
BOHR_TO_ANG = 0.529177211
PI = math.pi

version = '2.1'
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2019, 9, 1)


IToMult = {
    1: 'Singlet',
    2: 'Doublet',
    3: 'Triplet',
    4: 'Quartet',
    5: 'Quintet',
    6: 'Sextet',
    7: 'Septet',
    8: 'Octet',
    'Singlet': 1,
    'Doublet': 2,
    'Triplet': 3,
    'Quartet': 4,
    'Quintet': 5,
    'Sextet': 6,
    'Septet': 7,
    'Octet': 8
}

# ======================================================================= #

Interfaces = {
    1: {'script': 'SHARC_MOLPRO.py',
        'name': 'molpro',
        'description': 'MOLPRO (only CASSCF)',
        'get_routine': 'get_MOLPRO',
        'prepare_routine': 'prepare_MOLPRO',
        'features': {'overlap': ['wfoverlap'],
                     'dyson': ['wfoverlap'],
                     'nacdr': ['wfoverlap'],
                     'phases': ['wfoverlap'],
                     'soc': []},
        'pysharc': False
        },
    2: {'script': 'SHARC_COLUMBUS.py',
        'name': 'columbus',
        'description': 'COLUMBUS (CASSCF, RASSCF and MRCISD), using SEWARD integrals',
        'get_routine': 'get_COLUMBUS',
        'prepare_routine': 'prepare_COLUMBUS',
        'features': {'overlap': ['wfoverlap'],
                     'dyson': ['wfoverlap'],
                     'phases': ['wfoverlap'],
                     'nacdr': [],
                     'soc': []},
        'pysharc': False
        },
    3: {'script': 'SHARC_Analytical.py',
        'name': 'analytical',
        'description': 'Analytical PESs',
        'get_routine': 'get_Analytical',
        'prepare_routine': 'prepare_Analytical',
        'features': {'overlap': [],
                     'dipolegrad': [],
                     'phases': [],
                     'soc': []},
        'pysharc': False
        },
    4: {'script': 'SHARC_MOLCAS.py',
        'name': 'molcas',
        'description': 'MOLCAS (CASSCF, CASPT2, MS-CASPT2)',
        'get_routine': 'get_MOLCAS',
        'prepare_routine': 'prepare_MOLCAS',
        'features': {'overlap': [],
                     'dyson': ['wfoverlap'],
                     'dipolegrad': [],
                     'phases': [],
                     'soc': []},
        'pysharc': False
        },
    5: {'script': 'SHARC_AMS-ADF.py',
        'name': 'ams-adf',
        'description': 'AMS-ADF (DFT, TD-DFT)',
        'get_routine': 'get_AMS',
        'prepare_routine': 'prepare_AMS',
        'features': {'overlap': ['wfoverlap'],
                     'dyson': ['wfoverlap'],
                     'theodore': ['theodore'],
                     'phases': ['wfoverlap'],
                     'soc': []},
        'pysharc': False
        },
    6: {'script': 'SHARC_RICC2.py',
        'name': 'ricc2',
        'description': 'TURBOMOLE (ricc2 with CC2 and ADC(2))',
        'get_routine': 'get_RICC2',
        'prepare_routine': 'prepare_RICC2',
        'features': {'overlap': ['wfoverlap'],
                     'theodore': ['theodore'],
                     'phases': ['wfoverlap'],
                     'soc': []},
        'pysharc': False
        },
    7: {'script': 'SHARC_LVC.py',
        'name': 'lvc',
        'description': 'LVC Hamiltonian',
        'get_routine': 'get_LVC',
        'prepare_routine': 'prepare_LVC',
        'features': {'overlap': [],
                     'nacdr': [],
                     'phases': [],
                     'soc': []},
        'pysharc': True,
        'pysharc_driver': 'pysharc_lvc.py'
        },
    8: {'script': 'SHARC_GAUSSIAN.py',
        'name': 'gaussian',
        'description': 'GAUSSIAN (DFT, TD-DFT)',
        'get_routine': 'get_GAUSSIAN',
        'prepare_routine': 'prepare_GAUSSIAN',
        'features': {'overlap': ['wfoverlap'],
                     'dyson': ['wfoverlap'],
                     'theodore': ['theodore'],
                     'phases': ['wfoverlap']},
        'pysharc': False
        },
    9: {'script': 'SHARC_ORCA.py',
        'name': 'orca',
        'description': 'ORCA (DFT, TD-DFT, HF, CIS)',
        'get_routine': 'get_ORCA',
        'prepare_routine': 'prepare_ORCA',
        'features': {'overlap': ['wfoverlap'],
                     'dyson': ['wfoverlap'],
                     'theodore': ['theodore'],
                     'phases': ['wfoverlap'],
                     'soc': []},
        'pysharc': False
        },
    10: {'script': 'SHARC_BAGEL.py',
         'name': 'bagel',
         'description': 'BAGEL (CASSCF, CASPT2, (X)MS-CASPT2)',
         'get_routine': 'get_BAGEL',
         'prepare_routine': 'prepare_BAGEL',
         'features': {'overlap': ['wfoverlap'],
                      'dyson': ['wfoverlap'],
                      'nacdr': ['wfoverlap'],
                      'dipolegrad': [],
                      'phases': [], },
         'pysharc': False
         }
}


Couplings = {
    1: {'name': 'nacdt',
        'description': 'DDT     =  < a|d/dt|b >        Hammes-Schiffer-Tully scheme   '
        },
    2: {'name': 'nacdr',
        'description': 'DDR     =  < a|d/dR|b >        Original Tully scheme          '
        },
    3: {'name': 'overlap',
        'description': 'overlap = < a(t0)|b(t) >       Local Diabatization scheme     '
        }
}


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(13)
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
            sys.exit(14)
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(15)
# ======================================================================= #


def displaywelcome():
    print('Script for setup of optimizations with ORCA and SHARC started...\n')  # change
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Setup optimizations with ORCA and SHARC') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Author: Moritz Heindl, Sebastian Mai') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Version:' + version) + '||\n'
    string += '||' + '{:^80}'.format(versiondate.strftime("%d.%m.%y")) + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    string += '''
This script automatizes the setup of the input files ORCA+SHARC optimizations.
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
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.setup_orca_opt')

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
        line = re.sub('#.*$', '', line).strip()
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
                print("I didn't understand you.")
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


def itnmstates(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general():
    '''This routine questions from the user some general information:
    - initconds file
    - number of states
    - number of initial conditions
    - interface to use'''

    INFOS = {}


    print('\n' + '{:-^60}'.format('Path to ORCA') + '\n')
    path = os.getenv('ORCADIR')
    # path=os.path.expanduser(os.path.expandvars(path))
    if path == '':
        path = None
    else:
        path = '$ORCADIR/'
        # print('Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path))
        # if question('Do you want to use this MOLCAS installation?',bool,True):
        # INFOS['molcas']=path
        # if 'molcas' not in INFOS:
    print('Please specify path to ORCA directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['orca'] = question('Path to ORCA:', str, path)
    # print(INFOS['orca'])
    print


    print('{:-^60}'.format('Choose the quantum chemistry interface'))
    print('\nPlease specify the quantum chemistry interface (enter any of the following numbers):')
    for i in Interfaces:
        print('%i\t%s' % (i, Interfaces[i]['description']))
    print('')
    while True:
        num = question('Interface number:', int)[0]
        if num in Interfaces:
            break
        else:
            print('Please input one of the following: %s!' % ([i for i in Interfaces]))
    INFOS['interface'] = num
    print




    print('{:-^60}'.format('Geometry'))
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
        break
    INFOS['geom_location'] = path
    geometry_data = readfile(INFOS['geom_location'])
    ngeoms = len(geometry_data) // (natom + 2)
    if ngeoms > 1:
        print('Number of geometries: %i' % (ngeoms))
    INFOS['ngeom'] = ngeoms
    INFOS['natom'] = natom


    # Number of states
    print('\n' + '{:-^60}'.format('Number of states') + '\n')
    print('\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets.')
    while True:
        states = question('Number of states:', int)
        if len(states) == 0:
            continue
        if any(i < 0 for i in states):
            print('Number of states must be positive!')
            continue
        break
    print('')
    nstates = 0
    for mult, i in enumerate(states):
        nstates += (mult + 1) * i
    print('Number of states: ' + str(states))
    print('Total number of states: %i\n' % (nstates))
    INFOS['states'] = states
    INFOS['nstates'] = nstates
    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(INFOS['states']):
        statemap[i] = [imult, istate, ims]
        i += 1
    INFOS['statemap'] = statemap
    pprint.pprint(statemap)





    # states to optimize
    print('\n' + '{:-^60}'.format('States to optimize') + '\n')

    INFOS['maxmult'] = len(states)
    optmin = question('Do you want to optimize a minimum? (no=optimize crossing):', bool, True)
    if optmin:
        INFOS['opttype'] = 'min'
        print('\nPlease specify the state involved in the optimization\ne.g. 3 2 for the second triplet state.')
    else:
        INFOS['opttype'] = 'cross'
        print('\nPlease specify the first state involved in the optimization\ne.g. 3 2 for the second triplet state.')
    while True:
        rmult, rstate = tuple(question('State:', int, [1, 1]))
        # check
        if not 1 <= rmult <= INFOS['maxmult']:
            print('Multiplicity (%i) must be between 1 and %i!' % (rmult, INFOS['maxmult']))
            continue
        if not 1 <= rstate <= states[rmult - 1]:
            print('Only %i states of mult %i' % (states[rmult - 1], rmult))
            continue
        break
    INFOS['cas.root1'] = [rmult, rstate]

    if not optmin:
        print('\nPlease specify the second state involved in the optimization\ne.g. 3 2 for the second triplet state.')
        while True:
            rmult, rstate = tuple(question('Root:', int, [1, 2]))
            # check
            if not 1 <= rmult <= INFOS['maxmult']:
                print('%i must be between 1 and %i!' % (rmult, INFOS['maxmult']))
                continue
            if not 1 <= rstate <= states[rmult - 1]:
                print('Only %i states of mult %i' % (states[rmult - 1], rmult))
                continue
            INFOS['cas.root2'] = [rmult, rstate]
            if INFOS['cas.root1'] == INFOS['cas.root2']:
                print('Both states are identical!')
                continue
            # get type of optimization
            if INFOS['cas.root1'][0] == INFOS['cas.root2'][0]:
                print('Multiplicities of both states identical, optimizing a conical intersection.')
                INFOS['calc_ci'] = True
            else:
                print('Multiplicities of both states different, optimizing a minimum crossing point.')
                INFOS['calc_ci'] = False
            # find state 2 in statemap
            for i in statemap:
                if statemap[i][0:2] == INFOS['cas.root2']:
                    INFOS['cas.root2'] = i
                    break
            break

    # find state 1 in statemap
    for i in statemap:
        if statemap[i][0:2] == INFOS['cas.root1']:
            INFOS['cas.root1'] = i
            break



    INFOS['needed'] = []
    if not optmin and INFOS['calc_ci']:
        if 'nacdr' not in Interfaces[INFOS['interface']]['features']:
            print('{:-^60}'.format('Optimization parameter'))
            print('\nYou are optimizing a conical intersection, but the chosen interface cannot deliver nonadiabatic coupling vectors. The optimization will therefore employ the penalty function method of Levine, Coe, Martinez (DOI: 10.1021/jp0761618).\nIn this optimization scheme, there are two parameters, sigma and alpha, which affect how close to the true conical intersection the optimization will end up.')
            print('\nPlease enter the values for the sigma and alpha parameters.\n')
            print('A larger sigma makes convergence harder but optimization will go closer to the true CI.')
            sigma = question('Sigma: ', float, [3.5])[0]
            print('A smaller alpha makes convergence harder but optimization will go closer to the true CI.')
            alpha = question('Alpha: ', float, [0.02])[0]
            INFOS['sigma'] = sigma
            INFOS['alpha'] = alpha
        else:
            INFOS['needed'].extend(Interfaces[num]['features']['nacdr'])


    print('\nPlease enter the values for the maximum allowed displacement per timestep \n(choose smaller value if starting from a good guess and for large sigma or small alpha).')
    INFOS['maxstep'] = question('Maximum allowed step: ', float, [0.3])[0]









    # Add some simple keys
    INFOS['cwd'] = os.getcwd()
    print('')
    INFOS['needed'] = []

    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def checktemplate_MOLPRO(filename):
    necessary = ['basis', 'closed', 'occ', 'nelec', 'roots']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    i = 0
    for l in data:
        if necessary[i] in l:
            i += 1
            if i + 1 == len(necessary):
                return True
    print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
    return False

# =================================================


def get_MOLPRO(INFOS):
    ''' This routine asks for all questions specific to MOLPRO:
    - path to molpro
    - scratch directory
    - MOLPRO.template
    - wf.init
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('MOLPRO Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)


    # MOLPRO executable
    print('{:-^60}'.format('Path to MOLPRO') + '\n')
    path = os.getenv('MOLPRO')
    path = os.path.expanduser(os.path.expandvars(path))
    if not path == '':
        path = '$MOLPRO/'
    else:
        path = None
    # if path!='':
        # print('Environment variable $MOLPRO detected:\n$MOLPRO=%s\n' % (path))
        # if question('Do you want to use this MOLPRO installation?',bool,True):
        # INFOS['molpro']=path
    # if 'molpro' not in INFOS:
    print('\nPlease specify path to MOLPRO directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['molpro'] = question('Path to MOLPRO executable:', str, path)
    print('')


    # Scratch directory
    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    # MOLPRO input template
    print('{:-^60}'.format('MOLPRO input template file') + '\n')
    print('''Please specify the path to the MOLPRO.template file. This file must be a valid MOLPRO input file for a CASSCF calculation. It should contain the following settings:
- memory settings
- Basis set (possibly also Douglas-Kroll settings etc.)
- CASSCF calculation with:
  * Number of frozen, closed and occupied orbitals
  * wf and state cards for the specification of the wavefunction
MOLPRO.template files can easily be created using molpro_input.py (Open a second shell if you need to create one now).

The MOLPRO interface will generate the remaining MOLPRO input automatically.
''')
    if os.path.isfile('MOLPRO.template'):
        if checktemplate_MOLPRO('MOLPRO.template'):
            print('Valid file "MOLPRO.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['molpro.template'] = 'MOLPRO.template'
    if 'molpro.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_MOLPRO(filename):
                break
        INFOS['molpro.template'] = filename
    print('')


    # Initial wavefunction
    print('{:-^60}'.format('Initial wavefunction: MO Guess') + '\n')
    print('''Please specify the path to a MOLPRO wavefunction file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!

If you optimized your geometry with MOLPRO/CASSCF you can reuse the "wf" file from the optimization.
''')
    if question('Do you have an initial wavefunction file?', bool, True):
        while True:
            filename = question('Initial wavefunction file:', str, 'wf.init')
            if os.path.isfile(filename):
                break
            else:
                print('File not found!')
        INFOS['molpro.guess'] = filename
    else:
        print('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')
        time.sleep(2)
        INFOS['molpro.guess'] = False


    print('{:-^60}'.format('MOLPRO Ressource usage') + '\n')
    print('''Please specify the amount of memory available to MOLPRO (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
''')
    INFOS['molpro.mem'] = abs(question('MOLPRO memory:', int, [500])[0])
    print('''Please specify the number of CPUs to be used by EACH trajectory.
''')
    INFOS['molpro.ncpu'] = abs(question('Number of CPUs:', int, [1])[0])

    # Ionization
    # print(centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)

    # wfoverlap
    if 'wfoverlap' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wfoverlap code setup') + '\n')
        INFOS['molpro.wfpath'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')


    # Other settings
    INFOS['molpro.gradaccudefault'] = 1.e-7
    INFOS['molpro.gradaccumax'] = 1.e-4
    INFOS['molpro.ncore'] = -1
    INFOS['molpro.ndocc'] = 0

    return INFOS

# =================================================


def prepare_MOLPRO(INFOS, iconddir):
    # write MOLPRO.resources
    try:
        sh2pro = open('%s/MOLPRO.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareMOLPRO, iconddir=%s' % (iconddir))
        quit(1)
    string = '''molpro %s
scratchdir %s/%s/
savedir %s/%s/restart
gradaccudefault %.8f
gradaccumax %f
memory %i
ncpu %i
''' % (
        INFOS['molpro'],
        INFOS['scratchdir'],
        iconddir,
        INFOS['copydir'],
        iconddir,
        INFOS['molpro.gradaccudefault'],
        INFOS['molpro.gradaccumax'],
        INFOS['molpro.mem'],
        INFOS['molpro.ncpu']
    )
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfoverlap %s\n' % (INFOS['molpro.wfpath'])
    sh2pro.write(string)
    sh2pro.close()

    # copy MOs and template
    cpfrom = INFOS['molpro.template']
    cpto = '%s/MOLPRO.template' % (iconddir)
    shutil.copy(cpfrom, cpto)
    if INFOS['molpro.guess']:
        cpfrom = INFOS['molpro.guess']
        cpto = '%s/wf.init' % (iconddir)
        shutil.copy(cpfrom, cpto)

    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def checktemplate_COLUMBUS(TEMPLATE, mult):
    '''Checks whether TEMPLATE is a file or directory. If a file or does not exist, it quits with exit code 1, if it is a directory, it checks whether all important input files are there. Does not check for all input files, since runc does this, too.

    Arguments:
    1 string: path to TEMPLATE

    returns whether input is for isc keyword or socinr keyword
    and returns the DRT of the given multiplicity'''

    exist = os.path.exists(TEMPLATE)
    if exist:
        isfile = os.path.isfile(TEMPLATE)
        if isfile:
            # print('TEMPLATE=%s exists and is a file!' % (TEMPLATE))
            return None, None, None
        necessary = ['control.run', 'mcscfin', 'tranin', 'propin']
        lof = os.listdir(TEMPLATE)
        for i in necessary:
            if i not in lof:
                # print('Did not find input file %s! Did you prepare the input according to the instructions?' % (i))
                return None, None, None
        cidrtinthere = False
        ciudginthere = False
        for i in lof:
            if 'cidrtin' in i:
                cidrtinthere = True
            if 'ciudgin' in i:
                ciudginthere = True
        if not cidrtinthere or not ciudginthere:
            # print('Did not find input file %s.*! Did you prepare the input according to the instructions?' % (i))
            return None, None, None
    else:
        # print('Directory %s does not exist!' % (TEMPLATE))
        return None, None, None

    # get integral program
    try:
        intprog = open(TEMPLATE + '/intprogram')
        line = intprog.readline()
        if 'hermit' in line:
            INTPROG = 'dalton'
        elif 'seward' in line:
            INTPROG = 'seward'
        else:
            return None, None, None
    except IOError:
        return None, None, None

    # check cidrtin and cidrtin* for the multiplicity
    try:
        cidrtin = open(TEMPLATE + '/cidrtin')
        line = cidrtin.readline().split()
        if line[0].lower() == 'y':
            maxmult = int(cidrtin.readline().split()[0])
            cidrtin.readline()
            nelec = int(cidrtin.readline().split()[0])
            if mult <= maxmult and (mult + nelec) % 2 != 0:
                return 1, (mult + 1) // 2, INTPROG    # socinr=1, single=-1, isc=0
            else:
                return None, None, None
        else:
            mult2 = int(cidrtin.readline().split()[0])
            if mult != mult2:
                # print('Multiplicity %i cannot be treated in directory %s (single DRT)!'  % (mult,TEMPLATE))
                return None, None, None
            return -1, 1, INTPROG
    except IOError:
        # find out in which DRT the requested multiplicity is
        for i in range(1, 9):        # COLUMBUS can treat at most 8 DRTs
            try:
                cidrtin = open(TEMPLATE + '/cidrtin.%i' % i)
            except IOError:
                return None, None, None
            cidrtin.readline()
            mult2 = int(cidrtin.readline().split()[0])
            if mult == mult2:
                return 0, i, INTPROG
            cidrtin.close()

# =================================================


def get_COLUMBUS(INFOS):
    '''This routine asks for all questions specific to COLUMBUS:
    - path to COLUMBUS
    - scratchdir
    - path to template directory
    - mocoef
    - memory
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('COLUMBUS Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)


    # Path to COLUMBUS directory
    print('{:-^60}'.format('Path to COLUMBUS') + '\n')
    path = os.getenv('COLUMBUS')
    if path == '':
        path = None
    else:
        path = '$COLUMBUS/'
    # path=os.path.expanduser(os.path.expandvars(path))
    # if path!='':
        # print('Environment variable $COLUMBUS detected:\n$COLUMBUS=%s\n' % (path))
        # if question('Do you want to use this COLUMBUS installation?',bool,True):
        # INFOS['columbus']=path
    # if 'columbus' not in INFOS:
    print('\nPlease specify path to COLUMBUS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['columbus'] = question('Path to COLUMBUS:', str, path)
    print('')


    # Scratch directory
    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to temporally store all COLUMBUS files. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    # COLUMBUS template directory
    print('{:-^60}'.format('COLUMBUS input template directory') + '\n')
    print('''Please specify the path to the COLUMBUS template directory.
The directory must contain subdirectories with complete COLUMBUS input file sets for the following steps:
- Integrals with SEWARD/MOLCAS
- SCF
- MCSCF
- SO-MRCI (even if no Spin-Orbit couplings will be calculated)
The COLUMBUS interface will generate the remaining COLUMBUS input automatically, depending on the number of states.

In order to setup the COLUMBUS input, use COLUMBUS' input facility colinp. For further information, see the Spin-orbit tutorial for COLUMBUS [1].

[1] http://www.univie.ac.at/columbus/docs_COL70/tutorial-SO.pdf
''')
    while True:
        path = question('Path to templates:', str)
        path = os.path.expanduser(os.path.expandvars(path))
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            print('Directory %s does not exist!' % (path))
            continue

        content = os.listdir(path)
        multmap = {}
        allOK = True
        for mult in range(1, 1 + len(INFOS['states'])):
            if INFOS['states'][mult - 1] == 0:
                continue
            found = False
            for d in content:
                template = path + '/' + d
                socitype, drt, intprog = checktemplate_COLUMBUS(template, mult)
                if socitype is None:
                    continue
                if not d[-1] == '/':
                    d += '/'
                multmap[mult] = d
                found = True
                break
            if not found:
                print('No input directory for multiplicity %i!' % (mult))
                allOK = False
                continue
        if allOK:
            break
    print('')

    print('''Check whether the jobs are assigned correctly to the multiplicities. Use the following commands:
  mult job        make <mult> use the input in <job>
  show            show the mapping of multiplicities to jobs
  end             confirm this mapping
''')
    for i in multmap:
        print('%i ==> %s' % (i, multmap[i]))
    while True:
        line = question('Adjust job mapping:', str, 'end', False)
        if 'show' in line.lower():
            for i in multmap:
                print('%i ==> %s' % (i, multmap[i]))
            continue
        elif 'end' in line.lower():
            break
        else:
            f = line.split()
            try:
                m = int(f[0])
                j = f[1]
            except (ValueError, IndexError):
                continue
            if m not in multmap:
                print('Multiplicity %i not necessary!' % (m))
                continue
            if not os.path.isdir(path + '/' + j):
                print('No template subdirectory %s!' % (j))
                continue
            if not j[-1] == '/':
                j += '/'
            multmap[m] = j
    print('')

    mocoefmap = {}
    for job in set([multmap[i] for i in multmap]):
        mocoefmap[job] = multmap[1]
    print('''Check whether the mocoeffiles are assigned correctly to the jobs. Use the following commands:
  job mocoefjob   make <job> use the mocoeffiles from <mocoefjob>
  show            show the mapping of multiplicities to jobs
  end             confirm this mapping
''')
    width = max([len(i) for i in mocoefmap])
    for i in mocoefmap:
        print('%s' % (i) + ' ' * (width - len(i)) + ' <== %s' % (mocoefmap[i]))
    while True:
        line = question('Adjust mocoef mapping:', str, 'end', False)
        if 'show' in line.lower():
            for i in mocoefmap:
                print('%s <== %s' % (i, mocoefmap[i]))
            continue
        elif 'end' in line.lower():
            break
        else:
            f = line.split()
            try:
                j = f[0]
                m = f[1]
            except (ValueError, IndexError):
                continue
            if not m[-1] == '/':
                m += '/'
            if not j[-1] == '/':
                j += '/'
            mocoefmap[j] = m
    print('')

    INFOS['columbus.template'] = path
    INFOS['columbus.multmap'] = multmap
    INFOS['columbus.mocoefmap'] = mocoefmap
    INFOS['columbus.intprog'] = intprog

    INFOS['columbus.copy_template'] = question('Do you want to copy the template directory to each trajectory (Otherwise it will be linked)?', bool, False)
    if INFOS['columbus.copy_template']:
        INFOS['columbus.copy_template_from'] = INFOS['columbus.template']
        INFOS['columbus.template'] = './COLUMBUS.template/'


    # Initial mocoef
    print('{:-^60}'.format('Initial wavefunction: MO Guess') + '\n')
    print('''Please specify the path to a COLUMBUS mocoef file containing suitable starting MOs for the CASSCF calculation.
''')
    init = question('Do you have an initial mocoef file?', bool, True)
    if init:
        while True:
            line = question('Mocoef filename:', str, 'mocoef_mc.init')
            line = os.path.expanduser(os.path.expandvars(line))
            if os.path.isfile(line):
                break
            else:
                print('File not found!')
                continue
        INFOS['columbus.guess'] = line
    else:
        print('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')
        time.sleep(2)
        INFOS['columbus.guess'] = False
    print('')


    # Memory
    print('{:-^60}'.format('COLUMBUS Memory usage') + '\n')
    print('''Please specify the amount of memory available to COLUMBUS (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
''')
    INFOS['columbus.mem'] = abs(question('COLUMBUS memory:', int)[0])


    # need_wfoverlap=False
    # Ionization
    # print(centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)
    # if 'ion' in INFOS and INFOS['ion']:
    # need_wfoverlap=True
    # cioverlaps
    # if Couplings[INFOS['coupling']]['name']=='overlap':
    # need_wfoverlap=True

    # wfoverlap
    # if need_wfoverlap:
    if 'wfoverlap' in INFOS['needed']:
        if 'ion' in INFOS and INFOS['ion']:
            print('Dyson norms requested.')
        if Couplings[INFOS['coupling']]['name'] == 'overlap':
            print('Wavefunction overlaps requested.')
        print('\n' + '{:-^60}'.format('Wfoverlap code setup') + '\n')
        INFOS['columbus.wfpath'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')
        INFOS['columbus.wfthres'] = question('Determinant screening threshold:', float, [0.97])[0]
        INFOS['columbus.numfrozcore'] = question('Number of frozen core orbitals for overlaps (-1=as in template):', int, [-1])[0]
        if 'ion' in INFOS and INFOS['ion']:
            INFOS['columbus.numocc'] = question('Number of doubly occupied orbitals for Dyson:', int, [0])[0]

    return INFOS

# =================================================


def prepare_COLUMBUS(INFOS, iconddir):
    # write COLUMBUS.resources
    try:
        sh2col = open('%s/COLUMBUS.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareCOLUMBUS, directory=%i' % (iconddir))
        quit(1)
    string = '''columbus %s
scratchdir %s/%s/
savedir %s/%s/restart
memory %i
template %s
''' % (INFOS['columbus'], INFOS['scratchdir'], iconddir, INFOS['copydir'], iconddir, INFOS['columbus.mem'], INFOS['columbus.template'])
    string += 'integrals %s\n' % (INFOS['columbus.intprog'])
    for mult in INFOS['columbus.multmap']:
        string += 'DIR %i %s\n' % (mult, INFOS['columbus.multmap'][mult])
    string += '\n'
    for job in INFOS['columbus.mocoefmap']:
        string += 'MOCOEF %s %s\n' % (job, INFOS['columbus.mocoefmap'][job])
    string += '\n'
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfthres %f\n' % (INFOS['columbus.wfthres'])
        string += 'wfoverlap %s\n' % (INFOS['columbus.wfpath'])
        if INFOS['columbus.numfrozcore'] >= 0:
            string += 'numfrozcore %i\n' % (INFOS['columbus.numfrozcore'])
        if 'columbus.numocc' in INFOS:
            string += 'numocc %i\n' % (INFOS['columbus.numocc'])
    else:
        string += 'nooverlap\n'
    sh2col.write(string)
    sh2col.close()

    # copy MOs and template
    if INFOS['columbus.guess']:
        cpfrom = INFOS['columbus.guess']
        cpto = '%s/mocoef_mc.init' % (iconddir)
        shutil.copy(cpfrom, cpto)

    if INFOS['columbus.copy_template']:
        copy_from = INFOS['columbus.copy_template_from']
        copy_to = iconddir + '/COLUMBUS.template/'
        if os.path.exists(copy_to):
            shutil.rmtree(copy_to)
        shutil.copytree(copy_from, copy_to)


    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def check_Analytical_block(data, identifier, nstates, eMsg):
    iline = -1
    while True:
        iline += 1
        if iline == len(data):
            if eMsg:
                print('No matrix %s defined!' % (identifier))
            return False
        line = re.sub('#.*$', '', data[iline]).split()
        if line == []:
            continue
        ident = identifier.split()
        fits = True
        for i, el in enumerate(ident):
            if not el.lower() in line[i].lower():
                fits = False
                break
        if fits:
            break
    strings = data[iline + 1:iline + 1 + nstates]
    for i, el in enumerate(strings):
        a = el.strip().split(',')
        if len(a) < i + 1:
            if eMsg:
                print('%s matrix is not a lower triangular matrix with n=%i!' % (identifier, nstates))
            return False
    return True

# =================================================


def checktemplate_Analytical(filename, req_nstates, eMsg=True, dipolegrad=False):
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        if eMsg:
            print('Could not open %s' % (filename))
        return False

    # check whether first two lines are positive integers
    try:
        natom = int(data[0])
        nstates = int(data[1])
    except ValueError:
        if eMsg:
            print('First two lines must contain natom and nstates!')
        return False
    if natom < 1 or nstates < 1:
        if eMsg:
            print('natom and nstates must be positive!')
        return False
    if nstates != req_nstates:
        if eMsg:
            print('Template file is for %i states!' % (nstates))
        return False

    # check the next natom lines
    variables = set()
    for i in range(2, 2 + natom):
        line = data[i]
        match = re.match(r'\s*[a-zA-Z]*\\s+[a-zA-Z0][a-zA-Z0-9_]*\\s+[a-zA-Z0][a-zA-Z0-9_]*\\s+[a-zA-Z0][a-zA-Z0-9_]*', line)
        if not match:
            if eMsg:
                print('Line %i malformatted!' % (i + 1))
            return False
        else:
            a = line.split()
            for j in range(3):
                match = re.match(r'\s*[a-zA-Z][a-zA-Z0-9_]*', a[j + 1])
                if match:
                    variables.add(a[j + 1])

    # check variable blocks
    iline = -1
    while True:
        iline += 1
        if iline == len(data):
            break
        line = re.sub('#.*$', '', data[iline]).split()
        if line == []:
            continue
        if 'variables' in line[0].lower():
            while True:
                iline += 1
                if iline == len(data):
                    if eMsg:
                        print('Non-terminated variables block!')
                    return False
                line = re.sub('#.*$', '', data[iline]).split()
                if line == []:
                    continue
                if 'end' in line[0].lower():
                    break
                match = re.match(r'[a-zA-Z][a-zA-Z0-9_]*', line[0])
                if not match:
                    if eMsg:
                        print('Invalid variable name: %s' % (line[0]))
                    return False
                try:
                    a = float(line[1])
                except ValueError:
                    if eMsg:
                        print('Non-numeric value for variable %s' % (line[0]))
                    return False
                except IndexError:
                    if eMsg:
                        print('No value for variable %s' % (line[0]))
                    return False

    # check hamiltonian block
    line = 'hamiltonian'
    a = check_Analytical_block(data, line, nstates, eMsg)
    if not a:
        return False

    # check derivatives of each variable
    for v in variables:
        line = 'derivatives %s' % (v)
        a = check_Analytical_block(data, line, nstates, eMsg)
        if not a:
            return False

    # check dipole derivatives of each variable
    # can be zero, hence commented out
    # if dipolegrad:
        # for v in variables:
            # for p in range(3):
            #line='dipolederivatives %i %s' % (p+1,v)
            # a=check_Analytical_block(data,line,nstates,eMsg)
            # if not a:
            # return False

    return True

# =================================================


def get_Analytical(INFOS):

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('Analytical PES Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    if os.path.isfile('Analytical.template'):
        if checktemplate_Analytical('Analytical.template', INFOS['nstates'], eMsg=False, dipolegrad=INFOS['dipolegrad']):
            print('Valid file "Analytical.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['analytical.template'] = 'Analytical.template'
    if 'analytical.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_Analytical(filename, INFOS['nstates'], dipolegrad=INFOS['dipolegrad']):
                break
        INFOS['analytical.template'] = filename
    print('')

    return INFOS

# =================================================


def prepare_Analytical(INFOS, iconddir):
    # copy Analytical.template

    # copy MOs and template
    cpfrom = INFOS['analytical.template']
    cpto = '%s/Analytical.template' % (iconddir)
    shutil.copy(cpfrom, cpto)

    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_LVC(INFOS):
    # TODO: rename files for consistency with other interfaces

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('LVC Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    if os.path.isfile('LVC.template'):
        print('File "LVC.template" detected. ')
        usethisone = question('Use this template file?', bool, True)
        if usethisone:
            INFOS['LVC.template'] = 'LVC.template'
    if 'LVC.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue

            break
        INFOS['LVC.template'] = filename
    print('')

    return INFOS

# =================================================


def prepare_LVC(INFOS, iconddir):
    # copy LVC.template

    # copy MOs and template
    cpfrom = INFOS['LVC.template']
    cpto = '%s/LVC.template' % (iconddir)
    shutil.copy(cpfrom, cpto)


    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def check_MOLCAS_qmmm(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    for line in data:
        if 'qmmm' in line.lower():
            return True
    return False

# =================================================


def checktemplate_MOLCAS(filename, INFOS):
    necessary = ['basis', 'ras2', 'nactel', 'inactive']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for l in data:
            if i in re.sub('#.*$', '', l):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    roots_there = False
    for l in data:
        l = re.sub('#.*$', '', l).lower().split()
        if len(l) == 0:
            continue
        if 'roots' in l[0]:
            roots_there = True
    if not roots_there:
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            valid = []
            for l in data:
                if 'spin' in re.sub('#.*$', '', l).lower():
                    f = l.split()
                    if int(f[1]) == mult + 1:
                        valid.append(True)
                        break
            else:
                valid.append(False)
    if not all(valid):
        string = 'The template %s seems to be incomplete! It should contain the keyword "spin" for ' % (filename)
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            string += '%s, ' % (IToMult[mult + 1])
        string = string[:-2] + '!'
        print(string)
        return False
    return True

# =================================================


def get_MOLCAS(INFOS):
    '''This routine asks for all questions specific to MOLPRO:
    - path to molpro
    - scratch directory
    - MOLPRO.template
    - wf.init
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('MOLCAS Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Path to MOLCAS') + '\n')
    path = os.getenv('MOLCAS')
    # path=os.path.expanduser(os.path.expandvars(path))
    if path == '':
        path = None
    else:
        path = '$MOLCAS/'
        # print('Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path))
        # if question('Do you want to use this MOLCAS installation?',bool,True):
        # INFOS['molcas']=path
        # if 'molcas' not in INFOS:
    print('\nPlease specify path to MOLCAS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['molcas'] = question('Path to MOLCAS:', str, path)
    print('')


    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    print('{:-^60}'.format('MOLCAS input template file') + '\n')
    print('''Please specify the path to the MOLCAS.template file. This file must contain the following settings:

basis <Basis set>
ras2 <Number of active orbitals>
nactel <Number of active electrons>
inactive <Number of doubly occupied orbitals>
roots <Number of roots for state-averaging>

The MOLCAS interface will generate the appropriate MOLCAS input automatically.
''')
    if os.path.isfile('MOLCAS.template'):
        if checktemplate_MOLCAS('MOLCAS.template', INFOS):
            print('Valid file "MOLCAS.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['molcas.template'] = 'MOLCAS.template'
    if 'molcas.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_MOLCAS(filename, INFOS):
                break
        INFOS['molcas.template'] = filename
    print('')


    # QMMM
    if check_MOLCAS_qmmm(INFOS['molcas.template']):
        print('{:-^60}'.format('MOLCAS+TINKER QM/MM setup') + '\n')
        print('Your template specifies a QM/MM calculation. Please specify the path to TINKER.')
        path = os.getenv('TINKER')
        if path == '':
            path = None
        else:
            path = '$TINKER/'
        print('\nPlease specify path to TINKER bin/ directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        INFOS['tinker'] = question('Path to TINKER/bin:', str, path)
        print('Please give the key and connection table files.')
        while True:
            filename = question('Key file:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
            else:
                break
        INFOS['MOLCAS.fffile'] = filename
        while True:
            filename = question('Connection table file:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
            else:
                break
        INFOS['MOLCAS.ctfile'] = filename


    print('{:-^60}'.format('Initial wavefunction: MO Guess') + '\n')
    print('''Please specify the path to a MOLCAS JobIph file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    string = 'Do you have initial wavefunction files for '
    for mult, state in enumerate(INFOS['states']):
        if state <= 0:
            continue
        string += '%s, ' % (IToMult[mult + 1])
    string = string[:-2] + '?'
    if question(string, bool, True):
        while True:
            jobiph_or_rasorb = question('JobIph files (1) or RasOrb files (2)?', int)[0]
            if jobiph_or_rasorb in [1, 2]:
                break
        INFOS['molcas.jobiph_or_rasorb'] = jobiph_or_rasorb
        INFOS['molcas.guess'] = {}
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            while True:
                if jobiph_or_rasorb == 1:
                    guess_file = 'MOLCAS.%i.JobIph.init' % (mult + 1)
                else:
                    guess_file = 'MOLCAS.%i.RasOrb.init' % (mult + 1)
                filename = question('Initial wavefunction file for %ss:' % (IToMult[mult + 1]), str, guess_file)
                if os.path.isfile(filename):
                    INFOS['molcas.guess'][mult + 1] = filename
                    break
                else:
                    print('File not found!')
    else:
        print('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')
        time.sleep(2)
        INFOS['molcas.guess'] = {}


    print('{:-^60}'.format('MOLCAS Ressource usage') + '\n')
    print('''Please specify the amount of memory available to MOLCAS (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
''')
    INFOS['molcas.mem'] = abs(question('MOLCAS memory:', int)[0])
    print('''Please specify the number of CPUs to be used by EACH calculation.
''')
    INFOS['molcas.ncpu'] = abs(question('Number of CPUs:', int)[0])




    # Ionization
    # need_wfoverlap=False
    # print(centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)
    # if 'ion' in INFOS and INFOS['ion']:
    # need_wfoverlap=True

    # wfoverlap
    if 'wfoverlap' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wfoverlap code setup') + '\n')
        if 'ion' in INFOS and INFOS['ion']:
            print('Dyson norms requested.')
        INFOS['molcas.wfpath'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')
        # TODO not asked for: numfrozcore, numocc


    return INFOS

# =================================================


def prepare_MOLCAS(INFOS, iconddir):
    # write MOLCAS.resources
    try:
        sh2cas = open('%s/MOLCAS.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareMOLCAS, iconddir=%s' % (iconddir))
        quit(1)
    project = 'MOLCAS'
    string = '''molcas %s
scratchdir %s/%s/
savedir %s/%s/restart
memory %i
ncpu %i
project %s''' % (INFOS['molcas'],
                 INFOS['scratchdir'],
                 iconddir,
                 INFOS['copydir'],
                 iconddir,
                 INFOS['molcas.mem'],
                 INFOS['molcas.ncpu'],
                 project)
    if 'wfoverlap' in INFOS['needed']:
        string += '\nwfoverlap %s\n' % INFOS['molcas.wfpath']
    if 'tinker' in INFOS:
        string += 'tinker %s' % (INFOS['tinker'])
    sh2cas.write(string)
    sh2cas.close()

    # copy MOs and template
    cpfrom = INFOS['molcas.template']
    cpto = '%s/MOLCAS.template' % (iconddir)
    shutil.copy(cpfrom, cpto)
    if not INFOS['molcas.guess'] == {}:
        for i in INFOS['molcas.guess']:
            if INFOS['molcas.jobiph_or_rasorb'] == 1:
                cpfrom = INFOS['molcas.guess'][i]
                cpto = '%s/%s.%i.JobIph.init' % (iconddir, project, i)
            else:
                cpfrom = INFOS['molcas.guess'][i]
                cpto = '%s/%s.%i.RasOrb.init' % (iconddir, project, i)
            shutil.copy(cpfrom, cpto)

    if 'MOLCAS.fffile' in INFOS:
        cpfrom1 = INFOS['MOLCAS.fffile']
        cpto1 = '%s/MOLCAS.qmmm.key' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    if 'MOLCAS.ctfile' in INFOS:
        cpfrom1 = INFOS['MOLCAS.ctfile']
        cpto1 = '%s/MOLCAS.qmmm.table' % (iconddir)
        shutil.copy(cpfrom1, cpto1)
    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def checktemplate_AMS(filename, INFOS):
    necessary = ['basis', 'functional', 'charge']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for l in data:
            line = l.lower().split()
            if len(line) == 0:
                continue
            line = line[0]
            if i == re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    return True

# =================================================


def get_AMS(INFOS):
    '''This routine asks for all questions specific to AMS:
    - path to AMS
    - scratch directory
    - AMS-ADF.template
    - TAPE21
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||{:^80}||\n'.format("AMS Interface setup")
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Path to AMS'))
    path = os.getenv('AMSHOME')
    if path:
        path = '$AMSHOME/'
    amsbashrc = question('Setup from amsbashrc.sh file?', bool, True)
    if amsbashrc:
        if path:
            path = '$AMSHOME/amsbashrc.sh'
        print('\nPlease specify path to the amsbashrc.sh file (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        path = question('Path to amsbashrc.sh file:', str, path)
        INFOS['amsbashrc'] = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        print('Will use amsbashrc= %s' % INFOS['amsbashrc'])
        INFOS['ams'] = '$AMSHOME'
        INFOS['scmlicense'] = '$SCMLICENSE'
        print('')
    else:
        print('\nPlease specify path to AMS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        INFOS['ams'] = question('Path to AMS:', str, path)
        print('')
        print('{:-^60}'.format('Path to AMS license file') + '\n')
        path = os.getenv('SCMLICENSE')
        # path=os.path.expanduser(os.path.expandvars(path))
        if path == '':
            path = None
        else:
            path = '$SCMLICENSE'
        print('\nPlease specify path to AMS license.txt\n')
        INFOS['scmlicense'] = question('Path to license:', str, path)
        print('')


    # scratch
    print('{:^60}\n'.format('Scratch directory'))
    print('Please specify an appropriate scratch directory. This will be used to run the AMS calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    # template file
    print('{:-^60}'.format('AMS input template file') + '\n')
    print('''Please specify the path to the AMS-ADF.template file. This file must contain the following keywords:

basis <basis>
functional <type> <name>
charge <x> [ <x2> [ <x3> ...] ]

The AMS interface will generate the appropriate AMS input automatically.
''')
    if os.path.isfile('AMS-ADF.template'):
        if checktemplate_AMS('AMS-ADF.template', INFOS):
            print('Valid file "AMS-ADF.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['AMS-ADF.template'] = 'AMS-ADF.template'
    if 'AMS-ADF.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_AMS(filename, INFOS):
                break
        INFOS['AMS-ADF.template'] = filename
    print('')


    # initial MOs
    print('{:-^60}'.format('Initial restart: MO Guess') + '\n')
    print('''Please specify the path to an rkf file containing suitable starting MOs for the AMS calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    if question('Do you have a restart file?', bool, True):
        if True:
            filename = question('Restart file:', str, 'AMS.t21.init')
            INFOS['ams.guess'] = filename
    else:
        print('WARNING: Remember that the calculations may take longer without an initial guess for the MOs.')
        # time.sleep(2)
        INFOS['ams.guess'] = {}



    # Resources
    print('{:-^60}'.format('AMS Ressource usage') + '\n')
    print('''Please specify the number of CPUs to be used by EACH calculation.
''')
    INFOS['ams.ncpu'] = abs(question('Number of CPUs:', int)[0])

    if INFOS['ams.ncpu'] > 1:
        print('''Please specify how well your job will parallelize.
A value of 0 means that running in parallel will not make the calculation faster, a value of 1 means that the speedup scales perfectly with the number of cores.
Typical values for AMS are 0.90-0.98 for LDA/GGA functionals and 0.50-0.80 for hybrids (better if RIHartreeFock is used).''')
        INFOS['ams.scaling'] = min(1.0, max(0.0, question('Parallel scaling:', float, [0.8])[0]))
    else:
        INFOS['ams.scaling'] = 0.9


    # Ionization
    # need_wfoverlap=False
    # print(centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)
    # if 'ion' in INFOS and INFOS['ion']:
        # need_wfoverlap=True
    # if Couplings[INFOS['coupling']]['name']=='overlap':
        # need_wfoverlap=True


    # Overlaps
    # if need_wfoverlap:
    if 'wfoverlap' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wfoverlap code setup') + '\n')
        INFOS['ams.wfoverlap'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')
        print('')
        print('''State threshold for choosing determinants to include in the overlaps''')
        print('''For hybrids (and without TDA) one should consider that the eigenvector X may have a norm larger than 1''')
        INFOS['ams.ciothres'] = question('Threshold:', float, [0.998])[0]
        print('')
        INFOS['ams.mem'] = question('Memory for wfoverlap (MB):', int, [1000])[0]
        # TODO not asked: numfrozcore and numocc

        # print('Please state the number of core orbitals you wish to freeze for the overlaps (recommended to use for at least the 1s orbital and a negative number uses default values)?')
        # print('A value of -1 will use the defaults used by AMS for a small frozen core and 0 will turn off the use of frozen cores')
        #INFOS['frozcore_number']=question('How many orbital to freeze?',int,[-1])[0]


    # TheoDORE
    theodore_spelling = ['Om',
                         'PRNTO',
                         'Z_HE', 'S_HE', 'RMSeh',
                         'POSi', 'POSf', 'POS',
                         'PRi', 'PRf', 'PR', 'PRh',
                         'CT', 'CT2', 'CTnt',
                         'MC', 'LC', 'MLCT', 'LMCT', 'LLCT',
                         'DEL', 'COH', 'COHh']
    print('\n' + '{:-^60}'.format('Wave function analysis by TheoDORE') + '\n')
    # INFOS['theodore']=question('TheoDORE analysis?',bool,False)
    if 'theodore' in INFOS['needed']:

        INFOS['ams.theodore'] = question('Path to TheoDORE directory:', str, '$THEODIR')
        print('')

        print('Please give a list of the properties to calculate by TheoDORE.\nPossible properties:')
        string = ''
        for i, p in enumerate(theodore_spelling):
            string += '%s ' % (p)
            if (i + 1) % 8 == 0:
                string += '\n'
        print(string)
        li = question('TheoDORE properties:', str, 'Om  PRNTO  S_HE  Z_HE  RMSeh')
        if '[' in li:
            INFOS['theodore.prop'] = ast.literal_eval(li)
        else:
            INFOS['theodore.prop'] = li.split()
        print('')

        print('Please give a list of the fragments used for TheoDORE analysis.')
        print('You can use the list-of-lists from dens_ana.in')
        print('Alternatively, enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
        INFOS['theodore.frag'] = []
        while True:
            li = question('TheoDORE fragment:', str, 'end')
            if 'end' in li.lower():
                break
            if '[' in li:
                try:
                    INFOS['theodore.frag'] = ast.literal_eval(li)
                    break
                except ValueError:
                    continue
            f = [int(i) for i in li.split()]
            INFOS['theodore.frag'].append(f)
        INFOS['theodore.count'] = len(INFOS['theodore.prop']) + len(INFOS['theodore.frag'])**2
    else:
        INFOS['theodore'] = False


    return INFOS

# =================================================


def prepare_AMS(INFOS, iconddir):
    # write AMS-ADF.resources
    try:
        sh2cas = open('%s/AMS-ADF.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareAMS, iconddir=%s' % (iconddir))
        quit(1)
#  project='AMS'
    string = 'amshome %s\nscmlicense %s\nscratchdir %s/%s/\nsavedir %s/%s/restart\nncpu %i\nschedule_scaling %f\n' % (INFOS['ams'], INFOS['scmlicense'], INFOS['scratchdir'], iconddir, INFOS['copydir'], iconddir, INFOS['ams.ncpu'], INFOS['ams.scaling'])
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfoverlap %s\nwfthres %f\n' % (INFOS['ams.wfoverlap'], INFOS['ams.ciothres'])
        string += 'memory %i\n' % (INFOS['ams.mem'])
        #string+='numfrozcore %i\n' %(INFOS['frozcore_number'])
    else:
        string += 'nooverlap\n'
    if INFOS['theodore']:
        string += 'theodir %s\n' % (INFOS['ams.theodore'])
        string += 'theodore_prop %s\n' % (INFOS['theodore.prop'])
        string += 'theodore_fragment %s\n' % (INFOS['theodore.frag'])
    if 'AMS.fffile' in INFOS:
        string += 'qmmm_ff_file AMS.qmmm.ff\n'
    if 'AMS.ctfile' in INFOS:
        string += 'qmmm_table AMS.qmmm.table\n'
    sh2cas.write(string)
    sh2cas.close()

    # copy MOs and template
    cpfrom = INFOS['AMS-ADF.template']
    cpto = '%s/AMS-ADF.template' % (iconddir)
    shutil.copy(cpfrom, cpto)

    if INFOS['ams.guess']:
        cpfrom1 = INFOS['ams.guess']
        cpto1 = '%s/AMS.t21_init' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    if 'AMS.fffile' in INFOS:
        cpfrom1 = INFOS['AMS.fffile']
        cpto1 = '%s/AMS.qmmm.ff' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    if 'AMS.ctfile' in INFOS:
        cpfrom1 = INFOS['AMS.ctfile']
        cpto1 = '%s/AMS.qmmm.table' % (iconddir)
        shutil.copy(cpfrom1, cpto1)


    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def checktemplate_RICC2(filename, INFOS):
    necessary = ['basis']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for li in data:
            line = li.lower()
            if i in re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    return True

# =================================================


def get_RICC2(INFOS):
    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('Turbomole RICC2 Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Path to TURBOMOLE') + '\n')
    path = os.getenv('TURBODIR')
    if path == '':
        path = None
    else:
        path = '$TURBODIR/'
    print('\nPlease specify path to TURBOMOLE directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['turbomole'] = question('Path to TURBOMOLE:', str, path)
    print('')

    # print(centerstring('Path to ORCA',60,'-')+'\n')
    # path=os.getenv('ORCADIR')
    # if path=='':
    # path=None
    # else:
    # path='$ORCADIR/'
    # print('\nPlease specify path to ORCA directory (SHELL variables and ~ can be used, will be expanded when interface is started).\nORCA is necessary for the calculation of spin-orbit couplings with ricc2.\n')
    #INFOS['orca']=question('Path to ORCA:',str,path)
    # print('')


    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    print('{:-^60}'.format('RICC2 input template file') + '\n')
    print('''Please specify the path to the RICC2.template file. This file must contain the following settings:

basis <Basis set>

In addition, it can contain the following:

auxbasis <Basis set>
charge <integer>
method <"ADC(2)" or "CC2">                      # only ADC(2) can calculate spin-orbit couplings
frozen <number of frozen core orbitals>
spin-scaling <"none", "SCS", or "SOS">
douglas-kroll                                   # DKH is only used if this keyword is given

''')
    if os.path.isfile('RICC2.template'):
        if checktemplate_RICC2('RICC2.template', INFOS):
            print('Valid file "RICC2.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['ricc2.template'] = 'RICC2.template'
    if 'ricc2.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_RICC2(filename, INFOS):
                break
        INFOS['ricc2.template'] = filename
    print('')


    print('{:-^60}'.format('Initial wavefunction: MO Guess') + '\n')
    print('''Please specify the path to a Turbomole "mos" file containing suitable starting MOs for the calculation. Please note that this script cannot check whether the file and the input template are consistent!
''')
    string = 'Do you have an initial orbitals file?'
    if question(string, bool, True):
        while True:
            guess_file = 'mos'
            filename = question('Initial wavefunction file:', str, guess_file)
            if os.path.isfile(filename):
                INFOS['ricc2.guess'] = filename
                break
            else:
                print('File not found!')
    else:
        INFOS['ricc2.guess'] = []


    print('{:-^60}'.format('RICC2 Ressource usage') + '\n')
    print('''Please specify the amount of memory available to Turbomole.
''')
    INFOS['ricc2.mem'] = abs(question('RICC2 memory (MB):', int, [1000])[0])
    print('''Please specify the number of CPUs to be used by EACH trajectory.
''')
    INFOS['ricc2.ncpu'] = abs(question('Number of CPUs:', int, [1])[0])

    # if INFOS['laser']:
    # guess=2
    # else:
    guess = 1
    a = ['', '(recommended)']
    print('For response-based methods like CC2 and ADC(2), dipole moments and transition dipole moments carry a significant computational cost. In order to speed up calculations, the interface can restrict the calculation of these properties.')
    print('''Choose one of the following dipolelevels:
0       only calculate dipole moments which are for free                                %s
1       additionally, calculate transition dipole moments involving the ground state    %s
2       calculate all elements possible with the method                                 %s
''' % (a[guess == 0], a[guess == 1], a[guess == 2]))
    INFOS['ricc2.dipolelevel'] = question('Dipole level:', int, [guess])[0]


    if 'wfoverlap' in INFOS['needed']:
        print('Wavefunction overlaps requested.')
        INFOS['ricc2.wfpath'] = question('Path to wfoverlap executable:', str, '$SHARC/wfoverlap.x')
        print('')
        print('''Give threshold for choosing determinants to include in the overlaps''')
        INFOS['ricc2.wfthres'] = question('Threshold:', float, [0.998])[0]



    # TheoDORE
    theodore_spelling = ['Om',
                         'PRNTO',
                         'Z_HE', 'S_HE', 'RMSeh',
                         'POSi', 'POSf', 'POS',
                         'PRi', 'PRf', 'PR', 'PRh',
                         'CT', 'CT2', 'CTnt',
                         'MC', 'LC', 'MLCT', 'LMCT', 'LLCT',
                         'DEL', 'COH', 'COHh']
    #INFOS['theodore']=question('TheoDORE analysis?',bool,False)
    if 'theodore' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wave function analysis by TheoDORE') + '\n')

        INFOS['ricc2.theodore'] = question('Path to TheoDORE directory:', str, '$THEODIR')
        print('')

        print('Please give a list of the properties to calculate by TheoDORE.\nPossible properties:')
        string = ''
        for i, p in enumerate(theodore_spelling):
            string += '%s ' % (p)
            if (i + 1) % 8 == 0:
                string += '\n'
        print(string)
        l = question('TheoDORE properties:', str, 'Om  PRNTO  S_HE  Z_HE  RMSeh')
        if '[' in l:
            INFOS['theodore.prop'] = ast.literal_eval(l)
        else:
            INFOS['theodore.prop'] = l.split()
        print('')

        print('Please give a list of the fragments used for TheoDORE analysis.')
        print('You can use the list-of-lists from dens_ana.in')
        print('Alternatively, enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
        INFOS['theodore.frag'] = []
        while True:
            l = question('TheoDORE fragment:', str, 'end')
            if 'end' in l.lower():
                break
            if '[' in l:
                try:
                    INFOS['theodore.frag'] = ast.literal_eval(l)
                    break
                except ValueError:
                    continue
            f = [int(i) for i in l.split()]
            INFOS['theodore.frag'].append(f)
        INFOS['theodore.count'] = len(INFOS['theodore.prop']) + len(INFOS['theodore.frag'])**2

    return INFOS

# =================================================


def prepare_RICC2(INFOS, iconddir):
    # write RICC2.resources
    try:
        sh2cc2 = open('%s/RICC2.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepare_RICC2, iconddir=%s' % (iconddir))
        quit(1)
    string = '''turbodir %s
scratchdir %s/%s
memory %i
ncpu %i
dipolelevel %i
''' % (INFOS['turbomole'],
       INFOS['scratchdir'],
       iconddir,
       INFOS['ricc2.mem'],
       INFOS['ricc2.ncpu'],
       INFOS['ricc2.dipolelevel'])
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfoverlap %s\n' % (INFOS['ricc2.wfpath'])
        string += 'wfthres %f\n' % (INFOS['ricc2.wfthres'])
    else:
        string += 'nooverlap\n'
    if 'theodore' in INFOS['needed']:
        string += 'theodir %s\n' % (INFOS['ricc2.theodore'])
        string += 'theodore_prop %s\n' % (INFOS['theodore.prop'])
        string += 'theodore_fragment %s\n' % (INFOS['theodore.frag'])

    sh2cc2.write(string)
    sh2cc2.close()

    # copy MOs and template
    cpfrom = INFOS['ricc2.template']
    cpto = '%s/RICC2.template' % (iconddir)
    shutil.copy(cpfrom, cpto)
    if INFOS['ricc2.guess']:
        cpfrom1 = INFOS['ricc2.guess']
        cpto1 = '%s/mos.init' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    return


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def checktemplate_GAUSSIAN(filename, INFOS):
    necessary = ['basis', 'functional', 'charge']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for l in data:
            line = l.lower().split()
            if len(line) == 0:
                continue
            line = line[0]
            if i == re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    return True

# =================================================


def get_GAUSSIAN(INFOS):
    '''This routine asks for all questions specific to GAUSSIAN:
    - path to GAUSSIAN
    - scratch directory
    - GAUSSIAN.template
    - TAPE21
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('GAUSSIAN Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Path to GAUSSIAN') + '\n')
    tries = ['g16root', 'g09root', 'g03root']
    for i in tries:
        path = os.getenv(i)
        if path:
            path = '$%s/' % i
            break
    #gaussianprofile=question('Setup from gaussian.profile file?',bool,True)
    # if gaussianprofile:
        # if path:
            #path='%s/gaussian.profile' % path
        # print('\nPlease specify path to the gaussian.profile file (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        #path=question('Path to GAUSSIAN:',str,path)
        # INFOS['gaussianprofile']=os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        # print('Will use gaussianprofile= %s' % INFOS['gaussianprofile'])
        # INFOS['gaussian']='$GAUSSIANHOME'
        # print('')
    # else:
    print('\nPlease specify path to GAUSSIAN directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['groot'] = question('Path to GAUSSIAN:', str, path)
    print('')


    # scratch
    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to run the GAUSSIAN calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    # template file
    print('{:-^60}'.format('GAUSSIAN input template file') + '\n')
    print('''Please specify the path to the GAUSSIAN.template file. This file must contain the following keywords:

basis <basis>
functional <type> <name>
charge <x> [ <x2> [ <x3> ...] ]

The GAUSSIAN interface will generate the appropriate GAUSSIAN input automatically.
''')
    if os.path.isfile('GAUSSIAN.template'):
        if checktemplate_GAUSSIAN('GAUSSIAN.template', INFOS):
            print('Valid file "GAUSSIAN.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['GAUSSIAN.template'] = 'GAUSSIAN.template'
    if 'GAUSSIAN.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_GAUSSIAN(filename, INFOS):
                break
        INFOS['GAUSSIAN.template'] = filename
    print('')



    # initial MOs
    print('{:-^60}'.format('Initial restart: MO Guess') + '\n')
    print('''Please specify the path to an GAUSSIAN chk file containing suitable starting MOs for the GAUSSIAN calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    if question('Do you have a restart file?', bool, True):
        if True:
            while True:
                filename = question('Restart file:', str, 'GAUSSIAN.chk.init')
                if os.path.isfile(filename):
                    INFOS['gaussian.guess'] = filename
                    break
                else:
                    print('Could not find file "%s"!' % (filename))
    else:
        INFOS['gaussian.guess'] = {}


    # Resources
    print('{:-^60}'.format('GAUSSIAN Ressource usage') + '\n')
    print('''Please specify the number of CPUs to be used by EACH calculation.
''')
    INFOS['gaussian.ncpu'] = abs(question('Number of CPUs:', int)[0])

    if INFOS['gaussian.ncpu'] > 1:
        print('''Please specify how well your job will parallelize.
A value of 0 means that running in parallel will not make the calculation faster, a value of 1 means that the speedup scales perfectly with the number of cores.
Typical values for GAUSSIAN are 0.90-0.98.''')
        INFOS['gaussian.scaling'] = min(1.0, max(0.0, question('Parallel scaling:', float, [0.9])[0]))
    else:
        INFOS['gaussian.scaling'] = 0.9

    INFOS['gaussian.mem'] = question('Memory (MB):', int, [1000])[0]

    # Ionization
    # print('\n'+centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)
    # if INFOS['ion']:
    if 'wfoverlap' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wfoverlap code setup') + '\n')
        INFOS['gaussian.wfoverlap'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')
        print('')
        print('State threshold for choosing determinants to include in the overlaps')
        print('For hybrids without TDA one should consider that the eigenvector X may have a norm larger than 1')
        INFOS['gaussian.ciothres'] = question('Threshold:', float, [0.998])[0]
        print('')
        # TODO not asked: numfrozcore and numocc

        # print('Please state the number of core orbitals you wish to freeze for the overlaps (recommended to use for at least the 1s orbital and a negative number uses default values)?')
        # print('A value of -1 will use the defaults used by GAUSSIAN for a small frozen core and 0 will turn off the use of frozen cores')
        #INFOS['frozcore_number']=question('How many orbital to freeze?',int,[-1])[0]


    # TheoDORE
    theodore_spelling = ['Om',
                         'PRNTO',
                         'Z_HE', 'S_HE', 'RMSeh',
                         'POSi', 'POSf', 'POS',
                         'PRi', 'PRf', 'PR', 'PRh',
                         'CT', 'CT2', 'CTnt',
                         'MC', 'LC', 'MLCT', 'LMCT', 'LLCT',
                         'DEL', 'COH', 'COHh']
    #INFOS['theodore']=question('TheoDORE analysis?',bool,False)
    if 'theodore' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wave function analysis by TheoDORE') + '\n')

        INFOS['gaussian.theodore'] = question('Path to TheoDORE directory:', str, '$THEODIR')
        print('')

        print('Please give a list of the properties to calculate by TheoDORE.\nPossible properties:')
        string = ''
        for i, p in enumerate(theodore_spelling):
            string += '%s ' % (p)
            if (i + 1) % 8 == 0:
                string += '\n'
        print(string)
        l = question('TheoDORE properties:', str, 'Om  PRNTO  S_HE  Z_HE  RMSeh')
        if '[' in l:
            INFOS['theodore.prop'] = ast.literal_eval(l)
        else:
            INFOS['theodore.prop'] = l.split()
        print('')

        print('Please give a list of the fragments used for TheoDORE analysis.')
        print('You can use the list-of-lists from dens_ana.in')
        print('Alternatively, enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
        if qmmm_job(INFOS['GAUSSIAN.template'], INFOS):
            print('You should only include the atom numbers of QM and link atoms.')
        INFOS['theodore.frag'] = []
        while True:
            l = question('TheoDORE fragment:', str, 'end')
            if 'end' in l.lower():
                break
            if '[' in l:
                try:
                    INFOS['theodore.frag'] = ast.literal_eval(l)
                    break
                except ValueError:
                    continue
            f = [int(i) for i in l.split()]
            INFOS['theodore.frag'].append(f)
        INFOS['theodore.count'] = len(INFOS['theodore.prop']) + len(INFOS['theodore.frag'])**2


    return INFOS

# =================================================


def prepare_GAUSSIAN(INFOS, iconddir):
    # write GAUSSIAN.resources
    try:
        sh2cas = open('%s/GAUSSIAN.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareGAUSSIAN, iconddir=%s' % (iconddir))
        quit(1)
#  project='GAUSSIAN'
    string = 'groot %s\nscratchdir %s/%s/\nsavedir %s/%s/restart\nncpu %i\nschedule_scaling %f\n' % (INFOS['groot'], INFOS['scratchdir'], iconddir, INFOS['scratchdir'], iconddir, INFOS['gaussian.ncpu'], INFOS['gaussian.scaling'])
    string += 'memory %i\n' % (INFOS['gaussian.mem'])
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfoverlap %s\nwfthres %f\n' % (INFOS['gaussian.wfoverlap'], INFOS['gaussian.ciothres'])
        #string+='numfrozcore %i\n' %(INFOS['frozcore_number'])
    else:
        string += 'nooverlap\n'
    if 'theodore' in INFOS['needed']:
        string += 'theodir %s\n' % (INFOS['gaussian.theodore'])
        string += 'theodore_prop %s\n' % (INFOS['theodore.prop'])
        string += 'theodore_fragment %s\n' % (INFOS['theodore.frag'])
    sh2cas.write(string)
    sh2cas.close()

    # copy MOs and template
    cpfrom = INFOS['GAUSSIAN.template']
    cpto = '%s/GAUSSIAN.template' % (iconddir)
    shutil.copy(cpfrom, cpto)

    if INFOS['gaussian.guess']:
        cpfrom1 = INFOS['gaussian.guess']
        cpto1 = '%s/GAUSSIAN.chk.init' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    return


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def checktemplate_ORCA(filename, INFOS):
    necessary = ['basis', 'functional', 'charge']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for l in data:
            line = l.lower().split()
            if len(line) == 0:
                continue
            line = line[0]
            if i == re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    return True

# =================================================


def qmmm_job(filename, INFOS):
    necessary = ['qmmm']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for l in data:
            line = l.lower().split()
            if len(line) == 0:
                continue
            line = line[0]
            if i == re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        return False
    return True

# =================================================


def get_ORCA(INFOS):
    '''This routine asks for all questions specific to ORCA:
    - path to ORCA
    - scratch directory
    - ORCA.template
    - initial gbw file
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('ORCA Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Path to ORCA') + '\n')
    print('Using same ORCA installation as for the optimizer...')
    # print('\nPlease specify path to ORCA directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['orcadir'] = INFOS['orca']
    print('')




    # scratch
    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to run the ORCA calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    # template file
    print('{:-^60}'.format('ORCA input template file') + '\n')
    print('''Please specify the path to the ORCA.template file. This file must contain the following keywords:

basis <basis>
functional <type> <name>
charge <x> [ <x2> [ <x3> ...] ]

The ORCA interface will generate the appropriate ORCA input automatically.
''')
    if os.path.isfile('ORCA.template'):
        if checktemplate_ORCA('ORCA.template', INFOS):
            print('Valid file "ORCA.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['ORCA.template'] = 'ORCA.template'
    if 'ORCA.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_ORCA(filename, INFOS):
                break
        INFOS['ORCA.template'] = filename
    print('')


    # QMMM
    if qmmm_job(INFOS['ORCA.template'], INFOS):
        print('{:-^60}'.format('ORCA+TINKER QM/MM setup') + '\n')
        print('Your template specifies a QM/MM calculation. Please specify the path to TINKER.')
        path = os.getenv('TINKER')
        if path == '':
            path = None
        else:
            path = '$TINKER/'
        print('\nPlease specify path to TINKER bin/ directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        INFOS['tinker'] = question('Path to TINKER/bin:', str, path)
        while True:
            filename = question('Force field file:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
            else:
                break
        INFOS['ORCA.fffile'] = filename
        while True:
            filename = question('Connection table file:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
            else:
                break
        INFOS['ORCA.ctfile'] = filename


    # initial MOs
    print('{:-^60}'.format('Initial restart: MO Guess') + '\n')
    print('''Please specify the path to an ORCA gbw file containing suitable starting MOs for the ORCA calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    if question('Do you have a restart file?', bool, True):
        if True:
            while True:
                filename = question('Restart file:', str, 'ORCA.gbw')
                if os.path.isfile(filename):
                    INFOS['orca.guess'] = filename
                    break
                else:
                    print('Could not find file "%s"!' % (filename))
    else:
        INFOS['orca.guess'] = {}


    # Resources
    print('{:-^60}'.format('ORCA Ressource usage') + '\n')
    print('''Please specify the number of CPUs to be used by EACH calculation.
''')
    INFOS['orca.ncpu'] = abs(question('Number of CPUs:', int)[0])

    if INFOS['orca.ncpu'] > 1:
        print('''Please specify how well your job will parallelize.
A value of 0 means that running in parallel will not make the calculation faster, a value of 1 means that the speedup scales perfectly with the number of cores.''')
        INFOS['orca.scaling'] = min(1.0, max(0.0, question('Parallel scaling:', float, [0.8])[0]))
    else:
        INFOS['orca.scaling'] = 0.9
    INFOS['orca.mem'] = question('Memory (MB):', int, [1000])[0]


    # Ionization
    # print('\n'+centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)
    # if INFOS['ion']:
    if 'wfoverlap' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('WFoverlap setup') + '\n')
        INFOS['orca.wfoverlap'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')
        print('')
        print('State threshold for choosing determinants to include in the overlaps')
        print('For hybrids (and without TDA) one should consider that the eigenvector X may have a norm larger than 1')
        INFOS['orca.ciothres'] = question('Threshold:', float, [0.998])[0]
        print('')

        # PyQuante
        print('\n' + '{:-^60}'.format('PyQuante setup') + '\n')
        INFOS['orca.pyquante'] = question('Path to PyQuante lib directory:', str, '$PYQUANTE')
        print('')


    # TheoDORE
    theodore_spelling = ['Om',
                         'PRNTO',
                         'Z_HE', 'S_HE', 'RMSeh',
                         'POSi', 'POSf', 'POS',
                         'PRi', 'PRf', 'PR', 'PRh',
                         'CT', 'CT2', 'CTnt',
                         'MC', 'LC', 'MLCT', 'LMCT', 'LLCT',
                         'DEL', 'COH', 'COHh']
    #INFOS['theodore']=question('TheoDORE analysis?',bool,False)
    if 'theodore' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('Wave function analysis by TheoDORE') + '\n')

        INFOS['orca.theodore'] = question('Path to TheoDORE directory:', str, '$THEODIR')
        print('')

        print('Please give a list of the properties to calculate by TheoDORE.\nPossible properties:')
        string = ''
        for i, p in enumerate(theodore_spelling):
            string += '%s ' % (p)
            if (i + 1) % 8 == 0:
                string += '\n'
        print(string)
        l = question('TheoDORE properties:', str, 'Om  PRNTO  S_HE  Z_HE  RMSeh')
        if '[' in l:
            INFOS['theodore.prop'] = ast.literal_eval(l)
        else:
            INFOS['theodore.prop'] = l.split()
        print('')

        print('Please give a list of the fragments used for TheoDORE analysis.')
        print('You can use the list-of-lists from dens_ana.in')
        print('Alternatively, enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
        if qmmm_job(INFOS['ORCA.template'], INFOS):
            print('You should only include the atom numbers of QM and link atoms.')
        INFOS['theodore.frag'] = []
        while True:
            l = question('TheoDORE fragment:', str, 'end')
            if 'end' in l.lower():
                break
            if '[' in l:
                try:
                    INFOS['theodore.frag'] = ast.literal_eval(l)
                    break
                except ValueError:
                    continue
            f = [int(i) for i in l.split()]
            INFOS['theodore.frag'].append(f)
        INFOS['theodore.count'] = len(INFOS['theodore.prop']) + len(INFOS['theodore.frag'])**2
        if 'ORCA.ctfile' in INFOS:
            INFOS['theodore.count'] += 6



    return INFOS

# =================================================


def prepare_ORCA(INFOS, iconddir):
    # write ORCA.resources
    try:
        sh2cas = open('%s/ORCA.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareORCA, iconddir=%s' % (iconddir))
        quit(1)
#  project='ORCA'
    string = 'orcadir %s\nscratchdir %s/%s/\nsavedir %s/%s/restart\nncpu %i\nschedule_scaling %f\n' % (INFOS['orcadir'], INFOS['scratchdir'], iconddir, INFOS['copydir'], iconddir, INFOS['orca.ncpu'], INFOS['orca.scaling'])
    string += 'memory %i\n' % (INFOS['orca.mem'])
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfoverlap %s\nwfthres %f\npyquante %s\n' % (INFOS['orca.wfoverlap'], INFOS['orca.ciothres'], INFOS['orca.pyquante'])
        #string+='numfrozcore %i\n' %(INFOS['frozcore_number'])
    else:
        string += 'nooverlap\n'
    if 'theodore' in INFOS['needed']:
        string += 'theodir %s\n' % (INFOS['orca.theodore'])
        string += 'theodore_prop %s\n' % (INFOS['theodore.prop'])
        string += 'theodore_fragment %s\n' % (INFOS['theodore.frag'])
    if 'tinker' in INFOS:
        string += 'tinker %s\n' % (INFOS['tinker'])
    if 'ORCA.fffile' in INFOS:
        string += 'qmmm_ff_file ORCA.qmmm.ff\n'
    if 'ORCA.ctfile' in INFOS:
        string += 'qmmm_table ORCA.qmmm.table\n'
    sh2cas.write(string)
    sh2cas.close()

    # copy MOs and template
    cpfrom = INFOS['ORCA.template']
    cpto = '%s/ORCA.template' % (iconddir)
    shutil.copy(cpfrom, cpto)

    if INFOS['orca.guess']:
        cpfrom1 = INFOS['orca.guess']
        cpto1 = '%s/ORCA.gbw.init' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    if 'ORCA.fffile' in INFOS:
        cpfrom1 = INFOS['ORCA.fffile']
        cpto1 = '%s/ORCA.qmmm.ff' % (iconddir)
        shutil.copy(cpfrom1, cpto1)

    if 'ORCA.ctfile' in INFOS:
        cpfrom1 = INFOS['ORCA.ctfile']
        cpto1 = '%s/ORCA.qmmm.table' % (iconddir)
        shutil.copy(cpfrom1, cpto1)


    return


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def checktemplate_BAGEL(filename, INFOS):
    necessary = ['basis', 'df_basis', 'nact', 'nclosed']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        print('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for l in data:
            if i in re.sub('#.*$', '', l):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        print('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    roots_there = False
    for l in data:
        l = re.sub('#.*$', '', l).lower().split()
        if len(l) == 0:
            continue
        if 'nstate' in l[0]:
            roots_there = True
    if not roots_there:
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            valid = []
            for l in data:
                if 'spin' in re.sub('#.*$', '', l).lower():
                    f = l.split()
                    if int(f[1]) == mult + 1:
                        valid.append(True)
                        break
            else:
                valid.append(False)
    if not all(valid):
        string = 'The template %s seems to be incomplete! It should contain the keyword "spin" for ' % (filename)
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            string += '%s, ' % (IToMult[mult + 1])
        string = string[:-2] + '!'
        print(string)
        return False
    return True

# =================================================


def get_BAGEL(INFOS):
    '''This routine asks for all questions specific to BAGEL:
    - path to bagel
    - scratch directory
    - BAGEL.template
    - wf.init
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('BAGEL Interface setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Path to BAGEL') + '\n')
    path = os.getenv('BAGEL')
    # path=os.path.expanduser(os.path.expandvars(path))
    if path == '':
        path = None
    else:
        path = '$BAGEL/'
        # print('Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path))
        # if question('Do you want to use this MOLCAS installation?',bool,True):
        # INFOS['molcas']=path
        # if 'molcas' not in INFOS:
    print('\nPlease specify path to BAGEL directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['bagel'] = question('Path to BAGEL:', str, path)
    print('')


    print('{:-^60}'.format('Scratch directory') + '\n')
    print('Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str)
    print('')


    print('{:-^60}'.format('BAGEL input template file') + '\n')
    print('''Please specify the path to the BAGEL.template file. This file must contain the following settings:

basis <Basis set>
df_basis <Density fitting basis set>
nact <Number of active orbitals>
nclosed <Number of doubly occupied orbitals>
nstate <Number of states for state-averaging>

The BAGEL interface will generate the appropriate BAGEL input automatically.
''')
    if os.path.isfile('BAGEL.template'):
        if checktemplate_BAGEL('BAGEL.template', INFOS):
            print('Valid file "BAGEL.template" detected. ')
            usethisone = question('Use this template file?', bool, True)
            if usethisone:
                INFOS['bagel.template'] = 'BAGEL.template'
    if 'bagel.template' not in INFOS:
        while True:
            filename = question('Template filename:', str)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_BAGEL(filename, INFOS):
                break
        INFOS['bagel.template'] = filename
    print('')

    print('{:-^60}'.format('Dipole level') + '\n')
    print('Please specify the desired amount of calculated dipole moments:\n0 -only dipole moments that are for free are calculated\n1 -calculate all transition dipole moments between the (singlet) ground state and all singlet states for absorption spectra\n2 -calculate all dipole moments')
    INFOS['dipolelevel'] = question('Requested dipole level:', int, [0])[0]
    print('')


    # QMMM
#  if check_MOLCAS_qmmm(INFOS['molcas.template']):
#    print(centerstring('MOLCAS+TINKER QM/MM setup',60,'-')+'\n')
#    print('Your template specifies a QM/MM calculation. Please specify the path to TINKER.')
#    path=os.getenv('TINKER')
#    if path=='':
#      path=None
#    else:
#      path='$TINKER/'
#    print('\nPlease specify path to TINKER bin/ directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
#    INFOS['tinker']=question('Path to TINKER/bin:',str,path)
#    print('Please give the key and connection table files.')
#      while True:
#        filename=question('Key file:',str)
#      if not os.path.isfile(filename):
#        print('File %s does not exist!' % (filename))
#      else:
#        break
#    INFOS['MOLCAS.fffile']=filename
#    while True:
#      filename=question('Connection table file:',str)
#      if not os.path.isfile(filename):
#        print('File %s does not exist!' % (filename))
#      else:
#        break
#    INFOS['MOLCAS.ctfile']=filename



    print('{:-^60}'.format('Initial wavefunction: MO Guess') + '\n')
    print('''Please specify the path to a MOLCAS JobIph file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    INFOS['bagel.guess'] = {}
    string = 'Do you have initial wavefunction files for '
    for mult, state in enumerate(INFOS['states']):
        if state <= 0:
            continue
        string += '%s, ' % (IToMult[mult + 1])
    string = string[:-2] + '?'
    if question(string, bool, True):
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            while True:
                guess_file = 'archive.%i.init' % (mult + 1)
                filename = question('Initial wavefunction file for %ss:' % (IToMult[mult + 1]), str, guess_file)
                if os.path.isfile(filename):
                    INFOS['bagel.guess'][mult + 1] = filename
                    break
                else:
                    print('File not found!')
    else:
        print('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')
        time.sleep(1)

    print('{:-^60}'.format('BAGEL Ressource usage') + '\n')  # TODO

    print('''Please specify the number of CPUs to be used by EACH calculation.
''')
    INFOS['bagel.ncpu'] = abs(question('Number of CPUs:', int, [1])[0])

    if INFOS['bagel.ncpu'] > 1:
        INFOS['bagel.mpi'] = question('Use MPI mode (no=OpenMP)?', bool, False)
    else:
        INFOS['bagel.mpi'] = False





    # Ionization
    # need_wfoverlap=False
    # print(centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    #INFOS['ion']=question('Dyson norms?',bool,False)
    # if 'ion' in INFOS and INFOS['ion']:
        # need_wfoverlap=True

    # wfoverlap
    if 'wfoverlap' in INFOS['needed']:
        print('\n' + '{:-^60}'.format('WFoverlap setup') + '\n')
        INFOS['bagel.wfoverlap'] = question('Path to wavefunction overlap executable:', str, '$SHARC/wfoverlap.x')
        # TODO not asked for: numfrozcore, numocc
        print('''Please specify the path to the PyQuante directory.
''')
        INFOS['bagel.pyq'] = question('PyQuante path:', str)
        print('''Please specify the amount of memory available to wfoverlap.x (in MB). \n (Note that BAGEL's memory cannot be controlled)
''')
        INFOS['bagel.mem'] = abs(question('wfoverlap.x memory:', int, [1000])[0])
    else:
        INFOS['bagel.mem'] = 1000
        INFOS['bagel.pyq'] = ''

    return INFOS

# =================================================


def prepare_BAGEL(INFOS, iconddir):
    # write BAGEL.resources
    try:
        sh2cas = open('%s/BAGEL.resources' % (iconddir), 'w')
    except IOError:
        print('IOError during prepareBAGEL, iconddir=%s' % (iconddir))
        quit(1)
    project = 'BAGEL'
    string = 'bagel %s\npyquante %s\nscratchdir %s/%s/\nmemory %i\nncpu %i\ndipolelevel %i\nproject %s' % (INFOS['bagel'], INFOS['bagel.pyq'], INFOS['scratchdir'], iconddir, INFOS['bagel.mem'], INFOS['bagel.ncpu'], INFOS['dipolelevel'], project)

    if INFOS['bagel.mpi']:
        string += 'mpi\n'
    if 'wfoverlap' in INFOS['needed']:
        string += '\nwfoverlap %s\n' % INFOS['bagel.wfoverlap']
    else:
        string += '\nnooverlap\n'
#  if 'tinker' in INFOS:
#    string+='tinker %s' % (INFOS['tinker'])
    sh2cas.write(string)
    sh2cas.close()

    # copy MOs and template
    cpfrom = INFOS['bagel.template']
    cpto = '%s/BAGEL.template' % (iconddir)
    shutil.copy(cpfrom, cpto)
    if not INFOS['bagel.guess'] == {}:
        for i in INFOS['bagel.guess']:
            cpfrom = INFOS['bagel.guess'][i]
            cpto = '%s/%s.%i.init' % (iconddir, 'archive', i)
            shutil.copy(cpfrom, cpto)



    return
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_runscript_info(INFOS):
    ''''''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('Run mode setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    print('{:-^60}'.format('Run script') + '\n')

    INFOS['here'] = False
    print('\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid.')
    INFOS['copydir'] = question('Run directory?', str)
    print('')



    print('')
    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def make_directory(iconddir):
    '''Creates a directory'''

    iconddir = os.path.abspath(iconddir)
    if os.path.isfile(iconddir):
        print('\nWARNING: %s is a file!' % (iconddir))
        return -1
    if os.path.isdir(iconddir):
        if len(os.listdir(iconddir)) == 0:
            return 0
        else:
            print('\nWARNING: %s/ is not empty!' % (iconddir))
            if 'overwrite' not in globals():
                global overwrite
                overwrite = question('Do you want to overwrite files in this and all following directories? ', bool, False)
            if overwrite:
                return 0
            else:
                return -1
    else:
        try:
            os.mkdir(iconddir)
        except OSError:
            print('\nWARNING: %s cannot be created!' % (iconddir))
            return -1
        return 0


# ======================================================================================================================

def writeRunscript(INFOS, iconddir):
    '''writes the runscript in each subdirectory'''
    try:
        runscript = open('%s/run_EXTORCA.sh' % (iconddir), 'w')
    except IOError:
        print('IOError during writeRunscript, iconddir=%s' % (iconddir))
        quit(1)
    if 'proj' in INFOS:
        projname = '%4s_%5s' % (INFOS['proj'][0:4], iconddir[-6:-1])
    else:
        projname = 'orca_opt'

    # ================================
    intstring = ''
    if 'amsrc' in INFOS:
        intstring = '. %s\nexport PYTHONPATH=$AMSHOME/scripting:$PYTHONPATH' % (INFOS['amsrc'])


    string = '''#!/bin/bash

#$-N %s

PRIMARY_DIR=%s/
cd $PRIMARY_DIR

%s
export PATH=$SHARC:$PATH

$ORCADIR/orca orca.inp > orca.log

''' % (projname, os.path.abspath(iconddir), intstring)

    runscript.write(string)
    runscript.close()
    filename = '%s/run_EXTORCA.sh' % (iconddir)
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

    return


# ======================================================================================================================

def writeOrcascript(INFOS, iconddir):
    '''writes the orcascript in each subdirectory'''

    # geometry_data=readfile(INFOS['geom_location'])
    # natom=int(geometry_data[0])
    # ngeoms=len(geometry_data)/(natom+2)
    # print(ngeoms)
    # sys.exit(1)

    try:
        runscript = open('%s/orca.inp' % (iconddir), 'w')
    except IOError:
        print('IOError during writeRunscript, iconddir=%s' % (iconddir))
        quit(1)

    string = '''#
#SHARC: states %s
#SHARC: interface %s
#SHARC: opt %s %i''' % (' '.join([str(i) for i in INFOS['states']]),
                        Interfaces[INFOS['interface']]['name'],
                        INFOS['opttype'],
                        INFOS['cas.root1'])
    if INFOS['opttype'] == 'cross':
        string += ' %i' % INFOS['cas.root2']
    string += '\n'
    if INFOS['opttype'] == 'cross' and INFOS['calc_ci'] and 'nacdr' not in Interfaces[INFOS['interface']]['features']:
        string += '#SHARC: param %f %f\n' % (INFOS['sigma'], INFOS['alpha'])
    string += '''
! ExtOpt

%%geom
  maxstep %f
  Trust %f
  maxiter 200
end

* xyzfile 0 1 %s

''' % (INFOS['maxstep'], -INFOS['maxstep'], 'geom.xyz')



    runscript.write(string)
    runscript.close()
    filename = '%s/orca.inp' % (iconddir)


    return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_iconddir(istate, INFOS):
    if INFOS['diag']:
        dirname = 'State_%i' % (istate)
    else:
        mult, state, ms = INFOS['statemap'][istate]
        dirname = IToMult[mult] + '_%i' % (state - (mult == 1 or mult == 2))
    return dirname

# ====================================


def setup_all(INFOS):
    '''This routine sets up the directories for the initial calculations.'''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('Setting up directory...') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

    geometry_data = readfile(INFOS['geom_location'])
    natom = INFOS['natom']
    # replace all comment lines (orca can be picky about them)
    for igeom in range(INFOS['ngeom']):
        geometry_data[igeom * (natom + 2) + 1] = 'geometry:%i\n' % (igeom + 1)
    make_directory(INFOS['copydir'])

    for igeom in range(INFOS['ngeom']):
        iconddir = os.path.join(INFOS['copydir'], 'geom_%i' % (igeom + 1))
        make_directory(iconddir)
        writefile(os.path.join(iconddir, 'geom.xyz'), geometry_data[igeom * (natom + 2):(igeom + 1) * (natom + 2)])
        globals()[Interfaces[INFOS['interface']]['prepare_routine']](INFOS, iconddir)
        writeRunscript(INFOS, iconddir)
        writeOrcascript(INFOS, iconddir)

    print('\n')


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
    '''Main routine'''

    usage = '''
python setup_orca_opt.py

This interactive program prepares ORCA+SHARC optimizations.
'''

    description = ''
    parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()

    INFOS = get_general()
    INFOS = globals()[Interfaces[INFOS['interface']]['get_routine']](INFOS)
    INFOS = get_runscript_info(INFOS)

    print('\n' + '{:#^60}'.format('Full input') + '\n')
    for item in INFOS:
        print(item, ' ' * (25 - len(item)), INFOS[item])
    print('')
    setup = question('Do you want to setup the specified calculations?', bool, True)
    print('')

    if setup:
        setup_all(INFOS)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
