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

# Interactive script for the use of the ORCA external optimizer with SHARC
#
# usage: python setup_orca_opt.py #change

import math
import sys
import os
import stat
import shutil
import datetime
from optparse import OptionParser


from constants import IToMult
from utils import readfile, writefile, itnmstates, question
from SHARC_INTERFACE import SHARC_INTERFACE
import factory
from logger import log


# =========================================================0
# compatibility stuff

if sys.version_info[0] != 3:
    print('This is a script for Python 3!')
    sys.exit(0)

version = '4.0'
versionneeded = [0.2, 1.0, 2.0, 2.1, float(version)]
versiondate = datetime.date(2025, 4, 1)


# =========================================================0
# some constants
DEBUG = False
PI = math.pi
global KEYSTROKES

old_question = question

def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(question, typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def displaywelcome():
    log.info('Script for setup of optimizations with ORCA and SHARC started...\n')  # change
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
    log.info(string)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open("KEYSTROKES.tmp", "w")


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move("KEYSTROKES.tmp", "KEYSTROKES.setup_orca_opt")

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_interface() -> SHARC_INTERFACE:
    "asks for interface and instantiates it"
    Interfaces = factory.get_available_interfaces()
    log.info("")
    log.info("{:-^60}".format("Choose the quantum chemistry interface"))
    log.info("\nPlease specify the quantum chemistry interface (enter any of the following numbers):")
    possible_numbers = []
    for i, (name, interface, possible) in enumerate(Interfaces):
        if not possible:
            log.info("% 3i %-20s %s" % (i+1, name, interface))
        else:
            log.info("% 3i %-20s %s" % (i+1, name, interface.description()))
            possible_numbers.append(i+1)
    log.info("")
    while True:
        num = question("Interface number:", int)[0]
        if num in possible_numbers:
            break
        else:
            log.info("Please input one of the following: %s!" % (possible_numbers))
    log.info("")
    log.info("The following interface was selected:")
    log.info("% 3i %-20s %s" % (num, Interfaces[num-1][0], Interfaces[num-1][1].description()))
    return Interfaces[num-1][1]

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_general(INFOS) -> dict:
    '''This routine questions from the user some general information:
    - initconds file
    - number of states
    - number of initial conditions
    - interface to use'''

    log.info('\n' + '{:-^60}'.format('Path to ORCA') + '\n')
    path = os.getenv('ORCADIR')
    if path == '':
        path = None
    else:
        path = '$ORCADIR/'
    log.info('Please specify path to ORCA directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['orca'] = question('Path to ORCA:', str, path)
    log.info("")
    return INFOS

# ======================================================================================================================

def get_requests(INFOS, interface: SHARC_INTERFACE) -> list[str]:

    log.info('{:-^60}'.format('Geometry'))
    log.info('\nPlease specify the geometry file (xyz format, Angstroms):')
    while True:
        path = question('Geometry filename:', str, 'geom.xyz')
        try:
            gf = open(path, 'r')
        except IOError:
            log.info('Could not open: %s' % (path))
            continue
        g = gf.readlines()
        gf.close()
        try:
            natom = int(g[0])
        except ValueError:
            log.info('Malformatted: %s' % (path))
            continue
        break
    INFOS['geom_location'] = path
    geometry_data = readfile(INFOS['geom_location'])
    ngeoms = len(geometry_data) // (natom + 2)
    if ngeoms > 1:
        log.info('Number of geometries: %i' % (ngeoms))
    INFOS['ngeom'] = ngeoms
    INFOS['natom'] = natom


    # Number of states
    log.info('\n' + '{:-^60}'.format('Number of states') + '\n')
    log.info('\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets.')
    while True:
        states = question('Number of states:', int)
        if len(states) == 0:
            continue
        if any(i < 0 for i in states):
            log.info('Number of states must be positive!')
            continue
        break
    log.info('')



    log.info("\nPlease enter the molecular charge for each chosen multiplicity\ne.g. 0 +1 0 for neutral singlets and triplets and cationic doublets.")
    default = [i % 2 for i in range(len(states))]
    while True:
        charges = question("Molecular charges per multiplicity:", int, default)
        if not states:
            continue
        if len(charges) != len(states):
            log.info("Charges array must have same length as states array")
            continue
        break

    nstates = 0
    for mult, i in enumerate(states):
        nstates += (mult + 1) * i
    log.info('Number of states: ' + str(states))
    log.info('Total number of states: %i\n' % (nstates))
    INFOS['states'] = states
    INFOS['nstates'] = nstates
    INFOS["charge"] = charges
    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(INFOS['states']):
        statemap[i] = [imult, istate, ims]
        i += 1
    INFOS['statemap'] = statemap
    log.info(statemap)





    # states to optimize
    log.info('\n' + '{:-^60}'.format('States to optimize') + '\n')

    INFOS['maxmult'] = len(states)
    optmin = question('Do you want to optimize a minimum? (no=optimize crossing):', bool, True)
    if optmin:
        INFOS['opttype'] = 'min'
        log.info('\nPlease specify the state involved in the optimization\ne.g. 3 2 for the second triplet state.')
    else:
        INFOS['opttype'] = 'cross'
        log.info('\nPlease specify the first state involved in the optimization\ne.g. 3 2 for the second triplet state.')
    while True:
        rmult, rstate = tuple(question('State:', int, [1, 1]))
        # check
        if not 1 <= rmult <= INFOS['maxmult']:
            log.info('Multiplicity (%i) must be between 1 and %i!' % (rmult, INFOS['maxmult']))
            continue
        if not 1 <= rstate <= states[rmult - 1]:
            log.info('Only %i states of mult %i' % (states[rmult - 1], rmult))
            continue
        break
    INFOS['cas.root1'] = [rmult, rstate]

    if not optmin:
        log.info('\nPlease specify the second state involved in the optimization\ne.g. 3 2 for the second triplet state.')
        while True:
            rmult, rstate = tuple(question('Root:', int, [1, 2]))
            # check
            if not 1 <= rmult <= INFOS['maxmult']:
                log.info('%i must be between 1 and %i!' % (rmult, INFOS['maxmult']))
                continue
            if not 1 <= rstate <= states[rmult - 1]:
                log.info('Only %i states of mult %i' % (states[rmult - 1], rmult))
                continue
            INFOS['cas.root2'] = [rmult, rstate]
            if INFOS['cas.root1'] == INFOS['cas.root2']:
                log.info('Both states are identical!')
                continue
            # get type of optimization
            if INFOS['cas.root1'][0] == INFOS['cas.root2'][0]:
                log.info('Multiplicities of both states identical, optimizing a conical intersection.')
                INFOS['calc_ci'] = True
            else:
                log.info('Multiplicities of both states different, optimizing a minimum crossing point.')
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

    requests = ["h"]
    int_features = interface.get_features(KEYSTROKES=KEYSTROKES)
    if not optmin and INFOS['calc_ci']:
        if "nacdr" not in int_features:
            log.info('{:-^60}'.format('Optimization parameter'))
            log.info('\nYou are optimizing a conical intersection, but the chosen interface cannot deliver nonadiabatic coupling vectors. The optimization will therefore employ the penalty function method of Levine, Coe, Martinez (DOI: 10.1021/jp0761618).\nIn this optimization scheme, there are two parameters, sigma and alpha, which affect how close to the true conical intersection the optimization will end up.')
            log.info('\nPlease enter the values for the sigma and alpha parameters.\n')
            log.info('A larger sigma makes convergence harder but optimization will go closer to the true CI.')
            sigma = question('Sigma: ', float, [3.5])[0]
            log.info('A smaller alpha makes convergence harder but optimization will go closer to the true CI.')
            alpha = question('Alpha: ', float, [0.02])[0]
            INFOS['sigma'] = sigma
            INFOS['alpha'] = alpha
        else:
            requests.append("nacdr")
            INFOS["use_nacs"] = True

    log.info('\nPlease enter the values for the maximum allowed displacement per timestep \n(choose smaller value if starting from a good guess and for large sigma or small alpha).')
    INFOS['maxstep'] = question('Maximum allowed step: ', float, [0.3])[0]

    # Add some simple keys
    INFOS['cwd'] = os.getcwd()
    log.info('')

    return set(requests)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_runscript_info(INFOS):
    ''''''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('Run mode setup') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    log.info(string)

    log.info('{:-^60}'.format('Run script') + '\n')

    INFOS['here'] = False
    log.info('\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid.')
    INFOS['copydir'] = question('Run directory?', str)
    log.info('')



    log.info('')
    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def make_directory(iconddir):
    '''Creates a directory'''

    iconddir = os.path.abspath(iconddir)
    if os.path.isfile(iconddir):
        log.info('\nWARNING: %s is a file!' % (iconddir))
        return -1
    if os.path.isdir(iconddir):
        if len(os.listdir(iconddir)) == 0:
            return 0
        else:
            log.info('\nWARNING: %s/ is not empty!' % (iconddir))
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
            log.info('\nWARNING: %s cannot be created!' % (iconddir))
            return -1
        return 0


# ======================================================================================================================

def writeRunscript(INFOS, iconddir):
    '''writes the runscript in each subdirectory'''
    try:
        runscript = open('%s/run_EXTORCA.sh' % (iconddir), 'w')
    except IOError:
        log.info('IOError during writeRunscript, iconddir=%s' % (iconddir))
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
export ORCADIR=%s
export PATH=$SHARC:$ORCADIR:$PATH
export EXTOPTEXE=$SHARC/otool_external

orca orca.inp > orca.log

''' % (projname, os.path.abspath(iconddir), intstring, INFOS['orca'])

    runscript.write(string)
    runscript.close()
    filename = '%s/run_EXTORCA.sh' % (iconddir)
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

    return


# ======================================================================================================================

def writeOrcascript(INFOS, iconddir, interface):
    '''writes the orcascript in each subdirectory'''

    # geometry_data=readfile(INFOS['geom_location'])
    # natom=int(geometry_data[0])
    # ngeoms=len(geometry_data)/(natom+2)
    # print(ngeoms)
    # sys.exit(1)

    # try:
    #     runscript = open('%s/orca.inp' % (iconddir), 'w')
    # except IOError:
    #     log.info('IOError during writeRunscript, iconddir=%s' % (iconddir))
    #     quit(1)

    string = '''#
#SHARC: states %s
#SHARC: charge %s
#SHARC: interface %s
#SHARC: opt %s %i''' % (' '.join([str(i) for i in INFOS['states']]),
                       ' '.join([str(i) for i in INFOS['charge']]),
                       interface.name(),
                       INFOS['opttype'],
                       INFOS['cas.root1'])
    if INFOS['opttype'] == 'cross':
        string += ' %i' % INFOS['cas.root2']
    string += '\n'
    if INFOS['opttype'] == 'cross' and INFOS['calc_ci'] and not INFOS["use_nacs"]:
        string += '#SHARC: param %f %f\n' % (INFOS['sigma'], INFOS['alpha'])
    string += '''
! ExtOpt Opt

%%geom
  maxstep %f
  Trust %f
  maxiter 200
end

* xyzfile 0 1 %s

''' % (INFOS['maxstep'], -INFOS['maxstep'], 'geom.xyz')
    writefile(os.path.join(iconddir,"orca.inp"), string)

    # runscript.write(string)
    # runscript.close()

    # filename = '%s/otool_external.inp' % (iconddir)
    string = '''#
SHARC: states %s
SHARC: charge %s
SHARC: interface %s
SHARC: opt %s %i''' % (' '.join([str(i) for i in INFOS['states']]),
                       ' '.join([str(i) for i in INFOS['charge']]),
                       interface.name(),
                       INFOS['opttype'],
                       INFOS['cas.root1'])
    if INFOS['opttype'] == 'cross':
        string += ' %i' % INFOS['cas.root2']
    string += '\n'
    if INFOS['opttype'] == 'cross' and INFOS['calc_ci'] and not INFOS["use_nacs"]:
        string += 'SHARC: param %f %f\n' % (INFOS['sigma'], INFOS['alpha'])
    # runscript = open(filename, 'w')    
    # runscript.write(string)
    # runscript.close()
    writefile(os.path.join(iconddir,"otool_external.inp"), string)

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

def setup_all(INFOS, interface: SHARC_INTERFACE):
    '''This routine sets up the directories for the initial calculations.'''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('Setting up directory...') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    log.info(string)

    # add more things
    INFOS['link_files'] = False

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
        interface.prepare(INFOS, iconddir)
        writeRunscript(INFOS, iconddir)
        writeOrcascript(INFOS, iconddir, interface)

    log.info('\n')


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

    INFOS = {}
    INFOS = get_general(INFOS)
    chosen_interface = get_interface()()
    INFOS["needed_requests"] = get_requests(INFOS, chosen_interface)
    INFOS = chosen_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)
    INFOS = get_runscript_info(INFOS)

    log.info("\n" + f"{'Full input':#^60}" + "\n")
    for item in INFOS:
        log.info(f"{item:<25} {INFOS[item]}")
    log.info("")
    setup = question("Do you want to setup the specified calculations?", bool, True)
    log.info("")

    if setup:
        setup_all(INFOS, chosen_interface)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
