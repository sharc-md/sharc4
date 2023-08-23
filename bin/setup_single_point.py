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

# Interactive script to setup single point calculations using the SHARC interfaces
#
# usage: python setup_traj.py #change

import math
import os
import stat
import shutil
import datetime
from optparse import OptionParser
import pprint
from constants import IToMult
from utils import readfile, question, itnmstates
from SHARC_INTERFACE import SHARC_INTERFACE
import factory
from logger import log

version = '3.0'
versiondate = datetime.datetime(year=2023, month=8, day=22)

# =========================================================0
# some constants
DEBUG = False
PI = math.pi
global KEYSTROKES

old_question = question
def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(question, typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges)


Interfaces: list[SHARC_INTERFACE] = []
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
# ======================================================================= #


def displaywelcome():
    log.info('Script for single point setup with SHARC started...\n')  # change
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Setup single points with SHARC') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Author: Sebastian Mai, Severin Polonius') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Version:' + version) + '||\n'
    string += '||' + '{:^80}'.format(versiondate.strftime("%d.%m.%y")) + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '  ' + '{:=^80}'.format('') + '\n'
    string += '''
This script automatizes the setup of the input files for SHARC single point calculations.
  '''
    log.info(string)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.setup_single_point')

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_interface() -> SHARC_INTERFACE:
    'asks for interface and instantiates it'
    Interfaces = factory.get_available_interfaces()
    log.info('{:-^60}'.format('Choose the quantum chemistry interface'))
    log.info('\nPlease specify the quantum chemistry interface (enter any of the following numbers):')
    possible_numbers = []
    for i, (name, interface) in enumerate(Interfaces):
        if type(interface) == str:
            log.info('%i\t%s: %s' % (i, name, interface))
        else:
            log.info('%i\t%s: %s' % (i, name, interface.description()))
            possible_numbers.append(i)
    log.info('')
    while True:
        num = question('Interface number:', int)[0]
        if num in possible_numbers:
            break
        else:
            log.info('Please input one of the following: %s!' % (possible_numbers))
    log.info("")
    return Interfaces[num][1]

def get_requests(INFOS, interface: SHARC_INTERFACE) -> list[str]:
    """ get requests for every single point"""
    standard_requests = {'dm': "dipole moments", 'grad': "gradients", 'soc': "spin orbit couplings", 'nacdr': "nonadiabatic couplings", 'socdr': "derivatives of spin--orbit couplings", 'dmdr':
                         "derivates of dipole moments", 'multipolar_fit': "a distributed multipole expansion for all states", 'theodore': "THEODORE analysis"}
    int_features = interface.get_features()
    available_requests = sorted(set(*standard_requests.keys()).intersection(int_features))
    log.debug(available_requests)
    requests = ['h']
    log.info(f"{'Requests on every single point (additional to energy)':-^60-}")
    log.info("")
    for i in available_requests:
        if question(f"Calculate {standard_requests[i]}?:", bool, autocomplete=False, default=True):
            requests.append(i)

    return requests


def get_general(INFOS) -> dict:
    '''This routine questions from the user some general information:
    - initconds file
    - number of states
    - number of initial conditions
    - interface to use'''

    log.info('{:-^60}'.format('Geometry'))
    log.info('\nPlease specify the geometry file (xyz format, Angstroms):')
    while True:
        path = question('Geometry filename:', str, 'geom.xyz')
        try:
            path = os.path.expanduser(os.path.expandvars(path))
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
    nstates = 0
    for mult, i in enumerate(states):
        nstates += (mult + 1) * i
    log.info('Number of states: ' + str(states))
    log.info('Total number of states: %i\n' % (nstates))
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



    # Add some simple keys
    INFOS['cwd'] = os.getcwd()
    log.info('')
    INFOS['needed'] = []

    return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
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
    log.info(string)

    log.info('{:-^60}'.format('Run script') + '\n')

    INFOS['here'] = False
    log.info('\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid.')
    INFOS['copydir'] = question('Run directory?', str)
    if question('Do you have headers for the runscript?', bool):
        INFOS['headers'] = readfile(question('Path to header file:', str, 'header', autocomplete=True))

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
        runscript = open('%s/run_single_point.sh' % (iconddir), 'w')
    except IOError:
        log.info('IOError during writeRunscript, iconddir=%s' % (iconddir))
        quit(1)
    if 'proj' in INFOS:
        projname = '%4s_%5s' % (INFOS['proj'][0:4], iconddir[-6:-1])
    else:
        projname = 'singlep'

    headers = INFOS['headers'] if 'headers' in INFOS else ''


    # ================================

    string = '''#!/bin/bash
%s

PRIMARY_DIR=%s/
cd $PRIMARY_DIR


$SHARC/%s QM.in >> QM.log 2>> QM.err

''' % (projname, os.path.abspath(iconddir), headers, Interfaces[INFOS['interface']]['script'])

    runscript.write(string)
    runscript.close()
    filename = '%s/run_single_point.sh' % (iconddir)
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

    return


# ======================================================================================================================

def writeQMin(INFOS, iconddir, igeom, geometry_data):

    try:
        runscript = open('%s/QM.in' % (iconddir), 'w')
    except IOError:
        log.info('IOError during writeQMin, iconddir=%s' % (iconddir))
        quit(1)

    string = ''
    natom = INFOS['natom']
    for line in geometry_data[igeom * (natom + 2):(igeom + 1) * (natom + 2)]:
        string += line.strip() + '\n'

    string += '''
step 0
unit angstrom
states %s
''' % (' '.join([str(i) for i in INFOS['states']]))
    string += "\n".join(INFOS['requests'])

    runscript.write(string)
    runscript.close()

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

    geometry_data = readfile(INFOS['geom_location'])
    make_directory(INFOS['copydir'])

    for igeom in range(INFOS['ngeom']):
        iconddir = os.path.join(INFOS['copydir'], 'geom_%i' % (igeom + 1))
        make_directory(iconddir)
        interface.prepare(INFOS, iconddir)
        writeRunscript(INFOS, iconddir)
        writeQMin(INFOS, iconddir, igeom, geometry_data)

    log.info('\n')


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
    '''Main routine'''

    usage = '''
python setup_single_point.py

This interactive program prepares SHARC single point calculations.
'''

    description = ''
    parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()
    INFOS = {}
    chosen_interface = get_interface()()
    INFOS = get_general(INFOS)
    INFOS['requests'] = get_requests(INFOS)
    INFOS = chosen_interface.get_infos(INFOS)
    INFOS = get_runscript_info(INFOS)

    log.info('\n' + '{:#^60}'.format('Full input') + '\n')
    for item in INFOS:
        log.info(item, ' ' * (25 - len(item)), INFOS[item])
    log.info('')
    setup = question('Do you want to setup the specified calculations?', bool, True)
    log.info('')

    if setup:
        setup_all(INFOS)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
