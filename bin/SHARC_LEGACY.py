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
import datetime
import os
import stat
import shutil
from io import TextIOWrapper
import subprocess as sp
from typing import Optional
import re
import ast

import numpy as np
# internal
from constants import ATOMCHARGE, FROZENS, IToMult
from factory import factory
from SHARC_INTERFACE import SHARC_INTERFACE
from qmout import QMout
from utils import (ATOM, InDir, expand_path, itnmstates, mkdir, question,
                   readfile, writefile)
from logger import log


# TODO: define __all__

VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 24)

CHANGELOGSTRING = """
"""
np.set_printoptions(linewidth=400)

# ----------------------------------------------------
#             Legacy interface information
# ----------------------------------------------------


Interfaces = {
    1: {'script': 'SHARC_MOLPRO.py',
        'name': 'molpro',
        'description': 'MOLPRO (only CASSCF)',
        'get_routine': 'get_MOLPRO',
        'prepare_routine': 'prepare_MOLPRO',
        'features': {'h': [],
                     'soc': [],
                     'dm': [],
                     'grad': [],
                     'overlap': ['wfoverlap'],
                     'ion': ['wfoverlap'],
                     'nacdr': ['wfoverlap'],
                     'phases': ['wfoverlap'],
                     },
        },
    2: {'script': 'SHARC_COLUMBUS.py',
        'name': 'columbus',
        'description': 'COLUMBUS (CASSCF, RASSCF and MRCISD), using SEWARD integrals',
        'get_routine': 'get_COLUMBUS',
        'prepare_routine': 'prepare_COLUMBUS',
        'features': {'h': [],
                     'soc': [],
                     'dm': [],
                     'grad': [],
                     'overlap': ['wfoverlap'],
                     'ion': ['wfoverlap'],
                     'nacdr': [],
                     'phases': ['wfoverlap'],
                     },
        },
    3: {'script': 'SHARC_AMS_ADF.py',
        'name': 'AMS_ADF',
        'description': 'AMS_ADF (DFT, TD-DFT)',
        'get_routine': 'get_AMS',
        'prepare_routine': 'prepare_AMS',
        'features': {'h': [],
                     'soc': [],
                     'dm': [],
                     'grad': [],
                     'overlap': ['wfoverlap'],
                     'ion': ['wfoverlap'],
                     'phases': ['wfoverlap'],
                     'theodore': ['theodore'],
                     },
        },
    4: {'script': 'SHARC_BAGEL.py',
        'name': 'bagel',
        'description': 'BAGEL (CASSCF, CASPT2, (X)MS-CASPT2)',
        'get_routine': 'get_BAGEL',
        'prepare_routine': 'prepare_BAGEL',
        'features': {'h': [],
                     'dm': [],
                     'grad': [],
                     'overlap': ['wfoverlap'],
                     'ion': ['wfoverlap'],
                     'nacdr': [],
                     'phases': ['wfoverlap'], 
                     },
         },
}

def centerstring(string, n, pad=' '):
    length = len(string)
    if length >= n:
        return string
    else:
        return pad * ((n - length + 1) // 2) + string + pad * ((n - length) // 2)



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
        log.info('Could not open template file %s' % (filename))
        return False
    i = 0
    for line in data:
        if necessary[i] in line:
            i += 1
            if i + 1 == len(necessary):
                return True
    log.info('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
    return False

# =================================================


def get_MOLPRO(INFOS, KEYSTROKES: Optional[TextIOWrapper] = None):
    '''This routine asks for all questions specific to MOLPRO:
    - path to molpro
    - scratch directory
    - MOLPRO.template
    - wf.init
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + centerstring('MOLPRO Interface setup', 80) + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    log.info(string)

    # MOLPRO executable
    log.info(centerstring('Path to MOLPRO', 60, '-') + '\n')
    path = os.getenv('MOLPRO')
    path = os.path.expanduser(os.path.expandvars(path))
    if not path == '':
        path = '$MOLPRO/'
    else:
        path = None
    log.info('\nPlease specify path to MOLPRO directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['molpro'] = question('Path to MOLPRO executable:', str, KEYSTROKES=KEYSTROKES, default=path)
    log.info('')

    # Scratch directory
    log.info(centerstring('Scratch directory', 60, '-') + '\n')
    log.info('Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str, KEYSTROKES=KEYSTROKES)
    log.info('')

    # MOLPRO input template
    log.info(centerstring('MOLPRO input template file', 60, '-') + '\n')
    log.info('''Please specify the path to the MOLPRO.template file. This file must be a valid MOLPRO input file for a CASSCF calculation. It should contain the following settings:
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
            log.info('Valid file "MOLPRO.template" detected. ')
            usethisone = question('Use this template file?', bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                INFOS['molpro.template'] = 'MOLPRO.template'
    if 'molpro.template' not in INFOS:
        while True:
            filename = question('Template filename:', str, KEYSTROKES=KEYSTROKES)
            if not os.path.isfile(filename):
                print('File %s does not exist!' % (filename))
                continue
            if checktemplate_MOLPRO(filename):
                break
        INFOS['molpro.template'] = filename
    log.info('')

    # Initial wavefunction
    log.info(centerstring('Initial wavefunction: MO Guess', 60, '-') + '\n')
    log.info('''Please specify the path to a MOLPRO wavefunction file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!

If you optimized your geometry with MOLPRO/CASSCF you can reuse the "wf" file from the optimization.
''')
    if question('Do you have an initial wavefunction file?', bool, KEYSTROKES=KEYSTROKES, default=True):
        while True:
            filename = question('Initial wavefunction file:', str, KEYSTROKES=KEYSTROKES, default='wf.init')
            if os.path.isfile(filename):
                break
            else:
                log.info('File not found!')
        INFOS['molpro.guess'] = filename
    else:
        log.info('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')
        INFOS['molpro.guess'] = False


    log.info(centerstring('MOLPRO Ressource usage', 60, '-') + '\n')
    log.info('''Please specify the amount of memory available to MOLPRO (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
''')
    INFOS['molpro.mem'] = abs(question('MOLPRO memory:', int, KEYSTROKES=KEYSTROKES, default=[500])[0])
    log.info('''Please specify the number of CPUs to be used by EACH trajectory.
''')
    INFOS['molpro.ncpu'] = abs(question('Number of CPUs:', int, KEYSTROKES=KEYSTROKES, default=[1])[0])

    # wfoverlap
    if 'wfoverlap' in INFOS['needed']:
        log.info('\n' + centerstring('Wfoverlap code setup', 60, '-') + '\n')
        INFOS['molpro.wfpath'] = question('Path to wavefunction overlap executable:', str, KEYSTROKES=KEYSTROKES, default='$SHARC/wfoverlap.x')

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
        log.info('IOError during prepareMOLPRO, iconddir=%s' % (iconddir))
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
        cpto = '%s/QM/wf.init' % (iconddir)
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
            # log.info('TEMPLATE=%s exists and is a file!' % (TEMPLATE))
            return None, None, None
        necessary = ['control.run', 'mcscfin', 'tranin', 'propin']
        lof = os.listdir(TEMPLATE)
        for i in necessary:
            if i not in lof:
                # log.info('Did not find input file %s! Did you prepare the input according to the instructions?' % (i))
                return None, None, None
        cidrtinthere = False
        ciudginthere = False
        for i in lof:
            if 'cidrtin' in i:
                cidrtinthere = True
            if 'ciudgin' in i:
                ciudginthere = True
        if not cidrtinthere or not ciudginthere:
            # log.info('Did not find input file %s.*! Did you prepare the input according to the instructions?' % (i))
            return None, None, None
    else:
        # log.info('Directory %s does not exist!' % (TEMPLATE))
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
                # log.info('Multiplicity %i cannot be treated in directory %s (single DRT)!'  % (mult,TEMPLATE))
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


def get_COLUMBUS(INFOS, KEYSTROKES: Optional[TextIOWrapper] = None):
    '''This routine asks for all questions specific to COLUMBUS:
    - path to COLUMBUS
    - scratchdir
    - path to template directory
    - mocoef
    - memory
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + centerstring('COLUMBUS Interface setup', 80) + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    log.info(string)


    # Path to COLUMBUS directory
    log.info(centerstring('Path to COLUMBUS', 60, '-') + '\n')
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
    # if not 'columbus' in INFOS:
    log.info('\nPlease specify path to COLUMBUS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['columbus'] = question('Path to COLUMBUS:', str, KEYSTROKES=KEYSTROKES, default=path)
    log.info('')


    # Scratch directory
    log.info(centerstring('Scratch directory', 60, '-') + '\n')
    log.info('Please specify an appropriate scratch directory. This will be used to temporally store all COLUMBUS files. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str, KEYSTROKES=KEYSTROKES)
    log.info('')


    # COLUMBUS template directory
    log.info(centerstring('COLUMBUS input template directory', 60, '-') + '\n')
    log.info('''Please specify the path to the COLUMBUS template directory.
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
        path = question('Path to templates:', str, KEYSTROKES=KEYSTROKES)
        path = os.path.expanduser(os.path.expandvars(path))
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            log.info('Directory %s does not exist!' % (path))
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
    log.info('')

    log.info('''Check whether the jobs are assigned correctly to the multiplicities. Use the following commands:
  mult job        make <mult> use the input in <job>
  show            show the mapping of multiplicities to jobs
  end             confirm this mapping
''')
    for i in multmap:
        log.info('%i ==> %s' % (i, multmap[i]))
    while True:
        line = question('Adjust job mapping:', str, KEYSTROKES=KEYSTROKES, default='end', autocomplete=False)
        if 'show' in line.lower():
            for i in multmap:
                log.info('%i ==> %s' % (i, multmap[i]))
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
                log.info('Multiplicity %i not necessary!' % (m))
                continue
            if not os.path.isdir(path + '/' + j):
                log.info('No template subdirectory %s!' % (j))
                continue
            if not j[-1] == '/':
                j += '/'
            multmap[m] = j
    log.info('')

    mocoefmap = {}
    for job in set([multmap[i] for i in multmap]):
        mocoefmap[job] = multmap[1]
    log.info('''Check whether the mocoeffiles are assigned correctly to the jobs. Use the following commands:
  job mocoefjob   make <job> use the mocoeffiles from <mocoefjob>
  show            show the mapping of multiplicities to jobs
  end             confirm this mapping
''')
    width = max([len(i) for i in mocoefmap])
    for i in mocoefmap:
        log.info('%s' % (i) + ' ' * (width - len(i)) + ' <== %s' % (mocoefmap[i]))
    while True:
        line = question('Adjust mocoef mapping:', str, KEYSTROKES=KEYSTROKES, default='end', autocomplete=False)
        if 'show' in line.lower():
            for i in mocoefmap:
                log.info('%s <== %s' % (i, mocoefmap[i]))
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
    log.info('')

    INFOS['columbus.template'] = path
    INFOS['columbus.multmap'] = multmap
    INFOS['columbus.mocoefmap'] = mocoefmap
    INFOS['columbus.intprog'] = intprog

    INFOS['columbus.copy_template'] = question('Do you want to copy the template directory to each trajectory (Otherwise it will be linked)?', bool, KEYSTROKES=KEYSTROKES, default=False)
    if INFOS['columbus.copy_template']:
        INFOS['columbus.copy_template_from'] = INFOS['columbus.template']
        INFOS['columbus.template'] = './COLUMBUS.template/'


    # Initial mocoef
    log.info(centerstring('Initial wavefunction: MO Guess', 60, '-') + '\n')
    log.info('''Please specify the path to a COLUMBUS mocoef file containing suitable starting MOs for the CASSCF calculation.
''')
    init = question('Do you have an initial mocoef file?', bool, KEYSTROKES=KEYSTROKES, default=True)
    if init:
        while True:
            line = question('Mocoef filename:', str, KEYSTROKES=KEYSTROKES, default='mocoef_mc.init')
            line = os.path.expanduser(os.path.expandvars(line))
            if os.path.isfile(line):
                break
            else:
                log.info('File not found!')
                continue
        INFOS['columbus.guess'] = line
    else:
        log.info('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')
        INFOS['columbus.guess'] = False
    log.info('')


    # Memory
    log.info(centerstring('COLUMBUS Memory usage', 60, '-') + '\n')
    log.info('''Please specify the amount of memory available to COLUMBUS (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
''')
    INFOS['columbus.mem'] = abs(question('COLUMBUS memory:', int, KEYSTROKES=KEYSTROKES)[0])

    # wfoverlap
    if 'wfoverlap' in INFOS['needed']:
        log.info('\n' + centerstring('Wfoverlap code setup', 60, '-') + '\n')
        INFOS['columbus.wfpath'] = question('Path to wavefunction overlap executable:', str, KEYSTROKES=KEYSTROKES, default='$SHARC/wfoverlap.x')
        INFOS['columbus.wfthres'] = question('Determinant screening threshold:', float, KEYSTROKES=KEYSTROKES, default=[0.97])[0]
        INFOS['columbus.numfrozcore'] = question('Number of frozen core orbitals for overlaps (-1=as in template):', int, KEYSTROKES=KEYSTROKES, default=[-1])[0]
        if 'ion' in INFOS and INFOS['ion']:
            INFOS['columbus.numocc'] = question('Number of doubly occupied orbitals for Dyson:', int, KEYSTROKES=KEYSTROKES, default=[0])[0]

    return INFOS

# =================================================


def prepare_COLUMBUS(INFOS, iconddir):
    # write COLUMBUS.resources
    try:
        sh2col = open('%s/COLUMBUS.resources' % (iconddir), 'w')
    except IOError:
        log.info('IOError during prepareCOLUMBUS, directory=%i' % (iconddir))
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
        string += 'wfoverlap %s\n' % (INFOS['columbus.wfpath'])
        string += 'wfthres %f\n' % (INFOS['columbus.wfthres'])
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


def checktemplate_AMS(filename, INFOS):
    necessary = ['basis', 'functional', 'charge']
    try:
        f = open(filename)
        data = f.readlines()
        f.close()
    except IOError:
        log.info('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for li in data:
            line = li.lower().split()
            if len(line) == 0:
                continue
            line = line[0]
            if i == re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        log.info('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    return True


# =================================================


def get_AMS(INFOS, KEYSTROKES: Optional[TextIOWrapper] = None):
    '''This routine asks for all questions specific to AMS:
    - path to AMS
    - scratch directory
    - AMS_ADF.template
    - TAPE21
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + centerstring('AMS Interface setup', 80) + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    log.info(string)

    log.info(centerstring('Path to AMS', 60, '-') + '\n')
    path = os.getenv('AMSHOME')
    if path:
        path = '$AMSHOME/'
    amsbashrc = question('Setup from amsbashrc.sh file?', bool, KEYSTROKES=KEYSTROKES, default=True)
    if amsbashrc:
        if path:
            path = '$AMSHOME/amsbashrc.sh'
        log.info('\nPlease specify path to the amsbashrc.sh file (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        path = question('Path to amsbashrc.sh file:', str, KEYSTROKES=KEYSTROKES, default=path)
        INFOS['amsbashrc'] = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        log.info('Will use amsbashrc= %s' % INFOS['amsbashrc'])
        INFOS['ams'] = '$AMSHOME'
        INFOS['scmlicense'] = '$SCMLICENSE'
        log.info('')
    else:
        log.info('\nPlease specify path to AMS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        INFOS['ams'] = question('Path to AMS:', str, KEYSTROKES=KEYSTROKES, default=path)
        log.info('')
        log.info(centerstring('Path to AMS license file', 60, '-') + '\n')
        path = os.getenv('SCMLICENSE')
        # path=os.path.expanduser(os.path.expandvars(path))
        if path == '':
            path = None
        else:
            path = '$SCMLICENSE'
        log.info('\nPlease specify path to AMS license.txt\n')
        INFOS['scmlicense'] = question('Path to license:', str, KEYSTROKES=KEYSTROKES, default=path)
        log.info('')


    # scratch
    log.info(centerstring('Scratch directory', 60, '-') + '\n')
    log.info('Please specify an appropriate scratch directory. This will be used to run the AMS calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str, KEYSTROKES=KEYSTROKES)
    log.info('')


    # template file
    print(centerstring('AMS input template file', 60, '-') + '\n')
    print('''Please specify the path to the AMS_ADF.template file. This file must contain the following keywords:

basis <basis>
functional <type> <name>
charge <x> [ <x2> [ <x3> ...] ]

The AMS interface will generate the appropriate AMS input automatically.
''')
    if os.path.isfile('AMS_ADF.template'):
        if checktemplate_AMS('AMS_ADF.template', INFOS):
            log.info('Valid file "AMS_ADF.template" detected. ')
            usethisone = question('Use this template file?', bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                INFOS['AMS_ADF.template'] = 'AMS_ADF.template'
    if 'AMS_ADF.template' not in INFOS:
        while True:
            filename = question('Template filename:', str, KEYSTROKES=KEYSTROKES)
            if not os.path.isfile(filename):
                log.info('File %s does not exist!' % (filename))
                continue
            if checktemplate_AMS(filename, INFOS):
                break
        INFOS['AMS_ADF.template'] = filename
    log.info('')


    # initial MOs
    log.info(centerstring('Initial restart: MO Guess', 60, '-') + '\n')
    log.info('''Please specify the path to an AMS engine file (e.g. adf.rkf) containing suitable starting MOs for the AMS calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    if question('Do you have a restart file?', bool, KEYSTROKES=KEYSTROKES, default=True):
        if True:
            filename = question('Restart file:', str, KEYSTROKES=KEYSTROKES, default='AMS.t21.init')
            INFOS['ams.guess'] = filename
    else:
        log.info('WARNING: Remember that the calculations may take longer without an initial guess for the MOs.')
        # time.sleep(2)
        INFOS['ams.guess'] = {}



    # Resources
    log.info(centerstring('AMS Ressource usage', 60, '-') + '\n')
    log.info('''Please specify the number of CPUs to be used by EACH calculation.
''')
    INFOS['ams.ncpu'] = abs(question('Number of CPUs:', int, KEYSTROKES=KEYSTROKES)[0])

    if INFOS['ams.ncpu'] > 1:
        log.info('''Please specify how well your job will parallelize.
A value of 0 means that running in parallel will not make the calculation faster, a value of 1 means that the speedup scales perfectly with the number of cores.
Typical values for AMS are 0.90-0.98 for LDA/GGA functionals and 0.50-0.80 for hybrids (better if RIHartreeFock is used).''')
        INFOS['ams.scaling'] = min(1.0, max(0.0, question('Parallel scaling:', float, KEYSTROKES=KEYSTROKES, default=[0.8])[0]))
    else:
        INFOS['ams.scaling'] = 0.9


    # Overlaps
    # if need_wfoverlap:
    if 'wfoverlap' in INFOS['needed']:
        log.info('\n' + centerstring('Wfoverlap code setup', 60, '-') + '\n')
        INFOS['ams.wfoverlap'] = question('Path to wavefunction overlap executable:', str, KEYSTROKES=KEYSTROKES, default='$SHARC/wfoverlap.x')
        log.info('')
        log.info('''State threshold for choosing determinants to include in the overlaps''')
        log.info('''For hybrids (and without TDA) one should consider that the eigenvector X may have a norm larger than 1''')
        INFOS['ams.ciothres'] = question('Threshold:', float, KEYSTROKES=KEYSTROKES, default=[0.998])[0]
        log.info('')
        INFOS['ams.mem'] = question('Memory for wfoverlap (MB):', int, KEYSTROKES=KEYSTROKES, default=[1000])[0]
        # TODO not asked: numfrozcore and numocc

        # print('Please state the number of core orbitals you wish to freeze for the overlaps (recommended to use for at least the 1s orbital and a negative number uses default values)?')
        # print('A value of -1 will use the defaults used by AMS for a small frozen core and 0 will turn off the use of frozen cores')
        # INFOS['frozcore_number']=question('How many orbital to freeze?',int,[-1])[0]


    # TheoDORE
    theodore_spelling = ['Om',
                         'PRNTO',
                         'Z_HE', 'S_HE', 'RMSeh',
                         'POSi', 'POSf', 'POS',
                         'PRi', 'PRf', 'PR', 'PRh',
                         'CT', 'CT2', 'CTnt',
                         'MC', 'LC', 'MLCT', 'LMCT', 'LLCT',
                         'DEL', 'COH', 'COHh']
    log.info('\n' + centerstring('Wave function analysis by TheoDORE', 60, '-') + '\n')
    # INFOS['theodore']=question('TheoDORE analysis?',bool,False)
    if 'theodore' in INFOS['needed']:

        INFOS['ams.theodore'] = question('Path to TheoDORE directory:', str, KEYSTROKES=KEYSTROKES, default='$THEODIR')
        log.info('')

        log.info('Please give a list of the properties to calculate by TheoDORE.\nPossible properties:')
        string = ''
        for i, p in enumerate(theodore_spelling):
            string += '%s ' % (p)
            if (i + 1) % 8 == 0:
                string += '\n'
        log.info(string)
        line = question('TheoDORE properties:', str, KEYSTROKES=KEYSTROKES, default='Om  PRNTO  S_HE  Z_HE  RMSeh')
        if '[' in line:
            INFOS['theodore.prop'] = ast.literal_eval(line)
        else:
            INFOS['theodore.prop'] = line.split()
        log.info('')

        log.info('Please give a list of the fragments used for TheoDORE analysis.')
        log.info('You can use the list-of-lists from dens_ana.in')
        log.info('Alternatively, enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
        INFOS['theodore.frag'] = []
        while True:
            line = question('TheoDORE fragment:', str, KEYSTROKES=KEYSTROKES, default='end')
            if 'end' in line.lower():
                break
            if '[' in line:
                try:
                    INFOS['theodore.frag'] = ast.literal_eval(line)
                    break
                except ValueError:
                    continue
            f = [int(i) for i in line.split()]
            INFOS['theodore.frag'].append(f)
        INFOS['theodore.count'] = len(INFOS['theodore.prop']) + len(INFOS['theodore.frag'])**2
        if 'AMS.ctfile' in INFOS:
            INFOS['theodore.count'] += 7


    return INFOS

# =================================================

def prepare_AMS(INFOS, iconddir):
    # write AMS_ADF.resources
    try:
        sh2cas = open('%s/AMS_ADF.resources' % (iconddir), 'w')
    except IOError:
        log.info('IOError during prepareAMS, iconddir=%s' % (iconddir))
        quit(1)
#  project='AMS'
    string = 'amshome %s\nscmlicense %s\nscratchdir %s/%s/\nsavedir %s/%s/restart\nncpu %i\nschedule_scaling %f\n' % (INFOS['ams'], INFOS['scmlicense'], INFOS['scratchdir'], iconddir, INFOS['copydir'], iconddir, INFOS['ams.ncpu'], INFOS['ams.scaling'])
    if 'wfoverlap' in INFOS['needed']:
        string += 'wfoverlap %s\nwfthres %f\n' % (INFOS['ams.wfoverlap'], INFOS['ams.ciothres'])
        string += 'memory %i\n' % (INFOS['ams.mem'])
        # string+='numfrozcore %i\n' %(INFOS['frozcore_number'])
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
    cpfrom = INFOS['AMS_ADF.template']
    cpto = '%s/AMS_ADF.template' % (iconddir)
    shutil.copy(cpfrom, cpto)

    if INFOS['ams.guess']:
        cpfrom1 = INFOS['ams.guess']
        cpto1 = '%s/AMS.t21_init' % (iconddir)
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
        log.info('Could not open template file %s' % (filename))
        return False
    valid = []
    for i in necessary:
        for line in data:
            if i in re.sub('#.*$', '', line):
                valid.append(True)
                break
        else:
            valid.append(False)
    if not all(valid):
        log.info('The template %s seems to be incomplete! It should contain: ' % (filename) + str(necessary))
        return False
    roots_there = False
    for line in data:
        line = re.sub('#.*$', '', line).lower().split()
        if len(line) == 0:
            continue
        if 'nstate' in line[0]:
            roots_there = True
    if not roots_there:
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            valid = []
            for line in data:
                if 'spin' in re.sub('#.*$', '', line).lower():
                    f = line.split()
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
        log.info(string)
        return False
    return True

# =================================================


def get_BAGEL(INFOS, KEYSTROKES: Optional[TextIOWrapper] = None):
    '''This routine asks for all questions specific to BAGEL:
    - path to bagel
    - scratch directory
    - BAGEL.template
    - wf.init
    '''

    string = '\n  ' + '=' * 80 + '\n'
    string += '||' + centerstring('BAGEL Interface setup', 80) + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    log.info(string)

    log.info(centerstring('Path to BAGEL', 60, '-') + '\n')
    path = os.getenv('BAGEL')
    # path=os.path.expanduser(os.path.expandvars(path))
    if path == '':
        path = None
    else:
        path = '$BAGEL/'
        # log.info('Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path))
        # if question('Do you want to use this MOLCAS installation?',bool,True):
        # INFOS['molcas']=path
        # if 'molcas' not in INFOS:
    log.info('\nPlease specify path to BAGEL directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
    INFOS['bagel'] = question('Path to BAGEL:', str, KEYSTROKES=KEYSTROKES, default=path)
    log.info('')


    log.info(centerstring('Scratch directory', 60, '-') + '\n')
    log.info('Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
    INFOS['scratchdir'] = question('Path to scratch directory:', str, KEYSTROKES=KEYSTROKES)
    log.info('')


    log.info(centerstring('BAGEL input template file', 60, '-') + '\n')
    log.info('''Please specify the path to the BAGEL.template file. This file must contain the following settings:

basis <Basis set>
df_basis <Density fitting basis set>
nact <Number of active orbitals>
nclosed <Number of doubly occupied orbitals>
nstate <Number of states for state-averaging>

The BAGEL interface will generate the appropriate BAGEL input automatically.
''')
    if os.path.isfile('BAGEL.template'):
        if checktemplate_BAGEL('BAGEL.template', INFOS):
            log.info('Valid file "BAGEL.template" detected. ')
            usethisone = question('Use this template file?', bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                INFOS['bagel.template'] = 'BAGEL.template'
    if 'bagel.template' not in INFOS:
        while True:
            filename = question('Template filename:', str, KEYSTROKES=KEYSTROKES)
            if not os.path.isfile(filename):
                log.info('File %s does not exist!' % (filename))
                continue
            if checktemplate_BAGEL(filename, INFOS):
                break
        INFOS['bagel.template'] = filename
    log.info('')

    log.info(centerstring('Dipole level', 60, '-') + '\n')
    log.info('Please specify the desired amount of calculated dipole moments:\n0 -only dipole moments that are for free are calculated\n1 -calculate all transition dipole moments between the (singlet) ground state and all singlet states for absorption spectra\n2 -calculate all dipole moments')
    INFOS['dipolelevel'] = question('Requested dipole level:', int, KEYSTROKES=KEYSTROKES, default=[0])[0]
    log.info('')

    log.info(centerstring('Initial wavefunction: MO Guess', 60, '-') + '\n')
    log.info('''Please specify the path to a MOLCAS JobIph file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
''')
    INFOS['bagel.guess'] = {}
    string = 'Do you have initial wavefunction files for '
    for mult, state in enumerate(INFOS['states']):
        if state <= 0:
            continue
        string += '%s, ' % (IToMult[mult + 1])
    string = string[:-2] + '?'
    if question(string, bool, KEYSTROKES=KEYSTROKES, default=True):
        for mult, state in enumerate(INFOS['states']):
            if state <= 0:
                continue
            while True:
                guess_file = 'archive.%i.init' % (mult + 1)
                filename = question('Initial wavefunction file for %ss:' % (IToMult[mult + 1]), str, KEYSTROKES=KEYSTROKES, default=guess_file)
                if os.path.isfile(filename):
                    INFOS['bagel.guess'][mult + 1] = filename
                    break
                else:
                    log.info('File not found!')
    else:
        log.info('WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.')

    log.info(centerstring('BAGEL Ressource usage', 60, '-') + '\n')  # TODO

    log.info('''Please specify the number of CPUs to be used by EACH calculation.''')
    INFOS['bagel.ncpu'] = abs(question('Number of CPUs:', int, KEYSTROKES=KEYSTROKES, default=[1])[0])

    if INFOS['bagel.ncpu'] > 1:
        INFOS['bagel.mpi'] = question('Use MPI mode (no=OpenMP)?', bool, KEYSTROKES=KEYSTROKES, default=False)
    else:
        INFOS['bagel.mpi'] = False
    # Ionization
    # need_wfoverlap=False
    # log.info(centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
    # INFOS['ion']=question('Dyson norms?',bool,False)
    # if 'ion' in INFOS and INFOS['ion']:
        # need_wfoverlap=True

    # wfoverlap
    if 'wfoverlap' in INFOS['needed']:
        log.info('\n' + centerstring('WFoverlap setup', 60, '-') + '\n')
        INFOS['bagel.wfoverlap'] = question('Path to wavefunction overlap executable:', str, KEYSTROKES=KEYSTROKES, default='$SHARC/wfoverlap.x')
        # TODO not asked for: numfrozcore, numocc
        log.info('''Please specify the amount of memory available to wfoverlap.x (in MB). \n (Note that BAGEL's memory cannot be controlled)
''')
        INFOS['bagel.mem'] = abs(question('wfoverlap.x memory:', int, KEYSTROKES=KEYSTROKES, default=[1000])[0])
    else:
        INFOS['bagel.mem'] = 1000

    return INFOS

# =================================================


def prepare_BAGEL(INFOS, iconddir):
    # write BAGEL.resources
    try:
        sh2cas = open('%s/BAGEL.resources' % (iconddir), 'w')
    except IOError:
        log.info('IOError during prepareBAGEL, iconddir=%s' % (iconddir))
        quit(1)
    project = 'BAGEL'
    string = 'bagel %s\nscratchdir %s/%s/\nmemory %i\nncpu %i\ndipolelevel %i\nproject %s' % (INFOS['bagel'], INFOS['scratchdir'], iconddir, INFOS['bagel.mem'], INFOS['bagel.ncpu'], INFOS['dipolelevel'], project)

    if INFOS['bagel.mpi']:
        string += 'mpi\n'
    if 'wfoverlap' in INFOS['needed']:
        string += '\nwfoverlap %s\n' % INFOS['bagel.wfoverlap']
    else:
        string += '\nnooverlap\n'
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










class SHARC_LEGACY(SHARC_INTERFACE):
    _version = VERSION
    _versiondate = VERSIONDATE
    _changelogstring = CHANGELOGSTRING
    _step = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "child-dir": None,
                "child-program": None
            }
        )
        self.QMin.template.types.update(
            {
                "child-dir": str,
                "child-program": str
            }
        )

        self.legacy_interface = None



    @staticmethod
    def description():
        return "Basic interface to run legacy interfaces via file I/O"

    @staticmethod
    def version():
        return SHARC_LEGACY._version

    @staticmethod
    def name() -> str:
        return "LEGACY"
    
    @staticmethod
    def about():
        pass

    @staticmethod
    def versiondate():
        return SHARC_LEGACY._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_LEGACY._changelogstring

    @staticmethod
    def authors() -> str:
        return "Sebastian Mai"


    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:
        if self.legacy_interface is None:
            self.log.info('  '+"=" * 76)
            self.log.info(f"||{'LEGACY interface selection':^76}||")
            self.log.info('  '+"=" * 76)
            self.log.info("\n")

            self.log.info('Please specify the desired legacy interface')
            for i in Interfaces:
                self.log.info('%i\t%s' % (i, Interfaces[i]['description']))
            self.log.info('')
            
            while True:
                num = question('Interface number:', int, KEYSTROKES=KEYSTROKES)[0]
                if num in Interfaces:
                    break
                else:
                    self.log.info('Please input one of the following: %s!' % ([i for i in Interfaces]))
            self.legacy_interface = num
        
        features = set(Interfaces[self.legacy_interface]['features'])

        self.log.debug(features)
        return features
    


    def get_infos(self, INFOS, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info('  '+"=" * 76)
        self.log.info(f"||{'LEGACY interface setup':^76}||")
        self.log.info('  '+"=" * 76)
        self.log.info("\n")

        # get needed stuff
        INFOS['needed'] = set()
        for i in INFOS['needed_requests']:
            INFOS['needed'].update(Interfaces[self.legacy_interface]['features'][i])
        
        #self.log.info(INFOS)

        ## call get routine
        INFOS = globals()[Interfaces[self.legacy_interface]['get_routine']](INFOS, KEYSTROKES=KEYSTROKES)

        return INFOS



    def prepare(self, INFOS, dir_path) -> None:
        QMin = self.QMin

        # LEGACY.template
        NAME = Interfaces[self.legacy_interface]['name'].upper()
        string = 'child-dir %s\nchild-program %s' % (NAME, NAME)
        writefile(os.path.join(dir_path,'LEGACY.template'), string)

        # child setup
        qmdir = os.path.join(dir_path, NAME)
        mkdir(qmdir)
        
        # call prepare routine 
        globals()[Interfaces[self.legacy_interface]['prepare_routine']](INFOS, qmdir)

        # runQM.sh of child
        runname = os.path.join(qmdir, 'runQM.sh')
        runscript = open(runname, 'w')
        s = '''cd %s\n$SHARC/%s QM.in >> QM.log 2>> QM.err\nerr=$?\n\nexit $err''' % (NAME,Interfaces[self.legacy_interface]['script'])
        runscript.write(s)
        runscript.close()
        os.chmod(runname, os.stat(runname).st_mode | stat.S_IXUSR)



    def _step_logic(self):
        super()._step_logic()

    def write_step_file(self):
        super().write_step_file()
        
    def update_step(self, step: int = None):
        super().update_step(step)




    def read_template(self, template_file="LEGACY.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)
        if self.QMin.template['child-program'] is not None:
            for i in Interfaces:
                if self.QMin.template['child-program'].lower() == Interfaces[i]['name']:
                    self.legacy_interface = i
        else:
            self.log.error('No child interface given in LEGACY.template!')
            exit(1)

        



    def read_resources(self, resources_filename="LEGACY.resources"):
        # super().read_resources(resources_filename)
        self._read_resources = True


    def setup_interface(self):
        self.qm_savedir = os.path.join(self.QMin.save["savedir"], Interfaces[self.legacy_interface]['name'].upper())
        if not os.path.isdir(self.qm_savedir):
            mkdir(self.qm_savedir)
        return




    def run(self):
        # coordinates
        string = '%i\n\n' % self.QMin.molecule['natom']
        for i,atom in enumerate(self.QMin.coords["coords"]):
            string += '%s  %12.9f  %12.9f  %12.9f\n' % (self.QMin.molecule["elements"][i],atom[0],atom[1],atom[2])

        # information
        string += 'states '
        for i in self.QMin.molecule['states']:
            string += '%i ' % i
        string += '\nunit bohr\nsavedir %s\n' % self.qm_savedir
        if self.QMin.save['init']:
            string += 'init\n'
        elif self.QMin.save['samestep']:
            string += 'samestep\n'
        elif self.QMin.save['newstep']:
            pass
        string += 'retain %i\n' % self.QMin.requests['retain']

        # requests
        for key, value in self.QMin.requests.items():
            match key:
                case 'h' | 'soc' | 'dm' | 'overlap' | 'ion' | 'phases' | 'theodore':
                    if value:
                        string += key + '\n'
                case 'grad':
                    if self.QMin.requests[key] is not None:
                        string += key + ' ' + ' '.join([str(i) for i in value]) + '\n'
                case 'nacdr':
                    if self.QMin.requests[key] is not None:
                        string += key + ' select\n'
                        for pair in value:
                            string += '%i %i\n' % pair
                        string += 'end\n'
                
        # write QM.in file
        WORKDIR = os.path.abspath(self.QMin.template['child-dir'])
        filename = os.path.join(WORKDIR,'QM.in')
        writefile(filename, string)

        # call run script
        string = 'bash %s/runQM.sh' % self.QMin.template['child-dir']
        starttime = datetime.datetime.now()
        self.log.info('START:\t%s\t%s\t"%s"\n' % (WORKDIR, starttime, string))

        # No InDir() because runQM.sh contains the cd command
        stdoutfile = open(os.path.join(WORKDIR, 'runQM.out'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'runQM.err'), 'w')
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            self.log.error('Call have had some serious problems:', OSError)
            exit(22)
        stdoutfile.close()
        stderrfile.close()

        endtime = datetime.datetime.now()
        self.log.info('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (WORKDIR, endtime, endtime - starttime, runerror))
        
        return



    def getQMout(self):
        WORKDIR =  os.path.abspath(self.QMin.template['child-dir'])
        filename = os.path.join(WORKDIR,'QM.out')
        if not os.path.isfile(filename):
            self.log.error('No QM.out file found!')
            exit(1)
        
        # make QMout
        self.QMout = QMout(states = self.QMin.molecule["states"])
        requests = set()
        for k, v in self.QMin.requests.items():
            if v in (None, False, []):
                continue
            requests.add(k)
        self.QMout.allocate(states = self.QMin.molecule["states"],
                            natom = self.QMin.molecule['natom'],
                            npc = self.QMin.molecule['npc'],    
                            requests = requests)
        
        # get QMout of legacy child
        QMout2 = QMout(filepath = filename, 
                       states = self.QMin.molecule["states"],
                       natom = self.QMin.molecule['natom'],
                       npc = self.QMin.molecule['npc']
                       )
        self.log.debug(QMout2)
        
        # assign stuff
        items = ['h', 'dm', 'grad', 'overlap', 'phases', 'prop1d', 'prop2d', 'nacdr']
        for i in items:
            if i in self.QMout:
                self.QMout[i] = QMout2[i]

        self.QMout.runtime = self.clock.measuretime()
        return self.QMout



    def create_restart_files(self):
        pass





if __name__ == "__main__":
    from logger import loglevel

    try:
        legacy = SHARC_LEGACY(loglevel=loglevel)
        legacy.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
