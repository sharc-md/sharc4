#!/usr/bin/env python3


import os
import sys
import re
import ast
import math
import shutil
import pprint
import struct
import subprocess as sp
from datetime import datetime
from copy import deepcopy
from socket import gethostname

from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import au2a, kcal_to_Eh, NUMBERS, BASISSETS, IToMult, rcm_to_Eh
import civfl_ana

authors = 'Sebastian Mai, Severin Polonius'
version = '3.0'
versiondate = datetime(2021, 7, 15)

changelogstring = '''
12.02.2016:     Initial version 0.1
- CC2 and ADC(2) from Turbomole
- SOC from Orca call
- only singlets and triplets
  => Doublets and Dyson could be added later

15.03.2016:     0.1.1
- ridft can be used for the SCF calculation, but dscf has to be run afterwards anyways (for SOC and overlaps).
- Laplace-transformed SOS-CC2/ADC(2) can be used ("spin-scaling lt-sos"), but does not work for soc or trans-dm
- if ricc2 does not converge, it is rerun with different combinations of npre and nstart (until convergence or too many tries)

07.03.2017:
- wfthres is now interpreted as in other interfaces (default is 0.99 now)

25.04.2017:
- can use external basis set libraries

23.08.2017
- Resource file is now called "RICC2.resources" instead of "SH2CC2.inp" (old filename still works)

24.08.2017:
- numfrozcore in resources file can now be used to override number of frozen cores for overlaps
- added Theodore capabilities (compute descriptors, OmFrag, and NTOs (also activate MOLDEN key for that))

11.11.2020:
- COBRAMM can be used for QM/MM calculations
'''

shift_mask = {1: (+1, +1),
              2: (-2, -1),
              3: (+1, +1),
              4: (+1, +1),
              5: (+1, +1),
              6: (+1, +1)}


class RICC2(INTERFACE):
    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring

    @property
    def version(self):
        return self._version

    @property
    def versiondate(self):
        return self._versiondate

    @property
    def changelogstring(self):
        return self._changelogstring

    @property
    def authors(self):
        return self._authors

    def readQMin(self, QMinfilename):
        '''Reads the time-step dependent information from QMinfilename. This file contains all information from the current SHARC job: geometry, velocity, number of states, requested quantities along with additional information. The routine also checks this input and obtains a number of environment variables necessary to run COLUMBUS.

        Reads also the information from SH2COL

        Steps are:
        - open and read QMinfilename
        - Obtain natom, comment, geometry (, velocity)
        - parse remaining keywords from QMinfile
        - check keywords for consistency, calculate nstates, nmstates
        - obtain environment variables for path to COLUMBUS and scratch directory, and for error handling

        Arguments:
        1 string: name of the QMin file

        Returns:
        1 dictionary: QMin'''
        # read QMinfile
        QMinlines = readfile(QMinfilename)
        QMin = self._QMin

        # Get natom
        try:
            natom = int(QMinlines[0])
        except ValueError:
            print('first line must contain the number of atoms!')
            sys.exit(49)
        QMin['natom'] = natom
        if len(QMinlines) < natom + 4:
            print('Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task')
            sys.exit(50)

        # Save Comment line
        QMin['comment'] = QMinlines[1]

        # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
        QMin['geo'] = []
        QMin['veloc'] = []
        hasveloc = True
        for i in range(2, natom + 2):
            if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]):
                print('Input file does not comply to xyz file format! Maybe natom is just wrong.')
                sys.exit(51)
            fields = QMinlines[i].split()
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
            if len(args) >= 1 and 'select' in args[0]:
                pairs, i = self._get_pairs(QMinlines, i)
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
                sys.exit(52)
        else:
            factor = 1. / au2a

        for iatom in range(len(QMin['geo'])):
            for ixyz in range(3):
                QMin['geo'][iatom][ixyz + 1] *= factor


        if 'states' not in QMin:
            print('Keyword "states" not given!')
            sys.exit(53)

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
            sys.exit(54)

        possibletasks = ['h', 'soc', 'dm', 'grad', 'overlap', 'dmdr', 'socdr', 'ion', 'theodore', 'phases']
        if not any([i in QMin for i in possibletasks]):
            print('No tasks found! Tasks are "h", "soc", "dm", "grad","dmdr", "socdr", "overlap" and "ion".')
            sys.exit(55)

        if 'samestep' in QMin and 'init' in QMin:
            print('"Init" and "Samestep" cannot be both present in QM.in!')
            sys.exit(56)

        if 'phases' in QMin:
            QMin['overlap'] = []

        if 'overlap' in QMin and 'init' in QMin:
            print('"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"')
            sys.exit(57)

        if 'init' not in QMin and 'samestep' not in QMin and 'restart' not in QMin:
            QMin['newstep'] = []

        if not any([i in QMin for i in ['h', 'soc', 'dm', 'grad']]) and 'overlap' in QMin:
            QMin['h'] = []

        if len(QMin['states']) > 3:
            print('Higher multiplicities than triplets are not supported!')
            sys.exit(58)

        if len(QMin['states']) > 1 and QMin['states'][1] > 0:
            print('No doublet states supported currently!')
            sys.exit(59)

        if len(QMin['states']) == 1 and 'soc' in QMin:
            QMin = removekey(QMin, 'soc')
            QMin['h'] = []

        if 'h' in QMin and 'soc' in QMin:
            QMin = removekey(QMin, 'h')

        if 'nacdt' in QMin or 'nacdr' in QMin:
            print('Within the SHARC-RICC2 interface, couplings can only be calculated via the overlap method. "nacdr" and "nacdt" are not supported.')
            sys.exit(60)

        if 'socdr' in QMin or 'dmdr' in QMin:
            print('Within the SHARC-RICC2 interface, "dmdr" and "socdr" are not supported.')
            sys.exit(61)

        if 'ion' in QMin:
            print('Ionization probabilities not implemented!')
            sys.exit(62)

        if 'step' not in QMin:
            QMin['step'] = [0]

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
                        sys.exit(63)
                    if QMin['grad'][i] > nmstates:
                        print('State for requested gradient does not correspond to any state in QM input file state list!')
                        sys.exit(64)

        # Process the overlap requests
        # identically to the nac requests
        if 'overlap' in QMin:
            if len(QMin['overlap']) >= 1:
                nacpairs = QMin['overlap']
                for i in range(len(nacpairs)):
                    if nacpairs[i][0] > nmstates or nacpairs[i][1] > nmstates:
                        print('State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!')
                        sys.exit(65)
            else:
                QMin['overlap'] = [[j + 1, i + 1] for i in range(nmstates) for j in range(i + 1)]

        # obtain the statemap
        statemap = {}
        i = 1
        for imult, istate, ims in itnmstates(QMin['states']):
            statemap[i] = [imult, istate, ims]
            i += 1
        QMin['statemap'] = statemap

        # get the set of states for which gradients actually need to be calculated
        gradmap = set()
        if 'grad' in QMin:
            for i in QMin['grad']:
                gradmap.add(tuple(statemap[i][0:2]))
        gradmap = sorted(gradmap)
        QMin['gradmap'] = gradmap

        # --------------------------------------------- Resources ----------------------------------


        # environment setup

        QMin['pwd'] = os.getcwd()

        # open RICC2.resources
        filename = 'RICC2.resources'
        if os.path.isfile(filename):
            sh2cc2 = readfile(filename)
        else:
            print('HINT: reading resources from SH2CC2.inp')
            sh2cc2 = readfile('SH2CC2.inp')

        # ncpus for SMP-parallel turbomole and wfoverlap
        # this comes before the turbomole path determination
        QMin['ncpu'] = 1
        line = self.getsh2cc2key(sh2cc2, 'ncpu')
        if line[0]:
            try:
                QMin['ncpu'] = int(line[1])
                QMin['ncpu'] = max(1, QMin['ncpu'])
            except ValueError:
                print('Number of CPUs does not evaluate to numerical value!')
                sys.exit(66)
        os.environ['OMP_NUM_THREADS'] = str(QMin['ncpu'])
        if QMin['ncpu'] > 1:
            os.environ['PARA_ARCH'] = 'SMP'
            os.environ['PARNODES'] = str(QMin['ncpu'])


        # set TURBOMOLE paths
        QMin['turbodir'] = self.get_sh2cc2_environ(sh2cc2, 'turbodir')
        os.environ['TURBODIR'] = QMin['turbodir']
        arch = self.get_arch(QMin['turbodir'])
        os.environ['PATH'] = '%s/scripts:%s/bin/%s:' % (QMin['turbodir'], QMin['turbodir'], arch) + os.environ['PATH']

        # set ORCA paths
        if 'soc' in QMin:
            QMin['orcadir'] = self.get_sh2cc2_environ(sh2cc2, 'orcadir')
            os.environ['PATH'] = '%s:' % (QMin['orcadir']) + os.environ['PATH']
            os.environ['LD_LIBRARY_PATH'] = '%s:' % (QMin['orcadir']) + os.environ['LD_LIBRARY_PATH']


        # Set up scratchdir
        line = self.get_sh2cc2_environ(sh2cc2, 'scratchdir', False, False)
        if line is None:
            line = QMin['pwd'] + '/SCRATCHDIR/'
        line = os.path.expandvars(line)
        line = os.path.expanduser(line)
        line = os.path.abspath(line)
        # checkscratch(line)
        QMin['scratchdir'] = line


        # Set up savedir
        if 'savedir' in QMin:
            # savedir may be read from QM.in file
            line = QMin['savedir'][0]
        else:
            line = self.get_sh2cc2_environ(sh2cc2, 'savedir', False, False)
            if line is None:
                line = QMin['pwd'] + '/SAVEDIR/'
        line = os.path.expandvars(line)
        line = os.path.expanduser(line)
        line = os.path.abspath(line)
        if 'init' in QMin:
            self.checkscratch(line)
        QMin['savedir'] = line


        # debug keyword in SH2CC2
        line = self.getsh2cc2key(sh2cc2, 'debug')
        if line[0]:
            if len(line) <= 1 or 'true' in line[1].lower():
                global DEBUG
                DEBUG = True

        line = self.getsh2cc2key(sh2cc2, 'no_print')
        if line[0]:
            if len(line) <= 1 or 'true' in line[1].lower():
                global PRINT
                PRINT = False


        # memory for Turbomole, Orca, and wfoverlap
        QMin['memory'] = 100
        line = self.getsh2cc2key(sh2cc2, 'memory')
        if line[0]:
            try:
                QMin['memory'] = int(line[1])
                QMin['memory'] = max(100, QMin['memory'])
            except ValueError:
                print('Run memory does not evaluate to numerical value!')
                sys.exit(67)
        else:
            print('WARNING: Please set memory in RICC2.resources (in MB)! Using 100 MB default value!')


        # initial MO guess settings
        # if neither keyword is present, the interface will reuse MOs from savedir, or use the EHT guess
        line = self.getsh2cc2key(sh2cc2, 'always_orb_init')
        if line[0]:
            QMin['always_orb_init'] = []
        line = self.getsh2cc2key(sh2cc2, 'always_guess')
        if line[0]:
            QMin['always_guess'] = []
        if 'always_orb_init' in QMin and 'always_guess' in QMin:
            print('Keywords "always_orb_init" and "always_guess" cannot be used together!')
            sys.exit(68)


        # get the nooverlap keyword: no dets will be extracted if present
        line = self.getsh2cc2key(sh2cc2, 'nooverlap')
        if line[0]:
            if 'overlap' in QMin or 'phases' in QMin or 'ion' in QMin:
                print('"nooverlap" is incompatible with "overlap" or "phases"!')
                sys.exit(69)
            QMin['nooverlap'] = []


        # dipole moment calculation level
        QMin['dipolelevel'] = 2
        line = self.getsh2cc2key(sh2cc2, 'dipolelevel')
        if line[0]:
            try:
                QMin['dipolelevel'] = int(line[1])
            except ValueError:
                print('Run memory does not evaluate to numerical value!')
                sys.exit(70)


        # wfoverlaps setting
        QMin['wfthres'] = 0.99
        line = self.getsh2cc2key(sh2cc2, 'wfthres')
        if line[0]:
            QMin['wfthres'] = float(line[1])
        if 'overlap' in QMin:
            # QMin['wfoverlap']=get_sh2cc2_environ(sh2cc2,'wfoverlap')
            QMin['wfoverlap'] = self.get_sh2cc2_environ(sh2cc2, 'wfoverlap', False, False)
            if QMin['wfoverlap'] is None:
                ciopath = os.path.join(os.path.expandvars(os.path.expanduser('$SHARC')), 'wfoverlap.x')
                if os.path.isfile(ciopath):
                    QMin['wfoverlap'] = ciopath
                else:
                    print('Give path to wfoverlap.x in RICC2.resources!')
            line = self.getsh2cc2key(sh2cc2, 'numfrozcore')
            if line[0]:
                numfroz = int(line[1])
                if numfroz == 0:
                    QMin['ncore'] = 0
                elif numfroz > 0:
                    QMin['ncore'] = numfroz
                elif numfroz < 0:
                    pass        # here we rely on the frozen key from the template below


        # TheoDORE settings
        if 'theodore' in QMin:
            QMin['theodir'] = self.get_sh2cc2_environ(sh2cc2, 'theodir', False, False)
            if QMin['theodir'] is None or not os.path.isdir(QMin['theodir']):
                print('Give path to the TheoDORE installation directory in TURBOMOLE.resources!')
                sys.exit(71)
            os.environ['THEODIR'] = QMin['theodir']
            if 'PYTHONPATH' in os.environ:
                os.environ['PYTHONPATH'] += os.pathsep + os.path.join(QMin['theodir'], 'lib') + os.pathsep + QMin['theodir']
            else:
                os.environ['PYTHONPATH'] = os.path.join(QMin['theodir'], 'lib') + os.pathsep + QMin['theodir']


        # norestart setting
        line = self.getsh2cc2key(sh2cc2, 'no_ricc2_restart')
        if line[0]:
            QMin['no_ricc2_restart'] = []


        # neglected gradients
        QMin['neglected_gradient'] = 'zero'
        if 'grad' in QMin:
            line = self.getsh2cc2key(sh2cc2, 'neglected_gradient')
            if line[0]:
                if line[1].lower().strip() == 'zero':
                    QMin['neglected_gradient'] = 'zero'
                elif line[1].lower().strip() == 'gs':
                    QMin['neglected_gradient'] = 'gs'
                elif line[1].lower().strip() == 'closest':
                    QMin['neglected_gradient'] = 'closest'
                else:
                    print('Unknown argument to "neglected_gradient"!')
                    sys.exit(72)

        # --------------------------------------------- Template ----------------------------------

        # open template
        template = readfile('RICC2.template')

        QMin['template'] = {}
        integers = ['frozen', 'charge']
        strings = ['basis', 'auxbasis', 'method', 'scf', 'spin-scaling', 'basislib']
        floats = []
        booleans = ['douglas-kroll']
        for i in booleans:
            QMin['template'][i] = False
        QMin['template']['method'] = 'adc(2)'
        QMin['template']['scf'] = 'dscf'
        QMin['template']['spin-scaling'] = 'none'
        QMin['template']['basislib'] = ''
        QMin['template']['charge'] = 0
        QMin['template']['frozen'] = -1

        QMin['template']['theodore_prop'] = ['Om', 'PRNTO', 'S_HE', 'Z_HE', 'RMSeh']
        QMin['template']['theodore_fragment'] = []

        for line in template:
            line = re.sub('#.*$', '', line).lower().split()
            if len(line) == 0:
                continue
            elif line[0] in integers:
                QMin['template'][line[0]] = int(line[1])
            elif line[0] in booleans:
                QMin['template'][line[0]] = True
            elif line[0] in strings:
                QMin['template'][line[0]] = line[1]
            elif line[0] in floats:
                QMin['template'][line[0]] = float(line[1])

        necessary = ['basis']
        for i in necessary:
            if i not in QMin['template']:
                print('Key %s missing in template file!' % (i))
                sys.exit(73)

        # make basis set name in correct case, so that Turbomole recognizes them
        for basis in BASISSETS:
            if QMin['template']['basis'].lower() == basis.lower():
                QMin['template']['basis'] = basis
                break
        if 'auxbasis' in QMin['template']:
            for basis in BASISSETS:
                if QMin['template']['auxbasis'].lower() == basis.lower():
                    QMin['template']['auxbasis'] = basis
                    break
            if QMin['template']['basislib']:
                print('Keywords "basislib" and "auxbasis" cannot be used together in template!\nInstead, create a file for the auxbasis in /basislib/cbasen/')
                sys.exit(74)

        # go through sh2cc2 for the theodore settings
        for line in sh2cc2:
            orig = re.sub('#.*$', '', line).strip()
            line = orig.lower().split()
            if len(line) == 0:
                continue

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
                    l = []
                    for i in s:
                        l.append(int(i))
                    QMin['template']['theodore_fragment'].append(l)
# --    ------------------------------------------- QM/MM ----------------------------------

        # qmmm keyword
        QMin['qmmm'] = False
        QMin['template']['qmmm'] = False
        i = 0
        for line in template:
            line = re.sub('#.*$', '', line).lower().split()
            if len(line) < 1:
                continue
            if line[0] == 'qmmm':
                QMin['qmmm'] = True

        QMin['cobramm'] = False
        QMin['template']['cobramm'] = False
        i = 0
        for line in template:
            line = re.sub('#.*$', '', line).lower().split()
            if len(line) < 1:
                continue
            if line[0] == 'cobramm':
                QMin['cobramm'] = True
        if QMin['cobramm']:
            QMin['template']['cobramm'] = True

        # prepare everything
        if QMin['qmmm']:
            QMin['template']['qmmm'] = True

            # get settings from RICC2.resources
            # Tinker
            line = self.getsh2cc2key(sh2cc2, 'tinker')
            if not line[0]:
                print('TINKER path not given!')
                sys.exit(75)
            line = os.path.expandvars(line[1].strip())
            line = os.path.expanduser(line)
            line = os.path.abspath(line)
            QMin['tinker'] = line
            if not os.path.isfile(os.path.join(QMin['tinker'], 'bin', 'tkr2qm_s')):
                print('TINKER executable at "%s" not found!' % os.path.join(QMin['tinker'], 'bin', 'tkr2qm_s'))
                sys.exit(76)

            # table and ff files
            for line in sh2cc2:
                orig = re.sub('#.*$', '', line).strip()
                line = orig.lower().split()
                if len(line) == 0:
                    continue
                elif line[0] == 'qmmm_table':
                    line2 = orig.split(None, 1)
                    if len(line2) < 2:
                        print('Please specify a connection table file after "qmmm_table"!')
                        sys.exit(77)
                    filename = os.path.abspath(os.path.expandvars(os.path.expanduser(line2[1])))
                    QMin['template']['qmmm_table'] = filename
                elif line[0] == 'qmmm_ff_file':
                    line2 = orig.split(None, 1)
                    if len(line2) < 2:
                        print('Please specify a force field file after "qmmm_ff_file"!')
                        sys.exit(78)
                    filename = os.path.abspath(os.path.expandvars(os.path.expanduser(line2[1])))
                    QMin['template']['qmmm_ff_file'] = filename

            # prepare data structures and run Tinker
            QMin['qmmm'] = self.prepare_QMMM(QMin['template']['qmmm_table'])
            QMMMout = self.execute_tinker(QMin['template']['qmmm_ff_file'])

            # modify QMin dict
            QMin['geo_orig'] = QMin['geo']
            QMin['geo'] = QMin['qmmm']['QM_coords']
            QMin['natom_orig'] = QMin['natom']
            QMin['natom'] = len(QMin['geo'])
            QMin['pointcharges'] = deepcopy(QMin['qmmm']['pointcharges'])
            # QMin['pointcharges']=[]
            # for iatom in range(QMin['qmmm']['natom_table']):
            # atom=QMin['qmmm']['MM_coords'][iatom]
            # for iatom,atom in enumerate(QMin['qmmm']['MM_coords']):
            # QMin['pointcharges'].append( [atom[1],atom[2],atom[3],QMin['qmmm']['MMpc'][iatom]] )



# --    ------------------------------------------- logic checks ----------------------------------





        # logic checks:

        # find method
        allowed_methods = ['cc2', 'adc(2)']
        for m in allowed_methods:
            if QMin['template']['method'] == m:
                QMin['method'] = m
                break
        else:
            print('Unknown method "%s" given in RICC2.template' % (QMin['template']['method']))
            sys.exit(79)

        # find spin-scaling
        allowed_methods = ['scs', 'sos', 'lt-sos', 'none']
        if not any([QMin['template']['spin-scaling'] == i for i in allowed_methods]):
            print('Unknown spin-scaling "%s" given in RICC2.template' % (QMin['template']['spin-scaling']))
            sys.exit(80)

        # find SCF program
        allowed_methods = ['dscf', 'ridft']
        if not any([QMin['template']['scf'] == i for i in allowed_methods]):
            print('Unknown SCF program "%s" given in RICC2.template' % (QMin['template']['scf']))
            sys.exit(81)

        # get number of electrons
        nelec = 0
        for atom in QMin['geo']:
            nelec += NUMBERS[atom[0].title()]
        nelec -= QMin['template']['charge']
        QMin['nelec'] = nelec
        if nelec % 2 != 0:
            print('Currently, only even-electronic systems are possible in SHARC_RICC2.py!')
            sys.exit(82)

        # no soc for elements beyond Kr due to ECP
        if 'soc' in QMin and any([NUMBERS[atom[0].title()] > 36 for atom in QMin['geo']]):
            print('Spin-orbit couplings for elements beyond Kr do not work due to default ECP usage!')
            sys.exit(83)

        # soc and cc2 do not work together
        if 'soc' in QMin and 'cc2' in QMin['template']['method']:
            print('Currently, spin-orbit coupling is not possible at CC2 level. Please use ADC(2)!')
            sys.exit(84)

        # lt-sos-CC2/ADC(2) does not work in certain cases
        if 'lt-sos' in QMin['template']['spin-scaling']:
            if QMin['ncpu'] > 1:
                print('NOTE: Laplace-transformed SOS-%s is not fully SMP parallelized.' % (QMin['template']['method'].upper()))
                # sys.exit(85)
            if 'soc' in QMin:
                print('Laplace-transformed SOS-%s is not compatible with SOC calculation!' % (QMin['template']['method'].upper()))
                sys.exit(86)
            if QMin['dipolelevel'] == 2:
                print('Laplace-transformed SOS-%s is not compatible with dipolelevel=2!' % (QMin['template']['method'].upper()))
                sys.exit(87)

        # number of properties/entries calculated by TheoDORE
        if 'theodore' in QMin:
            QMin['template']['theodore_n'] = len(QMin['template']['theodore_prop']) + len(QMin['template']['theodore_fragment'])**2
        else:
            QMin['template']['theodore_n'] = 0


        # Check the save directory
        try:
            ls = os.listdir(QMin['savedir'])
            err = 0
        except OSError:
            err = 1
        if 'init' in QMin:
            err = 0
        elif 'overlap' in QMin:
            if 'newstep' in QMin:
                if 'mos' not in ls:
                    print('File "mos" missing in SAVEDIR!')
                    err += 1
                if 'coord' not in ls:
                    print('File "coord" missing in SAVEDIR!')
                    err += 1
                for imult, nstates in enumerate(QMin['states']):
                    if nstates < 1:
                        continue
                    if not 'dets.%i' % (imult + 1) in ls:
                        print('File "dets.%i.old" missing in SAVEDIR!' % (imult + 1))
                        err += 1
            elif 'samestep' in QMin or 'restart' in QMin:
                if 'mos.old' not in ls:
                    print('File "mos" missing in SAVEDIR!')
                    err += 1
                if 'coord.old' not in ls:
                    print('File "coord.old" missing in SAVEDIR!')
                    err += 1
                for imult, nstates in enumerate(QMin['states']):
                    if nstates < 1:
                        continue
                    if not 'dets.%i.old' % (imult + 1) in ls:
                        print('File "dets.%i.old" missing in SAVEDIR!' % (imult + 1))
                        err += 1
        if err > 0:
            print('%i files missing in SAVEDIR=%s' % (err, QMin['savedir']))
            sys.exit(88)

        if PRINT:
            self.printQMin(QMin)

        return

# ======================================================================= #
    # TODO potentially general -> SHARC_INTERFACE
    def printQMin(self):
        QMin = self._QMin()
        if DEBUG:
            pprint.pprint(QMin)
        if not PRINT:
            return
        print('==> QMin Job description for:\n%s' % (QMin['comment']))

        string = 'Tasks:  '
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
            string += '\tOverlaps'
        if 'angular' in QMin:
            string += '\tAngular'
        if 'ion' in QMin:
            string += '\tDyson norms'
        if 'dmdr' in QMin:
            string += '\tDM-Grad'
        if 'socdr' in QMin:
            string += '\tSOC-Grad'
        if 'theodore' in QMin:
            string += '\tTheoDORE'
        if 'phases' in QMin:
            string += '\tPhases'
        print(string)

        string = 'States: '
        for i in itmult(QMin['states']):
            string += '\t%i %s' % (QMin['states'][i - 1], IToMult[i])
        print(string)



        string = 'Method: \t'
        if QMin['template']['spin-scaling'] != 'none':
            string += QMin['template']['spin-scaling'].upper() + '-'
        string += QMin['template']['method'].upper()
        string += '/%s' % (QMin['template']['basis'])
        parts = []
        if QMin['template']['douglas-kroll']:
            parts.append('Douglas-Kroll')
        if QMin['template']['frozen'] != -1:
            parts.append('RICC2 frozen orbitals=%i' % (QMin['template']['frozen']))
        if QMin['template']['scf'] == 'ridft':
            parts.append('RI-SCF')
        if len(parts) > 0:
            string += '\t('
            string += ','.join(parts)
            string += ')'
        print(string)


        # if 'dm' in QMin and QMin['template']['method']=='adc(2)':
        # print('WARNING: excited-to-excited transition dipole moments in ADC(2) are zero!')


        if 'dm' in QMin and QMin['template']['method'] == 'cc2' and (QMin['states'][0] > 1 and len(QMin['states']) > 2 and QMin['states'][2] > 0):
            print('WARNING: will not calculate transition dipole moments! For CC2, please use only singlet states or ground state + triplets.')


        string = 'Found Geo'
        if 'veloc' in QMin:
            string += ' and Veloc! '
        else:
            string += '! '
        string += 'NAtom is %i.\n' % (QMin['natom'])
        print(string)

        string = '\nGeometry in Bohrs:\n'
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

        # if 'overlap' in QMin:
            # string='Overlaps:\n'
            # for i in range(1,QMin['nmstates']+1):
            # for j in range(1,QMin['nmstates']+1):
            # if [i,j] in QMin['overlap'] or [j,i] in QMin['overlap']:
            # string+='X '
            # else:
            # string+='. '
            # string+='\n'
            # print(string)

        for i in QMin:
            if not any([i == j for j in ['h', 'dm', 'soc', 'dmdr', 'socdr', 'theodore', 'geo', 'veloc', 'states', 'comment', 'LD_LIBRARY_PATH', 'grad', 'nacdr', 'ion', 'overlap', 'template', 'qmmm', 'cobramm', 'geo_orig', 'pointcharges']]):
                if not any([i == j for j in ['ionlist', 'ionmap']]) or DEBUG:
                    string = i + ': '
                    string += str(QMin[i])
                    print(string)
            else:
                string = i + ': ...'
                print(string)
        print('\n')
        sys.stdout.flush()


    # ====================== SUB ROUTINES ============================================+


    def get_RICC2out(self, job):
        QMin = self._QMin
        QMout = self._QMout
        # job contains: 'tmexc_soc','tmexc_dm', 'spectrum','exprop_dm','static_dm','gsgrad','exgrad', 'E'
        # reads ricc2.out and adds matrix elements to QMout
        ricc2 = readfile(QMin['scratchdir'] + '/JOB/ricc2.out')

        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']

        # Hamiltonian
        if 'h' in QMin or 'soc' in QMin:
            # make Hamiltonian
            if 'h' not in QMout:
                QMout['h'] = makecmatrix(nmstates, nmstates)
            # read the matrix elements from ricc2.out
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if 'soc' in QMin and 'tmexc_soc' in job:
                        # read SOC elements
                        QMout['h'][i][j] = self.getsocme(ricc2, i + 1, j + 1)
                    elif 'E' in job:
                        # read only diagonal elements
                        if i == j:
                            QMout['h'][i][j] = self.getenergy(ricc2, i + 1)

        # Dipole moments
        if 'dm' in QMin:
            # make dipole matrices
            if 'dm' not in QMout:
                QMout['dm'] = []
                for i in range(3):
                    QMout['dm'].append(makecmatrix(nmstates, nmstates))
            # read the elements from ricc2.out
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if m1 != m2:
                        continue
                    if ms1 != ms2:
                        continue
                    if (m1, s1) == (m2, s2) == (1, 1):
                        if 'static_dm' in job or ('gsgrad', 1, 1) in job:
                            for xyz in range(3):
                                QMout['dm'][xyz][i][j] = self.getdiagdm(ricc2, i + 1, xyz)
                    elif i == j:
                        if 'exprop_dm' in job or ('exgrad', m1, s1) in job:
                            for xyz in range(3):
                                QMout['dm'][xyz][i][j] = self.getdiagdm(ricc2, i + 1, xyz)
                    elif (m1, s1) == (1, 1) or (m2, s2) == (1, 1):
                        if 'spectrum' in job:
                            for xyz in range(3):
                                QMout['dm'][xyz][i][j] = self.gettransdm(ricc2, i + 1, j + 1, xyz)
                    else:
                        if 'tmexc_dm' in job:
                            for xyz in range(3):
                                QMout['dm'][xyz][i][j] = self.gettransdm(ricc2, i + 1, j + 1, xyz)

        # Gradients
        if 'grad' in QMin:
            # make vectors
            if 'grad' not in QMout:
                QMout['grad'] = [[[0., 0., 0.] for i in range(natom)] for j in range(nmstates)]
            if QMin['qmmm'] and 'pcgrad' not in QMout:
                QMout['pcgrad'] = [[[0. for i in range(3)] for j in QMin['pointcharges']] for k in range(nmstates)]
            if QMin['cobramm'] and 'pcgrad' not in QMout:
                ncharges = len(readfile('charge.dat'))  # -4
                # print(ncharges,"tot")
                QMout['pcgrad'] = [[[0. for i in range(3)] for j in range(ncharges)] for k in range(nmstates)]
                print(QMout['pcgrad'])
    #      pcgrad=os.path.join(QMin['scratchdir'],'JOB','pc_grad')
    #      specify_state=os.path.join(QMin['scratchdir'],'JOB','pc_grad.old.%s') % (nmstates)#% (mult,nexc)
            # shutil.copy(pcbrad,specify_state)
    #      shutil.move(pcgrad,specify_state)
    #    if QMin['cobramm'] and not 'pcgrad' in QMout: ##21.09.20
    #      QMout['pcgrad']=[ [ [ 0. for i in range(3) ] for j in QMin['pointcharges'] ] for k in range(nmstates) ]
            # read the elements from ricc2.out
            for i in range(nmstates):
                m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                if (m1, s1) == (1, 1):
                    tup = ('gsgrad', 1, 1)
                else:
                    tup = ('exgrad', m1, s1)
                if tup in job:
                    QMout['grad'][i] = self.getgrad(ricc2, i + 1)
                    if QMin['qmmm']:
                        QMout['pcgrad'][i - 1] = self.getpcgrad()
                    if QMin['cobramm']:
                        logfile = os.path.join(QMin['scratchdir'], 'JOB', 'pc_grad')
                        self.getcobrammpcgrad(logfile)
                        # gpc=getcobrammpcgrad(logfile,QMin)
                        # QMout['pcgrad'][i]=gpc
                        # print(QMout['pcgrad'][i])
                        pcgradold = os.path.join(QMin['scratchdir'], 'JOB', 'pc_grad')
                        specify_state = os.path.join(QMin['scratchdir'], 'JOB', 'pc_grad.%s.%s') % (m1, s1)  # % (mult,nexc)
            # shutil.copy(pcbrad,specify_state)
                        shutil.copy(pcgradold, specify_state)
            if QMin['neglected_gradient'] != 'zero' and 'samestep' not in QMin:
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    if not (m1, s1) in QMin['gradmap']:
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
                        if QMin['qmmm']:
                            QMout['pcgrad'][i] = QMout['pcgrad'][j]



        return QMout

    def getenergy(self, ricc2, istate):
        QMin = self._QMin
        mult, state, ms = tuple(QMin['statemap'][istate])

        # ground state energy
        if QMin['template']['method'] == 'cc2':
            string = 'Final CC2 energy'
        elif QMin['template']['method'] == 'adc(2)':
            string = 'Final MP2 energy'
        for line in ricc2:
            if string in line:
                e = float(line.split()[5])
                break
        else:
            print('"%s" not found in ricc2.out' % (string))
            sys.exit(14)

        # return gs energy if requested
        if mult == 1 and state == 1:
            return e

        # excited state
        if QMin['template']['method'] == 'cc2':
            string = '| sym | multi | state |          CC2 excitation energies       |  %t1   |  %t2   |'
        elif QMin['template']['method'] == 'adc(2)':
            string = '| sym | multi | state |          ADC(2) excitation energies    |  %t1   |  %t2   |'

        # find correct table
        iline = -1
        while True:
            iline += 1
            if iline == len(ricc2):
                print('"%s" not found in ricc2.out' % (string))
                sys.exit(15)
            line = ricc2[iline]
            if string in line:
                break
        iline += 3

        # find correct line
        if mult == 1:
            iline += state - 1
        elif mult == 3:
            if QMin['states'][0] >= 2:
                iline += QMin['states'][0]
            iline += state

        # get energy
        line = ricc2[iline]
        e = e + float(line.split()[7])
        return complex(e, 0.)


    def getsocme(self, ricc2, istate, jstate):
        QMin = self._QMin
        if istate == jstate:
            return self.getenergy(ricc2, istate)

        if istate not in QMin['statemap'] or jstate not in QMin['statemap']:
            print('States %i or %i are not in statemap!' % (istate, jstate))
            sys.exit(16)

        m1, s1, ms1 = tuple(QMin['statemap'][istate])
        m2, s2, ms2 = tuple(QMin['statemap'][jstate])

        # RICC2 does not calculate triplet-triplet SOC, singlet-singlet SOC is zero
        if m1 == m2:
            return complex(0., 0.)

        # RICC2 does not calculate SOC with S0
        if (m1, s1) == (1, 1) or (m2, s2) == (1, 1):
            return complex(0., 0.)

        # find the correct table
        string = '|        States           Operator    Excitation                 Transition             |'
        iline = 0
        while True:
            iline += 1
            if iline == len(ricc2):
                print('"%s" not found in ricc2.out' % (string))
                sys.exit(17)
            line = ricc2[iline]
            if string in line:
                break
        iline += 7

        # find the correct line
        if istate > jstate:
            m1, s1, ms1, m2, s2, ms2 = m2, s2, ms2, m1, s1, ms1
        nsing = QMin['states'][0] - 1
        ntrip = QMin['states'][2]
        ntot = nsing + ntrip
        if m1 == 1:
            x1 = s1 - 1
        else:
            x1 = nsing + s1
        if m2 == 1:
            x2 = s2 - 1
        else:
            x2 = nsing + s2
        for i in range(1, 1 + ntot):
            for j in range(i + 1, 1 + ntot):
                iline += 1
                if i == x1 and j == x2:
                    try:
                        s = ricc2[iline].split()
                        idx = int(10 + 2 * ms2)
                        soc = float(s[idx]) * rcm_to_Eh
                    except IndexError:
                        print('Could not find SOC matrix element with istate=%i, jstate=%i, line=%i' % (istate, jstate, iline))
                    return complex(soc, 0.)

    def getdiagdm(self, ricc2, istate, pol):
        QMin = self._QMin
        # finds and returns state dipole moments in ricc2.out
        m1, s1, ms1 = tuple(QMin['statemap'][istate])

        if (m1, s1) == (1, 1):
            start1string = ''
            start2string = '<<<<<<<<<<  GROUND STATE FIRST-ORDER PROPERTIES  >>>>>>>>>>>'
        else:
            start1string = '<<<<<<<<<<<<<<<  EXCITED STATE PROPERTIES  >>>>>>>>>>>>>>>>'
            start2string = 'number, symmetry, multiplicity:  % 3i a    %i' % (s1 - (m1 == 1), m1)
        findstring = '     dipole moment:'
        stopstring = 'Analysis of unrelaxed properties'

        # find correct section
        iline = -1
        while True:
            iline += 1
            if iline == len(ricc2):
                print('Could not find dipole moment of istate=%i, Fail=0' % (istate))
                sys.exit(18)
            line = ricc2[iline]
            if start1string in line:
                break

        # find correct state
        while True:
            iline += 1
            if iline == len(ricc2):
                print('Could not find dipole moment of istate=%i, Fail=1' % (istate))
                sys.exit(19)
            line = ricc2[iline]
            if start2string in line:
                break

        # find correct line
        while True:
            iline += 1
            if iline == len(ricc2):
                print('Could not find dipole moment of istate=%i, Fail=2' % (istate))
                sys.exit(20)
            line = ricc2[iline]
            if stopstring in line:
                print('Could not find dipole moment of istate=%i, Fail=3' % (istate))
                sys.exit(21)
            if findstring in line:
                break

        iline += 3 + pol
        s = ricc2[iline].split()
        dm = float(s[1])
        return complex(dm, 0.)

    def gettransdm(self, ricc2, istate, jstate, pol):
        QMin = self._QMin
        # finds and returns transition dipole moments in ricc2.out
        m1, s1, ms1 = tuple(QMin['statemap'][istate])
        m2, s2, ms2 = tuple(QMin['statemap'][jstate])

        if istate > jstate:
            m1, s1, ms1, m2, s2, ms2 = m2, s2, ms2, m1, s1, ms1

        # ground state to excited state
        if (m1, s1) == (1, 1):
            start1string = '<<<<<<<<<<<<  ONE-PHOTON ABSORPTION STRENGTHS  >>>>>>>>>>>>>'
            start2string = 'number, symmetry, multiplicity:  % 3i a    %i' % (s2 - (m2 == 1), m2)
            stopstring = '<<<<<<<<<<<<<<<  EXCITED STATE PROPERTIES  >>>>>>>>>>>>>>>>'

            # find correct section
            iline = -1
            while True:
                iline += 1
                if iline == len(ricc2):
                    print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=0' % (istate, jstate))
                    sys.exit(22)
                line = ricc2[iline]
                if start1string in line:
                    break

            # find correct state
            while True:
                iline += 1
                if iline == len(ricc2):
                    print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=1' % (istate, jstate))
                    sys.exit(23)
                line = ricc2[iline]
                if stopstring in line:
                    print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=2' % (istate, jstate))
                    sys.exit(24)
                if start2string in line:
                    break

            # find element
            iline += 7 + pol
            s = ricc2[iline].split()
            dm = 0.5 * (float(s[3]) + float(s[5]))
            return dm

        # excited to excited state
        else:
            start1string = '<<<<<<<<<<<  EXCITED STATE TRANSITION MOMENTS  >>>>>>>>>>>>'
            start2string = 'Transition moments for pair  % 3i a      % 3i a' % (s1 - (m1 == 1), s2 - (m2 == 1))
            stopstring = 'Model:'
            nostring = 'Transition and Operator of different multiplicity.'

            # find correct section
            iline = -1
            while True:
                iline += 1
                if iline == len(ricc2):
                    print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=4' % (istate, jstate))
                    sys.exit(25)
                line = ricc2[iline]
                if start1string in line:
                    break

            # find correct state
            while True:
                iline += 1
                if iline + 2 == len(ricc2):
                    print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=5' % (istate, jstate))
                    sys.exit(26)
                line = ricc2[iline]
                line2 = ricc2[iline + 2]
                if stopstring in line:
                    print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=6' % (istate, jstate))
                    sys.exit(27)
                if start2string in line and nostring not in line2:
                    break

            # find element
            iline += 2 + pol
            s = ricc2[iline].split()
            dm = 0.5 * (float(s[1]) + float(s[2]))
            return dm

        print('Could not find transition dipole moment of istate=%i,jstate=%i, Fail=7' % (istate, jstate))
        sys.exit(28)

    # ======================================================================= #
    def getgrad(self, ricc2, istate):
        QMin = self._QMin
        m1, s1, ms1 = tuple(QMin['statemap'][istate])
        natom = QMin['natom']

        if (m1, s1) == (1, 1):
            start1string = '<<<<<<<<<<  GROUND STATE FIRST-ORDER PROPERTIES  >>>>>>>>>>>'
            stop1string = '<<<<<<<<<<<<<<<  EXCITED STATE PROPERTIES  >>>>>>>>>>>>>>>>'
        else:
            start1string = '<<<<<<<<<<<<<<<  EXCITED STATE PROPERTIES  >>>>>>>>>>>>>>>>'
            stop1string = 'total wall-time'
        findstring = 'cartesian gradient of the energy (hartree/bohr)'

        # find correct section
        iline = -1
        while True:
            iline += 1
            if iline == len(ricc2):
                print('Could not find gradient of istate=%i, Fail=0' % (istate))
                sys.exit(29)
            line = ricc2[iline]
            if start1string in line:
                break
            if stop1string in line:
                print('Could not find gradient of istate=%i, Fail=1' % (istate))
                sys.exit(30)

        # find gradient
        while True:
            iline += 1
            if iline == len(ricc2):
                print('Could not find gradient of istate=%i, Fail=2' % (istate))
                sys.exit(31)
            line = ricc2[iline]
            if findstring in line:
                break
        iline += 3

        # get grad
        grad = []
        col = 0
        row = 0
        for iatom in range(natom):
            atom = []
            col += 1
            if col > 5:
                col = 1
                row += 1
            for xyz in range(3):
                line = ricc2[iline + 5 * row + xyz + 1]
                el = float(line.split()[col].replace('D', 'E'))
                atom.append(el)
            grad.append(atom)
        return grad

    # ======================================================================= #


    def getpcgrad(self):
        QMin = self._QMin
        pcgrad = readfile(os.path.join(QMin['scratchdir'], 'JOB', 'pc_grad'))

        grad = []
        iline = 0
        for ipc, pc in enumerate(QMin['pointcharges']):
            if pc[-1] != 0:
                g = []
                iline += 1
                s = pcgrad[iline].replace('D', 'E').split()
                for i in range(3):
                    g.append(float(s[i]))
                grad.append(g)
        return grad
# ======================================================================= #

    @staticmethod
    def getcobrammpcgrad(logfile):

        out = readfile(logfile)

        ncharges = len(out)
        gradpc = []
        out.pop(0)
        out.pop(-1)
        ncharges = len(out)
        string = ''
        # icharge=0
        for pc in range(ncharges):
            q = []
            xyz = out[pc].replace('D', 'E').split()
            # icharge+=1
            for i in range(3):
                q.append(float(xyz[i]))
            gradpc.append(q)
        filecharges = open("grad_charges", "a")
        string += '%i %i !\n' % (ncharges, 3)
        # string+='%i %i ! %i %i %i\n' % (natom,3,imult,istate,ims)
        for atom in range(ncharges):
            for xyz in range(3):
                string += '%s ' % (eformat((gradpc[atom][xyz]), 9, 3))
            string += "\n"
        filecharges.write(string)

# ======================================================================= #
    def get_jobs(self):
        # returns a list with the setup for the different ricc2 jobs
        # first, find the properties we need to calculate
        QMin = self._QMin
        prop = set()
        prop.add('E')
        if 'soc' in QMin:
            prop.add('tmexc_soc')
        if 'grad' in QMin:
            for i in QMin['gradmap']:
                if i == (1, 1):
                    prop.add(tuple(['gsgrad'] + list(i)))
                else:
                    prop.add(tuple(['exgrad'] + list(i)))
        if 'dm' in QMin:
            if QMin['dipolelevel'] >= 0:
                # make only dipole moments which are for free
                if 'soc' in QMin:
                    prop.add('tmexc_dm')
            if QMin['dipolelevel'] >= 1:
                # <S0|dm|T> only works for ADC(2)
                if QMin['states'][0] >= 1 and (QMin['template']['method'] == 'adc(2)' or len(QMin['states']) == 1):
                    prop.add('spectrum')
            if QMin['dipolelevel'] >= 2:
                prop.add('exprop_dm')
                prop.add('static_dm')
                # tmexc does not work for CC2 if excited singlets and triples are present
                if not (QMin['template']['method'] == 'cc2' and len(QMin['states']) > 1 and QMin['states'][0] > 1):
                    prop.add('tmexc_dm')

        # define the rules for distribution of jobs
        forbidden = {'E': [],
                     'tmexc_soc': ['gsgrad', 'exgrad', 'exprop_dm', 'static_dm'],
                     'tmexc_dm': [],
                     'spectrum': [],
                     'gsgrad': ['tmexc_soc'],
                     'exgrad': ['tmexc_soc', 'exgrad'],
                     'static_dm': ['tmexc_soc'],
                     'exprop_dm': ['tmexc_soc']
                     }
        if QMin['qmmm']:
            forbidden['gsgrad'].append('exgrad')
            forbidden['exgrad'].append('gsgrad')
        if QMin['cobramm']:  # 21.09.20#
            forbidden['gsgrad'].append('exgrad')
            forbidden['exgrad'].append('gsgrad')

        priority = ['E',
                    'tmexc_soc',
                    'tmexc_dm',
                    'spectrum',
                    'gsgrad',
                    'exgrad',
                    'static_dm',
                    'exprop_dm']

        # print prop

        # second, distribute the properties into jobs
        jobs = []
        # iterate until prop is empty
        while True:
            job = set()
            # process according to priority order
            for prior in priority:
                # print('Prior:',prior)
                # print('Job before iter: ',job)
                # print('Prop pefore iter:',prop)
                # cannot delete from prop during iteration, therefore make copy
                prop2 = deepcopy(prop)
                # check if prior is in prop
                for p in prop:
                    if prior in p:
                        # check if any forbidden task already in job
                        fine = True
                        for forb in forbidden[prior]:
                            for j in job:
                                if forb in j:
                                    fine = False
                        # if allowed, put task into job, delete from prop
                        if fine:
                            job.add(p)
                            prop2.remove(p)
                # copy back prop after iteration
                prop = deepcopy(prop2)
                # print('Job after iter: ',job)
                # print('Prop after iter:',prop)
                # print
            jobs.append(job)
            if len(prop) == 0:
                break

        # pprint.pprint(jobs)

        return jobs

    def gettasks(self):
        ''''''
        QMin = self._QMin
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']

        # Currently implemented keywords: h, soc, dm, grad, overlap, samestep, restart, init
        tasks = []
        # During initialization, create all temporary directories
        # and link them appropriately
        tasks.append(['mkdir', QMin['scratchdir']])
        tasks.append(['link', QMin['scratchdir'], QMin['pwd'] + '/SCRATCH', False])
        tasks.append(['mkdir', QMin['scratchdir'] + '/JOB'])
        if 'overlap' in QMin:
            tasks.append(['mkdir', QMin['scratchdir'] + '/OVERLAP'])
            tasks.append(['mkdir', QMin['scratchdir'] + '/AO_OVL'])

        if 'init' in QMin:
            tasks.append(['mkdir', QMin['savedir']])
            tasks.append(['link', QMin['savedir'], QMin['pwd'] + '/SAVE', False])

        if 'molden' in QMin and not os.path.isdir(QMin['savedir'] + '/MOLDEN'):
            tasks.append(['mkdir', QMin['savedir'] + '/MOLDEN'])

        if 'samestep' not in QMin and 'init' not in QMin and 'restart' not in QMin:
            tasks.append(['movetoold'])

        if 'backup' in QMin:
            tasks.append(['mkdir', QMin['savedir'] + '/backup/'])

        # do all TURBOMOLE calculations
        if 'overlaponly' not in QMin:

            tasks.append(['cleanup', QMin['scratchdir'] + '/JOB'])
            tasks.append(['writegeom', QMin['scratchdir'] + '/JOB'])
            tasks.append(['define', QMin['scratchdir'] + '/JOB'])
            tasks.append(['modify_control'])

            # initial orbitals
            if 'always_guess' in QMin:
                # no further action, define creates MOs
                pass
            elif 'always_orb_init' in QMin:
                # always take "mos.init" from the main directory
                tryfile = os.path.join(QMin['pwd'], 'mos.init')
                if os.path.isfile(tryfile):
                    tasks.append(['getmo', tryfile])
            else:
                if 'init' in QMin:
                    tryfile = os.path.join(QMin['pwd'], 'mos.init')
                    if os.path.isfile(tryfile):
                        tasks.append(['getmo', tryfile])
                elif 'samestep' in QMin:
                    tryfile = os.path.join(QMin['savedir'], 'mos')
                    if os.path.isfile(tryfile):
                        tasks.append(['getmo', tryfile])
                elif 'restart' in QMin:
                    tryfile = os.path.join(QMin['savedir'], 'mos.old')
                    if os.path.isfile(tryfile):
                        tasks.append
                elif 'newstep' in QMin:
                    tryfile = os.path.join(QMin['savedir'], 'mos.old')
                    if os.path.isfile(tryfile):
                        tasks.append(['getmo', tryfile])

            # SCF calculation
            if QMin['template']['scf'] == 'ridft':
                tasks.append(['ridft'])
            tasks.append(['dscf'])
            if 'molden' in QMin or 'theodore' in QMin:
                tasks.append(['copymolden'])

            # Orca call for SOCs
            if 'soc' in QMin:
                tasks.append(['orca_soc'])

            # ricc2 calls:
            # prop={'tmexc_soc','tmexc_dm', 'spectrum','exprop_dm','static_dm','gs_grad','exgrad'}
            jobs = self.get_jobs()
            for ijob, job in enumerate(jobs):
                tasks.append(['prep_control', job])
                tasks.append(['ricc2'])
                tasks.append(['get_RICC2out', job])
                if ijob == 0:
                    tasks.append(['save_data'])
                    if 'nooverlap' not in QMin:
                        mults = [i + 1 for i in range(len(QMin['states'])) if QMin['states'][i] > 0]
                        tasks.append(['get_dets', QMin['scratchdir'] + '/JOB', mults])
                    # if 'molden' in QMin:
                        # tasks.append(['molden'])

                    if 'theodore' in QMin:
                        tasks.append(['run_theodore'])
                        tasks.append(['get_theodore'])
                        if 'molden' in QMin:
                            tasks.append(['copy_ntos'])

            if 'overlap' in QMin:
                # get mixed AO overlap
                tasks.append(['cleanup', QMin['scratchdir'] + '/AO_OVL'])
                tasks.append(['get_AO_OVL', QMin['scratchdir'] + '/AO_OVL'])

                for imult in range(len(QMin['states'])):
                    if QMin['states'][imult] == 0:
                        continue
                    tasks.append(['cleanup', QMin['scratchdir'] + '/OVERLAP'])
                    tasks.append(['wfoverlap', QMin['scratchdir'] + '/OVERLAP', imult + 1])
                    tasks.append(['get_wfovlout', QMin['scratchdir'] + '/OVERLAP', imult + 1])

        if 'backup' in QMin:
            tasks.append(['backupdata', QMin['backup']])

        if 'cleanup' in QMin:
            tasks.append(['cleanup', QMin['savedir']])
        # if not DEBUG:
            # tasks.append(['cleanup',QMin['scratchdir']])

        return tasks


# ======================================================================= #


    def define(self, path, ricc2=True):
        QMin = self._QMin
        # first three sections
        if QMin['template']['basislib']:
            # write definrc
            string = '''basis=%s/basen
    basis=%s/cbasen
    ''' % (QMin['template']['basislib'], QMin['template']['basislib'])
            infile = os.path.join(path, '.definerc')
            writefile(infile, string)

        # write define input
        string = '''
    title: SHARC-RICC2 run
    a coord
    *
    no
    '''
        if QMin['template']['basislib']:
            string += '''lib
    3
    '''
        string += '''b
    all %s
    *
    eht
    y
    %i
    y
    ''' % (QMin['template']['basis'],
           QMin['template']['charge']
           )

        if ricc2:
            # cc section
            string += 'cc\n'

            # frozen orbitals
            if QMin['template']['frozen'] == 0:
                pass
            elif QMin['template']['frozen'] < 0:
                string += 'freeze\n*\n'
            elif QMin['template']['frozen'] > 0:
                string += 'freeze\ncore %i\n*\n' % (QMin['template']['frozen'])

            # auxilliary basis set: cbas
            # this is mandatory for ricc2, so if the user does not give an auxbasis, we take the default one
            if 'auxbasis' not in QMin['template']:
                if QMin['template']['basislib']:
                    string += 'cbas\n'
                    # skip error messages in define
                    elements = set()
                    for atom in QMin['geo']:
                        elements.add(atom[0])
                    string += '\n\n' * len(elements)
                    string += '''lib
    4
    b
    all %s
    *
    ''' % (QMin['template']['basis'])
                else:
                    string += 'cbas\n*\n'
            else:
                string += 'cbas\nb\nall %s\n*\n' % (QMin['template']['auxbasis'])

            # memory
            string += 'memory %i\n' % (QMin['memory'])

            # ricc2 section (here we set only the model)
            string += 'ricc2\n%s\n' % (QMin['template']['method'])
            if QMin['template']['spin-scaling'] == 'none':
                pass
            elif QMin['template']['spin-scaling'] == 'scs':
                string += 'scs\n'
            elif QMin['template']['spin-scaling'] == 'sos':
                string += 'sos\n'
            elif QMin['template']['spin-scaling'] == 'lt-sos':
                string += 'sos\n'
            # number of DIIS vectors
            ndiis = max(10, 5 * max(QMin['states']))
            string += 'mxdiis = %i\n' % (ndiis)
            string += '*\n'

            # leave cc input
            string += '*\n'

        # leave define
        string += '*\n'

        # string contains the input for define, now call it
        infile = os.path.join(path, 'define.input')
        writefile(infile, string)
        string = 'define < define.input'
        runerror = self.runProgram(string, path, 'define.output')

        if runerror != 0:
            print('define call failed! Error Code=%i Path=%s' % (runerror, path))
            sys.exit(97)


        return

    @staticmethod
    def add_section_to_control(path, section):
        # adds a section keyword to the control file, before $end
        # if section does not start with $, $ will be prepended
        if not section[0] == '$':
            section = '$' + section
        infile = readfile(path)

        # get iline of $end
        iline = -1
        while True:
            iline += 1
            line = infile[iline]
            if '$end' in line:
                break

        outfile = infile[:iline]
        outfile.append(section + '\n')
        outfile.extend(infile[iline:])
        writefile(path, outfile)
        return

# ======================================================================= #

    @staticmethod
    def add_option_to_control_section(path, section, newline):
        if not section[0] == '$':
            section = '$' + section
        infile = readfile(path)
        newline = '  ' + newline

        # get iline where section starts
        iline = -1
        while True:
            iline += 1
            if iline == len(infile):
                return
            line = infile[iline]
            if section in line:
                break

        # get jline where section ends
        jline = iline
        while True:
            jline += 1
            line = infile[jline]
            if '$' in line:
                break
            # do nothing if line is already there
            if newline + '\n' == line:
                return

        # splice together new file
        outfile = infile[:jline]
        outfile.append(newline + '\n')
        outfile.extend(infile[jline:])
        writefile(path, outfile)
        return

    @staticmethod
    def remove_section_in_control(path, section):
        # removes a keyword and its options from control file
        if not section[0] == '$':
            section = '$' + section
        infile = readfile(path)

        # get iline where section starts
        iline = -1
        while True:
            iline += 1
            if iline == len(infile):
                return
            line = infile[iline]
            if section in line:
                break

        # get jline where section ends
        jline = iline
        while True:
            jline += 1
            line = infile[jline]
            if '$' in line:
                break

        # splice together new file
        outfile = infile[:iline] + infile[jline:]
        writefile(path, outfile)
        return

    def modify_control(self):
        QMin = self._QMin
        # this adjusts the control file for the main JOB calculations
        control = os.path.join(QMin['scratchdir'], 'JOB/control')

        if 'soc' in QMin:
            self.add_section_to_control(control, '$mkl')
        if QMin['template']['douglas-kroll']:
            self.add_section_to_control(control, '$rdkh')

        # add laplace keyword for LT-SOS
        if QMin['template']['spin-scaling'] == 'lt-sos':
            self.add_section_to_control(control, '$laplace')
            self.add_option_to_control_section(control, '$laplace', 'conv=5')

        # remove_section_in_control(control,'$optimize')
        # add_option_to_control_section(control,'$ricc2','scs')
        self.remove_section_in_control(control, '$scfiterlimit')
        self.add_section_to_control(control, '$scfiterlimit 100')

        # QM/MM point charges
        if QMin['qmmm']:
            self.add_option_to_control_section(control, '$drvopt', 'point charges')
            self.add_section_to_control(control, '$point_charges file=pc')
            self.add_section_to_control(control, '$point_charge_gradients file=pc_grad')
            return

        # COBRAMM
        if QMin['cobramm']:
            self.add_option_to_control_section(control, '$drvopt', 'point charges')
            self.add_section_to_control(control, '$point_charges file=point_charges')  # inserire nome file quando deciso
            self.add_section_to_control(control, '$point_charge_gradients file=pc_grad')
            return

    def prep_control(self, job):
        # prepares the control file to calculate grad, soc, dm
        # job contains: 'tmexc_soc','tmexc_dm', 'spectrum','exprop_dm','static_dm','gsgrad','exgrad', 'E'
        QMin = self._QMin
        control = os.path.join(QMin['scratchdir'], 'JOB/control')

        # remove sections to cleanly rewrite them
        self.remove_section_in_control(control, '$response')
        self.remove_section_in_control(control, '$excitations')

        # add number of states
        self.add_section_to_control(control, '$excitations')
        self.add_option_to_control_section(control, '$ricc2', 'maxiter 45')
        nst = QMin['states'][0] - 1       # exclude ground state here
        if nst >= 1:
            string = 'irrep=a multiplicity=1 nexc=%i npre=%i nstart=%i' % (nst, nst + 1, nst + 1)
            self.add_option_to_control_section(control, '$excitations', string)
        if len(QMin['states']) >= 3:
            nst = QMin['states'][2]
            if nst >= 1:
                string = 'irrep=a multiplicity=3 nexc=%i npre=%i nstart=%i' % (nst, nst + 1, nst + 1)
                self.add_option_to_control_section(control, '$excitations', string)

        # add response section
        if 'static_dm' or 'gsgrad' in job:
            self.add_section_to_control(control, '$response')

        # add property lines
        if 'tmexc_soc' in job or 'tmexc_dm' in job:
            string = 'tmexc istates=all fstates=all operators='
            prop = []
            if 'tmexc_soc' in job:
                prop.append('soc')
            if 'tmexc_dm' in job:
                prop.append('diplen')
            string += ','.join(prop)
            self.add_option_to_control_section(control, '$excitations', string)
        if 'spectrum' in job:
            string = 'spectrum states=all operators=diplen'
            self.add_option_to_control_section(control, '$excitations', string)
        if 'exprop_dm' in job:
            string = 'exprop states=all relaxed operators=diplen'
            self.add_option_to_control_section(control, '$excitations', string)
        if 'static_dm' in job:
            string = 'static relaxed operators=diplen'
            self.add_option_to_control_section(control, '$response', string)

        if QMin['cobramm']:
            self.add_option_to_control_section(control, '$drvopt', ' point charges')
            self.add_section_to_control(control, '$point_charges file=point_charges')  # inserire nome file quando deciso
            self.add_section_to_control(control, '$point_charge_gradients file=pc_grad')

        # add gradients
        for j in job:
            if 'gsgrad' in j:
                string = 'gradient'
                self.add_option_to_control_section(control, '$response', string)
            if 'exgrad' in j:
                string = 'xgrad states=(a{%i} %i)' % (j[1], j[2] - (j[1] == 1))
                self.add_option_to_control_section(control, '$excitations', string)


        # ricc2 restart
        if 'E' not in job and 'no_ricc2_restart' not in QMin:
            self.add_option_to_control_section(control, '$ricc2', 'restart')
            restartfile = os.path.join(QMin['scratchdir'], 'JOB/restart.cc')
            try:
                os.remove(restartfile)
            except OSError:
                pass

        # D1 and D2 diagnostic
        if DEBUG and 'E' in job:
            self.add_option_to_control_section(control, '$ricc2', 'd1diag')
            self.add_section_to_control(control, '$D2-diagnostic')

        return

    # TODO possibly general -> SHARC_INTERFACE
    def get_dets(self, path, mults):
        # read all determinant expansions from working directory and put them into the savedir
        QMin = self._QMin
        for imult in mults:
            ca = civfl_ana(path, imult, maxsqnorm=QMin['wfthres'], filestr='CCRE0')
            for istate in range(1, 1 + QMin['states'][imult - 1]):
                ca.get_state_dets(istate)
            writename = os.path.join(QMin['savedir'], 'dets.%i' % (imult))
            ca.write_det_file(QMin['states'][imult - 1], wname=writename)

            # for CC2, also save the left eigenvectors
            if QMin['template']['method'] == 'cc2':
                ca = civfl_ana(path, imult, maxsqnorm=QMin['wfthres'], filestr='CCLE0')
                for istate in range(1, 1 + QMin['states'][imult - 1]):
                    ca.get_state_dets(istate)
                writename = os.path.join(QMin['savedir'], 'dets_left.%i' % (imult))
                ca.write_det_file(QMin['states'][imult - 1], wname=writename)

            if 'frozenmap' not in QMin:
                QMin['frozenmap'] = {}
            QMin['frozenmap'][imult] = ca.nfrz
        return QMin

    def get_AO_OVL(self, path):
        QMin = self._QMin
        # get double geometry
        oldgeom = readfile(os.path.join(QMin['savedir'], 'coord.old'))
        newgeom = readfile(os.path.join(QMin['savedir'], 'coord'))
        string = '$coord\n'
        wrt = False
        for line in oldgeom:
            if '$coord' in line:
                wrt = True
                continue
            elif '$' in line:
                wrt = False
                continue
            if wrt:
                string += line
        for line in newgeom:
            if '$coord' in line:
                wrt = True
                continue
            elif '$' in line:
                wrt = False
                continue
            if wrt:
                string += line
        string += '$end\n'
        tofile = os.path.join(path, 'coord')
        writefile(tofile, string)

        # call define and then add commands to control file
        self.define(path, ricc2=False)
        controlfile = os.path.join(path, 'control')
        self.remove_section_in_control(controlfile, '$scfiterlimit')
        self.add_section_to_control(controlfile, '$scfiterlimit 0')
        self.add_section_to_control(controlfile, '$intsdebug sao')
        self.add_section_to_control(controlfile, '$closed shells')
        self.add_option_to_control_section(controlfile, '$closed shells', 'a 1-2')
        self.add_section_to_control(controlfile, '$scfmo none')

        # write geometry again because define tries to be too clever with the double geometry
        tofile = os.path.join(path, 'coord')
        writefile(tofile, string)

        # call dscf
        string = 'dscf'
        runerror = self.runProgram(string, path, 'dscf.out')

        # get AO overlap matrix from dscf.out
        dscf = readfile(os.path.join(path, 'dscf.out'))
        for line in dscf:
            if ' number of basis functions   :' in line:
                nbas = int(line.split()[-1])
                break
        else:
            print('Could not find number of basis functions in dscf.out!')
            sys.exit(98)
        iline = -1
        while True:
            iline += 1
            line = dscf[iline]
            if 'OVERLAP(SAO)' in line:
                break
        iline += 1
        ao_ovl = makermatrix(nbas, nbas)
        x = 0
        y = 0
        while True:
            iline += 1
            line = dscf[iline].split()
            for el in line:
                ao_ovl[x][y] = float(el)
                ao_ovl[y][x] = float(el)
                x += 1
                if x > y:
                    x = 0
                    y += 1
            if y >= nbas:
                break
        # the SAO overlap in dscf output is a LOWER triangular matrix
        # hence, the off-diagonal block must be transposed

        # write AO overlap matrix to savedir
        string = '%i %i\n' % (nbas // 2, nbas // 2)
        for irow in range(nbas // 2, nbas):
            for icol in range(0, nbas // 2):
                string += '% .15e ' % (ao_ovl[icol][irow])          # note the exchanged indices => transposition
            string += '\n'
        filename = os.path.join(QMin['savedir'], 'ao_ovl')
        writefile(filename, string)
        return

    def run_dscf(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')
        if QMin['ncpu'] > 1:
            string = 'dscf_omp'
        else:
            string = 'dscf'
        runerror = self.runProgram(string, workdir, 'dscf.out')
        if runerror != 0:
            print('DSCF calculation crashed! Error code=%i' % (runerror))
            sys.exit(99)

        return

    # ======================================================================= #


    def run_ridft(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')

        # add RI settings to control file
        controlfile = os.path.join(workdir, 'control')
        self.remove_section_in_control(controlfile, '$maxcor')
        self.add_section_to_control(controlfile, '$maxcor %i' % (int(QMin['memory'] * 0.6)))
        self.add_section_to_control(controlfile, '$ricore %i' % (int(QMin['memory'] * 0.4)))
        self.add_section_to_control(controlfile, '$jkbas file=auxbasis')
        self.add_section_to_control(controlfile, '$rij')
        self.add_section_to_control(controlfile, '$rik')

        if QMin['ncpu'] > 1:
            string = 'ridft_smp'
        else:
            string = 'ridft'
        runerror = self.runProgram(string, workdir, 'ridft.out')
        if runerror != 0:
            print('RIDFT calculation crashed! Error code=%i' % (runerror))
            sys.exit(100)

        # remove RI settings from control file
        controlfile = os.path.join(workdir, 'control')
        self.remove_section_in_control(controlfile, '$maxcor')
        self.add_section_to_control(controlfile, '$maxcor %i' % (QMin['memory']))
        self.remove_section_in_control(controlfile, '$rij')
        self.remove_section_in_control(controlfile, '$rik')

        return

    # ======================================================================= #


    def run_orca(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')
        string = 'orca_2mkl soc -gbw'
        runerror = self.runProgram(string, workdir, 'orca_2mkl.out', 'orca_2mkl.err')
        if runerror != 0:
            print('orca_2mkl calculation crashed! Error code=%i' % (runerror))
            sys.exit(101)

        string = '''soc.gbw
    soc.psoc
    soc.soc
    3
    1 2 3 0 4 0 0 4
    0
    '''
        writefile(os.path.join(workdir, 'soc.socinp'), string)

        string = 'orca_soc soc.socinp -gbw'
        runerror = self.runProgram(string, workdir, 'orca_soc.out', 'orca_soc.err')
        if runerror != 0:
            print('orca_soc calculation crashed! Error code=%i' % (runerror))
            sys.exit(102)

        return


    # ======================================================================= #


    @staticmethod
    def change_pre_states(workdir, itrials):

        filename = os.path.join(workdir, 'control')
        data = readfile(filename)
        data2 = deepcopy(data)
        for i, line in enumerate(data):
            if 'irrep' in line:
                s = line.replace('=', ' ').split()
                mult = s[3]
                nexc = s[5]
                npre = str(int(s[7]) + shift_mask[itrials][0])
                nstart = str(int(s[9]) + shift_mask[itrials][1])
                data2[i] = 'irrep=a multiplicity=%s nexc=%s npre=%s nstart=%s\n' % (mult, nexc, npre, nstart)
            if 'maxiter 45' in line:
                data2[i] = 'maxiter 100\n'
        writefile(filename, data2)

    # ======================================================================= #


    def run_ricc2(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')

        # enter loop until convergence of CC2/ADC(2)
        itrials = 0
        while True:
            if QMin['ncpu'] > 1:
                string = 'ricc2_omp'
            else:
                string = 'ricc2'
            runerror = self.runProgram(string, workdir, 'ricc2.out')
            if runerror != 0:
                print('RICC2 calculation crashed! Error code=%i' % (runerror))
                ok = False
            # check for convergence in output file
            filename = os.path.join(workdir, 'ricc2.out')
            data = readfile(filename)
            ok = True
            for line in data:
                if 'NO CONVERGENCE' in line:
                    ok = False
                    break
            if ok:
                break
            # go only here if no convergence
            itrials += 1
            if itrials > max(shift_mask):
                print('Not able to obtain convergence in RICC2. Aborting...')
                sys.exit(103)
            print('No convergence of excited-state calculations! Restarting with modified number of preoptimization states...')
            self.change_pre_states(workdir, itrials)

        return

    def runeverything(self, tasks):
        QMin = self._QMin
        if PRINT or DEBUG:
            print('=============> Entering RUN section <=============\n\n')

        QMout = {}
        for task in tasks:
            if DEBUG:
                print(task)
            if task[0] == 'movetoold':
                movetoold(QMin['savedir'])
            if task[0] == 'mkdir':
                mkdir(task[1])
            if task[0] == 'link':
                if len(task) == 4:
                    link(task[1], task[2], task[3])
                else:
                    link(task[1], task[2])
            if task[0] == 'getmo':
                getmo(task[1], QMin['scratchdir'])
            if task[0] == 'define':
                self.define(task[1], )
            if task[0] == 'modify_control':
                self.modify_control()
            if task[0] == 'writegeom':
                self.writegeom()
            if task[0] == 'save_data':
                save_data()
            # if task[0]=='backupdata':
                # backupdata(task[1],QMin)
            if task[0] == 'copymolden':
                self.copymolden()
            if task[0] == 'get_dets':
                QMin = self.get_dets(task[1], task[2], QMin)
            if task[0] == 'cleanup':
                cleandir(task[1])
            if task[0] == 'dscf':
                self.run_dscf()
            if task[0] == 'ridft':
                self.run_ridft()
            if task[0] == 'orca_soc':
                self.run_orca()
            if task[0] == 'prep_control':
                self.prep_control(task[1])
            if task[0] == 'ricc2':
                self.run_ricc2()
            if task[0] == 'get_RICC2out':
                QMout = self.get_RICC2out(task[1])
            if task[0] == 'get_AO_OVL':
                self.get_AO_OVL(task[1])
            if task[0] == 'wfoverlap':
                self.wfoverlap(task[1], task[2])
            if task[0] == 'get_wfovlout':
                QMout = self.get_wfovlout(task[1], task[2])
            if task[0] == 'run_theodore':
                self.setupWORKDIR_TH()
                self.run_theodore()
            if task[0] == 'get_theodore':
                QMout = self.get_theodore()
            if task[0] == 'copy_ntos':
                self.copy_ntos()

        # if no dyson pairs were calculated because of selection rules, put an empty matrix
        if 'prop' not in QMout and 'ion' in QMin:
            QMout['prop'] = makecmatrix(QMin['nmstates'], QMin['nmstates'])

        # Phases from overlaps
        if 'phases' in QMin:
            if 'phases' not in QMout:
                QMout['phases'] = [complex(1., 0.) for i in range(QMin['nmstates'])]
            if 'overlap' in QMout:
                for i in range(QMin['nmstates']):
                    if QMout['overlap'][i][i].real < 0.:
                        QMout['phases'][i] = complex(-1., 0.)

        # transform back from QM to QM/MM
        if QMin['template']['qmmm']:
            QMin, QMout = self.transform_QM_QMMM(QMin, QMout)


        return QMin, QMout
# ======================================================================= #


    def execute_tinker(self, ff_file_path):
        '''
        run tinker to get:
        * MM energy
        * MM gradient
        * point charges

        is only allowed to read the following keys from QMin:
        qmmm
        scratchdir
        savedir
        tinker
        '''
        QMin = self._QMin
        QMMM = QMin['qmmm']

        # prepare Workdir
        WORKDIR = os.path.join(QMin['scratchdir'], 'TINKER')
        mkdir(WORKDIR)

        print('Writing TINKER inputs ...      ', datetime.datetime.now())
        # key file
        string = 'parameters %s\nQMMM %i\n' % (ff_file_path, QMMM['natom_table'] + len(QMMM['linkbonds']))
        string += 'QM %i %i\n' % (-1, len(QMMM['QM_atoms']))
        if len(QMMM['linkbonds']) > 0:
            string += 'LA %s\n' % (
                ' '.join([str(QMMM['reorder_input_MM'][i] + 1) for i in QMMM['LI_atoms']]))
        string += 'MM %i %i\n' % (-(1 + len(QMMM['QM_atoms']) + len(QMMM['linkbonds'])),
                                  QMMM['natom_table'] + len(QMMM['linkbonds']))
        # if DEBUG:
        # string+='\nDEBUG\n'
        if QMin['ncpu'] > 1:
            string += '\nOPENMP-THREADS %i\n' % QMin['ncpu']
        if len(QMMM['linkbonds']) > 0:
            string += 'atom    999    99    HLA     "Hydrogen Link Atom"        1      1.008     0\n'
        # string+='CUTOFF 1.0\n'
        string += '\n'
        filename = os.path.join(WORKDIR, 'TINKER.key')
        writefile(filename, string)


        # xyz/type/connection file
        string = '%i\n' % (len(QMMM['MM_coords']))
        for iatom_MM in range(len(QMMM['MM_coords'])):
            iatom_input = QMMM['reorder_MM_input'][iatom_MM]
            string += '% 5i  %3s  % 16.12f % 16.12f % 16.12f  %4s  %s\n' % (
                iatom_MM + 1,
                QMMM['MM_coords'][iatom_input][0],
                QMMM['MM_coords'][iatom_input][1],
                QMMM['MM_coords'][iatom_input][2],
                QMMM['MM_coords'][iatom_input][3],
                QMMM['atomtype'][iatom_input],
                ' '.join([str(QMMM['reorder_input_MM'][i] + 1) for i in sorted(QMMM['connect'][iatom_input])])
            )
        filename = os.path.join(WORKDIR, 'TINKER.xyz')
        writefile(filename, string)


        # communication file
        string = 'SHARC 0 -1\n'
        for iatom_MM in range(len(QMMM['MM_coords'])):
            iatom_input = QMMM['reorder_MM_input'][iatom_MM]
            string += '% 16.12f % 16.12f % 16.12f\n' % tuple(QMMM['MM_coords'][iatom_input][1:4])
        filename = os.path.join(WORKDIR, 'TINKER.qmmm')
        writefile(filename, string)


        # standard input file
        string = 'TINKER.xyz'
        filename = os.path.join(WORKDIR, 'TINKER.in')
        writefile(filename, string)


        # run TINKER
        self.runTINKER(WORKDIR, QMin['tinker'], QMin['savedir'], strip=False, ncpu=QMin['ncpu'])


        # read out TINKER
        filename = os.path.join(WORKDIR, 'TINKER.qmmm')
        output = readfile(filename)

        # check success
        if 'MMisOK' not in output[0]:
            print('TINKER run not successful!')
            sys.exit(39)

        # get MM energy (convert from kcal to Hartree)
        print('Searching MMEnergy ...         ', datetime.datetime.now())
        QMMM['MMEnergy'] = float(output[1].split()[-1]) * kcal_to_Eh

        # get MM gradient (convert from kcal/mole/A to Eh/bohr)
        print('Searching MMGradient ...       ', datetime.datetime.now())
        QMMM['MMGradient'] = {}
        for line in output:
            if 'MMGradient' in line:
                s = line.split()
                iatom_MM = int(s[1]) - 1
                iatom_input = QMMM['reorder_MM_input'][iatom_MM]
                grad = [float(i) * kcal_to_Eh * au2a for i in s[2:5]]
                QMMM['MMGradient'][iatom_input] = grad
            if 'MMq' in line:
                break

        # get MM point charges
        print('Searching MMpc_raw ...         ', datetime.datetime.now())
        QMMM['MMpc_raw'] = {}
        for i in range(QMMM['natom_table']):
            QMMM['MMpc_raw'][i] = 0.
        iline = 0
        while True:
            iline += 1
            line = output[iline]
            if 'MMq' in line:
                break
        iatom_MM = len(QMMM['QM_atoms']) + len(QMMM['LI_atoms']) - 1
        while True:
            iline += 1
            iatom_MM += 1
            line = output[iline]
            if 'NMM' in line:
                break
            s = line.split()
            q = float(s[-1])
            QMMM['MMpc_raw'][QMMM['reorder_MM_input'][iatom_MM]] = q

        # compute actual charges (including redistribution)
        print('Redistributing charges ...     ', datetime.datetime.now())
        QMMM['MMpc'] = {}
        for i in range(QMMM['natom_table']):
            s = 0.
            for factor, iatom in QMMM['charge_distr'][i]:
                s += factor * QMMM['MMpc_raw'][iatom]
            QMMM['MMpc'][i] = s

        # make list of pointcharges without QM atoms and zero-charge MM atoms
        print('Finalizing charges ...         ', datetime.datetime.now())
        QMMM['pointcharges'] = []
        QMMM['reorder_pc_input'] = {}
        ipc = 0
        for iatom_input in QMMM['MMpc']:
            q = QMMM['MMpc'][iatom_input]
            if q != 0:
                atom = QMMM['MM_coords'][iatom_input]
                QMMM['pointcharges'].append(atom[1:4] + [q])
                QMMM['reorder_pc_input'][ipc] = iatom_input
                ipc += 1

        # Get energy components from standard out (debug print)
        filename = os.path.join(WORKDIR, 'TINKER.out')
        output = readfile(filename)
        QMMM['MMEnergy_terms'] = {}
        for line in output:
            if 'kcal/mol' in line:
                s = line.split()
                QMMM['MMEnergy_terms'][s[0]] = float(s[2])

        print('====================================')
        print('\n')
        return QMMM

# ======================================================================= #

    def runTINKER(WORKDIR, tinker, savedir, strip=False, ncpu=1):
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = os.path.join(tinker, 'bin', 'tkr2qm_s') + ' '
        string += ' < TINKER.in'
        os.environ['OMP_NUM_THREADS'] = str(ncpu)
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
            sys.stdout.flush()
        stdoutfile = open(os.path.join(WORKDIR, 'TINKER.out'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'TINKER.err'), 'w')
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(41)
        stdoutfile.close()
        stderrfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
            sys.stdout.flush()
        if DEBUG and runerror != 0:
            copydir = os.path.join(savedir, 'debug_TINKER_stdout')
            if not os.path.isdir(copydir):
                mkdir(copydir)
            outfile = os.path.join(WORKDIR, 'TINKER.out')
            tofile = os.path.join(copydir, "TINKER_problems.out")
            shutil.copy(outfile, tofile)
            print('Error in %s! Copied TINKER output to %s' % (WORKDIR, tofile))
        os.chdir(prevdir)
        # if strip and not DEBUG and runerror==0:
        # stripWORKDIR(WORKDIR)
        return runerror


    @staticmethod
    def get_arch(turbodir):
        os.environ['TURBODIR'] = turbodir
        string = os.path.join(turbodir, 'scripts', 'sysname')
        proc = sp.Popen([string], stdout=sp.PIPE)
        output = proc.communicate()[0].decode().strip()
        print('Architecture: %s' % output)
        return output

    @staticmethod
    def getsh2cc2key(sh2cc2, key):
        i = -1
        while True:
            i += 1
            try:
                line = re.sub(r'#.*$', '', sh2cc2[i])
            except IndexError:
                break
            line = line.strip().split(None, 1)
            if line == []:
                continue
            if key.lower() in line[0].lower():
                return line
        return ['', '']

    # ======================================================================= #

    @staticmethod
    def get_sh2cc2_environ(sh2cc2, key, environ=True, crucial=True):
        line = RICC2.getsh2cc2key(sh2cc2, key)
        if line[0]:
            LINE = line[1]
        else:
            if environ:
                LINE = os.getenv(key.upper())
                if not LINE:
                    print('Either set $%s or give path to %s in SH2COL.inp!' % (key.upper(), key.upper()))
                    if crucial:
                        sys.exit(44)
                    else:
                        return None
            else:
                print('Give path to %s in SH2COL.inp!' % (key.upper()))
                if crucial:
                    sys.exit(45)
                else:
                    return None
        LINE = os.path.expandvars(LINE)
        LINE = os.path.expanduser(LINE)
        LINE = os.path.abspath(LINE)
        LINE = RICC2.removequotes(LINE).strip()
        if containsstring(';', LINE):
            print("$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (key.upper(), key.upper()))
            sys.exit(46)
        return LINE

    # ======================= main ====================================
    def main(self):

        # Process Command line arguments
        if len(sys.argv) != 2:
            print('Usage:\n./SHARC_RICC2.py <QMin>\n')
            print('version:', self.version)
            print('date: {:%d.%m.%Y}'.format(self.versiondate))
            print('changelog:\n', self.changelogstring)
            sys.exit(111)
        QMinfilename = sys.argv[1]

        # Print header
        self.printheader()

        # # Read QMinfile
        self.readQMin(QMinfilename)

        # # Process Tasks
        # Tasks = self._gettasks(QMin)
        # if self.DEBUG:
        #     pprint.pprint(Tasks)

        # # do all runs
        # QMin, QMout = self._runeverything(Tasks, QMin)

        # self.printQMout(QMin, QMout)

        # # Measure time
        # runtime = self.clock.measuretime()
        # QMout['runtime'] = runtime

        # # Write QMout
        # self.writeQMout(QMin, QMout, QMinfilename)

        if self.PRINT or self.DEBUG:
            print(datetime.now())
            print('#================ END ================#')



# =============================================================================================== #
# =============================================================================================== #
# ========================================= Main ================================================ #
# =============================================================================================== #
# =============================================================================================== #


def main():
    # Retrieve PRINT and DEBUG
    PRINT = True
    DEBUG = False
    try:
        envPRINT = os.getenv('SH2CC2_PRINT')
        if envPRINT and envPRINT.lower() == 'false':
            PRINT = False
        envDEBUG = os.getenv('SH2CC2_DEBUG')
        if envDEBUG and envDEBUG.lower() == 'true':
            DEBUG = True
    except ValueError:
        print('PRINT or DEBUG environment variables do not evaluate to logical values!')
        sys.exit(110)

    interface = RICC2(PRINT, DEBUG)
    interface.main()


if __name__ == '__main__':
    main()
