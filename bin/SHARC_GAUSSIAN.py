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
# *****************************************

# IMPORTS
# external
from itertools import chain
import pprint
import sys
import os
import shutil
import subprocess as sp
import re
import math
import time
import datetime
from multiprocessing import Pool
from copy import deepcopy
from socket import gethostname
import traceback
import numpy as np

# internal
from resp import Resp
from tdm import es2es_tdm
from SHARC_INTERFACE import INTERFACE
from utils import mkdir, readfile, containsstring, shorten_DIR, makermatrix, makecmatrix, build_basis_dict, get_pyscf_order_from_gaussian, removekey, writefile, itmult, safe_cast, get_bool_from_env, get_cart2sph_matrix
from globals import DEBUG, PRINT
from constants import IToMult, au2eV, IAn2AName
from error import Error, exception_hook

sys.excepthook = exception_hook

authors = 'Sebastian Mai, Maximilian F.S.J. Menger and Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2022, 2, 10)

changelogstring = '''
30.11.2017: INITIAL VERSION
- only singlets
- h, dm, grad, overlap are working

01.12.2017:
- all features of ADF interface, minus SOCs

08.01.2019:
- added "basis_external" keyword

10.02.2022:
- ported to new interface class
- groot -> gaussiandir (consistency)
'''


class GAUSSIAN(INTERFACE):

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

    def read_template(self, template_filename='ORCA.template'):
        '''reads the template file
        has to be called after setup_mol!'''

        if not self._read_resources:
            raise Error('Interface is not set up correctly. Call read_resources with the .resources file first!', 23)
        QMin = self._QMin
        # define classes and defaults
        bools = {'denfit': False, 'no_tda': False, 'unrestricted_triplets': False, 'qmmm': False}
        strings = {
            'basis': '6-31G',
            'functional': 'PBEPBE',
            'dispersion': '',
            'grid': 'finegrid',
            'scrf': '',
            'scf': '',
            'qmmm_table': 'GAUSSIAN.qmmm.table',
            'qmmm_ff_file': 'GAUSSIAN.ff',
            'iop': '',
            'keys': '',
            'basis_external': ''
        }
        integers = {}
        floats = {}
        special = {
            'paddingstates': [0 for i in QMin['states']],
            'charge': [i % 2 for i in range(len(QMin['states']))],
        }

        lines = readfile(template_filename)
        QMin['template'] = {
            **bools,
            **strings,
            **integers,
            **floats,
            **special,
            **self.parse_keywords(
                lines, bools=bools, strings=strings, integers=integers, floats=floats, special=special
            )
        }
        # read external basis set
        if QMin['template']['basis_external']:
            QMin['template']['basis'] = 'gen'
            QMin['template']['basis_external'] = readfile(QMin['template']['basis_external'])
        # do logic checks
        if not QMin['template']['unrestricted_triplets']:
            if len(QMin['template']['charge']) >= 3 and QMin['template']['charge'][0] != QMin['template']['charge'][2]:
                raise Error(
                    'Charges of singlets and triplets differ. Please enable the "unrestricted_triplets" option!', 54
                )

        self._read_template = True
        return

    def read_resources(self, resources_filename="ORCA.resources"):
        super().read_resources(resources_filename)
        QMin = self._QMin
        QMin['Gversion'] = GAUSSIAN.getVersion(QMin['gaussiandir'])

        os.environ['g%sroot' % QMin['Gversion']] = QMin['gaussiandir']
        os.environ['GAUSS_EXEDIR'] = QMin['gaussiandir']
        os.environ['GAUSS_SCRDIR'] = '.'
        os.environ['PATH'] = '$GAUSS_EXEDIR:' + os.environ['PATH']
        QMin['GAUSS_EXEDIR'] = QMin['gaussiandir']
        QMin['GAUSS_EXE'] = os.path.join(QMin['gaussiandir'], 'g%s' % QMin['Gversion'])
        self._read_resources = True
        print('Detected GAUSSIAN version %s' % QMin['Gversion'])
        return

    @staticmethod
    def getVersion(gaussiandir):
        tries = {'g09': '09', 'g16': '16'}
        ls = os.listdir(gaussiandir)
        for i in tries:
            if i in ls:
                return tries[i]
        else:
            raise Error('Found no executable (possible names: %s) in $gaussiandir!' % (list(tries)), 17)

    def _jobs(self):
        QMin = self._QMin
        # make the jobs
        jobs = {}
        if QMin['states_to_do'][0] > 0:
            jobs[1] = {'mults': [1], 'restr': True}
        if len(QMin['states_to_do']) >= 2 and QMin['states_to_do'][1] > 0:
            jobs[2] = {'mults': [2], 'restr': False}
        if len(QMin['states_to_do']) >= 3 and QMin['states_to_do'][2] > 0:
            if not QMin['template']['unrestricted_triplets'] and QMin['states_to_do'][0] > 0:
                # jobs[1]['mults'].append(3)
                jobs[3] = {'mults': [1, 3], 'restr': True}
            else:
                jobs[3] = {'mults': [3], 'restr': False}
        if len(QMin['states_to_do']) >= 4:
            for imult, nstate in enumerate(QMin['states_to_do'][3:]):
                if nstate > 0:
                    jobs[len(jobs) + 1] = {'mults': [imult + 4], 'restr': False}
        QMin['jobs'] = jobs

    def _states_to_do(self):
        QMin = self._QMin
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

    @staticmethod
    def _initorbs(QMin):
        # check for initial orbitals
        initorbs = {}
        step = QMin['step']
        if 'always_guess' in QMin:
            QMin['initorbs'] = {}
        elif 'init' in QMin or 'always_orb_init' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['pwd'], 'GAUSSIAN.chk.init')
                if os.path.isfile(filename):
                    initorbs[job] = filename
            for job in QMin['joblist']:
                filename = os.path.join(QMin['pwd'], f'GAUSSIAN.chk.{job}.init')
                if os.path.isfile(filename):
                    initorbs[job] = filename
            if 'always_orb_init' in QMin and len(initorbs) < QMin['njobs']:
                raise Error('Initial orbitals missing for some jobs!', 59)
            QMin['initorbs'] = initorbs
        elif 'newstep' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['savedir'], f'GAUSSIAN.chk.{job}.{step-1}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    raise Error('File %s missing in savedir!' % (filename), 60)
            QMin['initorbs'] = initorbs
        elif 'samestep' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['savedir'], f'GAUSSIAN.chk.{job}.{step}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    raise Error('File %s missing in savedir!' % (filename), 61)
            QMin['initorbs'] = initorbs
        elif 'restart' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['savedir'], f'GAUSSIAN.chk.{job}.{step}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    raise Error('File %s missing in savedir!' % (filename), 62)
            QMin['initorbs'] = initorbs

    def generate_joblist(self):
        QMin = self._QMin
        # sort the gradients into the different jobs
        gradjob = {}
        for ijob in QMin['joblist']:
            gradjob['master_%i' % ijob] = {}
        for grad in QMin['gradmap']:
            ijob = QMin['multmap'][grad[0]]
            isgs = False
            istates = QMin['states_to_do'][grad[0] - 1]
            if not QMin['jobs'][ijob]['restr']:
                if grad[1] == 1:
                    isgs = True
            else:
                if grad == (1, 1):
                    isgs = True
            if isgs and istates > 1:
                gradjob['grad_%i_%i' % grad] = {}
                gradjob['grad_%i_%i' % grad][grad] = {'gs': True}
            else:
                if len(gradjob['master_%i' % ijob]) > 0:
                    gradjob['grad_%i_%i' % grad] = {}
                    gradjob['grad_%i_%i' % grad][grad] = {'gs': False}
                else:
                    gradjob['master_%i' % ijob][grad] = {'gs': False}

        # make map for states onto gradjobs
        jobgrad = {}
        for job in gradjob:
            for state in gradjob[job]:
                jobgrad[state] = (job, gradjob[job][state]['gs'])
        QMin['jobgrad'] = jobgrad

        densjob = {}
        if 'multipolar_fit' in QMin:
            jobdens = {}
            # detect where the densities will be calculated
            # gs and first es always accessible from master if gs is not other mult
            for dens in QMin['densmap']:
                ijob = QMin['multmap'][dens[0]]
                gsmult = QMin['multmap'][-ijob][0]
                if gsmult == dens[0] and dens[1] == 2:
                    jobdens[dens] = f'master_{ijob}'
                    # check if only ground state is asked
                    densjob[f'master_{ijob}'] = {'scf': True, 'es': True, 'gses': True}
                elif dens in jobgrad:
                    if jobgrad[dens][1]:    # this is a gs only calculation
                        jobdens[dens] = f'master_{ijob}'
                        continue
                    j = jobgrad[dens][0]
                    jobdens[dens] = j
                    if 'master' in j:    # parse excited-state from gradient calc
                        if dens[0] == 3 and QMin['jobs'][ijob]['restr']:
                            # gs already read by from singlet calc, gses for singlet to triplet = 0
                            densjob[j] = {'scf': False, 'es': True, 'gses': True}
                        else:
                            if QMin['states'][gsmult - 1] == 1:
                                densjob[j] = {'scf': True, 'es': False, 'gses': False}
                            else:
                                densjob[j] = {'scf': True, 'es': True, 'gses': True}
                    else:
                        densjob[j] = {'scf': False, 'es': True, 'gses': False}
                elif dens[1] == 1:
                    jobdens[dens] = f'master_{ijob}'
                    if dens[0] == 3 and QMin['jobs'][ijob]['restr']:
                        # gs already read by from singlet calc, gses for singlet to triplet = 0
                        densjob[f'master_{ijob}'] = {'scf': False, 'es': True, 'gses': True}
                    else:
                        densjob[f'master_{ijob}'] = {'scf': True, 'es': False, 'gses': False}
                else:
                    jobdens[dens] = f'dens_{dens[0]}_{dens[1]}'
                    densjob[f'dens_{dens[0]}_{dens[1]}'] = {'scf': False, 'es': True, 'gses': False}

            QMin['jobdens'] = jobdens
            QMin['densjob'] = densjob
        # add the master calculations
        schedule = []
        QMin['nslots_pool'] = []
        ntasks = 0
        for i in gradjob:
            if 'master' in i:
                ntasks += 1
        nrounds, nslots, cpu_per_run = INTERFACE.divide_slots(QMin['ncpu'], ntasks, QMin['schedule_scaling'])
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
                # get the rootstate for the multiplicity as the first excited state
                QMin1['rootstate'] = min(1, QMin['states'][QMin['multmap'][-QMin1['IJOB']][-1] - 1] - 1)
                if 3 in QMin['multmap'][-QMin1['IJOB']] and QMin['jobs'][QMin1['IJOB']]['restr']:
                    QMin1['rootstate'] = 1
                    QMin1['states'][0] = 1
                    QMin1['states_to_do'][0] = 1
                icount += 1
                schedule[-1][i] = QMin1

        # add the gradient calculations
        ntasks = 0
        for i in gradjob:
            if 'grad' in i:
                ntasks += 1
        for i in densjob:
            if 'dens' in i:
                ntasks += 1
        if ntasks > 0:
            nrounds, nslots, cpu_per_run = INTERFACE.divide_slots(QMin['ncpu'], ntasks, QMin['schedule_scaling'])
            QMin['nslots_pool'].append(nslots)
            schedule.append({})
            icount = 0
            for i in gradjob:
                if 'grad' in i:
                    QMin1 = deepcopy(QMin)
                    mult, state = (int(x) for x in i.split('_')[1:])
                    ijob = QMin['multmap'][mult]
                    QMin1['IJOB'] = ijob
                    gsmult = QMin['multmap'][-ijob][0]
                    remove = [
                        'gradmap', 'ncpu', 'h', 'soc', 'dm', 'overlap', 'ion', 'always_guess', 'always_orb_init', 'init'
                    ]
                    for r in remove:
                        QMin1 = removekey(QMin1, r)
                    QMin1['gradmap'] = list(gradjob[i])
                    QMin1['ncpu'] = cpu_per_run[icount]
                    QMin1['gradonly'] = []
                    QMin1['rootstate'] = state - 1 if gsmult == mult else state    # 1 is first excited state of mult
                    icount += 1
                    schedule[-1][i] = QMin1

            for i in densjob:
                if 'dens' in i:
                    QMin1 = deepcopy(QMin)
                    mult, state = (int(x) for x in i.split('_')[1:])
                    ijob = QMin['multmap'][mult]
                    QMin1['IJOB'] = ijob
                    gsmult = QMin['multmap'][-ijob][0]
                    remove = [
                        'gradmap', 'ncpu', 'h', 'soc', 'dm', 'overlap', 'ion', 'always_guess', 'always_orb_init', 'init'
                    ]
                    for r in remove:
                        QMin1 = removekey(QMin1, r)
                    QMin1['ncpu'] = cpu_per_run[icount]
                    QMin1['rootstate'] = state - 1 if gsmult == mult else state    # 1 is first excited state of mult
                    QMin1['densonly'] = True
                    icount += 1
                    schedule[-1][i] = QMin1
        QMin['schedule'] = schedule
        return

    def _backupdir(self):
        QMin = self._QMin
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
            QMin['backup'] = backupdir1


# =============================================================================================== #
# =============================================================================================== #
# ==================================== GAUSSIAN Job Execution =================================== #
# =============================================================================================== #
# =============================================================================================== #

    def run(self):
        self.generate_joblist()
        if DEBUG:
            print('SCHEDULE:')
            pprint.pprint(self._QMin['schedule'], depth=2)

        errorcodes = {}
        # run all the jobs
        errorcodes = self.runjobs(self._QMin['schedule'])

        # do all necessary overlap and Dyson calculations
        errorcodes = self.run_wfoverlap(errorcodes)

        # do all necessary Theodore calculations
        errorcodes = self.run_theodore(errorcodes)

    def dry_run(self):
        self.generate_joblist()
        if DEBUG:
            print('SCHEDULE:')
            pprint.pprint(self._QMin['schedule'], depth=2)

    def runjobs(self, schedule):
        QMin = self._QMin

        print('>>>>>>>>>>>>> Starting the GAUSSIAN job execution')

        errorcodes = {}
        for ijobset, jobset in enumerate(schedule):
            if not jobset:
                continue
            pool = Pool(processes=QMin['nslots_pool'][ijobset])
            for job in jobset:
                QMin1 = jobset[job]
                WORKDIR = os.path.join(QMin['scratchdir'], job)

                errorcodes[job] = pool.apply_async(GAUSSIAN.run_calc, [WORKDIR, QMin1])
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
        if any((i != 0 for i in errorcodes.values())):
            print('Some subprocesses did not finish successfully!')
            raise Error('See %s:%s for error messages in GAUSSIAN output.' % (gethostname(), QMin['scratchdir']), 64)
        self.create_restart_files()
        return errorcodes

    def create_restart_files(self):
        QMin = self._QMin
        if self._PRINT:
            print('>>>>>>>>>>>>> Saving files')
            starttime = datetime.datetime.now()
        for ijobset, jobset in enumerate(QMin['schedule']):
            if not jobset:
                continue
            for job in jobset:
                if 'master' in job:
                    WORKDIR = os.path.join(QMin['scratchdir'], job)
                    if 'samestep' not in QMin:
                        GAUSSIAN.saveFiles(WORKDIR, jobset[job])
                    if 'ion' in QMin and ijobset == 0:
                        GAUSSIAN.saveAOmatrix(WORKDIR, QMin)
        GAUSSIAN.saveGeometry(QMin)
        if self._PRINT:
            endtime = datetime.datetime.now()
            print('Saving Runtime: %s' % (endtime - starttime))
        print

    # ======================================================================= #

    @staticmethod
    def run_calc(WORKDIR, QMin):
        try:
            GAUSSIAN.setupWORKDIR(WORKDIR, QMin)
            strip = True
            err = GAUSSIAN.runGaussian(WORKDIR, QMin['GAUSS_EXE'], strip)
            err = 0
        except Exception as problem:
            print('*' * 50 + '\nException in run_calc(%s)!' % (WORKDIR))
            traceback.print_exc()
            print('*' * 50 + '\n')
            raise problem

        return err

    # ======================================================================= #

    def setupWORKDIR(WORKDIR, QMin):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
        # then put the GAUSSIAN.com file
        GAUSSIAN._initorbs(QMin)
        # setup the directory
        mkdir(WORKDIR)

        # write GAUSSIAN.com
        inputstring = GAUSSIAN.writeGAUSSIANinput(QMin)
        filename = os.path.join(WORKDIR, 'GAUSSIAN.com')
        writefile(filename, inputstring)
        if DEBUG:
            print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
            print(inputstring)
            print('GAUSSIAN input written to: %s' % (filename))
            print('====================================================================')

        # wf file copying
        if 'master' in QMin:
            job = QMin['IJOB']
            if job in QMin['initorbs']:
                fromfile = QMin['initorbs'][job]
                tofile = os.path.join(WORKDIR, 'GAUSSIAN.chk')
                shutil.copy(fromfile, tofile)
        elif 'grad' in QMin or 'densonly' in QMin:
            job = QMin['IJOB']
            fromfile = os.path.join(QMin['scratchdir'], 'master_%i' % job, 'GAUSSIAN.chk')
            tofile = os.path.join(WORKDIR, 'GAUSSIAN.chk')
            shutil.copy(fromfile, tofile)

        # force field file copying
        # if QMin['template']['qmmm']:
        # fromfile=QMin['template']['qmmm_ff_file']
        # tofile=os.path.join(WORKDIR,'ADF.ff')
        # shutil.copy(fromfile,tofile)

        return

    # ======================================================================= #

    @staticmethod
    def writeGAUSSIANinput(QMin):

        # general setup
        job = QMin['IJOB']
        gsmult = QMin['multmap'][-job][0]
        restr = QMin['jobs'][job]['restr']
        charge = QMin['chargemap'][gsmult]

        # determine the root in case it was not determined in schedule jobs
        if 'rootstate' not in QMin:
            QMin['rootstate'] = min(1, QMin['states'][QMin['multmap'][-QMin['IJOB']][-1] - 1] - 1)
            if 3 in QMin['multmap'][-QMin['IJOB']] and QMin['jobs'][QMin['IJOB']]['restr']:
                QMin['rootstate'] = 1

        # excited states to calculate
        states_to_do = QMin['states_to_do']
        for imult in range(len(states_to_do)):
            if not imult + 1 in QMin['multmap'][-job]:
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
            if sing and trip:
                mults_td = ',50-50'
            elif sing and not trip:
                mults_td = ',singlets'
            elif trip and not sing:
                mults_td = ',triplets'
        else:
            ncalc = max(states_to_do)
            mults_td = ''

        # gradients
        if 'gradmap' in QMin:
            dograd = True
            root = QMin['rootstate']
        else:
            dograd = False

        dodens = False
        if 'multipolar_fit' in QMin:
            dodens = True
            root = QMin['rootstate']

        # construct the input string TODO
        string = ''

        # link 0
        string += '%%MEM=%iMB\n' % (QMin['memory'])
        string += '%%NProcShared=%i\n' % (QMin['ncpu'])
        string += '%%Chk=%s\n' % ('GAUSSIAN.chk')
        if 'AOoverlap' in QMin or 'ion' in QMin:
            string += '%%Rwf=%s\n' % ('GAUSSIAN.rwf')
            if 'AOoverlap' in QMin:
                string += '%KJob l302\n'
        string += '\n'

        # Route section
        data = ['p', 'nosym', 'unit=AU', QMin['template']['functional']]
        if not QMin['template']['functional'].lower() == 'dftba':
            data.append(QMin['template']['basis'])
        if dograd:
            data.append('force')
        if 'AOoverlap' in QMin:
            data.append('IOP(2/12=3)')
        if QMin['template']['dispersion']:
            data.append('EmpiricalDispersion=%s' % QMin['template']['dispersion'])
        if QMin['template']['grid']:
            data.append('int(grid=%s)' % QMin['template']['grid'])
        if QMin['template']['denfit']:
            data.append('denfit')
        if ncalc > 0:
            if not QMin['template']['no_tda']:
                s = 'tda'
            else:
                s = 'td'
            if 'master' in QMin:
                s += '(nstates=%i%s' % (ncalc, mults_td)
            else:
                s += '(read'
            if dograd and root > 0:
                s += f',root={root}'
            elif dodens and root > 0:
                s += f',root={root}'
            s += ') density=Current'
            data.append(s)
        if QMin['template']['scrf']:
            s = ','.join(QMin['template']['scrf'].split())
            data.append('scrf(%s)' % s)
        if QMin['template']['scf']:
            s = ','.join(QMin['template']['scf'].split())
            data.append('scf(%s)' % s)
        if QMin['template']['iop']:
            s = ','.join(QMin['template']['iop'].split())
            data.append('iop(%s)' % s)
        if QMin['template']['keys']:
            data.extend([QMin['template']['keys']])
        if 'densonly' in QMin:
            data.append('pop=Regular')    # otherwise CI density will not be printed
            data.append('Guess=read')
        if 'theodore' in QMin:
            data.append('pop=full')
            data.append('IOP(9/40=3)')
        data.append('GFPRINT')
        string += '#'
        for i in data:
            string += i + '\n'
        # title
        string += '\nSHARC-GAUSSIAN job\n\n'

        # charge/mult and geometry
        if 'AOoverlap' in QMin:
            string += '%i %i\n' % (2. * charge, 1)
        else:
            string += '%i %i\n' % (charge, gsmult)
        for label, coords in zip(QMin['elements'], QMin['coords']):
            string += '%4s %16.9f %16.9f %16.9f\n' % (label, coords[0], coords[1], coords[2])
        string += '\n'
        if QMin['template']['functional'].lower() == 'dftba':
            string += '@GAUSS_EXEDIR:dftba.prm\n'
        if QMin['template']['basis_external']:
            for line in QMin['template']['basis_external']:
                string += line
        string += '\n\n'

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
    @staticmethod
    def runGaussian(WORKDIR, GAUSS_EXE, strip=False):
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = GAUSS_EXE + ' '
        string += '< GAUSSIAN.com'
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
            sys.stdout.flush()
        stdoutfile = open(os.path.join(WORKDIR, 'GAUSSIAN.log'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'GAUSSIAN.err'), 'w')
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            raise Error('Call have had some serious problems:', OSError, 65)
        stdoutfile.close()
        stderrfile.close()

        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write(
                'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' %
                (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror)
            )
            sys.stdout.flush()
        os.chdir(prevdir)
        if strip and not DEBUG and runerror == 0:
            GAUSSIAN.stripWORKDIR(WORKDIR)
        return runerror

    # ======================================================================= #
    @staticmethod
    def stripWORKDIR(WORKDIR):
        ls = os.listdir(WORKDIR)
        keep = ['GAUSSIAN.com$', 'GAUSSIAN.err$', 'GAUSSIAN.log$', 'GAUSSIAN.chk', 'GAUSSIAN.fchk', 'GAUSSIAN.rwf']
        for ifile in ls:
            delete = True
            for k in keep:
                if containsstring(k, ifile):
                    delete = False
            if delete:
                rmfile = os.path.join(WORKDIR, ifile)
                if not DEBUG:
                    os.remove(rmfile)

    # ======================================================================= #

    @staticmethod
    def saveGeometry(QMin):
        string = ''
        for label, atom in zip(QMin['elements'], QMin['coords']):
            string += '%4s %16.9f %16.9f %16.9f\n' % (label, atom[0], atom[1], atom[2])
        filename = os.path.join(QMin['savedir'], f'geom.dat.{QMin["step"]}')
        writefile(filename, string)
        if PRINT:
            print(shorten_DIR(filename))
        return

    # ======================================================================= #

    @staticmethod
    def saveFiles(WORKDIR, QMin):

        # copy the TAPE21 from master directories
        job = QMin['IJOB']
        step = QMin['step']
        fromfile = os.path.join(WORKDIR, 'GAUSSIAN.chk')
        tofile = os.path.join(QMin['savedir'], f'GAUSSIAN.chk.{job}.{step}')
        shutil.copy(fromfile, tofile)
        if PRINT:
            print(shorten_DIR(tofile))

        # if necessary, extract the MOs and write them to savedir
        if 'ion' in QMin or not QMin['nooverlap']:
            f = os.path.join(WORKDIR, 'GAUSSIAN.chk')
            string = GAUSSIAN.get_MO_from_chk(f, QMin)
            mofile = os.path.join(QMin['savedir'], f'mos.{job}.{step}')
            writefile(mofile, string)
            if PRINT:
                print(shorten_DIR(mofile))

        # if necessary, extract the TDDFT coefficients and write them to savedir
        if 'ion' in QMin or not QMin['nooverlap']:
            f = os.path.join(WORKDIR, 'GAUSSIAN.chk')
            strings = GAUSSIAN.get_dets_from_chk(f, QMin)
            for f in strings:
                writefile(f, strings[f])
                if PRINT:
                    print(shorten_DIR(f))

    # ======================================================================= #

    @staticmethod
    def get_rwfdump(groot, filename, number):
        WORKDIR = os.path.dirname(filename)
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        dumpname = 'rwfdump.txt'
        string = '%s/rwfdump %s %s %s' % (groot, os.path.basename(filename), dumpname, number)
        # print(string)
        try_shells = ['sh', 'bash', 'csh', 'tcsh']
        ok = False
        for shell in try_shells:
            try:
                runerror = sp.call(string, shell=True, executable=shell)
                ok = True
            except OSError:
                pass
        if not ok:
            raise Error('Gaussian rwfdump has serious problems:', OSError, 68)
        string = readfile(dumpname)
        os.chdir(prevdir)
        return string

    # ======================================================================= #
    @staticmethod
    def get_MO_from_chk(filename, QMin):

        job = QMin['IJOB']
        restr = QMin['jobs'][job]['restr']

        # extract alpha orbitals
        data = GAUSSIAN.get_rwfdump(QMin['gaussiandir'], filename, '524R')
        for iline, line in enumerate(data):
            if "Dump of file" in line:
                break
        mocoef_A = []
        while True:
            iline += 1
            if iline >= len(data):
                break
            s = data[iline].split()
            for i in s:
                mocoef_A.append(float(i.replace('D', 'E')))
        NAO = int(math.sqrt(len(mocoef_A)))
        NMO_A = NAO
        MO_A = [mocoef_A[NAO * i:NAO * (i + 1)] for i in range(NAO)]

        # extract beta orbitals
        if not restr:
            data = GAUSSIAN.get_rwfdump(QMin['gaussiandir'], filename, '526R')
            for iline, line in enumerate(data):
                if "Dump of file" in line:
                    break
            mocoef_B = []
            while True:
                iline += 1
                if iline >= len(data):
                    break
                s = data[iline].split()
                for i in s:
                    mocoef_B.append(float(i.replace('D', 'E')))
            if not NAO == int(math.sqrt(len(mocoef_B))):
                raise Error('Problem in orbital reading!', 69)
            NMO_B = NAO
            MO_B = [mocoef_B[NAO * i:NAO * (i + 1)] for i in range(NAO)]

        NMO = NMO_A - QMin['frozcore']
        if restr:
            NMO = NMO_A - QMin['frozcore']
        else:
            NMO = NMO_A + NMO_B - 2 * QMin['frozcore']

        # make string
        string = '''2mocoef
    header
     1
    MO-coefficients from Gaussian
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

    @staticmethod
    def get_dets_from_chk(filename, QMin):

        # get general infos
        job = QMin['IJOB']
        restr = QMin['jobs'][job]['restr']
        mults = QMin['jobs'][job]['mults']
        if 3 in mults:
            mults = [3]
        gsmult = QMin['multmap'][-job][0]
        nstates_to_extract = deepcopy(QMin['states'])
        for i in range(len(nstates_to_extract)):
            if not i + 1 in mults:
                nstates_to_extract[i] = 0
            elif i + 1 == gsmult:
                nstates_to_extract[i] -= 1

        # get infos from logfile
        logfile = os.path.join(os.path.dirname(filename), 'GAUSSIAN.log')
        data = readfile(logfile)
        infos = {}
        for iline, line in enumerate(data):
            if 'NBsUse=' in line:
                s = line.split()
                infos['nbsuse'] = int(s[1])
            if 'Range of M.O.s used for correlation:' in line:
                for i in [1, 2]:
                    s = data[iline + i].replace('=', ' ').split()
                    for j in range(5):
                        infos[s[2 * j]] = int(s[2 * j + 1])

        if 'NOA' not in infos:
            nstates_onfile = 0
            charge = QMin['chargemap'][gsmult]
            nelec = float(QMin['Atomcharge'] - charge)
            infos['NOA'] = int(nelec / 2. + float(gsmult - 1) / 2.)
            infos['NOB'] = int(nelec / 2. - float(gsmult - 1) / 2.)
            infos['NVA'] = infos['nbsuse'] - infos['NOA']
            infos['NVB'] = infos['nbsuse'] - infos['NOB']
            infos['NFC'] = 0
        else:
            # get all info from checkpoint
            data = GAUSSIAN.get_rwfdump(QMin['gaussiandir'], filename, '635R')
            for iline, line in enumerate(data):
                if "Dump of file" in line:
                    break
            eigenvectors_array = []
            while True:
                iline += 1
                if iline >= len(data):
                    break
                s = data[iline].split()
                for i in s:
                    try:
                        eigenvectors_array.append(float(i.replace('D', 'E')))
                    except ValueError:
                        eigenvectors_array.append(float('NaN'))
            nstates_onfile = (len(eigenvectors_array) -
                              12) // (4 + 8 * (infos['NOA'] * infos['NVA'] + infos['NOB'] * infos['NVB']))
        # print(nstates_onfile)
        # print(len(eigenvectors_array))
        # print(infos)

        # get ground state configuration
        # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
        if restr:
            occ_A = [3 for i in range(infos['NFC'] + infos['NOA'])] + [0 for i in range(infos['NVA'])]
        if not restr:
            occ_A = [1 for i in range(infos['NFC'] + infos['NOA'])] + [0 for i in range(infos['NVA'])]
            occ_B = [2 for i in range(infos['NFC'] + infos['NOB'])] + [0 for i in range(infos['NVB'])]
        occ_A = tuple(occ_A)
        if not restr:
            occ_B = tuple(occ_B)

        # get infos
        nocc_A = infos['NOA']
        nvir_A = infos['NVA']
        nocc_B = infos['NOB']
        nvir_B = infos['NVB']

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
                # get X+Y vector
                startindex = 12 + istate * (nvir_A * nocc_A + nvir_B * nocc_B)
                endindex = startindex + nvir_A * nocc_A + nvir_B * nocc_B
                eig = [i for i in eigenvectors_array[startindex:endindex]]
                # get X-Y vector
                startindex = 12 + istate * (nvir_A * nocc_A +
                                            nvir_B * nocc_B) + 4 * nstates_onfile * (nvir_A * nocc_A + nvir_B * nocc_B)
                endindex = startindex + nvir_A * nocc_A + nvir_B * nocc_B
                eigl = [i for i in eigenvectors_array[startindex:endindex]]
                # get X vector
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
                            index = iocc * nvir_B + ivirt + nvir_A * nocc_A
                            dets[(iocc, ivirt, 2)] = eig[index]
                # truncate vectors
                norm = 0.
                for k in sorted(dets, key=lambda x: dets[x]**2, reverse=True):
                    if restr:
                        factor = 0.5
                    else:
                        factor = 1.
                    if norm > factor * QMin['wfthres']:
                        del dets[k]
                        continue
                    norm += dets[k]**2
                # create strings and expand singlets
                dets2 = {}
                if restr:
                    for iocc, ivirt, dummy in dets:
                        # singlet
                        if mult == 1:
                            # alpha excitation
                            key = list(occ_A)
                            key[infos['NFC'] + iocc] = 2
                            key[infos['NFC'] + nocc_A + ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                            # beta excitation
                            key[infos['NFC'] + iocc] = 1
                            key[infos['NFC'] + nocc_A + ivirt] = 2
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                        # triplet
                        elif mult == 3:
                            key = list(occ_A)
                            key[infos['NFC'] + iocc] = 1
                            key[infos['NFC'] + nocc_A + ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] * math.sqrt(2.)
                else:
                    for iocc, ivirt, dummy in dets:
                        if dummy == 1:
                            key = list(occ_A + occ_B)
                            key[infos['NFC'] + iocc] = 0
                            key[infos['NFC'] + nocc_A + ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                        elif dummy == 2:
                            key = list(occ_A + occ_B)
                            key[2 * infos['NFC'] + nocc_A + nvir_A + iocc] = 0
                            key[2 * infos['NFC'] + nocc_A + nvir_A + nocc_B + ivirt] = 2
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
                        if any(
                            [
                                key[i] != 2 for i in
                                range(nocc_A + nvir_A + QMin['frozcore'], nocc_A + nvir_A + 2 * QMin['frozcore'])
                            ]
                        ):
                            problem = True
                    if problem:
                        print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                        continue
                        # sys.exit(70)
                    if restr:
                        key2 = key[QMin['frozcore']:]
                    else:
                        key2 = key[QMin['frozcore']:QMin['frozcore'] + nocc_A + nvir_A] + key[nocc_A + nvir_A +
                                                                                              2 * QMin['frozcore']:]
                    dets3[key2] = dets2[key]
                # append
                eigenvectors[mult].append(dets3)

        strings = {}
        step = QMin['step']
        for imult, mult in enumerate(mults):
            filename = os.path.join(QMin['savedir'], f'dets.{mult}.{step}')
            strings[filename] = GAUSSIAN.format_ci_vectors(eigenvectors[mult])

        return strings

    # ======================================================================= #

    @staticmethod
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

    @staticmethod
    def saveAOmatrix(WORKDIR, QMin):
        filename = os.path.join(WORKDIR, 'GAUSSIAN.rwf')
        NAO, Smat = GAUSSIAN.get_smat(filename, QMin['gaussiandir'])

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

    @staticmethod
    def get_smat(filename, groot):

        # get all info from checkpoint
        data = GAUSSIAN.get_rwfdump(groot, filename, '514R')

        # extract matrix
        for iline, line in enumerate(data):
            if "Dump of file" in line:
                break
        Smat = []
        while True:
            iline += 1
            if iline >= len(data):
                break
            s = data[iline].split()
            for i in s:
                Smat.append(float(i.replace('D', 'E')))
        NAO = int(math.sqrt(2. * len(Smat) + 0.25) - 0.5)

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

    # =============================================================================================== #
    # =============================================================================================== #
    # =======================================  Dyson and overlap calcs ============================== #
    # =============================================================================================== #
    # =============================================================================================== #

    # ======================================================================= #

    def run_theodore(self, errorcodes):
        QMin = self._QMin

        if 'theodore' in QMin:
            print('>>>>>>>>>>>>> Starting the TheoDORE job execution')

            for ijob in QMin['jobs']:
                if not QMin['jobs'][ijob]['restr']:
                    if self._DEBUG:
                        print('Skipping Job %s because it is unrestricted.' % (ijob))
                    continue
                else:
                    mults = QMin['jobs'][ijob]['mults']
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += QMin['states'][i - 1] - (i == gsmult)
                    if ns == 0:
                        if self._DEBUG:
                            print('Skipping Job %s because it contains no excited states.' % (ijob))
                        continue
                WORKDIR = os.path.join(QMin['scratchdir'], 'master_%i' % ijob)
                self.setupWORKDIR_TH(WORKDIR)
                os.environ
                errorcodes['theodore_%i' % ijob] = GAUSSIAN.runTHEODORE(WORKDIR, QMin['theodir'])

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
            if any((i != 0 for i in errorcodes.values())):
                print('Some subprocesses did not finish successfully!')
                sys.exit(76)

            print('')

        return errorcodes

    def setupWORKDIR_TH(self, WORKDIR):
        QMin = self._QMin
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

        # write dens_ana.in
        inputstring = 'rtype="cclib"\nrfile="GAUSSIAN.log"\njmol_orbitals=False\nmolden_orbitals=False\nOm_formula=2\neh_pop=1\ncomp_ntos=True\nprint_OmFrag=True\noutput_file="tden_summ.txt"\nprop_list=%s\nat_lists=%s' % (
            str(QMin['theodore_prop']), str(QMin['theodore_fragment'])
        )

        filename = os.path.join(WORKDIR, 'dens_ana.in')
        writefile(filename, inputstring)
        if DEBUG:
            print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
            print(inputstring)
            print('TheoDORE input written to: %s' % (filename))
            print('====================================================================')

        return

    @staticmethod
    def runTHEODORE(WORKDIR, THEODIR):
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = os.path.join(THEODIR, 'bin', 'analyze_tden.py')
        stdoutfile = open(os.path.join(WORKDIR, 'theodore.out'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'theodore.err'), 'w')
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
            sys.stdout.flush()
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(77)
        stdoutfile.close()
        stderrfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write(
                'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' %
                (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror)
            )
            sys.stdout.flush()
        os.chdir(prevdir)
        return runerror

    # ======================================================================= #
    def get_Double_AOovl(self):
        QMin = self._QMin

        # get geometries
        filename1 = os.path.join(QMin['savedir'], f'geom.dat.{QMin["step"]-1}')
        oldgeo = GAUSSIAN.get_geometry(filename1)
        filename2 = os.path.join(QMin['savedir'], f'geom.dat.{QMin["step"]}')
        newgeo = GAUSSIAN.get_geometry(filename2)

        # apply shift
        # shift=1e-5
        # for iatom in range(len(oldgeo)):
        # for ixyz in range(3):
        # oldgeo[iatom][1+ixyz]+=shift

        # build QMin   # TODO: always singlet for AOoverlaps
        QMin1 = deepcopy(QMin)
        QMin1['elements'] = [x[0] for x in chain(oldgeo, newgeo)]
        QMin1['coords'] = [x[1:] for x in chain(oldgeo, newgeo)]
        QMin1['AOoverlap'] = [filename1, filename2]
        QMin1['IJOB'] = QMin['joblist'][0]
        QMin1['natom'] = len(newgeo)
        remove = ['nacdr', 'grad', 'h', 'soc', 'dm', 'overlap', 'ion']
        for r in remove:
            QMin1 = removekey(QMin1, r)

        # run the calculation
        WORKDIR = os.path.join(QMin['scratchdir'], 'AOoverlap')
        err = GAUSSIAN.run_calc(WORKDIR, QMin1)

        # get output
        filename = os.path.join(WORKDIR, 'GAUSSIAN.rwf')
        NAO, Smat = GAUSSIAN.get_smat(filename, QMin['gaussiandir'])

        # adjust the diagonal blocks for DFTB-A
        if QMin['template']['functional'] == 'dftba':
            Smat = GAUSSIAN.adjust_DFTB_Smat(Smat, NAO, QMin)

        # Smat is now full matrix NAO*NAO
        # we want the lower left quarter, but transposed
        string = '%i %i\n' % (NAO // 2, NAO // 2)
        for irow in range(NAO // 2, NAO):
            for icol in range(0, NAO // 2):
                string += '% .15e ' % (Smat[icol][irow])    # note the exchanged indices => transposition
            string += '\n'
        filename = os.path.join(QMin['savedir'], 'AO_overl.mixed')
        writefile(filename, string)

        return

    # ======================================================================= #

    @staticmethod
    def get_geometry(filename):
        data = readfile(filename)
        geometry = []
        for line in data:
            s = line.split()
            geometry.append([s[0], float(s[1]), float(s[2]), float(s[3])])
        return geometry

    # ======================================================================= #

    @staticmethod
    def adjust_DFTB_Smat(Smat, NAO, QMin):
        # list with the number of basis functions for basis set VSTO-6G* (used for DFTBA in Gaussian)
        nbasis = {1: ['h', 'he'], 2: ['li', 'be'], 4: ['b', 'c', 'n', 'o', 'f', 'ne']}
        nbs = {}
        for i in nbasis:
            for el in nbasis[i]:
                nbs[el] = i
        Nb = 0
        itot = 0
        mapping = {}
        for ii, i in enumerate(QMin['geo']):
            try:
                Nb += nbs[i[0].lower()]
            except KeyError:
                raise Error('Error: Overlaps with DFTB need further testing!', 80)
            for j in range(nbs[i[0].lower()]):
                mapping[itot] = ii
                itot += 1
        # print mapping
        # sys.exit(81)
        # make interatomic overlap blocks unit matrices
        for i in range(Nb):
            ii = mapping[i]
            for j in range(Nb):
                jj = mapping[j]
                if ii != jj:
                    continue
                if i == j:
                    Smat[i][j + Nb] = 1.
                    Smat[i + Nb][j] = 1.
                else:
                    Smat[i][j + Nb] = 0.
                    Smat[i + Nb][j] = 0.
        return Smat

    # =============================================================================================== #
    # =============================================================================================== #
    # ====================================== GAUSSIAN output parsing ================================ #
    # =============================================================================================== #
    # =============================================================================================== #

    def getQMout(self):
        QMin = self._QMin

        if PRINT:
            print('>>>>>>>>>>>>> Reading output files')
        starttime = datetime.datetime.now()

        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        joblist = QMin['joblist']

        # TODO:
        # excited state energies and transition moments could be read from rwfdump "770R"
        # KS orbital energies: "522R"
        # geometry SEEMS TO BE in "507R"
        # 1TDM might be in "633R"
        # Hamiltonian
        if 'h' in QMin:    # or 'soc' in QMin:
            # make Hamiltonian
            if 'h' not in QMout:
                QMout['h'] = makecmatrix(nmstates, nmstates)
            # go through all jobs
            for job in joblist:
                # first get energies from TAPE21
                logfile = os.path.join(QMin['scratchdir'], 'master_%i/GAUSSIAN.log' % (job))
                energies = GAUSSIAN.getenergy(logfile, job, QMin)
                # also get SO matrix and mapping
                # if 'soc' in QMin and QMin['jobs'][job]['restr']:
                #outfile=os.path.join(QMin['scratchdir'],'master_%i/ADF.out' % (job))
                # submatrix,invstatemap=getsocm(outfile,t21file,job,QMin)
                mults = QMin['multmap'][-job]
                if 3 in mults:
                    mults = [3]
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                        m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                        if m1 not in mults or m2 not in mults:
                            continue
                        if i == j:
                            QMout['h'][i][j] = energies[(m1, s1)]
                        # elif 'soc' in QMin and QMin['jobs'][job]['restr']:
                        # if m1==m2==1:
                        # continue
                        # x=invstatemap[(m1,s1,ms1)]
                        # y=invstatemap[(m2,s2,ms2)]
                        # QMout['h'][i][j]=submatrix[x-1][y-1]

        # Dipole Moments
        if 'dm' in QMin:
            # make matrix
            if 'dm' not in QMout:
                QMout['dm'] = [makecmatrix(nmstates, nmstates) for i in range(3)]
            # go through all jobs
            for job in joblist:
                logfile = os.path.join(QMin['scratchdir'], 'master_%i/GAUSSIAN.log' % (job))
                dipoles = GAUSSIAN.gettdm(logfile, job, QMin)
                mults = QMin['multmap'][-job]
                mults = QMin['multmap'][-job]
                if 3 in mults:
                    mults = [3]
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
                            logfile = os.path.join(QMin['scratchdir'], path, 'GAUSSIAN.log')
                            dm = GAUSSIAN.getdm(logfile)
                            for ixyz in range(3):
                                QMout['dm'][ixyz][i][j] = dm[ixyz]
                        if i == j:
                            continue
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
                logfile = os.path.join(QMin['scratchdir'], path, 'GAUSSIAN.log')
                g = GAUSSIAN.getgrad(logfile, QMin)
                for istate in QMin['statemap']:
                    state = QMin['statemap'][istate]
                    if (state[0], state[1]) == grad:
                        QMout['grad'][istate - 1] = g
            if QMin['neglected_gradient'] != 'zero':
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
                        QMout['overlap'][i][j] = GAUSSIAN.getsmate(out, s1, s2)

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
                        QMout['prop'][i][j] = GAUSSIAN.getDyson(out, s1, s2) * factor

        # multipolar_density fits
        if 'multipolar_fit' in QMin:
            if 'multipolar_fit' not in QMout:
                QMout['multipolar_fit'] = {}

            for dens in QMin['densjob']:
                workdir = os.path.join(QMin['scratchdir'], dens)
                self.get_fchk(workdir, QMin['gaussiandir'])
            # sort densjobs
            density_map = {}    # map for (mult, state, state): position in densities
            sorted_densjobs = []
            jobfiles = set()
            i = 0
            for dens in QMin['densmap']:
                job = QMin['jobdens'][dens]
                if job not in jobfiles:
                    jobfiles.add(job)
                    flags = QMin['densjob'][job]
                    sorted_densjobs.append((job, flags))
                    scf, es, gses = map(flags.get, ('scf', 'es', 'gses'))
                    if scf:
                        density_map[(*dens, *dens)] = i
                        i += 1
                    if es:
                        edens = (dens[0], dens[1] + 1) if scf else dens
                        density_map[(*edens, *edens)] = i
                        i += 1
                    if gses:
                        # determine the excited states for this multiplicity
                        nstates = QMin['states'][dens[0] - 1]
                        ijob = QMin['multmap'][dens[0]]
                        gsmult = QMin['multmap'][-ijob][0]
                        first = 2
                        if gsmult != dens[0]:
                            first = 1
                        last = nstates
                        for es in range(first, last + 1):
                            state = dens[1]
                            density_map[(gsmult, 1, dens[0], es)] = i
                            i += 1
            # read basis
            fchkfile = os.path.join(QMin['scratchdir'], sorted_densjobs[0][0], 'GAUSSIAN.fchk')
            basis, n_bf, cartesian_d, cartesian_f, p_eq_s_shell = self.get_basis(fchkfile)
            print("basis information: P(S=P):", p_eq_s_shell, " cartesian d:", cartesian_d, "cartesian_f", cartesian_f)
            ECPs = self.parse_ecp(fchkfile)
            # collect all densities from the file in densjob (file: bools) and jobdens (state: file)
            densities = self.get_dens_from_fchks(
                sorted_densjobs,
                basis,
                n_bf,
                cartesian_d=cartesian_d,
                cartesian_f=cartesian_f,
                p_eq_s_shell=p_eq_s_shell
            )
            #  after reordering the densities on needs to align the d and f orbitals for spherical or cartesian basis
            #  if cartesian_d != cartesian_f:
            #  nao = len(densities[0])
            #  # change both to spherical
            #  cartesian_basis = False
            #  cart2sph_matrix = get_cart2sph_matrix(3 if cartesian_f else 2, nao, QMin['elements'], basis)
            #  for i in range(len(densities)):
            #  print(np.linalg.norm(densities[i]))
            #  densities[i] = cart2sph_matrix.T @ densities[i] @ cart2sph_matrix
            #  print(np.linalg.norm(densities[i]))

            #  else:
            #  cartesian_basis = cartesian_d
            cartesian_basis = cartesian_d
            fits = Resp(
                QMin['coords'],
                QMin['elements'],
                QMin['resp_vdw_radii'],
                QMin['resp_density'],
                QMin['resp_shells'],
                grid=QMin['resp_grid'],
                beta=QMin['resp_beta']
            )
            gsmult = QMin['statemap'][1][0]
            charge = QMin['chargemap'][gsmult]
            pprint.pprint(ECPs)
            fits.prepare(
                basis, gsmult - 1, charge, ecps=ECPs, cart_basis=cartesian_basis
            )    # the charge of the atom does not affect integrals
            # obtain normalization coefficients of pyscf overlap
            ao_sqrt_norms = np.sqrt(np.diag(fits.Sao))
            # obtain new order of the AO orbitals
            new_order = get_pyscf_order_from_gaussian(QMin['elements'], basis, cartesian_d=cartesian_d, cartesian_f=cartesian_f)
            print("reordering atomic orbitals according to")
            print(new_order)
            if len(new_order) != len(densities[0]):
                raise Error("The list with the new order of the AOs has a different length!", 45)
            # reorder all densities and renormalize them
            for i in range(len(densities)):
                densities[i] = densities[i][:, new_order][new_order, :]
                densities[i] = (densities[i] / ao_sqrt_norms[:, None]) / ao_sqrt_norms[None, :]

            fits_map = {}
            for i, d_i in enumerate(QMin['densmap']):
                # do gs density
                key = (*d_i, *d_i)
                fits_map[key] = fits.multipoles_from_dens(
                    densities[density_map[key]],
                    include_core_charges=True,
                    order=QMin['resp_fit_order'],
                    charge=QMin['chargemap'][d_i[0]]
                )

                for d_j in QMin['densmap'][i + 1:]:
                    if d_i[0] != d_j[0]:
                        continue
                    # do gses and eses density
                    key = (*d_i, *d_j)
                    if key in density_map:
                        fits_map[key] = fits.multipoles_from_dens(
                            densities[density_map[key]], include_core_charges=False, order=QMin['resp_fit_order']
                        )
                    else:
                        ijob = QMin['multmap'][d_i[0]]    # the multiplicity is the same -> ijob same
                        gsmult = QMin['multmap'][-ijob][0]
                        dmI = densities[density_map[(gsmult, 1, *d_i)]]
                        dmJ = densities[density_map[(gsmult, 1, *d_j)]]
                        trans_dens = es2es_tdm(dmI, dmJ, fits.Sao)
                        fits_map[key] = fits.multipoles_from_dens(
                            trans_dens, include_core_charges=False, order=QMin['resp_fit_order']
                        )
            QMout['multipolar_fit'] = fits_map

        # TheoDORE
        if 'theodore' in QMin:
            if 'theodore' not in QMout:
                QMout['theodore'] = makecmatrix(QMin['theodore_n'], nmstates)
            for job in joblist:
                if not QMin['jobs'][job]['restr']:
                    continue
                else:
                    mults = QMin['jobs'][job]['mults']
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += QMin['states'][i - 1] - (i == gsmult)
                    if ns == 0:
                        continue
                sumfile = os.path.join(QMin['scratchdir'], 'master_%i/tden_summ.txt' % job)
                omffile = os.path.join(QMin['scratchdir'], 'master_%i/OmFrag.txt' % job)
                props = GAUSSIAN.get_theodore(sumfile, omffile, QMin)
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    if (m1, s1) in props:
                        for j in range(QMin['theodore_n']):
                            QMout['theodore'][i][j] = props[(m1, s1)][j]

        endtime = datetime.datetime.now()
        if PRINT:
            print("Readout Runtime: %s" % (endtime - starttime))

        if DEBUG:
            copydir = os.path.join(QMin['savedir'], 'debug_GAUSSIAN_stdout')
            if not os.path.isdir(copydir):
                mkdir(copydir)
            for job in joblist:
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/GAUSSIAN.log' % (job))
                shutil.copy(outfile, os.path.join(copydir, "GAUSSIAN_%i.log" % job))
                if QMin['jobs'][job]['restr'] and 'theodore' in QMin:
                    outfile = os.path.join(QMin['scratchdir'], 'master_%i/tden_summ.txt' % job)
                    try:
                        shutil.copy(outfile, os.path.join(copydir, 'THEO_%i.out' % (job)))
                    except IOError:
                        pass
                    outfile = os.path.join(QMin['scratchdir'], 'master_%i/OmFrag.txt' % job)
                    try:
                        shutil.copy(outfile, os.path.join(copydir, 'THEO_OMF_%i.out' % (job)))
                    except IOError:
                        pass
            if 'grad' in QMin:
                for grad in QMin['gradmap']:
                    path, isgs = QMin['jobgrad'][grad]
                    outfile = os.path.join(QMin['scratchdir'], path, 'GAUSSIAN.log')
                    shutil.copy(outfile, os.path.join(copydir, "GAUSSIAN_GRAD_%i_%i.log" % grad))
            if 'overlap' in QMin:
                for mult in itmult(QMin['states']):
                    job = QMin['multmap'][mult]
                    outfile = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
                    shutil.copy(outfile, os.path.join(copydir, 'WFOVL_%i_%i.out' % (mult, job)))
            if 'ion' in QMin:
                for ion in QMin['ionmap']:
                    outfile = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
                    shutil.copy(outfile, os.path.join(copydir, 'Dyson_%i_%i_%i_%i.out' % ion))

        return

    # ======================================================================= #

    @staticmethod
    def getenergy(logfile, ijob, QMin):

        # open file
        f = readfile(logfile)
        if PRINT:
            print('Energy:   ' + shorten_DIR(logfile))

        # read ground state
        for line in f:
            if ' SCF Done:' in line:
                gsenergy = float(line.split()[4])

        # figure out the excited state settings
        mults = QMin['jobs'][ijob]['mults']
        restr = QMin['jobs'][ijob]['restr']
        gsmult = mults[0]
        estates_to_extract = deepcopy(QMin['states'])
        estates_to_extract[gsmult - 1] -= 1
        for imult in range(len(estates_to_extract)):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0
        for imult in range(len(estates_to_extract)):
            if imult + 1 in mults:
                estates_to_extract[imult] = max(estates_to_extract)

        # extract excitation energies
        # loop also works if no energies should be extracted
        energies = {(gsmult, 1): gsenergy}
        for imult in mults:
            nstates = estates_to_extract[imult - 1]
            if nstates > 0:
                istate = 0
                for line in f:
                    if 'Excited State' in line:
                        if restr:
                            if not IToMult[imult] in line:
                                continue
                        energies[(imult, istate + 1 + (gsmult == imult))] = float(line.split()[4]) / au2eV + gsenergy
                        istate += 1
                        if istate >= nstates:
                            break
        return energies

    # ======================================================================= #
    @staticmethod
    def gettdm(logfile, ijob, QMin):

        # open file
        f = readfile(logfile)
        if PRINT:
            print('Dipoles:  ' + shorten_DIR(logfile))

        # figure out the excited state settings
        mults = QMin['jobs'][ijob]['mults']
        if 3 in mults:
            mults = [3]
        restr = QMin['jobs'][ijob]['restr']
        gsmult = mults[0]
        estates_to_extract = deepcopy(QMin['states'])
        estates_to_extract[gsmult - 1] -= 1
        for imult in range(len(estates_to_extract)):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0

        # get ordering of states in Gaussian output
        istate = [int(i + 1 == gsmult) for i in range(len(QMin['states']))]
        index = 0
        gaustatemap = {}
        for iline, line in enumerate(f):
            if 'Excited State' in line:
                if restr:
                    s = line.replace('-', ' ').split()
                    imult = IToMult[s[3]]
                    istate[imult - 1] += 1
                    gaustatemap[(imult, istate[imult - 1])] = index
                    index += 1
                else:
                    imult = gsmult
                    istate[imult - 1] += 1
                    gaustatemap[(imult, istate[imult - 1])] = index
                    index += 1

        # extract transition dipole moments
        dipoles = {}
        for imult in mults:
            if not imult == gsmult:
                continue
            nstates = estates_to_extract[imult - 1]
            if nstates > 0:
                for iline, line in enumerate(f):
                    if 'Ground to excited state transition electric dipole moments ' in line:
                        for istate in range(nstates):
                            shift = gaustatemap[(imult, istate + 1 + (gsmult == imult))]
                            s = f[iline + 2 + shift].split()
                            dipoles[(imult, istate + 1 + (gsmult == imult))] = [float(i) for i in s[1:4]]
        return dipoles

    # ======================================================================= #
    @staticmethod
    def get_fchk(workdir, gaussiandir=''):
        prevdir = os.getcwd()
        os.chdir(workdir)
        string = os.path.join(gaussiandir, 'formchk') + ' GAUSSIAN.chk'
        try:
            sp.call(string, shell=True, stdout=sys.stderr, stderr=sys.stderr)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(77)
        print('Generated .fchk file in', workdir)
        os.chdir(prevdir)

    @staticmethod
    def get_basis(fchkfile: str):
        cartesian = False
        p_eq_s = False
        n_bf = 0
        f = open(fchkfile, 'r')
        lines = f.readlines()
        f.close()
        shell_types = []
        n_prim = []
        s_a_map = []
        prim_exp = []
        contr_coeff = []
        ps_contr_coeff = None
        atom_symbols = []
        i = 0
        while i != len(lines):
            if 'Atomic numbers' in lines[i]:
                natom = int(lines[i].split()[-1])
                n_lines = ((natom - 1) // 6 + 1)
                i += 1
                atom_symbols = list(
                    map(lambda x: IAn2AName[int(x)], chain(*map(lambda x: x.split(), lines[i:i + n_lines])))
                )
            if 'Number of basis functions' in lines[i]:
                n_bf = int(lines[i].split()[-1])
            if 'Pure/Cartesian d shells' in lines[i]:
                cartesian_d = int(lines[i].split()[-1]) == 1
            if 'Pure/Cartesian f shells' in lines[i]:
                cartesian_f = int(lines[i].split()[-1]) == 1
            if 'Shell types' in lines[i]:
                n = int(lines[i].split()[-1])
                n_lines = (n - 1) // 6 + 1
                i += 1
                shell_types = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                i += n_lines
            if 'Number of primitives per shell' in lines[i]:
                n = int(lines[i].split()[-1])
                n_lines = (n - 1) // 6 + 1
                i += 1
                n_prim = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                i += n_lines
            if 'Shell to atom map' in lines[i]:
                n = int(lines[i].split()[-1])
                n_lines = (n - 1) // 6 + 1
                i += 1
                s_a_map = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                i += n_lines
            if 'Primitive exponents' in lines[i]:
                n = int(lines[i].split()[-1])
                n_lines = (n - 1) // 5 + 1
                i += 1
                prim_exp = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                i += n_lines
            if 'Contraction coefficients' in lines[i]:
                n = int(lines[i].split()[-1])
                n_lines = (n - 1) // 5 + 1
                i += 1
                contr_coeff = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                i += n_lines
                if 'P(S=P) Contraction coefficients' in lines[i]:
                    p_eq_s = True
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    ps_contr_coeff = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                break
            i += 1
        return build_basis_dict(
            atom_symbols, shell_types, n_prim, s_a_map, prim_exp, contr_coeff, ps_contr_coeff
        ), n_bf, cartesian_d, cartesian_f, p_eq_s

    @staticmethod
    def parse_ecp(fchkfile: str):
        props = {
            'Number of atoms': None,
            'Atomic numbers': None,
            'ECP-MaxLECP': None,
            'ECP-KFirst': None,
            'ECP-KLast': None,
            'ECP-LMax': None,
            'ECP-LPSkip': None,
            'ECP-RNFroz': None,
            'ECP-NLP': None,
            'ECP-CLP1': None,
            'ECP-ZLP': None
        }
        types = {'I': int, 'R': float, 'C': str}

        with open(fchkfile, 'r') as f:
            line = f.readline()

            def parse_num(llst, cast):
                return cast(llst[-1])

            def parse_array(n, cast, buf):
                npl = 6 if cast == int else 5
                nl = (n - 1) // npl + 1
                return np.fromiter(
                    chain(*map(lambda x: x.split(), map(lambda _: buf.readline(), range(nl)))), count=n, dtype=cast
                )

            while line:
                for k in filter(lambda k: props[k] is None, props.keys()):
                    if k in line:
                        llst = line[len(k):].split()
                        cast = types[llst[0]]
                        n = parse_num(llst, cast)
                        if llst[1] == 'N=':
                            props[k] = parse_array(int(llst[-1]), cast, f)
                        else:
                            props[k] = n
                # ----------------------------
                line = f.readline()

        # ++++++++++++++++++ Start making things
        natom = props['Number of atoms']
        if props['ECP-NLP'] is None:
            print("no ECPS found!")
            return {}
        skips = props['ECP-LPSkip'] == 0
        kfirst = props['ECP-KFirst'].reshape((-1, natom))[:, skips]
        klast = props['ECP-KLast'].reshape((-1, natom))[:, skips]
        lmax = props['ECP-LMax'][skips]
        froz = props['ECP-RNFroz'][skips].astype(int)
        nlp = props['ECP-NLP']
        clp1 = props['ECP-CLP1']
        zlp = props['ECP-ZLP']

        atom_ids = np.where(skips)[0]
        symbols = [IAn2AName[props['Atomic numbers'][x]] for x in atom_ids]
        fun_sym = 'SPDFGHIJKLMNOTU'

        ECPs = {}
        # loop over all atoms
        for (i, a), s, lm in zip(enumerate(atom_ids), symbols, lmax):
            ecp_string = f'{s} nelec {froz[i]: d}\n'

            # build the momentum list (with highest momentum first labeled as u1)
            funs = [fun_sym[x] for x in reversed(range(lm))]
            funs.append('ul')

            # loop over all angular momentums
            for j, (fi, la, fun) in enumerate(zip(kfirst[:, i], klast[:, i], reversed(funs))):
                ecp_string += f'{s} {fun}\n'
                for y in range(fi - 1, la):
                    ecp_string += f'{nlp[y]:2d}    {zlp[y]: 12.7f}       {clp1[y]: 12.7f}\n'
            ECPs[a] = ecp_string

        return ECPs

    def get_dens_from_fchks(
        self,
        densjobs: list[tuple[str, dict[str, bool]]],
        basis,
        n_bf,
        cartesian_d=False,
        cartesian_f=False,
        p_eq_s_shell=False
    ):
        QMin = self._QMin
        densities = []
        atom_symbols = QMin['elements']
        for dens, flags in densjobs:
            scf, es, gses = map(flags.get, ('scf', 'es', 'gses'))
            fchkfile = os.path.join(QMin['scratchdir'], dens, 'GAUSSIAN.fchk')
            scf_read = es_read = gses_read = False
            f = open(fchkfile, 'r')
            lines = f.readlines()
            f.close()
            new_dens = 0
            i = 0
            while i < len(lines):
                if scf and 'SCF Density' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    d = np.fromiter(
                        map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                    )
                    # d is triangular -> fold into full matrix (assuming lower triangular)
                    d_tril = np.zeros((n_bf, n_bf))
                    idx = np.tril_indices(n_bf)
                    d_tril[idx] = d
                    density = d_tril.T + d_tril
                    np.fill_diagonal(density, np.diag(d_tril))
                    densities.append(density)
                    new_dens += 1
                    i += n_lines
                    scf_read = True
                if es and 'CI Density' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    d = np.fromiter(
                        map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                    )
                    # d is triangular -> fold into full matrix (assuming lower triangular)
                    d_tril = np.zeros((n_bf, n_bf))
                    idx = np.tril_indices(n_bf)
                    d_tril[idx] = d
                    density = d_tril.T + d_tril
                    np.fill_diagonal(density, np.diag(d_tril))
                    densities.append(density)
                    new_dens += 1
                    i += n_lines
                    es_read = True
                if 'Excited state NLR' in lines[i]:
                    i += 1
                    n_es = int(lines[i].split()[-1])
                    i += 1
                    n_g2e = int(lines[i].split()[-1])
                    i += 1
                    if n_es != 0:
                        i += 1
                if gses and 'G to E trans densities' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    d = np.fromiter(
                        map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                    ).reshape((2 * n_g2e, n_bf, n_bf))
                    #TODO average over the two density matrices: X+Y + X-Y /2 = X
                    for i_d in range(0, 2 * n_g2e, 2):
                        tmp = (d[i_d, ...] + d[i_d + 1, ...]) * math.sqrt(2)
                        densities.append(tmp)
                        new_dens += 1
                    i += n_lines
                    gses_read = True
                i += 1
            if scf and not scf_read:
                raise Error(f'Missing "SCF Density" in checkpoint file for job {dens}!\nfchk-file:  {fchkfile}', 33)
            if es and not es_read:
                raise Error(f'Missing "CI Density" in checkpoint file for job {dens}!\nfchk-file:  {fchkfile}', 33)
            if gses and not gses_read:
                raise Error(
                    f'Missing "G to E trans densities" in checkpoint file for job {dens}!\nfchk-file:  {fchkfile}', 33
                )
        # read densities
        return densities

    # ======================================================================= #

    def getdm(logfile):

        # open file
        f = readfile(logfile)
        if PRINT:
            print('Dipoles:  ' + shorten_DIR(logfile))

        for iline, line in enumerate(f):
            if 'Forces (Hartrees/Bohr)' in line:
                s = f[iline - 2].split('=')[1].replace('D', 'E')
                dmx = float(s[0:15])
                dmy = float(s[15:30])
                dmz = float(s[30:45])
                dm = [dmx, dmy, dmz]
                return dm

    # ======================================================================= #

    @staticmethod
    def getgrad(logfile, QMin):

        # read file and check if ego is active
        out = readfile(logfile)
        if PRINT:
            print('Gradient: ' + shorten_DIR(logfile))

        # initialize
        natom = QMin['natom']
        g = [[0. for i in range(3)] for j in range(natom)]

        # get gradient
        string = 'Forces (Hartrees/Bohr)'
        shift = 3
        for iline, line in enumerate(out):
            if string in line:
                for iatom in range(natom):
                    s = out[iline + shift + iatom].split()
                    for i in range(3):
                        g[iatom][i] = -float(s[2 + i])

        return g

    # ======================================================================= #
    @staticmethod
    def getsmate(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                raise Error('Overlap of states %i - %i not found!' % (s1, s2), 82)
            if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    # ======================================================================= #

    @staticmethod
    def getDyson(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                raise Error('Dyson norm of states %i - %i not found!' % (s1, s2), 83)
            if containsstring('Dyson norm matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    # ======================================================================= #

    @staticmethod
    def get_theodore(sumfile, omffile, QMin):    # TODO!
        def theo_float(i):
            return safe_cast(i, float, 0.)

        out = readfile(sumfile)
        if PRINT:
            print('TheoDORE: ' + shorten_DIR(sumfile))
        props = {}
        for line in out[2:]:
            s = line.replace('?', ' ').split()
            if len(s) == 0:
                continue
            n = int(re.search('([0-9]+)', s[0]).groups()[0])
            m = re.search('([a-zA-Z]+)', s[0]).groups()[0]
            for i in IToMult:
                if isinstance(i, str) and m in i:
                    m = IToMult[i]
                    break
            props[(m, n + (m == 1))] = [theo_float(i) for i in s[4:]]

        out = readfile(omffile)
        if PRINT:
            print('TheoDORE: ' + shorten_DIR(omffile))
        for line in out[1:]:
            s = line.replace('(', ' ').replace(')', ' ').split()
            if len(s) == 0:
                continue
            n = int(re.search('([0-9]+)', s[0]).groups()[0])
            m = re.search('([a-zA-Z]+)', s[0]).groups()[0]
            for i in IToMult:
                if isinstance(i, str) and m in i:
                    m = IToMult[i]
                    break
            props[(m, n + (m == 1))].extend([theo_float(i) for i in s[2:]])

        return props

if __name__ == '__main__':

    gaussian = GAUSSIAN(get_bool_from_env('DEBUG', False), get_bool_from_env('PRINT'))
    gaussian.main()
