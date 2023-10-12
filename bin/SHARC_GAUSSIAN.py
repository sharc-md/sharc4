#!/usr/bin/env python3

# external

# internal
from resp import Resp
from qmin import QMin
from qmout import QMout
from utils import mkdir, readfile, containsstring, shorten_DIR, makermatrix, makecmatrix, build_basis_dict, get_pyscf_order_from_gaussian, writefile, itmult, safe_cast, strip_dir
from constants import IToMult, au2eV, IAn2AName
from tdm import es2es_tdm
from SHARC_ABINITIO import SHARC_ABINITIO

import os
import sys
import shutil
import pprint
import re
import math
import time
import datetime
import traceback
from io import TextIOWrapper
from textwrap import dedent
from typing import Optional
from itertools import chain
import subprocess as sp
from multiprocessing import Pool
from socket import gethostname
from copy import deepcopy

from numpy import np

__all__ = ["SHARC_GAUSSIAN"]

AUTHORS = "Sebastian Mai, Severin Polonius, Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "GAUSSIAN"
DESCRIPTION = "SHARC interface for the GAUSSIAN16 program suite"

CHANGELOGSTRING = """
"""
all_features = {"h", "dm", "grad", "overlap", "phases", "ion", "multipolar_fit", "theodore", "point_charges",
                # raw data request
                "basis_set", "wave_functions", "density_matrices", }

class SHARC_GAUSSIAN(SHARC_ABINITIO):
    """
    SHARC interface for gaussian
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION
    _theodore_settings = {
        'rtype': 'cclib',
        'rfile': 'GAUSSIAN.log',
        'read_binary': False,
        'jmol_orbitals': False,
        'molden_orbitals': False,
        'Om_formula': 2,
        'eh_pop': 1,
        'comp_ntos': True,
        'print_OmFrag': True,
        'output_file': 'tden_summ.txt',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "basis": "6-31G",
                "functional": "PBEPBE",
                "dispersion": None,
                "scrf": None,
                "grid": None,
                "denfit": False,
                "scf": None,
                "no_tda": False,
                "unrestricted_triplets": False,
                "iop": None,
                "paste_input_file": None,
                "external_basis": None,
                "noneqsolv": False
            }
        )
        self.QMin.template.types.update(
            {
                "basis": str,
                "functional": str,
                "dispersion": str,
                "scrf": str,
                "grid": str,
                "denfit": bool,
                "scf": str,
                "no_tda": bool,
                "unrestricted_triplets": bool,
                "iop": str,
                "paste_input_file": str,
                "external_basis": str,
                "noneqsolv": bool
            }
        )

        # Add resource keys
        self.QMin.resources.update(
            {
                "groot": None,
                "wfoverlap": None,
                "wfthres": None,
                "numfrozcore": 0,
                "numocc": None,
                "schedule_scaling": 0.9,
                "delay": 1
            }
        )

        self.QMin.resources.types.update(
            {
                "groot": str,
                "wfoverlap": str,
                "wfthres": float,
                "numfrozcore": int,
                "numocc": int,
                "schedule_scaling": float,
                "delay": int
            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_GAUSSIAN._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_GAUSSIAN._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_GAUSSIAN._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_GAUSSIAN._authors

    @staticmethod
    def name() -> str:
        return SHARC_GAUSSIAN._name

    @staticmethod
    def description() -> str:
        return SHARC_GAUSSIAN._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_GAUSSIAN._name}\n{SHARC_GAUSSIAN._description}"

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS


    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

    def read_resources(self, resources_filename="GAUSSIAN.resources"):
        super().read_resources(resources_filename)
        self._Gversion = SHARC_GAUSSIAN.getVersion(QMin.resources['groot'])

        os.environ['groot'] = self.QMin['groot']
        os.environ['GAUSS_EXEDIR'] = self.QMin['groot']
        os.environ['GAUSS_SCRDIR'] = '.'
        os.environ['PATH'] = '$GAUSS_EXEDIR:' + os.environ['PATH']
        self.QMin['GAUSS_EXEDIR'] = self.QMin['groot']
        self.QMin['GAUSS_EXE'] = os.path.join(QMin['groot'], 'g%s' % self._Gversion)
        self._read_resources = True
        self.log.info('Detected GAUSSIAN version %s' % self._Gversion)

    def read_template(self, template_file: str) -> None:
        super().read_template(template_file)

        # Convert keys to string if list
        if isinstance(self.QMin.template["keys"], list):
            self.QMin.template["keys"] = " ".join(self.QMin.template["keys"])

        # read external basis set
        if self.QMin.template['basis_external']:
            self.QMin.template['basis'] = 'gen'
            self.QMin.template['basis_external'] = readfile(self.QMin.template['basis_external'])
        if self.QMin.template['paste_input_file']:
            self.QMin.template['paste_input_file'] = readfile(self.QMin.template['paste_input_file'])
        # do logic checks
        if not self.QMin.template['unrestricted_triplets']:
            if len(self.QMin.template['charge']) >= 3 and self.QMin.template['charge'][0] != self.QMin.template['charge'][2]:
                raise RuntimeError(
                    'Charges of singlets and triplets differ. Please enable the "unrestricted_triplets" option!'
                )

        self._read_template = True

    # TODO
    def setup_interface(self):

        # make the chargemap
        self.QMin.maps['chargemap'] = {i + 1: c for i, c in enumerate(self.QMin.template['charge'])}
        self._states_to_do()    # can be different in interface -> general method here with possibility to overwrite
        # make the jobs
        self._jobs()
        jobs = QMin.control['jobs']
        # make the multmap (mapping between multiplicity and job)
        multmap = {}
        for ijob, job in jobs.items():
            for imult in job['mults']:
                multmap[imult] = ijob
            multmap[-(ijob)] = job['mults']
        multmap[1] = 1
        self.QMin.maps['multmap'] = multmap

        # get the joblist
        self.QMin.control['joblist'] = sorted(jobs.keys())
        self.QMin.control['njobs'] = len(QMin['joblist'])

        # make the gsmap
        gsmap = {}
        for i in range(self.QMin.molecule['nmstates']):
            m1, s1, ms1 = tuple(self.QMin.molecule['statemap'][i + 1])
            gs = (m1, 1, ms1)
            job = self.QMin.maps['multmap'][m1]
            if m1 == 3 and self.QMin.control['jobs'][job]['restr']:
                gs = (1, 1, 0.0)
            for j in range(self.QMin.molecule['nmstates']):
                m2, s2, ms2 = tuple(self.QMin.maps['statemap'][j + 1])
                if (m2, s2, ms2) == gs:
                    break
            gsmap[i + 1] = j + 1
        self.QMin.maps['gsmap'] = gsmap
        pass

    def _request_logic(self):
        super()._request_logic()
        # make the ionmap
        if self.QMin.requests['ion']:
            ionmap = []
            for m1 in itmult(self.QMin.molecule['states']):
                job1 = self.QMin.maps['multmap'][m1]
                el1 = self.QMin.maps['chargemap'][m1]
                for m2 in itmult(self.QMin.molecule['states']):
                    if m1 >= m2:
                        continue
                    job2 = self.QMin.maps['multmap'][m2]
                    el2 = self.QMin.maps['chargemap'][m2]
                    # print m1,job1,el1,m2,job2,el2
                    if abs(m1 - m2) == 1 and abs(el1 - el2) == 1:
                        ionmap.append((m1, job1, m2, job2))
            self.QMin.maps['ionmap'] = ionmap

    @staticmethod
    def getVersion(groot):
        tries = {'g09': '09', 'g16': '16'}
        ls = os.listdir(groot)
        for i in tries:
            if i in ls:
                return tries[i]
        else:
            # self.log.error('Found no executable (possible names: %s) in $groot!' % (list(tries)))
            raise RuntimeError('Found no executable (possible names: %s) in $groot!' % (list(tries)))

    
    def _jobs(self):
        self.QMin = self._QMin
        # make the jobs
        jobs = {}
        if self.QMin.control['states_to_do'][0] > 0:
            jobs[1] = {'mults': [1], 'restr': True}
        if len(QMin.control['states_to_do']) >= 2 and self.QMin.control['states_to_do'][1] > 0:
            jobs[2] = {'mults': [2], 'restr': False}
        if len(QMin.control['states_to_do']) >= 3 and self.QMin.control['states_to_do'][2] > 0:
            if not self.QMin.template['unrestricted_triplets'] and self.QMin.control['states_to_do'][0] > 0:
                # jobs[1]['mults'].append(3)
                jobs[3] = {'mults': [1, 3], 'restr': True}
            else:
                jobs[3] = {'mults': [3], 'restr': False}
        if len(QMin.control['states_to_do']) >= 4:
            for imult, nstate in enumerate(QMin.control['states_to_do'][3:]):
                if nstate > 0:
                    jobs[len(jobs) + 1] = {'mults': [imult + 4], 'restr': False}
        self.QMin.control['jobs'] = jobs

    def _states_to_do(self):
        self.QMin = self._QMin
        # obtain the states to actually compute
        states_to_do = deepcopy(QMin.molecule['states'])
        for i in range(len(QMin.molecule['states'])):
            if states_to_do[i] > 0:
                states_to_do[i] += self.QMin.template['paddingstates'][i]
        if not self.QMin.template['unrestricted_triplets']:
            if len(QMin.molecule['states']) >= 3 and self.QMin.molecule['states'][2] > 0 and self.QMin.molecule['states'][0] <= 1:
                if 'soc' in self.QMin:
                    states_to_do[0] = 2
                else:
                    states_to_do[0] = 1
        self.QMin.control['states_to_do'] = states_to_do

    def _initorbs(QMin):
        # check for initial orbitals
        initorbs = {}
        step = QMin.save['step']
        if 'always_guess' in QMin:
            QMin['initorbs'] = {}
        elif 'init' in QMin or 'always_orb_init' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin.resources['pwd'], 'GAUSSIAN.chk.init')
                if os.path.isfile(filename):
                    initorbs[job] = filename
            for job in QMin['joblist']:
                filename = os.path.join(QMin.resources['pwd'], f'GAUSSIAN.chk.{job}.init')
                if os.path.isfile(filename):
                    initorbs[job] = filename
            if 'always_orb_init' in QMin and len(initorbs) < QMin.control['njobs']:
                raise RuntimeError('Initial orbitals missing for some jobs!')
            QMin['initorbs'] = initorbs
        elif 'newstep' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin.save['savedir'], f'GAUSSIAN.chk.{job}.{step-1}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    raise RuntimeError('File %s missing in savedir!' % (filename))
            QMin['initorbs'] = initorbs
        elif 'samestep' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin.save['savedir'], f'GAUSSIAN.chk.{job}.{step}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    raise RuntimeError('File %s missing in savedir!' % (filename))
            QMin['initorbs'] = initorbs
        elif 'restart' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin.save['savedir'], f'GAUSSIAN.chk.{job}.{step}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    raise RuntimeError('File %s missing in savedir!' % (filename))
            QMin['initorbs'] = initorbs

    def generate_joblist(self):
        # sort the gradients into the different jobs
        gradjob = {}
        for ijob in self.QMin.control['joblist']:
            gradjob['master_%i' % ijob] = {}
        for grad in self.QMin.maps['gradmap']:
            ijob = self.QMin.maps['multmap'][grad[0]]
            isgs = False
            istates = self.QMin.control['states_to_do'][grad[0] - 1]
            if not self.QMin.control['jobs'][ijob]['restr']:
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
        self.QMin.control['jobgrad'] = jobgrad

        densjob = {}
        if self.QMin.request['multipolar_fit']:
            jobdens = {}
            # detect where the densities will be calculated
            # gs and first es always accessible from master if gs is not other mult
            for dens in self.QMin.maps['densmap']:
                ijob = self.QMin.maps['multmap'][dens[0]]
                gsmult = self.QMin.maps['multmap'][-ijob][0]
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
                        if dens[0] == 3 and self.QMin.control['jobs'][ijob]['restr']:
                            # gs already read by from singlet calc, gses for singlet to triplet = 0
                            densjob[j] = {'scf': False, 'es': True, 'gses': True}
                        else:
                            if self.QMin.molecule['states'][gsmult - 1] == 1:
                                densjob[j] = {'scf': True, 'es': False, 'gses': False}
                            else:
                                densjob[j] = {'scf': True, 'es': True, 'gses': True}
                    else:
                        densjob[j] = {'scf': False, 'es': True, 'gses': False}
                elif dens[1] == 1:
                    jobdens[dens] = f'master_{ijob}'
                    if dens[0] == 3 and self.QMin.control['jobs'][ijob]['restr']:
                        # gs already read by from singlet calc, gses for singlet to triplet = 0
                        densjob[f'master_{ijob}'] = {'scf': False, 'es': True, 'gses': True}
                    else:
                        densjob[f'master_{ijob}'] = {'scf': True, 'es': False, 'gses': False}
                else:
                    jobdens[dens] = f'dens_{dens[0]}_{dens[1]}'
                    densjob[f'dens_{dens[0]}_{dens[1]}'] = {'scf': False, 'es': True, 'gses': False}

            self.QMin.control['jobdens'] = jobdens
            self.QMin.control['densjob'] = densjob
        # add the master calculations
        schedule = []
        self.QMin.scheduling['nslots_pool'] = []
        ntasks = 0
        for i in gradjob:
            if 'master' in i:
                ntasks += 1
        nrounds, nslots, cpu_per_run = SHARC_ABINITIO.divide_slots(QMin.resources['ncpu'], ntasks, self.QMin.resources['schedule_scaling'])
        memory_per_core = self.QMin.resources['memory'] // self.QMin.resources['ncpu']
        self.QMin.scheduling['nslots_pool'].append(nslots)
        schedule.append({})
        icount = 0
        for i in sorted(gradjob):
            if 'master' in i:
                QMin1 = deepcopy(self.QMin)
                del QMin1.scheduling
                QMin1.control['master'] = True
                QMin1.control['jobid'] = int(i.split('_')[1])
                QMin1.maps['gradmap'] = list(gradjob[i])
                QMin1.resources['ncpu'] = cpu_per_run[icount]
                QMin1.resources['memory'] = memory_per_core * cpu_per_run[icount]
                # get the rootstate for the multiplicity as the first excited state
                QMin1.control['rootstate'] = min(1, QMin.molecule['states'][QMin.maps['multmap'][-QMin1['jobid']][-1] - 1] - 1)
                if 3 in QMin.maps['multmap'][-QMin1.control['jobid']] and QMin.control['jobs'][QMin1.control['jobid']]['restr']:
                    QMin1.control['rootstate'] = 1
                    QMin1.molecule['states'][0] = 1
                    QMin1.control['states_to_do'][0] = 1
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
            nrounds, nslots, cpu_per_run = SHARC_ABINITIO.divide_slots(QMin.resources['ncpu'], ntasks, QMin['schedule_scaling'])
            self.QMin.scheduling['nslots_pool'].append(nslots)
            schedule.append({})
            icount = 0
            for i in gradjob:
                if 'grad' in i:
                    QMin1 = deepcopy(QMin)
                    del QMin1.scheduling
                    mult, state = (int(x) for x in i.split('_')[1:])
                    ijob = QMin.maps['multmap'][mult]
                    QMin1.control['jobid'] = ijob
                    gsmult = QMin.maps['multmap'][-ijob][0]
                    for i in ['h', 'soc', 'dm', 'overlap', 'ion']:
                        QMin1.requests[i] = False
                    for i in ['always_guess', 'always_orb_init', 'init']:
                        QMin1.save[i] = False
                    QMin1.maps['gradmap'] = list(gradjob[i])
                    QMin1.resources['ncpu'] = cpu_per_run[icount]
                    QMin1.resources['memory'] = memory_per_core * cpu_per_run[icount]
                    QMin1.control['gradonly'] = []
                    QMin1.control['rootstate'] = state - 1 if gsmult == mult else state    # 1 is first excited state of mult
                    icount += 1
                    schedule[-1][i] = QMin1

            for i in densjob:
                if 'dens' in i:
                    QMin1 = deepcopy(QMin)
                    del QMin1.scheduling
                    mult, state = (int(x) for x in i.split('_')[1:])
                    ijob = QMin.maps['multmap'][mult]
                    QMin1.control['jobid'] = ijob
                    gsmult = QMin.maps['multmap'][-ijob][0]
                    for i in ['h', 'soc', 'dm', 'overlap', 'ion']:
                        QMin1.requests[i] = False
                    for i in ['always_guess', 'always_orb_init', 'init']:
                        QMin1.save[i] = False
                    QMin1.resources['ncpu'] = cpu_per_run[icount]
                    QMin1.resources['memory'] = memory_per_core * cpu_per_run[icount]
                    QMin1.control['rootstate'] = state - 1 if gsmult == mult else state    # 1 is first excited state of mult
                    QMin1.control['densonly'] = True
                    icount += 1
                    schedule[-1][i] = QMin1
        self.QMin.scheduling['schedule'] = schedule
        return

    def _backupdir(self):
        QMin = self._QMin
        # make name for backup directory
        if QMin.requests['backup']:
            backupdir = QMin.save['savedir'] + '/backup'
            backupdir1 = backupdir
            i = 0
            while os.path.isdir(backupdir1):
                i += 1
                if 'step' in QMin:
                    backupdir1 = backupdir + '/step%s_%i' % (QMin.save['step'][0], i)
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
        self.log.debug(f"{self.QMin.scheduling['schedule']}")

        # run all the jobs
        self.log.info('>>>>>>>>>>>>> Starting the GAUSSIAN job execution')
        self.runjobs(self.QMin.scheduling['schedule'])

        # do all necessary overlap and Dyson calculations
        self._run_wfoverlap()

        # do all necessary Theodore calculations
        self._run_theodore()

    def dry_run(self):
        self.generate_joblist()
        self.log.debug(f"{self.QMin.scheduling['schedule']}")
        # do all necessary overlap and Dyson calculations
        self._run_wfoverlap()
        # do all necessary Theodore calculations
        self._run_theodore()

    def create_restart_files(self):
        self.log.print('>>>>>>>>>>>>> Saving files')
        starttime = datetime.datetime.now()
        for ijobset, jobset in enumerate(self.QMin.scheduling['schedule']):
            if not jobset:
                continue
            for job in jobset:
                if 'master' in job:
                    WORKDIR = os.path.join(self.QMin.resources['scratchdir'], job)
                    if QMin.save['samestep']:
                        self.saveFiles(WORKDIR, jobset[job])
                    if QMin.requests['ion'] and ijobset == 0:
                        self.saveAOmatrix(WORKDIR, QMin)
        self.saveGeometry(QMin)
        endtime = datetime.datetime.now()
        self.log.print('Saving Runtime: %s\n' % (endtime - starttime))

    # ======================================================================= #

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        - sets up the workdir
        - runs GAUSSIAN
        - checks error code
        - strips the workdir of unwanted files
        """
        try:
            self.setupWORKDIR(workdir, qmin)
            strip = True
            starttime = datetime.datetime.now()
            exit_code = self.run_program(workdir, f"{qmin['GAUSS_EXE']} < GAUSSIAN.com", os.path.join(workdir, "GAUSSIAN.log"), os.path.join(workdir, "GAUSSIAN.err"))
            endtime = datetime.datetime.now()
        except Exception as problem:
            self.log.info('*' * 50 + '\nException in run_calc(%s)!' % (workdir))
            self.log.info(f"{traceback.format_exc()}")
            self.log.info('*' * 50 + '\n')
            raise problem
        if strip and exit_code == 0:
            keep = ['GAUSSIAN.com$', 'GAUSSIAN.err$', 'GAUSSIAN.log$', 'GAUSSIAN.chk', 'GAUSSIAN.fchk', 'GAUSSIAN.rwf']
            strip_dir(workdir, keep_files=keep)

        return exit_code, endtime - starttime

    # ======================================================================= #

    def setupWORKDIR(self, WORKDIR, QMin):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
        # then put the GAUSSIAN.com file
        SHARC_GAUSSIAN._initorbs(QMin)
        # setup the directory
        mkdir(WORKDIR)

        # write GAUSSIAN.com
        inputstring = SHARC_GAUSSIAN.writeGAUSSIANinput(QMin)
        filename = os.path.join(WORKDIR, 'GAUSSIAN.com')
        writefile(filename, inputstring)
        self.log.debug('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        self.log.debug(inputstring)
        self.log.debug('GAUSSIAN input written to: %s' % (filename))
        self.log.debug('====================================================================')

        # wf file copying
        if 'master' in QMin:
            job = QMin['jobid']
            if job in QMin['initorbs']:
                fromfile = QMin['initorbs'][job]
                tofile = os.path.join(WORKDIR, 'GAUSSIAN.chk')
                shutil.copy(fromfile, tofile)
        elif QMin.requests['grad'] or 'densonly' in QMin:
            job = QMin['jobid']
            fromfile = os.path.join(QMin.resources['scratchdir'], 'master_%i' % job, 'GAUSSIAN.chk')
            tofile = os.path.join(WORKDIR, 'GAUSSIAN.chk')
            shutil.copy(fromfile, tofile)

        return

    # ======================================================================= #

    @staticmethod
    def writeGAUSSIANinput(QMin):

        # general setup
        job = QMin.control['jobid']
        gsmult = QMin.maps['multmap'][-job][0]
        restr = QMin.control['jobs'][job]['restr']
        charge = QMin.maps['chargemap'][gsmult]

        # determine the root in case it was not determined in schedule jobs
        if 'rootstate' not in QMin:
            QMin['rootstate'] = min(1, QMin.molecule['states'][QMin.maps['multmap'][-QMin.control['jobid']][-1] - 1] - 1)
            if 3 in QMin.maps['multmap'][-QMin.control['jobid']] and QMin.control['jobs'][QMin.control['jobid']]['restr']:
                QMin['rootstate'] = 1

        # excited states to calculate
        states_to_do = QMin.control['states_to_do']
        for imult in range(len(states_to_do)):
            if not imult + 1 in QMin.maps['multmap'][-job]:
                states_to_do[imult] = 0
        states_to_do[gsmult - 1] -= 1

        # do minimum number of states for gradient jobs
        if 'gradonly' in QMin:
            gradmult = QMin.maps['gradmap'][0][0]
            gradstat = QMin.maps['gradmap'][0][1]
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
        if QMin.requests['multipolar_fit']:
            dodens = True
            root = QMin['rootstate']

        # construct the input string TODO
        string = ''

        # link 0
        string += '%%MEM=%iMB\n' % (QMin.resources['memory'])
        string += '%%NProcShared=%i\n' % (QMin.resources['ncpu'])
        string += '%%Chk=%s\n' % ('GAUSSIAN.chk')
        if 'AOoverlap' in QMin.control or QMin.requests['ion']:
            string += '%%Rwf=%s\n' % ('GAUSSIAN.rwf')
            if 'AOoverlap' in QMin.control:
                string += '%KJob l302\n'
        string += '\n'

        # Route section
        data = ['p', 'nosym', 'unit=AU', QMin.template['functional']]
        if not QMin.template['functional'].lower() == 'dftba':
            data.append(QMin.template['basis'])
        if dograd:
            data.append('force')
        if 'AOoverlap' in QMin.control:
            data.append('IOP(2/12=3)')
        if QMin.template['dispersion']:
            data.append('EmpiricalDispersion=%s' % QMin.template['dispersion'])
        if QMin.template['grid']:
            data.append('int(grid=%s)' % QMin.template['grid'])
        if QMin.template['denfit']:
            data.append('denfit')
        if ncalc > 0:
            if not QMin.template['no_tda']:
                s = 'tda'
            else:
                s = 'td'
            if QMin.control['master']:
                s += '(nstates=%i%s' % (ncalc, mults_td)
            else:
                s += '(read'
            if dograd and root > 0:
                s += f',root={root}'
            elif dodens and root > 0:
                s += f',root={root}'
            if QMin.template['noneqsolv']:
                s += ',noneqsolv'
            s += ') density=Current'
            data.append(s)
        if QMin.template['scrf']:
            s = ','.join(QMin.template['scrf'].split())
            data.append('scrf(%s)' % s)
        if QMin.template['scf']:
            s = ','.join(QMin.template['scf'].split())
            data.append('scf(%s)' % s)
        if QMin.template['iop']:
            s = ','.join(QMin.template['iop'].split())
            data.append('iop(%s)' % s)
        if QMin.template['keys']:
            data.extend([QMin.template['keys']])
        if QMin.control['densonly']:
            data.append('pop=Regular')    # otherwise CI density will not be printed
            data.append('Guess=read')
        if QMin.requests['theodore']:
            data.append('pop=full')
            data.append('IOP(9/40=3)')
        data.append('GFPRINT')
        string += '#'
        for i in data:
            string += i + '\n'
        # title
        string += '\nSHARC-GAUSSIAN job\n\n'

        # charge/mult and geometry
        if 'AOoverlap' in QMin.control:
            string += '%i %i\n' % (2. * charge, 1)
        else:
            string += '%i %i\n' % (charge, gsmult)
        for label, coords in zip(QMin.molecule['elements'], QMin.molecule['coords']):
            string += '%4s %16.9f %16.9f %16.9f\n' % (label, coords[0], coords[1], coords[2])
        string += '\n'
        if QMin.template['functional'].lower() == 'dftba':
            string += '@GAUSS_EXEDIR:dftba.prm\n'
        if QMin.template['basis_external']:
            for line in QMin.template['basis_external']:
                string += line
            string += '\n'
        if QMin.template['paste_input_file']:
            # string += '\n'
            for line in QMin.template['paste_input_file']:
                string += line
        string += '\n\n'

        return string

    # ======================================================================= #

    def saveGeometry(self, QMin):
        string = ''
        for label, atom in zip(QMin.molecule['elements'], QMin.molecule['coords']):
            string += '%4s %16.9f %16.9f %16.9f\n' % (label, atom[0], atom[1], atom[2])
        filename = os.path.join(QMin.save['savedir'], f'geom.dat.{QMin.save["step"]}')
        writefile(filename, string)
        self.log.print(shorten_DIR(filename))
        return

    # ======================================================================= #

    def saveFiles(self,WORKDIR, QMin):

        # copy the TAPE21 from master directories
        job = QMin.control['jobid']
        step = QMin.save['step']
        fromfile = os.path.join(WORKDIR, 'GAUSSIAN.chk')
        tofile = os.path.join(QMin.save['savedir'], f'GAUSSIAN.chk.{job}.{step}')
        shutil.copy(fromfile, tofile)
        self.log.print(shorten_DIR(tofile))

        # if necessary, extract the MOs and write them to savedir
        if QMin.requests['ion'] or not QMin['nooverlap']:
            f = os.path.join(WORKDIR, 'GAUSSIAN.chk')
            string = SHARC_GAUSSIAN.get_MO_from_chk(f, QMin)
            mofile = os.path.join(QMin.save['savedir'], f'mos.{job}.{step}')
            writefile(mofile, string)
            self.log.print(shorten_DIR(mofile))

        # if necessary, extract the TDDFT coefficients and write them to savedir
        if QMin.requests['ion'] or not QMin['nooverlap']:
            f = os.path.join(WORKDIR, 'GAUSSIAN.chk')
            strings = SHARC_GAUSSIAN.get_dets_from_chk(f, QMin)
            for f in strings:
                writefile(f, strings[f])
                self.log.print(shorten_DIR(f))

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
        for shell in try_shells:
            try:
                runerror = sp.call(string, shell=True, executable=shell)
            except OSError as e:
                raise RuntimeError('Gaussian rwfdump has serious problems:', OSError, 68)
        string = readfile(dumpname)
        os.chdir(prevdir)
        return string

    # ======================================================================= #
    @staticmethod
    def get_MO_from_chk(filename, QMin):

        job = QMin.control['jobid']
        restr = QMin.control['jobs'][job]['restr']

        # extract alpha orbitals
        data = SHARC_GAUSSIAN.get_rwfdump(QMin.resources['groot'], filename, '524R')
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
            data = SHARC_GAUSSIAN.get_rwfdump(QMin.resources['groot'], filename, '526R')
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
                raise RuntimeError('Problem in orbital reading!')
            NMO_B = NAO
            MO_B = [mocoef_B[NAO * i:NAO * (i + 1)] for i in range(NAO)]

        NMO = NMO_A - QMin.molecule['frozcore']
        if restr:
            NMO = NMO_A - QMin.molecule['frozcore']
        else:
            NMO = NMO_A + NMO_B - 2 * QMin.molecule['frozcore']

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
            if imo < QMin.molecule['frozcore']:
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
                if imo < QMin.molecule['frozcore']:
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
        job = QMin.control['jobid']
        restr = QMin.control['jobs'][job]['restr']
        mults = QMin.control['jobs'][job]['mults']
        if 3 in mults:
            mults = [3]
        gsmult = QMin.maps['multmap'][-job][0]
        nstates_to_extract = deepcopy(QMin.molecule['states'])
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
            charge = QMin.maps['chargemap'][gsmult]
            nelec = float(QMin.molecule['Atomcharge'] - charge)
            infos['NOA'] = int(nelec / 2. + float(gsmult - 1) / 2.)
            infos['NOB'] = int(nelec / 2. - float(gsmult - 1) / 2.)
            infos['NVA'] = infos['nbsuse'] - infos['NOA']
            infos['NVB'] = infos['nbsuse'] - infos['NOB']
            infos['NFC'] = 0
        else:
            # get all info from checkpoint
            data = SHARC_GAUSSIAN.get_rwfdump(QMin.resources['groot'], filename, '635R')
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
                    key = tuple(occ_A[QMin.molecule['frozcore']:])
                else:
                    key = tuple(occ_A[QMin.molecule['frozcore']:] + occ_B[QMin.molecule['frozcore']:])
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
                    if norm > factor * QMin.resources['wfthres']:
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
                        if any([key[i] != 3 for i in range(QMin.molecule['frozcore'])]):
                            problem = True
                    else:
                        if any([key[i] != 1 for i in range(QMin.molecule['frozcore'])]):
                            problem = True
                        if any(
                            [
                                key[i] != 2 for i in
                                range(nocc_A + nvir_A + QMin.molecule['frozcore'], nocc_A + nvir_A + 2 * QMin.molecule['frozcore'])
                            ]
                        ):
                            problem = True
                    if problem:
                        print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                        continue
                        # sys.exit(70)
                    if restr:
                        key2 = key[QMin.molecule['frozcore']:]
                    else:
                        key2 = key[QMin.molecule['frozcore']:QMin.molecule['frozcore'] + nocc_A + nvir_A] + key[nocc_A + nvir_A +
                                                                                              2 * QMin.molecule['frozcore']:]
                    dets3[key2] = dets2[key]
                # append
                eigenvectors[mult].append(dets3)

        strings = {}
        step = QMin.save['step']
        for imult, mult in enumerate(mults):
            filename = os.path.join(QMin.save['savedir'], f'dets.{mult}.{step}')
            strings[filename] = SHARC_GAUSSIAN.format_ci_vectors(eigenvectors[mult])

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

    def saveAOmatrix(self, WORKDIR, QMin):
        filename = os.path.join(WORKDIR, 'GAUSSIAN.rwf')
        NAO, Smat = SHARC_GAUSSIAN.get_smat(filename, QMin['groot'])

        string = '%i %i\n' % (NAO, NAO)
        for irow in range(NAO):
            for icol in range(NAO):
                string += '% .15e ' % (Smat[icol][irow])
            string += '\n'
        filename = os.path.join(QMin.save['savedir'], 'AO_overl')
        writefile(filename, string)
        self.log.print(shorten_DIR(filename))

    # ======================================================================= #

    @staticmethod
    def get_smat(filename, groot):

        # get all info from checkpoint
        data = SHARC_GAUSSIAN.get_rwfdump(groot, filename, '514R')

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

    def _create_aoovl(self):

        # get geometries
        filename1 = os.path.join(self.QMin.save['savedir'], f'geom.dat.{self.QMin.save["step"]-1}')
        oldgeo = SHARC_GAUSSIAN.get_geometry(filename1)
        filename2 = os.path.join(self.QMin.save['savedir'], f'geom.dat.{self.QMin.save["step"]}')
        newgeo = SHARC_GAUSSIAN.get_geometry(filename2)

        # build QMin   # TODO: always singlet for AOoverlaps
        QMin1 = deepcopy(self.QMin)
        del QMin.scheduling
        QMin1.molecule['elements'] = [x[0] for x in chain(oldgeo, newgeo)]
        QMin1.coords['coords'] = [x[1:] for x in chain(oldgeo, newgeo)]
        QMin1.control['AOoverlap'] = [filename1, filename2]
        QMin1.control['jobid'] = QMin['joblist'][0]
        QMin1.molecule['natom'] = len(newgeo)
        remove = ['nacdr', 'grad', 'h', 'soc', 'dm', 'overlap', 'ion']
        for r in remove:
            QMin1.requests[r] = False

        # run the calculation
        WORKDIR = os.path.join(self.QMin.resources['scratchdir'], 'AOoverlap')
        self.execute_from_qmin(WORKDIR, QMin1)

        # get output
        filename = os.path.join(WORKDIR, 'GAUSSIAN.rwf')
        NAO, Smat = SHARC_GAUSSIAN.get_smat(filename, self.QMin['groot'])

        # adjust the diagonal blocks for DFTB-A
        if self.QMin.template['functional'] == 'dftba':
            Smat = SHARC_GAUSSIAN.adjust_DFTB_Smat(Smat, NAO, QMin)

        # Smat is now full matrix NAO*NAO
        # we want the lower left quarter, but transposed
        string = '%i %i\n' % (NAO // 2, NAO // 2)
        for irow in range(NAO // 2, NAO):
            for icol in range(0, NAO // 2):
                string += '% .15e ' % (Smat[icol][irow])    # note the exchanged indices => transposition
            string += '\n'
        filename = os.path.join(self.QMin.save['savedir'], 'AO_overl.mixed')
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
                raise RuntimeError('Error: Overlaps with DFTB need further testing!')
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
        self.log.print('>>>>>>>>>>>>> Reading output files')
        starttime = datetime.datetime.now()

        states = self.QMin.molecule['states']
        nstates = self.QMin.molecule['nstates']
        nmstates = self.QMin.molecule['nmstates']
        natom = self.QMin.molecule['natom']
        joblist = self.QMin['joblist']
        self.QMout.allocate(states, natom, self.QMin.molecule['npc'], {r for r in self.QMin.requests.keys() if self.QMin.requests[r]})

        # TODO:
        # excited state energies and transition moments could be read from rwfdump "770R"
        # KS orbital energies: "522R"
        # geometry SEEMS TO BE in "507R"
        # 1TDM might be in "633R"
        # Hamiltonian
        for job in joblist:
            logfile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/GAUSSIAN.log' % (job))
            log_content = readfile(logfile)

            if self.QMin.requests['h']:    # or 'soc' in self.QMin:
                # go through all jobs
                for job in joblist:
                    # first get energies from TAPE21
                    logfile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/GAUSSIAN.log' % (job))
                    energies = self.getenergy(log_content, job)
                    mults = self.QMin.maps['multmap'][-job]
                    if 3 in mults:
                        mults = [3]
                    for i in range(nmstates):
                        m1, s1, ms1 = tuple(self.QMin.maps['statemap'][i + 1])
                        if m1 not in mults:
                            continue
                        self.QMout['h'][i][i] = energies[(m1, s1)]

            # Dipole Moments
            if self.QMin.requests['dm']:
            # go through all jobs
                dipoles = self.gettdm(log_content, job)
                mults = self.QMin.maps['multmap'][-job]
                mults = self.QMin.maps['multmap'][-job]
                if 3 in mults:
                    mults = [3]
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps['statemap'][i + 1])
                    if m1 not in self.QMin.control['jobs'][job]['mults']:
                        continue
                    for j in range(nmstates):
                        m2, s2, ms2 = tuple(self.QMin.maps['statemap'][j + 1])
                        if m2 not in self.QMin.control['jobs'][job]['mults']:
                            continue
                        if i == j and (m1, s1) in self.QMin.maps['gradmap']:
                            path, isgs = self.QMin['jobgrad'][(m1, s1)]
                            logfile = os.path.join(self.QMin.resources['scratchdir'], path, 'GAUSSIAN.log')
                            dm = SHARC_GAUSSIAN.getdm(logfile)
                            for ixyz in range(3):
                                self.QMout['dm'][ixyz][i][j] = dm[ixyz]
                        if i == j:
                            continue
                        if not m1 == m2 == mults[0] or not ms1 == ms2:
                            continue
                        if s1 == 1:
                            for ixyz in range(3):
                                self.QMout['dm'][ixyz][i][j] = dipoles[(m2, s2)][ixyz]
                        elif s2 == 1:
                            for ixyz in range(3):
                                self.QMout['dm'][ixyz][i][j] = dipoles[(m1, s1)][ixyz]

        # Gradients
        if self.QMin.requests['grad']:
            if 'grad' not in QMout:
                self.QMout['grad'] = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
            for grad in self.QMin.maps['gradmap']:
                path, isgs = self.QMin['jobgrad'][grad]
                logfile = os.path.join(self.QMin.resources['scratchdir'], path, 'GAUSSIAN.log')
                g = SHARC_GAUSSIAN.getgrad(logfile, self.QMin)
                for istate in self.QMin.maps['statemap']:
                    state = self.QMin.maps['statemap'][istate]
                    if (state[0], state[1]) == grad:
                        self.QMout['grad'][istate - 1] = g
            if self.QMin['neglected_gradient'] != 'zero':
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps['statemap'][i + 1])
                    if not (m1, s1) in self.QMin.maps['gradmap']:
                        if self.QMin['neglected_gradient'] == 'gs':
                            j = self.QMin.maps['gsmap'][i + 1] - 1
                        elif self.QMin['neglected_gradient'] == 'closest':
                            e1 = self.QMout['h'][i][i]
                            de = 999.
                            for grad in self.QMin.maps['gradmap']:
                                for k in range(nmstates):
                                    m2, s2, ms2 = tuple(self.QMin.maps['statemap'][k + 1])
                                    if grad == (m2, s2):
                                        break
                                e2 = self.QMout['h'][k][k]
                                if de > abs(e1 - e2):
                                    de = abs(e1 - e2)
                                    j = k
                        self.QMout['grad'][i] = QMout['grad'][j]

        # Regular Overlaps
        if self.QMin.requests['overlap']:
            if 'overlap' not in QMout:
                self.QMout['overlap'] = makecmatrix(nmstates, nmstates)
            for mult in itmult(self.QMin.molecule['states']):
                job = self.QMin.maps['multmap'][mult]
                outfile = os.path.join(self.QMin.resources['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
                out = readfile(outfile)
                self.log.print('Overlaps: ' + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(self.QMin.maps['statemap'][i + 1])
                        m2, s2, ms2 = tuple(self.QMin.maps['statemap'][j + 1])
                        if not m1 == m2 == mult:
                            continue
                        if not ms1 == ms2:
                            continue
                        self.QMout['overlap'][i][j] = SHARC_GAUSSIAN.getsmate(out, s1, s2)

        # Phases from overlaps
        if self.QMin.requests['phases']:
            if 'phases' not in QMout:
                self.QMout['phases'] = [complex(1., 0.) for i in range(nmstates)]
            if 'overlap' in QMout:
                for i in range(nmstates):
                    if self.QMout['overlap'][i][i].real < 0.:
                        self.QMout['phases'][i] = complex(-1., 0.)

        # Dyson norms
        if self.QMin.requests['ion']:
            if 'prop' not in QMout:
                self.QMout['prop'] = makecmatrix(nmstates, nmstates)
            for ion in self.QMin.maps['ionmap']:
                outfile = os.path.join(self.QMin.resources['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
                out = readfile(outfile)
                self.log.print('Dyson:    ' + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(self.QMin.maps['statemap'][i + 1])
                        m2, s2, ms2 = tuple(self.QMin.maps['statemap'][j + 1])
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
                        self.QMout['prop'][i][j] = SHARC_GAUSSIAN.getDyson(out, s1, s2) * factor

        # multipolar_density fits
        if self.QMin.requests['multipolar_fit']:
            if 'multipolar_fit' not in QMout:
                self.QMout['multipolar_fit'] = {}

            for dens in self.QMin['densjob']:
                workdir = os.path.join(self.QMin.resources['scratchdir'], dens)
                self.get_fchk(workdir, self.QMin['groot'])
            # sort densjobs
            density_map = {}    # map for (mult, state, state): position in densities
            sorted_densjobs = []
            jobfiles = set()
            i = 0
            for dens in self.QMin.maps['densmap']:
                job = self.QMin['jobdens'][dens]
                if job not in jobfiles:
                    jobfiles.add(job)
                    flags = self.QMin['densjob'][job]
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
                        nstates = self.QMin.molecule['states'][dens[0] - 1]
                        ijob = self.QMin.maps['multmap'][dens[0]]
                        gsmult = self.QMin.maps['multmap'][-ijob][0]
                        first = 2
                        if gsmult != dens[0]:
                            first = 1
                        last = nstates
                        for es in range(first, last + 1):
                            state = dens[1]
                            density_map[(gsmult, 1, dens[0], es)] = i
                            i += 1
            # read basis
            fchkfile = os.path.join(self.QMin.resources['scratchdir'], sorted_densjobs[0][0], 'GAUSSIAN.fchk')
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
            #  cart2sph_matrix = get_cart2sph_matrix(3 if cartesian_f else 2, nao, self.QMin.molecule['elements'], basis)
            #  for i in range(len(densities)):
            #  print(np.linalg.norm(densities[i]))
            #  densities[i] = cart2sph_matrix.T @ densities[i] @ cart2sph_matrix
            #  print(np.linalg.norm(densities[i]))

            #  else:
            #  cartesian_basis = cartesian_d
            cartesian_basis = cartesian_d
            fits = Resp(
                self.QMin.molecule['coords'],
                self.QMin.molecule['elements'],
                self.QMin.resources['resp_vdw_radii'],
                self.QMin.resources['resp_density'],
                self.QMin.resources['resp_shells'],
                grid=self.QMin.resources['resp_grid']
            )
            gsmult = self.QMin.maps['statemap'][1][0]
            charge = self.QMin.maps['chargemap'][gsmult]
            pprint.pprint(ECPs)
            fits.prepare(
                basis, gsmult - 1, charge, ecps=ECPs, cart_basis=cartesian_basis
            )    # the charge of the atom does not affect integrals
            # obtain normalization coefficients of pyscf overlap
            ao_sqrt_norms = np.sqrt(np.diag(fits.Sao))
            # obtain new order of the AO orbitals
            new_order = get_pyscf_order_from_gaussian(self.QMin.molecule['elements'], basis, cartesian_d=cartesian_d, cartesian_f=cartesian_f)
            print("reordering atomic orbitals according to")
            print(new_order)
            if len(new_order) != len(densities[0]):
                raise RuntimeError("The list with the new order of the AOs has a different length!")
            # reorder all densities and renormalize them
            for i in range(len(densities)):
                densities[i] = densities[i][:, new_order][new_order, :]
                densities[i] = (densities[i] / ao_sqrt_norms[:, None]) / ao_sqrt_norms[None, :]

            fits_map = {}
            D = fits.mol.intor('int1e_r')
            for i, d_i in enumerate(self.QMin.maps['densmap']):
                # do gs density
                key = (*d_i, *d_i)
                fits_map[key] = fits.multipoles_from_dens(
                    densities[density_map[key]],
                    include_core_charges=True,
                    order=self.QMin.resources['resp_fit_order'],
                    charge=self.QMin.maps['chargemap'][d_i[0]],
                    betas=self.QMin.resources['resp_betas']
                )

                for d_j in self.QMin.maps['densmap'][i + 1:]:
                    if d_i[0] != d_j[0]:
                        continue
                    # do gses and eses density
                    key = (*d_i, *d_j)
                    if key in density_map:
                        fits_map[key] = fits.multipoles_from_dens(
                            densities[density_map[key]], include_core_charges=False, order=self.QMin.resources['resp_fit_order'],
                            betas=self.QMin.resources['resp_betas']
                        )
                        print(f"check: {key[0]}_{key[1]}->{key[2]}_{key[3]} transition dipole from real density",(-np.einsum('xij,ij->x', D, densities[density_map[key]])).tolist())
                        #  np.savetxt("_".join(map(str, key))+"_trans_dens", densities[density_map[key]])
                    else:
                        ijob = self.QMin.maps['multmap'][d_i[0]]    # the multiplicity is the same -> ijob same
                        gsmult = self.QMin.maps['multmap'][-ijob][0]
                        dmI = densities[density_map[(gsmult, 1, *d_i)]]
                        dmJ = densities[density_map[(gsmult, 1, *d_j)]]
                        trans_dens = es2es_tdm(dmI, dmJ, fits.Sao)
                        fits_map[key] = fits.multipoles_from_dens(
                            trans_dens, include_core_charges=False, order=self.QMin.resources['resp_fit_order'],
                            betas=self.QMin.resources['resp_betas']
                        )
                        #  np.savetxt("_".join(map(str, key))+"_trans_dens", trans_dens)
                        print(f"check: {key[0]}_{key[1]}->{key[2]}_{key[3]} transition dipole from appr. density",(-np.einsum('xij,ij->x', D, trans_dens)).tolist())
            self.QMout['multipolar_fit'] = fits_map

        # TheoDORE
        if self.QMin.requests['theodore']:
            if 'theodore' not in QMout:
                self.QMout['theodore'] = makecmatrix(self.QMin['theodore_n'], nmstates)
            for job in joblist:
                if not self.QMin.control['jobs'][job]['restr']:
                    continue
                else:
                    mults = self.QMin.control['jobs'][job]['mults']
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += self.QMin.molecule['states'][i - 1] - (i == gsmult)
                    if ns == 0:
                        continue
                sumfile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/tden_summ.txt' % job)
                omffile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/OmFrag.txt' % job)
                props = SHARC_GAUSSIAN.get_theodore(sumfile, omffile, self.QMin)
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps['statemap'][i + 1])
                    if (m1, s1) in props:
                        for j in range(self.QMin['theodore_n']):
                            self.QMout['theodore'][i][j] = props[(m1, s1)][j]

        endtime = datetime.datetime.now()
        self.log.print("Readout Runtime: %s" % (endtime - starttime))

        if DEBUG:
            copydir = os.path.join(self.QMin.save['savedir'], 'debug_GAUSSIAN_stdout')
            if not os.path.isdir(copydir):
                mkdir(copydir)
            for job in joblist:
                outfile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/GAUSSIAN.log' % (job))
                shutil.copy(outfile, os.path.join(copydir, "GAUSSIAN_%i.log" % job))
                if self.QMin.control['jobs'][job]['restr'] and 'theodore' in self.QMin:
                    outfile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/tden_summ.txt' % job)
                    try:
                        shutil.copy(outfile, os.path.join(copydir, 'THEO_%i.out' % (job)))
                    except IOError:
                        pass
                    outfile = os.path.join(self.QMin.resources['scratchdir'], 'master_%i/OmFrag.txt' % job)
                    try:
                        shutil.copy(outfile, os.path.join(copydir, 'THEO_OMF_%i.out' % (job)))
                    except IOError:
                        pass
            if self.QMin.requests['grad']:
                for grad in self.QMin.maps['gradmap']:
                    path, isgs = self.QMin['jobgrad'][grad]
                    outfile = os.path.join(self.QMin.resources['scratchdir'], path, 'GAUSSIAN.log')
                    shutil.copy(outfile, os.path.join(copydir, "GAUSSIAN_GRAD_%i_%i.log" % grad))
            if self.QMin.requests['overlap']:
                for mult in itmult(self.QMin.molecule['states']):
                    job = self.QMin.maps['multmap'][mult]
                    outfile = os.path.join(self.QMin.resources['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
                    shutil.copy(outfile, os.path.join(copydir, 'WFOVL_%i_%i.out' % (mult, job)))
            if self.QMin.requests['ion']:
                for ion in self.QMin.maps['ionmap']:
                    outfile = os.path.join(self.QMin.resources['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
                    shutil.copy(outfile, os.path.join(copydir, 'Dyson_%i_%i_%i_%i.out' % ion))

        return

    # ======================================================================= #

    def getenergy(self, log_content, ijob):

        # open file
        self.log.print('Energy:   ' + shorten_DIR(logfile))

        # read ground state
        for line in log_content:
            if ' SCF Done:' in line:
                gsenergy = float(line.split()[4])

        # figure out the excited state settings
        mults = QMin.control['jobs'][ijob]['mults']
        restr = QMin.control['jobs'][ijob]['restr']
        gsmult = mults[0]
        estates_to_extract = deepcopy(QMin.molecule['states'])
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
                for line in log_content:
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
    def gettdm(log_content, ijob):

        # open file
        self.log.print('Dipoles:  ' + shorten_DIR(logfile))

        # figure out the excited state settings
        mults = QMin.control['jobs'][ijob]['mults']
        if 3 in mults:
            mults = [3]
        restr = QMin.control['jobs'][ijob]['restr']
        gsmult = mults[0]
        estates_to_extract = deepcopy(QMin.molecule['states'])
        estates_to_extract[gsmult - 1] -= 1
        for imult in range(len(estates_to_extract)):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0

        # get ordering of states in Gaussian output
        istate = [int(i + 1 == gsmult) for i in range(len(QMin.molecule['states']))]
        index = 0
        gaustatemap = {}
        for iline, line in enumerate(log_content):
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
                for iline, line in enumerate(log_content):
                    if 'Ground to excited state transition electric dipole moments ' in line:
                        for istate in range(nstates):
                            shift = gaustatemap[(imult, istate + 1 + (gsmult == imult))]
                            s = f[iline + 2 + shift].split()
                            dipoles[(imult, istate + 1 + (gsmult == imult))] = [float(i) for i in s[1:4]]
        return dipoles

    # ======================================================================= #
    @staticmethod
    def get_fchk(workdir, groot=''):
        prevdir = os.getcwd()
        os.chdir(workdir)
        string = os.path.join(groot, 'formchk') + ' GAUSSIAN.chk'
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
    def parse_fchk(fchkfile: str, properties:dict):
        """
        Parse some keywords from an fchkfile raw

            fchkfile: str  name of the FCHK file
            properties: list[str]  list of keywords you want to parse
        Returns:
        ---
            dict[str,]  dictionary with properties as keys and values
        """
        types = {'I': int, 'R': float, 'C': str}
        res = {k: None for k in properties}

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
                for k in filter(lambda k: res[k] is None, res.keys()):
                    if k in line:
                        llst = line[len(k):].split()
                        cast = types[llst[0]]
                        n = parse_num(llst, cast)
                        if llst[1] == 'N=':
                            res[k] = parse_array(int(llst[-1]), cast, f)
                        else:
                            res[k] = n
                # ----------------------------
                line = f.readline()

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
        atom_symbols = QMin.molecule['elements']
        for dens, flags in densjobs:
            scf, es, gses = map(flags.get, ('scf', 'es', 'gses'))
            fchkfile = os.path.join(QMin.resources['scratchdir'], dens, 'GAUSSIAN.fchk')
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
                    for i_d in range(0, 2 * n_g2e, 2):
                        #  tmp = (d[i_d, ...] + d[i_d + 1, ...]) / 2
                        tmp = d[i_d, ...] * math.sqrt(2)
                        densities.append(tmp)
                        new_dens += 1
                    i += n_lines
                    gses_read = True
                i += 1
            if scf and not scf_read:
                raise RuntimeError(f'Missing "SCF Density" in checkpoint file for job {dens}!\nfchk-file:  {fchkfile}')
            if es and not es_read:
                raise RuntimeError(f'Missing "CI Density" in checkpoint file for job {dens}!\nfchk-file:  {fchkfile}')
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
        self.log.print('Dipoles:  ' + shorten_DIR(logfile))

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
        self.log.print('Gradient: ' + shorten_DIR(logfile))

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

if __name__ == "__main__":
    SHARC_GAUSSIAN(loglevel=10).main()
