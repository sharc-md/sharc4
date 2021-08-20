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
from datetime import date, datetime
import math
import sys
import os
import re
import shutil
import ast
from time import time
import numpy as np
import subprocess as sp
from abc import ABC, abstractmethod, abstractproperty
from functools import reduce, singledispatchmethod
from socket import gethostname
import pprint
from textwrap import wrap

# internal
from error import Error
from printing import printcomplexmatrix, printgrad, printtheodore
from utils import *
from constants import *
from parse_keywords import KeywordParser

# NOTE: Error handling especially import for processes in pools (error_callback)
# NOTE: gradient calculation necessitates multiple parallel calls (either inside interface) or
# one interface = one calculation (i.e. interface spawns multiple instances of itself)
# NOTE: logic checks in read_template and read_resources and in run() if required (LVC won't need check in run())


class INTERFACE(ABC):
    _QMin = {}
    _QMout = {}

    # internal status indicators
    _setup_mol = False
    _read_resources = False
    _read_template = False
    _DEBUG = False
    _PRINT = True

    # TODO: set Debug and Print flag
    # TODO: set persistant flag for file-io vs in-core
    def __init__(self, debug=False, print=True, persistent=False):
        self.clock = clock(verbose=print)
        # self.printheader()
        self._DEBUG = debug
        self._PRINT = print
        self._persistent = persistent
        self._QMin['pwd'] = os.getcwd()

    # ================== abstract methods and properties ===================

    @abstractproperty
    def authors(self) -> str:
        return 'Severin Polonius, Sebastian Mai'

    @abstractproperty
    def version(self) -> str:
        return '3.0'

    @abstractproperty
    def versiondate(self) -> date:
        return date(2021, 7, 15)

    @abstractproperty
    def changelogstring(self) -> str:
        return 'This is the changelog string'

    @property
    def QMin(self) -> dict:
        return self._QMin

    @QMin.setter
    def QMin(self, value: dict) -> None:
        self._QMin = value

    @property
    def QMout(self) -> dict:
        return self._QMout

    @QMout.setter
    def QMout(self, value: dict) -> None:
        self._QMout = value

    def main(self):

        name = self.__class__.__name__
        args = sys.argv
        if len(args) != 2:
            print(
                'Usage:',
                f'./SHARC_{name} <QMin>',
                f'version: {self.version}',
                f'date: {self.versiondate}',
                f'changelog: {self.changelogstring}',
                sep='\n'
            )
            sys.exit(106)
        QMinfilename = sys.argv[1]
        pwd = os.getcwd()
        self.printheader()
        self.setup_mol(os.path.join(pwd, QMinfilename))
        self.read_resources(os.path.join(pwd, f"{name}.resources"))
        self.read_template(os.path.join(pwd, f"{name}.template"))
        self.set_coords(os.path.join(pwd, QMinfilename))
        self.read_requests(os.path.join(pwd, QMinfilename))
        self.setup_run()
        self.run()
        if PRINT or DEBUG:
            self.printQMout()
        self.writeQMout()

    @abstractmethod
    def read_template(self, template_filename):
        pass

    @abstractmethod
    def read_resources(self, resources_filename):
        pass

    @abstractmethod
    def run(self):
        pass

        # ============================ Implemented public methods ========================

    def setup_mol(self, QMinfilename: str):
        QMin = self._QMin
        self._QMinfilename = QMinfilename
        QMinlines = readfile(QMinfilename)
        QMin['comment'] = QMinlines[1]
        QMin['elements'] = INTERFACE.read_elements(QMinlines)
        QMin['Atomcharge'] = sum(map(lambda x: ATOMCHARGE[x], QMin['elements']))
        QMin['frozcore'] = sum(map(lambda x: FROZENS[x], QMin['elements']))
        QMin['natom'] = len(QMin['elements'])

        # replaces all comments with white space. filters all empty lines
        filtered = filter(
            lambda x: not re.match(r'^\s*$', x), map(lambda x: re.sub(r'#.*$', '', x), QMinlines[QMin['natom'] + 2:])
        )

        # naively parse all key argument pairs from QM.in
        for line in filtered:
            llist = line.split(None, 1)
            key = llist[0].lower()
            if key == 'states':
                QMin.update(self.parseStates(llist[1]))
            elif key == 'unit':
                self.set_unit(llist[1].strip().lower())
            elif key == 'savedir':
                QMin[key] = llist[1].strip()
        if 'savedir' not in QMin:
            QMin['savedir'] = './SAVEDIR/'
        QMin['savedir'] = os.path.abspath(os.path.expanduser(os.path.expandvars(QMin['savedir'])))
        self._setup_mol = True
        # NOTE: Quantity requests (tasks) are dealt with later and potentially re-assigned
        return

    def set_unit(self, unit: str):
        if unit in ['bohr', 'angstrom']:
            self._QMin['unit'] = unit
            self._factor = 1. if unit == 'bohr' else 1. / BOHR_TO_ANG
        else:
            raise Error('unknown unit specified', 23)

    @staticmethod
    def parseStates(states: str) -> dict:
        res = {}
        try:
            res['states'] = list(map(int, states.split()))
        except (ValueError, IndexError):
            # get traceback of currently handled exception
            tb = sys.exc_info()[2]
            raise Error('Keyword "states" has to be followed by integers!', 37).with_traceback(tb)
        reduc = 0
        for i in reversed(res['states']):
            if i == 0:
                reduc += 1
            else:
                break
        for i in range(reduc):
            del res['states'][-1]
        nstates = 0
        nmstates = 0
        for i in range(len(res['states'])):
            nstates += res['states'][i]
            nmstates += res['states'][i] * (i + 1)
        res['nstates'] = nstates
        res['nmstates'] = nmstates
        return res

    # enables function overloads for different types (call detects type and calls corresponding version of function)

    @singledispatchmethod
    def set_coords(self, xyz):
        raise NotImplementedError("'set_coords' is only implemented for str, list[list[float]] or numpy.ndarray type")

    @set_coords.register
    def _(self, xyz: str):
        lines = readfile(xyz)
        try:
            natom = int(lines[0])
        except ValueError:
            raise Error('first line must contain the number of atoms!', 2)
        self._QMin["coords"
                   ] = np.asarray([INTERFACE._parse_xyz(x)[1] for x in lines[2:natom + 2]], dtype=float) * self._factor

    @set_coords.register
    def _(self, xyz: list):
        self._QMin["coords"] = np.asarray(xyz) * self._factor

    @set_coords.register
    def _(self, xyz: np.ndarray):
        if xyz.shape != (self._QMin['natoms'], 3):
            raise Error(f"Shape of coords does not match current system: {xyz.shape} {(self._QMin['natoms'], 3)}")
        self._QMin["coords"] = xyz * self._factor

    @singledispatchmethod
    def set_requests(self, requests):
        raise NotImplementedError("'set_requests' is only implemented for str or dict type!")

    @set_requests.register
    def _(self, requests: str):
        raise NotImplementedError()

    @set_requests.register
    def _(self, requests: dict):
        # delete all old requests
        self._reset_requests()
        # logic for raw tasks object from pysharc interface
        if 'tasks' in requests and type(requests['tasks']) is str:
            requests.update({k.lower(): True for k in requests['tasks'].split()})
            del requests['tasks']
        for task in ['nacdr', 'overlap', 'grad', 'ion']:
            if task in requests and type(requests[task]) is str:
                if task == requests[task].lower() or requests[task] == 'all':
                    requests[task] = True
                else:
                    requests[task] = [int(i) for i in requests[task].split()]

        self._QMin.update(requests)
        self._request_logic()

    def read_requests(self, requests_filename: str = "QM.in"):
        # delete all old requests
        self._reset_requests()
        if not self._read_template:
            raise Error('Interface is not set up correctly. Call read_template with the .template file first!', 23)
        QMin = self._QMin

        lines = readfile(requests_filename)
        filtered = filter(
            lambda x: not re.match(r'^\s*$', x),
            map(lambda x: re.sub(r'#.*$', '', x).strip(), lines[QMin['natom'] + 2:])
        )
        file_str = '\n'.join(filtered)

        def format_match(x: re.Match) -> str:
            return re.sub(r'\n+', "','", "['{}']".format(x.group(3)))

        lines = re.sub(r'(s(elect|tart).*\n)([^end]+)(\nend)', format_match, file_str).split('\n')

        def parse(line: str):
            llist = line.split(None, 1)
            if len(llist) == 1:
                return llist[0].lower(), True
            args = llist[1]
            if args[0] == '[':
                args = ast.literal_eval(args)
                if type(args[0]) == str:
                    args = list(map(lambda x: [int(i) for i in x.split()], args))
                return llist[0].lower(), args
            args = args.split()
            if len(args) == 1:
                args = args[0]
            return llist[0].lower(), args

        # NOTE: old QMin read stuff is not overwritten. Problem with states?
        self._QMin = {**dict(map(parse, lines)), **QMin}
        self._request_logic()

    def _reset_requests(self):
        for k in [
            'init', 'samestep', 'newstep', 'restart', 'cleanup', 'backup', 'h', 'soc', 'dm', 'grad', 'overlap', 'dmdr',
            'socdr', 'ion', 'theodore', 'phases'
        ]:
            if k in self._QMin:
                del self._QMin[k]

    def _request_logic(self):
        QMin = self._QMin
        possibletasks = {'h', 'soc', 'dm', 'grad', 'overlap', 'dmdr', 'socdr', 'ion', 'theodore', 'phases'}
        tasks = possibletasks & QMin.keys()
        if len(tasks) == 0:
            raise Error(f'No tasks found! Tasks are {possibletasks}.', 39)

        if 'h' not in tasks and 'soc' not in tasks:
            QMin['h'] = True

        if 'soc' in tasks and (len(QMin['states']) < 3 or QMin['states'][2] <= 0):
            del QMin['soc']
            QMin['h'] = True
            print('HINT: No triplet states requested, turning off SOC request.')

        if 'samestep' in QMin and 'init' in QMin:
            raise Error('"Init" and "Samestep" cannot be both present in QM.in!', 41)

        if 'restart' in QMin and 'init' in QMin:
            raise Error('"Init" and "Samestep" cannot be both present in QM.in!', 42)

        if 'phases' in tasks:
            QMin['overlap'] = True

        if 'overlap' in tasks and 'init' in tasks:
            raise Error(
                '"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"',
                43
            )
        if QMin.keys().isdisjoint({'init', 'samestep', 'restart'}):
            QMin['newstep'] = True

        if not tasks.isdisjoint({'h', 'soc', 'dm', 'grad'}) and 'overlap' in tasks:
            QMin['h'] = True

        if 'dmdr' in tasks:
            raise Error('Dipole derivatives ("dmdr") not currently supported', 45)

        if 'socdr' in tasks:
            raise Error('Spin-orbit coupling derivatives ("socdr") are not implemented', 46)

        # Check for correct gradient list
        if 'grad' in tasks:
            grad = QMin['grad']
            if grad is True or grad == 'all':
                grad = [i + 1 for i in range(QMin['nmstates'])]
                # pass
            else:
                try:
                    grad = [int(i) for i in grad]
                except ValueError:
                    raise Error('Arguments to keyword "grad" must be "all" or a list of integers!', 47)
                if len(grad) > QMin['nmstates']:
                    raise Error(
                        'State for requested gradient does not correspond to any state in QM input file state list!', 48
                    )
            QMin['grad'] = grad

        # wfoverlap settings
        if ('overlap' in QMin or 'ion' in QMin) and self.__class__.__name__ != 'LVC':
            # WFoverlap

            if not os.path.isfile(QMin['resources']['wfoverlap']):
                print('Give path to wfoverlap.x in ORCA.resources!')
                sys.exit(54)

        if 'theodore' in QMin:
            if QMin['resources']['theodir'] is not None and not os.path.isdir(QMin['resources']['theodir']):
                print('Give path to the TheoDORE installation directory in ORCA.resources!')
                sys.exit(56)
            os.environ['THEODIR'] = QMin['resources']['theodir']
            if 'PYTHONPATH' in os.environ:
                os.environ['PYTHONPATH'] = os.path.join(
                    QMin['resources']['theodir'], 'lib'
                ) + os.pathsep + QMin['resources']['theodir'] + os.pathsep + os.environ['PYTHONPATH']
                # print os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = os.path.join(QMin['theodir'],
                                                        'lib') + os.pathsep + QMin['resources']['theodir']

    # NOTE: generalize the parsing of keyword based input files, with lines as input

    def setup_run(self):
        QMin = self._QMin
        # obtain the statemap
        QMin['statemap'] = {i + 1: [*v] for i, v in enumerate(itnmstates(QMin['states']))}

        # obtain the states to actually compute
        states_to_do = [v + QMin['template']['paddingstates'][i] if v > 0 else v for i, v in enumerate(QMin['states'])]
        if not QMin['template']['unrestricted_triplets']:
            if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                states_to_do[0] = max(QMin['states'][0], 1)
                req = max(QMin['states'][0] - 1, QMin['states'][2])
                states_to_do[0] = req + 1
                states_to_do[2] = req
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
                    # jobs[len(jobs)+1]={'mults':[imult+4],'restr':False}
                    jobs[imult + 4] = {'mults': [imult + 4], 'restr': False}
        QMin['jobs'] = jobs

        # make the multmap (mapping between multiplicity and job)
        # multmap[imult]=ijob
        # multmap[-ijob]=[imults]
        multmap = {}
        for ijob, job in jobs.items():
            for imult in job['mults']:
                multmap[imult] = ijob
            multmap[-(ijob)] = job['mults']
        multmap[1] = 1
        QMin['multmap'] = multmap

        # get the joblist
        QMin['joblist'] = sorted(jobs.keys())
        QMin['njobs'] = len(QMin['joblist'])

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
            gradmap = {tuple(QMin['statemap'][i][0:2]) for i in QMin['grad']}
        QMin['gradmap'] = sorted(gradmap)

        # make the chargemap
        QMin['chargemap'] = {i + 1: c for i, c in enumerate(QMin['template']['charge'])}

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
            QMin['resources']['theodore_n'] = len(QMin['resources']['theodore_prop']
                                                  ) + len(QMin['resources']['theodore_fragment'])**2
        else:
            QMin['resources']['theodore_n'] = 0

        # TODO: QMMM
        QMin['qmmm'] = False

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

    def parse_keywords(
        self,
        lines: list[str],
        bools: dict[str, bool] = {},
        strings: dict[str, str] = {},
        paths: dict[str, str] = {},
        integers: dict[str, int] = {},
        floats: dict[str, float] = {},
        special: dict[str] = {}
    ) -> dict:
        '''
        Returns the parsed arguments of a set of lines as a dict with the help of the functions declared in
        parse_template.py.

                Parameters:
                        bools (dict[bool]): Dictionary with all keywords and their defaults with type bool
                        strings (dict[str]): Dictionary with all keywords and their defaults with type string
                        paths (dict[str]): Dictionary with all keywords and their defaults that are paths
                        integers (dict[int]): Dictionary with all keywords and their defaults with type int
                        floats (dict[float]): Dictionary with all keywords and their defaults with type float
                        special (dict): Dictionary with all keywords and their defaults with complex type
                        lines (list[str]): lines to parse

                Returns:
                        keyword_dict (dict): dict with parsed keywords
        '''
        # replaces all comments with white space. filters all empty lines
        filtered = filter(lambda x: not re.match(r'^\s*$', x), map(lambda x: re.sub(r'#.*$', '', x).strip(), lines))

        # concat all lines for select keyword:
        # 1 join lines to full file string,
        # 2 match all select/start ... end blocks,
        # 3 replace all \n with ',' in the matches,
        # 4 return matches between [' and ']
        file_str = '\n'.join(filtered)
        if not file_str or file_str.isspace():  # check is there is only whitespace left!
            return {}

        QMin = self._QMin
        template_parser = KeywordParser(len(QMin['states']), QMin['Atomcharge'])
        # prepare dict with parsers for every value type
        bool_parser = {k: lambda x: True for k in bools}
        string_parser = {k: lambda x: x for k in strings}
        path_parser = {k: lambda x: template_parser.path(x) for k in paths}
        integer_parser = {k: lambda x: int(float(x)) for k in integers}
        float_parser = {k: lambda x: float(x) for k in floats}
        special_parser = {k: getattr(template_parser, k) for k in special}

        def format_match(x: re.Match) -> str:
            return re.sub(r'\n+', "','", "['{}']".format(x.group(3)))

        lines = re.sub(r'(s(elect|tart).*\n)([^end]+)(\nend)', format_match, file_str).split('\n')

        def parse(d: dict, line: str) -> dict:
            return self._parse_to_dict(
                d, line, {
                    **bool_parser,
                    **string_parser,
                    **path_parser,
                    **integer_parser,
                    **float_parser,
                    **special_parser
                }
            )

        return reduce(parse, lines, {})

    # split line into key and args, calls parser for args and adds key: parser(args) to dict
    @staticmethod
    def _parse_to_dict(d: dict, line: str, parsers: dict) -> dict:
        llist = line.strip().split(None, 1)
        key = llist[0].lower()
        args = ' '
        if len(llist) == 2:
            args = llist[1]
        try:
            if key in d:
                dk = d[key]
                if isinstance(dk, dict):
                    dk.update(parsers[key](args))
                elif isinstance(dk, list):
                    dk.extend(parsers[key](args))
            else:
                d[key] = parsers[key](args)
        except Error:
            raise
        except Exception:
            ty, val, tb = sys.exc_info()
            raise Error(
                f'Something went wrong while parsing the keyword: {key} {args}:\n\
                {ty.__name__}: {val}\nPlease consult the examples folder in the $SHARCDIR for more information!'
            ).with_traceback(tb)
        return d

    def generate_joblist(self):
        QMin = self._QMin
        # sort the gradients into the different jobs
        gradjob = {}
        for ijob in QMin['joblist']:
            gradjob[f'master_{ijob}'] = {}
        for grad in QMin['gradmap']:
            ijob = QMin['multmap'][grad[0]]
            isgs = False
            if not QMin['jobs'][ijob]['restr']:
                if grad[1] == 1:
                    isgs = True
            else:
                if grad == (1, 1):
                    isgs = True
            istates = QMin['states_to_do'][grad[0] - 1]
            gradjob['master_%i' % ijob][grad] = {'gs': isgs}
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
        nrounds, nslots, cpu_per_run = self.divide_slots(QMin['ncpu'], ntasks, QMin['schedule_scaling'])
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
                if QMin1['template']['qmmm']:
                    QMin1['qmmm'] = True
                icount += 1
                schedule[-1][i] = QMin1
        # add the gradient calculations
        ntasks = 0
        for i in gradjob:
            if 'grad' in i:
                ntasks += 1
        if ntasks > 0:
            nrounds, nslots, cpu_per_run = self.divide_slots(QMin['ncpu'], ntasks, QMin['schedule_scaling'])
            QMin['nslots_pool'].append(nslots)
            schedule.append({})
            icount = 0
            for i in gradjob:
                if 'grad' in i:
                    QMin1 = deepcopy(QMin)
                    mult = list(gradjob[i])[0][0]
                    QMin1['IJOB'] = QMin['multmap'][mult]
                    remove = [
                        'gradmap', 'ncpu', 'h', 'soc', 'dm', 'overlap', 'ion', 'always_guess', 'always_orb_init', 'init'
                    ]
                    for r in remove:
                        QMin1 = removekey(QMin1, r)
                    QMin1['gradmap'] = list(gradjob[i])
                    QMin1['ncpu'] = cpu_per_run[icount]
                    QMin1['gradonly'] = []
                    if QMin1['template']['qmmm']:
                        QMin1['qmmm'] = True
                    icount += 1
                    schedule[-1][i] = QMin1
        return schedule

    @staticmethod
    def read_coords(xyz):
        lines = readfile(xyz)
        try:
            natom = int(lines[0])
        except ValueError:
            raise Error('first line must contain the number of atoms!', 2)
        return [[x[0], *x[1]] for x in map(INTERFACE._parse_xyz, lines[2:natom + 2])]

    @staticmethod
    def read_elements(QMinlines: list[str]) -> list[str]:

        try:
            natom = int(QMinlines[0])
        except ValueError:
            raise Error('first line must contain the number of atoms!', 2)
        if len(QMinlines) < natom + 4:
            raise Error(
                'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task', 3
            )
        atomlist = list(map(lambda x: INTERFACE._parse_xyz(x)[0], (QMinlines[2:natom + 2])))
        return atomlist

    # ======================================================================= #

    @staticmethod
    def _parse_xyz(line: str) -> tuple[str, list[float]]:
        match = re.match(r'([a-zA-Z]{1,2}\d?)((\s+-?\d+\.\d*){3,6})', line.strip())
        if match:
            return match[1], list(map(float, match[2].split()[:3]))
        else:
            raise Error(f"line is not xyz\n\n{line}", 43)

    @staticmethod
    def _get_pairs(QMinlines, i):
        nacpairs = []
        while True:
            i += 1
            try:
                line = QMinlines[i].lower()
            except IndexError:
                raise Error('"keyword select" has to be completed with an "end" on another line!', 47)
            if 'end' in line:
                break
            fields = line.split()
            try:
                nacpairs.append([int(fields[0]), int(fields[1])])
            except ValueError:
                raise Error('"nacdr select" is followed by pairs of state indices, each pair on a new line!', 48)
        return nacpairs, i

    # ======================================================================= #

    @staticmethod
    def checkscratch(SCRATCHDIR):
        '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1,
        if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

        Arguments:
        1 string: path to SCRATCHDIR'''

        exist = os.path.exists(SCRATCHDIR)
        if exist:
            isfile = os.path.isfile(SCRATCHDIR)
            if isfile:
                raise Error('$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR), 42)
        else:
            try:
                os.makedirs(SCRATCHDIR)
            except OSError:
                raise Error('Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR), 43)

    @staticmethod
    def removequotes(string):
        if string.startswith("'") and string.endswith("'"):
            return string[1:-1]
        elif string.startswith('"') and string.endswith('"'):
            return string[1:-1]
        else:
            return string
# ======================================================================= #

    @staticmethod
    def get_smatel(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                raise Error('Overlap of states %i - %i not found!' % (s1, s2), 32)
            if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    @staticmethod
    def coords_same(coord1, coord2):
        thres = 1e-5
        s = 0.
        for i in range(3):
            s += (coord1[i] - coord2[i])**2
        s = math.sqrt(s)
        return s <= thres

    def runProgram(self, string, workdir, outfile, errfile=''):
        prevdir = os.getcwd()
        PRINT = self._PRINT
        DEBUG = self._DEBUG
        if DEBUG:
            print(workdir)
        os.chdir(workdir)
        if PRINT or DEBUG:
            starttime = time()
            sys.stdout.write('%s\n\t%s' % (string, starttime))
            sys.stdout.flush()
        stdoutfile = open(os.path.join(workdir, outfile), 'w')
        if errfile:
            stderrfile = open(os.path.join(workdir, errfile), 'w')
        else:
            stderrfile = sp.STDOUT
        try:
            exit_code = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            t, v, tb = sys.exc_info()
            raise Error(f'Call has had some serious problems:\nWORKDIR:{workdir}\n{t}: {v}', 96).with_traceback(tb)
        stdoutfile.close()
        if errfile:
            stderrfile.close()
        if PRINT or DEBUG:
            endtime = time()
            sys.stdout.write(
                '\t{:%d.%m.%Y %H:%M}\t\tRuntime: {:3f}s\t\tExit Code: {}\n\n'.format(
                    datetime.datetime.now(), endtime - starttime, exit_code
                )
            )
        os.chdir(prevdir)
        return exit_code

    @staticmethod
    def parallel_speedup(N, scaling):
        # computes the parallel speedup from Amdahls law
        # with scaling being the fraction of parallelizable work and (1-scaling) being the serial part
        return 1. / ((1 - scaling) + scaling / N)

    @staticmethod
    def divide_slots(ncpu, ntasks, scaling):
        # this routine figures out the optimal distribution of the tasks over the CPU cores
        #   returns the number of rounds (how many jobs each CPU core will contribute to),
        #   the number of slots which should be set in the Pool,
        #   and the number of cores for each job.
        ntasks_per_round = min(ncpu, ntasks)
        optimal = {}
        for i in range(1, 1 + ntasks_per_round):
            nrounds = int(math.ceil(float(ntasks) // i))
            ncores = ncpu // i
            optimal[i] = nrounds // INTERFACE.parallel_speedup(ncores, scaling)
        # print optimal
        best = min(optimal, key=optimal.get)
        nrounds = int(math.ceil(float(ntasks) // best))
        ncores = ncpu // best

        cpu_per_run = [0] * ntasks
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

    def stripWORKDIR(WORKDIR, keep):
        for ifile in os.listdir(WORKDIR):
            if any([containsstring(k, ifile) for k in keep]):
                break
            rmfile = os.path.join(WORKDIR, ifile)
            os.remove(rmfile)

    def writegeom(self):
        QMin = self._QMin
        factor = au2a
        fname = QMin['scratchdir'] + '/JOB/geom.xyz'
        string = '%i\n\n' % (QMin['natom'])
        for atom in QMin['geo']:
            string += atom[0]
            for xyz in range(1, 4):
                string += '  %f' % (atom[xyz] * factor)
            string += '\n'
        writefile(fname, string)

        os.chdir(QMin['scratchdir'] + '/JOB')
        error = sp.call('x2t geom.xyz > coord', shell=True)
        if error != 0:
            raise Error('xyz2col call failed!', 95)
        os.chdir(QMin['pwd'])

        # QM/MM
        if QMin['qmmm']:
            string = '$point_charges nocheck\n'
            for atom in QMin['pointcharges']:
                string += '%16.12f %16.12f %16.12f %12.9f\n' % (atom[0] / au2a, atom[1] / au2a, atom[2] / au2a, atom[3])
            string += '$end\n'
            filename = QMin['scratchdir'] + '/JOB/pc'
            writefile(filename, string)

        # COBRAMM
        if QMin['cobramm']:
            # chargefiles='charge.dat'
            # tocharge=os.path.join(QMin['scratchdir']+'/JOB/point_charges')
            # shutil.copy(chargefiles,tocharge)
            cobcharges = open('charge.dat', 'r')
            charges = cobcharges.read()
            only_atom = charges.split()
            only_atom.pop(0)
            filename = QMin['scratchdir'] + '/JOB/point_charges'
            string = '$point_charges nocheck\n'
            string += charges
            # counter=0
            # for atom in only_atom:
            #   	string+=atom
            #    string+=' '
            #    counter+=1
            #    if counter == 4:
            #      string+='\n'
            #      counter=0
            #    #string+='\n'
            string += '$end'
            writefile(filename, string)

    def get_wfovlout(self, path, mult):

        QMin = self._QMin
        QMout = self._QMout
        outfile = os.path.join(path, 'wfovl.out')
        out = readfile(outfile)

        if 'overlap' in QMin:
            nmstates = QMin['nmstates']
            if 'overlap' not in QMout:
                QMout['overlap'] = makecmatrix(nmstates, nmstates)
            # read the overlap matrix
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if not m1 == m2 == mult:
                        continue
                    if not ms1 == ms2:
                        continue
                    QMout['overlap'][i][j] = self.get_smatel(out, s1, s2)

        return

    def wfoverlap(self, scradir, mult):
        QMin = self._QMin
        # link all input files for wfoverlap
        savedir = QMin['savedir']
        link(os.path.join(savedir, 'ao_ovl'), os.path.join(scradir, 'ao_ovl'), crucial=True, force=True)
        link(os.path.join(savedir, 'mos.old'), os.path.join(scradir, 'mos.a'), crucial=True, force=True)
        link(os.path.join(savedir, 'mos'), os.path.join(scradir, 'mos.b'), crucial=True, force=True)
        if QMin['template']['method'] == 'cc2':
            link(
                os.path.join(savedir, 'dets_left.%i.old' % (mult)),
                os.path.join(scradir, 'dets.a'),
                crucial=True,
                force=True
            )
        else:
            link(
                os.path.join(savedir, 'dets.%i.old' % (mult)),
                os.path.join(scradir, 'dets.a'),
                crucial=True,
                force=True
            )
        link(os.path.join(savedir, 'dets.%i' % (mult)), os.path.join(scradir, 'dets.b'), crucial=True, force=True)

        # write input file for wfoverlap
        string = '''mix_aoovl=ao_ovl
    a_mo=mos.a
    b_mo=mos.b
    a_det=dets.a
    b_det=dets.b
    a_mo_read=2
    b_mo_read=2
    '''
        if 'ncore' in QMin:
            icore = QMin['ncore']
        elif 'frozenmap' in QMin:
            icore = QMin['frozenmap'][mult]
        else:
            icore = 0
        string += 'ncore=%i' % (icore)
        writefile(os.path.join(scradir, 'wfovl.inp'), string)

        # run wfoverlap
        string = '%s -f wfovl.inp -m %i' % (QMin['wfoverlap'], QMin['memory'])
        self.runProgram(string, scradir, 'wfovl.out')

    def copymolden(self):
        QMin = self._QMin
        # run tm2molden in scratchdir
        string = 'molden.input\nY\n'
        filename = os.path.join(QMin['scratchdir'], 'JOB', 'tm2molden.input')
        writefile(filename, string)
        string = 'tm2molden < tm2molden.input'
        path = os.path.join(QMin['scratchdir'], 'JOB')
        self.runProgram(string, path, 'tm2molden.output')

        if 'molden' in QMin:
            # create directory
            moldendir = QMin['savedir'] + '/MOLDEN/'
            if not os.path.isdir(moldendir):
                mkdir(moldendir)

            # save the molden.input file
            f = QMin['scratchdir'] + '/JOB/molden.input'
            fdest = moldendir + '/step_%s.molden' % (QMin['step'][0])
            shutil.copy(f, fdest)

    # ======================================================================= #
    def run_theodore(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')
        string = 'python2 %s/bin/analyze_tden.py' % (QMin['theodir'])
        runerror = self.runProgram(string, workdir, 'theodore.out')
        if runerror != 0:
            raise Error('Theodore calculation crashed! Error code=%i' % (runerror), 105)
        return

    def setupWORKDIR_TH(self):
        raise NotImplementedError()

    @staticmethod
    def get_theodore(sumfile, omffile):
        def theo_float(i):
            return safe_cast(i, float, 0.)

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

    def run_wfoverlap(self, errorcodes):
        QMin = self._QMin
        print('>>>>>>>>>>>>> Starting the WFOVERLAP job execution')

        # do Dyson calculations
        if 'ion' in QMin:
            for ionpair in QMin['ionmap']:
                WORKDIR = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i' % ionpair)
                files = {
                    'aoovl': 'AO_overl',
                    'det.a': 'dets.%i' % ionpair[0],
                    'det.b': 'dets.%i' % ionpair[2],
                    'mo.a': 'mos.%i' % ionpair[1],
                    'mo.b': 'mos.%i' % ionpair[3]
                }
                INTERFACE.setupWORKDIR_WF(WORKDIR, QMin, files, self._DEBUG)
                errorcodes[
                    'Dyson_%i_%i_%i_%i' % ionpair
                ] = INTERFACE.runWFOVERLAP(WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])

        # do overlap calculations
        if 'overlap' in QMin:
            self.get_Double_AOovl()
            for m in itmult(QMin['states']):
                job = QMin['multmap'][m]
                WORKDIR = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i' % (m, job))
                files = {
                    'aoovl': 'AO_overl.mixed',
                    'det.a': 'dets.%i.old' % m,
                    'det.b': 'dets.%i' % m,
                    'mo.a': 'mos.%i.old' % job,
                    'mo.b': 'mos.%i' % job
                }
                INTERFACE.setupWORKDIR_WF(WORKDIR, QMin, files, self._DEBUG)
                errorcodes[
                    'WFOVL_%i_%i' % (m, job)
                ] = INTERFACE.runWFOVERLAP(WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])

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
        if any((i != 0 for i in errorcodes.values())):
            print('Some subprocesses did not finish successfully!')
            sys.exit(100)

        print('')

        return errorcodes

    # ======================================================================= #

    def setupWORKDIR_WF(WORKDIR, QMin, files, DEBUG=False):
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

    @staticmethod
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
            sys.exit(101)
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

    @staticmethod
    def get_props(sumfile, omffile):
        out = readfile(sumfile)
        props = {}

        def theo_float(x):
            return safe_cast(x, float, 0.)

        for line in out[2:]:
            s = line.replace('(', ' ').replace(')', ' ').split()
            if len(s) == 0:
                continue
            n = int(s[0])
            m = int(s[1])
            props[(m, n + (m == 1))] = [theo_float(i) for i in s[5:]]

        out = readfile(omffile)
        for line in out[1:]:
            s = line.replace('(', ' ').replace(')', ' ').split()
            if len(s) == 0:
                continue
            n = int(s[0])
            m = int(s[1])
            props[(m, n + (m == 1))].extend([theo_float(i) for i in s[4:]])

        return props

    # ======================================================================= #

    def get_Double_AOovl(self):
        raise NotImplementedError("Implement into derived interface!")

    def copy_ntos(self):
        QMin = self._QMin
        # create directory
        moldendir = QMin['savedir'] + '/MOLDEN/'
        if not os.path.isdir(moldendir):
            mkdir(moldendir)

        # save the nto_x-x.a.mld files
        for i in QMin['statemap']:
            m, s, ms = QMin['statemap'][i]
            if m == 1 and s == 1:
                continue
            if m > 1 and ms != float(m - 1) / 2:
                continue
            f = os.path.join(QMin['scratchdir'], 'JOB', 'nto_%i-%i-a.mld' % (s - (m == 1), m))
            fdest = moldendir + '/step_%s__nto_%i_%i.molden' % (QMin['step'], m, s)
            shutil.copy(f, fdest)

# =============================================================================================== #
# =============================================================================================== #
# =========================================== QM/MM ============================================= #
# =============================================================================================== #
# =============================================================================================== #

    def prepare_QMMM(self, table_file):
        ''' creates dictionary with:
        MM coordinates (including connectivity and atom types)
        QM coordinates (including Link atom stuff)
        point charge data (including redistribution for Link atom neighbors)
        reorder arrays (for internal processing, all QM, then all LI, then all MM)

        is only allowed to read the following keys from QMin:
        geo
        natom
        QM/MM related infos from template
        '''
        QMin = self._QMin
        table = readfile(table_file)

        # read table file
        print('===== Running QM/MM preparation ====')
        print('Reading table file ...         ', datetime.now())
        QMMM = {}
        QMMM['qmmmtype'] = []
        QMMM['atomtype'] = []
        QMMM['connect'] = []
        allowed = ['qm', 'mm']
        # read table file
        for iline, line in enumerate(table):
            s = line.split()
            if len(s) == 0:
                continue
            if not s[0].lower() in allowed:
                raise Error('Not allowed QMMM-type "%s" on line %i!' % (s[0], iline + 1), 34)
            QMMM['qmmmtype'].append(s[0].lower())
            QMMM['atomtype'].append(s[1])
            QMMM['connect'].append(set())
            for i in s[2:]:
                QMMM['connect'][-1].add(int(i) - 1)    # internally, atom numbering starts at 0
        QMMM['natom_table'] = len(QMMM['qmmmtype'])

        # list of QM and MM atoms
        QMMM['QM_atoms'] = []
        QMMM['MM_atoms'] = []
        for iatom in range(QMMM['natom_table']):
            if QMMM['qmmmtype'][iatom] == 'qm':
                QMMM['QM_atoms'].append(iatom)
            elif QMMM['qmmmtype'][iatom] == 'mm':
                QMMM['MM_atoms'].append(iatom)

        # make connections redundant and fill bond array
        print('Checking connection table ...  ', datetime.now())
        QMMM['bonds'] = set()
        for iatom in range(QMMM['natom_table']):
            for jatom in QMMM['connect'][iatom]:
                QMMM['bonds'].add(tuple(sorted([iatom, jatom])))
                QMMM['connect'][jatom].add(iatom)
        QMMM['bonds'] = sorted(list(QMMM['bonds']))

        # find link bonds
        print('Finding link bonds ...         ', datetime.now())
        QMMM['linkbonds'] = []
        QMMM['LI_atoms'] = []
        for i, j in QMMM['bonds']:
            if QMMM['qmmmtype'][i] != QMMM['qmmmtype'][j]:
                link = {}
                if QMMM['qmmmtype'][i] == 'qm':
                    link['qm'] = i
                    link['mm'] = j
                elif QMMM['qmmmtype'][i] == 'mm':
                    link['qm'] = j
                    link['mm'] = i
                link['scaling'] = {'qm': 0.3, 'mm': 0.7}
                link['element'] = 'H'
                link['atom'] = [link['element'], 0., 0., 0.]
                for xyz in range(3):
                    link['atom'][xyz + 1] += link['scaling']['mm'] * QMin['geo'][link['mm']][xyz + 1]
                    link['atom'][xyz + 1] += link['scaling']['qm'] * QMin['geo'][link['qm']][xyz + 1]
                QMMM['linkbonds'].append(link)
                QMMM['LI_atoms'].append(QMMM['natom_table'] - 1 + len(QMMM['linkbonds']))
                QMMM['atomtype'].append('999')
                QMMM['connect'].append(set([link['qm'], link['mm']]))

        # check link bonds
        mm_in_links = []
        qm_in_links = []
        mm_in_link_neighbors = []
        for link in QMMM['linkbonds']:
            mm_in_links.append(link['mm'])
            qm_in_links.append(link['qm'])
            for j in QMMM['connect'][link['mm']]:
                if QMMM['qmmmtype'][j] == 'mm':
                    mm_in_link_neighbors.append(j)
        mm_in_link_neighbors.extend(mm_in_links)
        # no QM atom is allowed to be bonded to two MM atoms
        if not len(qm_in_links) == len(set(qm_in_links)):
            raise Error('Some QM atom is involved in more than one link bond!', 35)
        # no MM atom is allowed to be bonded to two QM atoms
        if not len(mm_in_links) == len(set(mm_in_links)):
            raise Error('Some MM atom is involved in more than one link bond!', 36)
        # no neighboring MM atoms are allowed to be involved in link bonds
        if not len(mm_in_link_neighbors) == len(set(mm_in_link_neighbors)):
            raise Error('An MM-link atom is bonded to another MM-link atom!', 37)

        # check geometry and connection table
        if not QMMM['natom_table'] == QMin['natom']:
            raise Error('Number of atoms in table file does not match number of atoms in QMin!', 38)

        # process MM geometry (and convert to angstrom!)
        QMMM['MM_coords'] = []
        for atom in QMin['geo']:
            QMMM['MM_coords'].append([atom[0]] + [i * au2a for i in atom[1:4]])
        for ilink, link in enumerate(QMMM['linkbonds']):
            QMMM['MM_coords'].append(['HLA'] + link['atom'][1:4])

        # create reordering dicts
        print('Creating reorder mappings ...  ', datetime.now())
        QMMM['reorder_input_MM'] = {}
        QMMM['reorder_MM_input'] = {}
        j = -1
        for i, t in enumerate(QMMM['qmmmtype']):
            if t == 'qm':
                j += 1
                QMMM['reorder_MM_input'][j] = i
        for ilink, link in enumerate(QMMM['linkbonds']):
            j += 1
            QMMM['reorder_MM_input'][j] = QMMM['natom_table'] + ilink
        for i, t in enumerate(QMMM['qmmmtype']):
            if t == 'mm':
                j += 1
                QMMM['reorder_MM_input'][j] = i
        for i in QMMM['reorder_MM_input']:
            QMMM['reorder_input_MM'][QMMM['reorder_MM_input'][i]] = i

        # process QM geometry (including link atoms), QM coords in bohr!
        QMMM['QM_coords'] = []
        QMMM['reorder_input_QM'] = {}
        QMMM['reorder_QM_input'] = {}
        j = -1
        for iatom in range(QMMM['natom_table']):
            if QMMM['qmmmtype'][iatom] == 'qm':
                QMMM['QM_coords'].append(deepcopy(QMin['geo'][iatom]))
                j += 1
                QMMM['reorder_input_QM'][iatom] = j
                QMMM['reorder_QM_input'][j] = iatom
        for ilink, link in enumerate(QMMM['linkbonds']):
            QMMM['QM_coords'].append(link['atom'])
            j += 1
            QMMM['reorder_input_QM'][-(ilink + 1)] = j
            QMMM['reorder_QM_input'][j] = -(ilink + 1)

        # process charge redistribution around link bonds
        # point charges are in input geometry ordering
        print('Charge redistribution ...      ', datetime.now())
        QMMM['charge_distr'] = []
        for iatom in range(QMMM['natom_table']):
            if QMMM['qmmmtype'][iatom] == 'qm':
                QMMM['charge_distr'].append([(0., 0)])
            elif QMMM['qmmmtype'][iatom] == 'mm':
                if iatom in mm_in_links:
                    QMMM['charge_distr'].append([(0., 0)])
                else:
                    QMMM['charge_distr'].append([(1., iatom)])
        for link in QMMM['linkbonds']:
            mm_neighbors = []
            for j in QMMM['connect'][link['mm']]:
                if QMMM['qmmmtype'][j] == 'mm':
                    mm_neighbors.append(j)
            if len(mm_neighbors) > 0:
                factor = 1. / len(mm_neighbors)
                for j in QMMM['connect'][link['mm']]:
                    if QMMM['qmmmtype'][j] == 'mm':
                        QMMM['charge_distr'][j].append((factor, link['mm']))

        # pprint.pprint(QMMM)
        return QMMM

    def transform_QM_QMMM(self):
        QMin = self._QMin
        QMout = self._QMout
        # Meta data
        QMin['natom'] = QMin['natom_orig']
        QMin['geo'] = QMin['geo_orig']

        # Hamiltonian
        if 'h' in QMout:
            for i in range(QMin['nmstates']):
                QMout['h'][i][i] += QMin['qmmm']['MMEnergy']

        # Gradients
        if 'grad' in QMout:
            nmstates = QMin['nmstates']
            natom = QMin['natom_orig']
            grad = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
            # QM gradient
            for iqm in QMin['qmmm']['reorder_QM_input']:
                iqmmm = QMin['qmmm']['reorder_QM_input'][iqm]
                if iqmmm < 0:
                    ilink = -iqmmm - 1
                    link = QMin['qmmm']['linkbonds'][ilink]
                    for istate in range(nmstates):
                        for ixyz in range(3):
                            grad[istate][link['qm']][ixyz] += QMout['grad'][istate][iqm][ixyz] * link['scaling']['qm']
                            grad[istate][link['mm']][ixyz] += QMout['grad'][istate][iqm][ixyz] * link['scaling']['mm']
                else:
                    for istate in range(nmstates):
                        for ixyz in range(3):
                            grad[istate][iqmmm][ixyz] += QMout['grad'][istate][iqm][ixyz]
            # PC gradient
            # for iqm,iqmmm in enumerate(QMin['qmmm']['MM_atoms']):
            for iqm in QMin['qmmm']['reorder_pc_input']:
                iqmmm = QMin['qmmm']['reorder_pc_input'][iqm]
                for istate in range(nmstates):
                    for ixyz in range(3):
                        grad[istate][iqmmm][ixyz] += QMout['pcgrad'][istate][iqm][ixyz]
            # MM gradient
            for iqmmm in range(QMin['qmmm']['natom_table']):
                for istate in range(nmstates):
                    for ixyz in range(3):
                        grad[istate][iqmmm][ixyz] += QMin['qmmm']['MMGradient'][iqmmm][ixyz]
            QMout['grad'] = grad

        # pprint.pprint(QMout)
        return

    @staticmethod
    def write_pccoord_file(pointcharges):
        '''Writes pointcharges as file'''
        string = '%i\n' % len(pointcharges)
        for atom in pointcharges:
            string += f'{atom[3]} {atom[0]} {atom[1]} {atom[2]}\n'
        return string

    # ============================PRINTING ROUTINES========================== #

    def printheader(self):
        '''Prints the formatted header of the log file. Prints version number and version date
        Takes nothing, returns nothing.'''

        print(self.clock.starttime, gethostname(), os.getcwd())
        rule = '=' * 76
        lines = [
            f'  {rule}', '', f'SHARC - {self.__class__.__name__} - Interface', '', f'Authors: {self.authors}', '',
            f'Version: {self.version}', 'Date: {:%d.%m.%Y}'.format(self.versiondate), '', f'  {rule}'
        ]
        # wraps Authors line in case its too long
        lines[4:5] = wrap(lines[4], width=70)
        lines[1:-1] = map(lambda s: '||{:^76}||'.format(s), lines[1:-1])
        print(*lines, sep='\n')
        print('\n')

    def printQMin(self):

        QMin = self._QMin
        PRINT = self._PRINT
        DEBUG = self._DEBUG
        if not PRINT:
            return
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
        string += QMin['template']['functional'].split()[0].upper()
        string += '/%s' % (QMin['template']['basis'])
        parts = []
        if QMin['template']['dispersion']:
            parts.append(QMin['template']['dispersion'].split()[0].upper())
        if QMin['template']['qmmm']:
            parts.append('QM/MM')
        if len(parts) > 0:
            string += '\t('
            string += ','.join(parts)
            string += ')'
        print(string)
        # TODO: remove after implementing QMMM
        QMin['geo_orig'] = QMin['coords']
        QMin['natom_orig'] = QMin['natom']
        QMin['frozcore_orig'] = QMin['frozcore']
        QMin['Atomcharge_orig'] = QMin['Atomcharge']
        string = 'Found Geo'
        if 'veloc' in QMin:
            string += ' and Veloc! '
        else:
            string += '! '
        string += 'NAtom is %i.\n' % (QMin['natom'])
        print(string)

        string = 'Geometry in Bohrs (%i atoms):\n' % QMin['natom']
        if DEBUG:
            for i in range(QMin['natom_orig']):
                string += '%2s ' % (QMin['elements'][i])
                for j in range(3):
                    string += '% 7.4f ' % (QMin['geo_orig'][i][j])
                string += '\n'
        else:
            for i in range(min(QMin['natom_orig'], 5)):
                string += '%2s ' % (QMin['elements'][i])
                for j in range(3):
                    string += '% 7.4f ' % (QMin['geo_orig'][i][j])
                string += '\n'
            if QMin['natom_orig'] > 5:
                string += '..     ...     ...     ...\n'
                string += '%2s ' % (QMin['elements'][-1])
                for j in range(3):
                    string += '% 7.4f ' % (QMin['geo_orig'][-1][j])
                string += '\n'
        print(string)

        if 'veloc' in QMin and DEBUG:
            string = ''
            for i in range(QMin['natom_orig']):
                string += '%s ' % (QMin['geo_orig'][i][0])
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

        for i in sorted(QMin):
            if not any(
                [
                    i == j for j in [
                        'h', 'dm', 'soc', 'dmdr', 'socdr', 'theodore', 'geo', 'veloc', 'states', 'comment', 'grad',
                        'nacdr', 'ion', 'overlap', 'template', 'statemap', 'pointcharges', 'geo_orig', 'qmmm'
                    ]
                ]
            ):
                if not any([i == j for j in ['ionlist']]) or DEBUG:
                    string = i + ': '
                    string += str(QMin[i])
                    print(string)
        print('\n')
        sys.stdout.flush()

    def printQMout(self):
        '''If PRINT, prints a summary of all requested QM output values.
        Matrices are formatted using printcomplexmatrix, vectors using printgrad.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout'''
        QMin = self._QMin
        QMout = self._QMout
        if not self._PRINT:
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
                            print(
                                '%s\tStates %i - %i\tMs= % .1f\tPolarization %s:' %
                                (IToMult[imult], i, j, msi, IToPol[ipol])
                            )
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
    def printgrad(self, grad, natom, geo):
        '''Prints a gradient or nac vector. Also prints the atom elements.
        If the gradient is identical zero, just prints one line.

        Arguments:
        1 list of list of float: gradient
        2 integer: natom
        3 list of list: geometry specs'''

        string = ''
        iszero = True
        for atom in range(natom):
            if not self.DEBUG:
                if atom == 5:
                    string += '...\t...\t     ...\t     ...\t     ...\n'
                if 5 <= atom < natom - 1:
                    continue
            string += '%i\t%s\t' % (atom + 1, geo[atom][0])
            for xyz in range(3):
                if grad[atom][xyz] != 0:
                    iszero = False
                string += '% .5f\t' % (grad[atom][xyz])
            string += '\n'
        if iszero:
            print('\t\t...is identical zero...\n')
        else:
            print(string)

    def printtheodore(matrix, QMin):
        string = '%6s ' % 'State'
        for i in QMin['template']['theodore_prop']:
            string += '%6s ' % i
        for i in range(len(QMin['template']['theodore_fragment'])):
            for j in range(len(QMin['template']['theodore_fragment'])):
                string += '  Om%1i%1i ' % (i + 1, j + 1)
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


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #


    def writeQMout(self):
        '''Writes the requested quantities to the file which SHARC reads in.
        The filename is QMinfilename with everything after the first dot replaced by "out".

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout
        3 string: QMinfilename'''
        QMin = self._QMin
        QMinfilename = self._QMinfilename
        k = QMinfilename.find('.')
        if k == -1:
            outfilename = QMinfilename + '.out'
        else:
            outfilename = QMinfilename[:k] + '.out'
        if self._PRINT:
            print('===> Writing output to file %s in SHARC Format\n' % (outfilename))
        string = ''
        if 'h' in QMin or 'soc' in QMin:
            string += self.writeQMoutsoc()
        if 'dm' in QMin:
            string += self.writeQMoutdm()
        if 'grad' in QMin:
            string += self.writeQMoutgrad()
        if 'overlap' in QMin:
            string += self.writeQMoutnacsmat()
        if 'nacdr' in QMin:
            string += self.writeQMoutnacana()
        if 'socdr' in QMin:
            string += self.writeQMoutsocdr()
        if 'dmdr' in QMin:
            string += self.writeQMoutdmdr()
        if 'ion' in QMin:
            string += self.writeQMoutprop()
        if 'theodore' in QMin or QMin['template']['qmmm']:
            string += self.writeQMoutTHEODORE()
        if 'phases' in QMin:
            string += self.writeQmoutPhases()
        if 'grad' in QMin:
            if QMin['template']['cobramm']:
                self.writeQMoutgradcobramm()
        string += self.writeQMouttime()
        outfile = os.path.join(QMin['pwd'], outfilename)
        writefile(outfile, string)
        return

    def writeQMoutsoc(self):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the SOC matrix'''
        QMin = self._QMin
        QMout = self._QMout
        nmstates = QMin['nmstates']
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

    def writeQMoutdm(self):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line. The string contains three such matrices.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the DM matrices'''
        QMin = self._QMin
        QMout = self._QMout
        nmstates = QMin['nmstates']
        string = ''
        string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (
                        eformat(QMout['dm'][xyz][i][j].real, 12, 3), eformat(QMout['dm'][xyz][i][j].imag, 12, 3)
                    )
                string += '\n'
            string += ''
        return string

    # ======================================================================= #
    def writeQMoutdmdr(self):

        QMin = self._QMin
        QMout = self._QMout
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
                        natom, 3, imult, istate, ims, jmult, jstate, jms, ipol
                    )
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

    def writeQMoutsocdr(self):

        QMin = self._QMin
        QMout = self._QMout
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
                    natom, 3, imult, istate, ims, jmult, jstate, jms
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s %s ' % (
                            eformat(QMout['socdr'][i][j][atom][xyz].real, 12,
                                    3), eformat(QMout['socdr'][i][j][atom][xyz].imag, 12, 3)
                        )
                string += '\n'
                string += ''
                j += 1
            i += 1
        string += '\n'
        return string

    def writeQMoutang(self):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line. The string contains three such matrices.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the DM matrices'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Angular Momentum Matrices (3x%ix%i, complex)\n' % (9, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (
                        eformat(QMout['angular'][xyz][i][j].real, 12,
                                3), eformat(QMout['angular'][xyz][i][j].imag, 12, 3)
                    )
                string += '\n'
            string += ''
        return string

    # ======================================================================= #

    def writeQMoutgrad(self):
        '''Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the Gradient vectors'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += '%i %i ! m1 %i s1 %i ms1 %i\n' % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s ' % (eformat(QMout['grad'][i][atom][xyz], 12, 3))
                string += '\n'
            string += ''
            i += 1
        return string

    # ======================================================================= #

    def writeQMoutnacnum(self):
        '''Generates a string with the NAC matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the NAC matrix'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Non-adiabatic couplings (ddt) (%ix%i, complex)\n' % (4, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(QMout['nacdt'][i][j].real, 12, 3), eformat(QMout['nacdt'][i][j].imag, 12, 3)
                )
            string += '\n'
        string += ''
        # also write wavefunction phases
        string += '! %i Wavefunction phases (%i, complex)\n' % (7, nmstates)
        for i in range(nmstates):
            string += '%s %s\n' % (eformat(QMout['phases'][i], 12, 3), eformat(0., 12, 3))
        string += '\n\n'
        return string

    # ======================================================================= #

    def writeQMoutnacana(self):
        '''Generates a string with the NAC vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the NAC vectors'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                # string+='%i %i ! %i %i %i %i %i %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms)
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (
                    natom, 3, imult, istate, ims, jmult, jstate, jms
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s ' % (eformat(QMout['nacdr'][i][j][atom][xyz], 12, 3))
                    string += '\n'
                string += ''
                j += 1
            i += 1
        return string

    # ======================================================================= #

    def writeQMoutnacsmat(self):
        '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the transformation matrix'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for j in range(nmstates):
            for i in range(nmstates):
                string += '%s %s ' % (
                    eformat(QMout['overlap'][j][i].real, 12, 3), eformat(QMout['overlap'][j][i].imag, 12, 3)
                )
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #

    def writeQMouttime(self):
        '''Generates a string with the quantum mechanics total runtime in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the runtime is given.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the runtime'''

        QMout = self._QMout
        string = '! 8 Runtime\n%s\n' % (eformat(QMout['runtime'], 9, 3))
        return string

    # ======================================================================= #

    def writeQMoutprop(self):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the SOC matrix'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Property Matrix (%ix%i, complex)\n' % (11, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3)
                )
            string += '\n'
        string += '\n'

        # print(property matrices (flag 20) in new format)
        string += '! %i Property Matrices\n' % (20)
        string += '%i    ! number of property matrices\n' % (1)

        string += '! Property Matrix Labels (%i strings)\n' % (1)
        string += 'Dyson norms\n'

        string += '! Property Matrices (%ix%ix%i, complex)\n' % (1, nmstates, nmstates)
        string += '%i %i   ! Dyson norms\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3)
                )
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #

    def writeQMoutTHEODORE(self):

        QMin = self._QMin
        QMout = self._QMout
        nmstates = QMin['nmstates']
        nprop = QMin['template']['theodore_n']
        if QMin['template']['qmmm']:
            nprop += len(QMin['qmmm']['MMEnergy_terms'])
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
        if QMin['template']['qmmm']:
            for label in sorted(QMin['qmmm']['MMEnergy_terms']):
                descriptors.append(label)
                string += label + '\n'

        string += '! Property Vectors (%ix%i, real)\n' % (nprop, nmstates)
        if 'theodore' in QMin:
            for i in range(QMin['template']['theodore_n']):
                string += '! TheoDORE descriptor %i (%s)\n' % (i + 1, descriptors[i])
                for j in range(nmstates):
                    string += '%s\n' % (eformat(QMout['theodore'][j][i].real, 12, 3))
        if QMin['template']['qmmm']:
            for label in sorted(QMin['qmmm']['MMEnergy_terms']):
                string += '! QM/MM energy contribution (%s)\n' % (label)
                for j in range(nmstates):
                    string += '%s\n' % (eformat(QMin['qmmm']['MMEnergy_terms'][label], 12, 3))
        string += '\n'

        return string

    # ======================================================================= #

    def writeQmoutPhases(self):

        QMin = self._QMin
        QMout = self._QMout
        string = '! 7 Phases\n%i ! for all nmstates\n' % (QMin['nmstates'])
        for i in range(QMin['nmstates']):
            string += '%s %s\n' % (eformat(QMout['phases'][i].real, 9, 3), eformat(QMout['phases'][i].imag, 9, 3))
        return string

    def writeQMoutgradcobramm(self):
        '''Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the Gradient vectors'''
        QMin = self._QMin
        QMout = self._QMout
        ncharges = len(readfile(os.path.join(QMin['scratchdir'], 'JOB', 'pc_grad'))) - 2
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = len(QMout['pcgrad'][0])
        print(QMout['pcgrad'][1])
        string = ''
        print(natom)
        # string+='! %i Gradient Vectors (%ix%ix3, real)\n' % (3,nmstates,natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    print((QMout['pcgrad'][i][atom], 9, 3), i, atom)
                    string += '%s ' % (eformat(QMout['pcgrad'][i][atom][xyz], 9, 3))
                string += '\n'
            # string+='\n'
            i += 1
        string += '\n'
        writefile("grad_charges", string)

    # ======================================================================= #

    def backupdata(self, backupdir):
        # save all files in savedir, except which have 'old' in their name
        QMin = self._QMin
        ls = os.listdir(self._QMin['savedir'])
        for f in ls:
            ff = self._QMin['savedir'] + '/' + f
            if os.path.isfile(ff) and 'old' not in ff:
                step = int(self._QMin['step'])
                fdest = backupdir + '/' + f + '.stp' + str(step)
                shutil.copy(ff, fdest)
        # save molden files
        if 'molden' in QMin:
            ff = os.path.join(QMin['savedir'], 'MOLDEN', 'step_%s.molden' % (QMin['step']))
            fdest = os.path.join(backupdir, 'step_%s.molden' % (QMin['step']))
            shutil.copy(ff, fdest)
