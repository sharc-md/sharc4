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
from error import Error, exception_hook
from printing import printcomplexmatrix, printgrad, printtheodore
from utils import *
from globals import DEBUG, PRINT
from constants import *
from parse_keywords import KeywordParser

sys.excepthook = exception_hook

# NOTE: Error handling especially import for processes in pools (error_callback)
# NOTE: gradient calculation necessitates multiple parallel calls (either inside interface) or
# one interface = one calculation (i.e. interface spawns multiple instances of itself)
# NOTE: logic checks in read_template and read_resources and in run() if required (LVC won't need check in run())


class INTERFACE(ABC):

    # internal status indicators
    _setup_mol = False
    _read_resources = False
    _read_template = False
    _DEBUG = False
    _PRINT = True

    # TODO: set Debug and Print flag
    # TODO: set persistant flag for file-io vs in-core
    def __init__(self, debug=False, print=True, persistent=False):
        # all the input and info for the calculation is stored here
        self._QMin = {}
        # all the output from the calculation will be stored here
        self._QMout = {}
        self.clock = clock(verbose=print)
        self._DEBUG = debug
        DEBUG.set(debug)
        PRINT.set(print)
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
        '''
        main routine for all interfaces.
        This routine containes all functions that will be accessed when any interface is calculating a single point. All of these functions have to be defined in the derived class if not available in this base class
        '''

        args = sys.argv
        name = self.__class__.__name__
        self.printheader()
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
        # set up the system (i.e. molecule, states, unit...)
        self.setup_mol(os.path.join(pwd, QMinfilename))
        # read in the resources available for this computation (program path, cores, memory)
        self.read_resources(os.path.join(pwd, f"{name}.resources"))
        # read in the specific template file for the interface with all keywords
        self.read_template(os.path.join(pwd, f"{name}.template"))
        # set the coordinates of the molecular system
        self.set_coords(os.path.join(pwd, QMinfilename))
        # read the property requests that have to be calculated
        self.read_requests(os.path.join(pwd, QMinfilename))
        # setup the folders for the computation
        self.setup_run()
        # perform the calculation and parse the output, do subsequent calculations with other tools
        self.run()
        # writes a STEP file in the SAVEDIR (marks this step as succesfull)
        self.write_step_file()
        # printing and output generation
        if self._PRINT or self._DEBUG:
            self.printQMout()
        self._QMout['runtime'] = self.clock.measuretime()
        self.writeQMout()

    @abstractmethod
    def read_template(self, template_filename):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def read_resources(self, resources_filename):
        '''
        a template for the read resources method.
        This function might already be sufficient for an interface to use but can be extended by calling this function as `super.read_resources()`.
        '''
        if not self._setup_mol:
            raise Error('Interface is not set up for this template. Call setup_mol with the QM.in file first!', 23)
        QMin = self._QMin

        pwd = os.getcwd()
        QMin['pwd'] = pwd

        paths = {
            self.__class__.__name__.lower() + 'dir': '',
            'scratchdir': '',
            'savedir': '',    # NOTE: savedir from QMin
            'theodir': '',
            'wfoverlap': os.path.join(os.path.expandvars(os.path.expanduser('$SHARC')), 'wfoverlap.x'),
        }
        bools = {
            'debug': False,
            'save_stuff': False,
            'no_print': False,
            'nooverlap': False,
            'always_orb_init': False,
            'always_guess': False,
        }
        integers = {'ncpu': 1, 'memory': 100, 'numfrozcore': -1, 'numocc': 0, 'theodore_n': 0, 'resp_layers': 4, 'resp_tdm_fit_order': 2}
        floats = {'delay': 0.0, 'schedule_scaling': 0.9, 'wfthres': 0.99, 'resp_density': 1., 'resp_first_layer': 1.4}
        special = {
            'neglected_gradient': 'zero',
            'theodore_prop': ['Om', 'PRNTO', 'S_HE', 'Z_HE', 'RMSeh'],
            'theodore_fragment': [],
            'resp_shells': False  # default calculated from other values = [1.4, 1.6, 1.8, 2.0]
        }
        lines = readfile(resources_filename)
        # assign defaults first, which get updated by the parsed entries, which are updated by the entries that were already in QMin
        QMin['resources'] = {
            **bools,
            **paths,
            **integers,
            **floats,
            **special,
            **self.parse_keywords(
                lines,
                bools=bools,
                paths=paths,
                integers=integers,
                floats=floats,
                special=special,
            )
        }
        print('DEBUG:', QMin['resources']['debug'])
        self._DEBUG = QMin['resources']['debug']
        self._PRINT = QMin['resources']['no_print'] is False
        if not DEBUG:
            DEBUG.set(self._DEBUG)
        if not PRINT:
            PRINT.set(self._PRINT)

        # NOTE: This is really optional
        ncpu = QMin['resources']['ncpu']
        if os.environ.get('NSLOTS') is not None:
            ncpu = int(os.environ.get('NSLOTS'))
            print(f'Detected $NSLOTS variable. Will use ncpu={ncpu}')
        elif os.environ.get('SLURM_NTASKS_PER_NODE') is not None:
            ncpu = int(os.environ.get('SLURM_NTASKS_PER_NODE'))
            print('Detected $SLURM_NTASKS_PER_NODE variable. Will use ncpu={ncpu}')
        QMin['resources']['ncpu'] = max(1, ncpu)

        if 0 > QMin['resources']['schedule_scaling'] or QMin['resources']['schedule_scaling'] > 1.:
            QMin['resources']['schedule_scaling'] = 0.9
        if 'always_orb_init' in QMin and 'always_guess' in QMin:
            raise Error('Keywords "always_orb_init" and "always_guess" cannot be used together!', 53)

        if QMin['resources']['numfrozcore'] >= 0:
            QMin['frozcore'] = QMin['resources']['numfrozcore']

        if QMin['resources']['numocc'] <= 0:
            QMin['numocc'] = 0
        else:
            QMin['numocc'] = max(0, QMin['resources']['numocc'] - QMin['frozcore'])

        # construct shells
        shells, first, nlayers = map(QMin['resources'].get, ('resp_shells', 'resp_first_layer', 'resp_layers'))
        if not shells:
            if DEBUG:
                print(f"Calculating resp layers as: {first} + 4/sqrt({nlayers})")
            incr = 0.4 / math.sqrt(nlayers)
            QMin['resources']['resp_shells'] = [first + incr * x for x in range(nlayers)]

        self._QMin = {**QMin['resources'], **QMin}
        return
        # ============================ Implemented public methods ========================

    def setup_mol(self, QMinfilename: str):
        '''
        Sets up the molecular system from a `QM.in` file.
        parses the elements, states, and savedir and prepare the QMin object accordingly.
        '''
        self._QMinfilename = QMinfilename
        QMin = self._QMin
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
        if '$' in QMin['savedir']:
            raise Error(f'undefined env variable in "savedir"! {QMin["savedir"]}')
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
        self._QMin["coords"] = np.asarray([parse_xyz(x)[1] for x in lines[2:natom + 2]], dtype=float) * self._factor

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

        self._QMin = {**dict(map(parse, lines)), **QMin}
        QMin = self._QMin
        # NOTE: old QMin read stuff is not overwritten. Problem with states?
        self._request_logic()

    def _reset_requests(self):
        for k in [
            'init', 'samestep', 'newstep', 'restart', 'cleanup', 'backup', 'h', 'soc', 'dm', 'grad', 'overlap', 'dmdr',
            'nac', 'nacdr', 'socdr', 'ion', 'theodore', 'phases', 'multipolar_fit'
        ]:
            if k in self._QMin:
                del self._QMin[k]

    def _request_logic(self):
        QMin = self._QMin
        # prepare savedir
        if not os.path.isdir(QMin['savedir']):
            mkdir(QMin['savedir'])

        possibletasks = {
            'h', 'soc', 'dm', 'grad', 'overlap', 'dmdr', 'socdr', 'ion', 'theodore', 'phases', 'multipolar_fit'
        }
        tasks = possibletasks & QMin.keys()
        if len(tasks) == 0:
            raise Error(f'No tasks found! Tasks are {possibletasks}.', 39)

        if 'h' not in tasks and 'soc' not in tasks:
            QMin['h'] = True

        if 'soc' in tasks and (len(QMin['states']) < 3 or QMin['states'][2] <= 0):
            del QMin['soc']
            QMin['h'] = True
            print('HINT: No triplet states requested, turning off SOC request.')

        # remove old keywords:
        for i in ['restart', 'init', 'samestep', 'newstep']:
            removekey(QMin, i)
        # check for savedir and STEP file
        last_step = None
        stepfile = os.path.join(QMin['savedir'], 'STEP')
        if os.path.isfile(stepfile):
            try:
                last_step = int(readfile(stepfile)[0])
            except (IndexError, ValueError):
                print(f'Warning: "STEP" file found in {stepfile}\nLast step index could not be read!\n')
        # checking scheme: determined last step in combination with specified step variable
        # (-1 -> newstep; None -> newstep)
        if 'step' not in QMin:
            if last_step is not None:
                QMin['newstep'] = True
                QMin['step'] = last_step + 1
            else:
                QMin['init'] = True
                QMin['step'] = 0
        else:
            QMin['step'] = int(QMin['step'])
            if last_step is None:
                if QMin['step'] == 0:
                    QMin['init'] = True
                else:
                    raise Error(
                        f'Specified step ({QMin["step"]}) could not be restarted from!\nCheck your savedir and "STEP" file in {QMin["savedir"]}'
                    )
            elif QMin['step'] == -1:
                QMin['newstep'] = True
                QMin['step'] = last_step + 1
            elif QMin['step'] == last_step:
                QMin['samestep'] = True
            elif QMin['step'] == last_step + 1:
                QMin['newstep'] = True
            else:
                raise Error(
                    f'Determined last step ({last_step}) from savedir and specified step ({QMin["step"]}) do not fit!\nPrepare your savedir and "STEP" file accordingly before starting again or choose "step -1" if you want to proceed from last successful step!'
                )

        if 'phases' in tasks:
            QMin['overlap'] = True

        if 'overlap' in tasks and 'init' in tasks:
            raise Error(
                '"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"',
                43
            )

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

        # Check for correct density list
        if 'multipolar_fit' in tasks:
            mf = QMin['multipolar_fit']
            if mf is True or mf == 'all':
                mf = [i + 1 for i in range(QMin['nmstates'])]
            else:
                try:
                    grad = [int(i) for i in mf]
                except ValueError:
                    raise Error('Arguments to keyword "multipolar_fit" must be "all" or a list of integers!', 49)
                if len(grad) > QMin['nmstates']:
                    raise Error(
                        'State for requested gradient does not correspond to any state in QM input file state list!', 50
                    )
            QMin['multipolar_fit'] = sorted(mf)

        # wfoverlap settings
        if ('overlap' in QMin or 'ion' in QMin) and self.__class__.__name__ != 'LVC':
            # WFoverlap

            if not os.path.isfile(QMin['resources']['wfoverlap']):
                raise Error('Give path to wfoverlap.x in resources file!', 54)

        if 'theodore' in QMin:
            if QMin['resources']['theodir'] is None or not os.path.isdir(QMin['resources']['theodir']):
                raise Error('Give path to the TheoDORE installation directory in resources file!', 56)
            os.environ['THEODIR'] = QMin['resources']['theodir']
            if 'PYTHONPATH' in os.environ:
                os.environ['PYTHONPATH'] = os.path.join(
                    QMin['resources']['theodir'], 'lib'
                ) + os.pathsep + QMin['resources']['theodir'] + os.pathsep + os.environ['PYTHONPATH']
                # print os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = os.path.join(QMin['theodir'],
                                                        'lib') + os.pathsep + QMin['resources']['theodir']
        if 'pc_file' in QMin:
            QMin['point_charges'] = [[float(x[0])*self._factor, float(x[1])*self._factor, float(x[2])*self._factor, float(x[3])] for x in map(lambda x: x.split(), readfile(QMin['pc_file']))]


    def setup_run(self):
        QMin = self._QMin
        # obtain the statemap
        QMin['statemap'] = {i + 1: [*v] for i, v in enumerate(itnmstates(QMin['states']))}

        self._states_to_do()    # can be different in interface -> general method here with possibility to overwrite
        # make the jobs
        self._jobs()
        jobs = QMin['jobs']
        # make the multmap (mapping between multiplicity and job)
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

        densmap = set()
        if 'multipolar_fit' in QMin:
            densmap = {tuple(QMin['statemap'][i][0:2]) for i in QMin['multipolar_fit']}
        QMin['densmap'] = sorted(densmap)

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
            QMin['theodore_n'] = len(QMin['resources']['theodore_prop']
                                     ) + len(QMin['resources']['theodore_fragment'])**2
        else:
            QMin['theodore_n'] = 0

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

        if self._PRINT:
            self.printQMin()

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
        if not file_str or file_str.isspace():    # check if there is only whitespace left!
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
            ty, val, tb = sys.exc_info()
            raise Error(
                f'Something went wrong while parsing the keyword: {key} {args}:\n\
                {ty.__name__}: {val}\nPlease consult the examples folder in the $SHARCDIR for more information!'
            ).with_traceback(None)
        except KeyError as e:
            ty, val, tb = sys.exc_info()
            keys = '\n    '.join(parsers.keys())
            available_keys = f'Available Keywords:\n    {keys}'
            raise Error(
                f'The keyword {val} is not known!' +
                f'\nPlease consult the examples folder in the $SHARCDIR for more information!\n{available_keys}', 34
            )
        return d

    def write_step_file(self):
        QMin = self.QMin
        if 'cleanup' in QMin:
            return
        savedir = QMin['savedir']
        stepfile = os.path.join(savedir, 'STEP')
        writefile(stepfile, str(QMin['step']))

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
            gradjob[f'master_{ijob}'][grad] = {'gs': isgs}
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
        return [[x[0], *x[1]] for x in map(parse_xyz, lines[2:natom + 2])]

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
        atomlist = list(map(lambda x: parse_xyz(x)[0], (QMinlines[2:natom + 2])))
        return atomlist

    # ======================================================================= #

    def _jobs(self):
        QMin = self.QMin
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

    def _states_to_do(self):
        QMin = self.QMin
        # obtain the states to actually compute
        states_to_do = [v + QMin['template']['paddingstates'][i] if v > 0 else v for i, v in enumerate(QMin['states'])]
        if not QMin['template']['unrestricted_triplets']:
            if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                states_to_do[0] = max(QMin['states'][0], 1)
                req = max(QMin['states'][0] - 1, QMin['states'][2])
                states_to_do[0] = req + 1
                states_to_do[2] = req
        QMin['states_to_do'] = states_to_do

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
    def parallel_speedup(N, scaling) -> float:
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
            nrounds = int(math.ceil(ntasks / i))
            ncores = ncpu // i
            optimal[i] = nrounds / INTERFACE.parallel_speedup(ncores, scaling)
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
        return nrounds, nslots, cpu_per_run

    @staticmethod
    def stripWORKDIR(WORKDIR, keep):
        for ifile in os.listdir(WORKDIR):
            if any([containsstring(k, ifile) for k in keep]):
                continue
            rmfile = os.path.join(WORKDIR, ifile)
            os.remove(rmfile)


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

    # ======================================================================= #
    def run_theodore(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')
        string = 'python %s/bin/analyze_tden.py' % (QMin['theodir'])
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
        step = QMin['step']
        # do Dyson calculations
        if 'ion' in QMin:
            for ionpair in QMin['ionmap']:
                WORKDIR = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i' % ionpair)
                files = {
                    'aoovl': 'AO_overl',
                    'det.a': f'dets.{ionpair[0]}.{step}',
                    'det.b': f'dets.{ionpair[2]}.{step}',
                    'mo.a': f'mos.{ionpair[1]}.{step}',
                    'mo.b': f'mos.{ionpair[3]}.{step}'
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
                    'det.a': f'dets.{m}.{step - 1}',
                    'det.b': f'dets.{m}.{step}',
                    'mo.a': f'mos.{job}.{step - 1}',
                    'mo.b': f'mos.{job}.{step}'
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
            raise Error('Some subprocesses did not finish successfully!', 100)

        print('')

        return errorcodes

    # ======================================================================= #

    def setupWORKDIR_WF(WORKDIR, QMin, files, DEBUG=False):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

        # setup the directory
        mkdir(WORKDIR)

        # write wfovl.inp
        inputstring = 'mix_aoovl=aoovl\na_mo=mo.a\nb_mo=mo.b\na_det=det.a\nb_det=det.b\na_mo_read=0\nb_mo_read=0\nao_read=0'
        if 'ion' in QMin:
            if QMin['numocc'] > 0:
                inputstring += 'numocc=%i\n' % (QMin['numocc'])
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
        except OSError as e:
            raise Error(f'Call have had some serious problems:\n\t{e}', 101)
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
        if 'comment' in QMin:
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
        '''
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
                printgrad(QMout['grad'][istate], natom, QMin['elements'], self._DEBUG)
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
        for i in QMin['resources']['theodore_prop']:
            string += '%6s ' % i
        for i in range(len(QMin['resources']['theodore_fragment'])):
            for j in range(len(QMin['resources']['theodore_fragment'])):
                string += '  Om%1i%1i ' % (i + 1, j + 1)
        string += '\n' + '-------' * (1 + QMin['resources']['theodore_n']) + '\n'
        istate = 0
        for imult, i, ms in itnmstates(QMin['states']):
            istate += 1
            string += '%6i ' % istate
            for i in matrix[istate - 1]:
                string += '%6.4f ' % i.real
            string += '\n'
        print(string)


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
        k = QMinfilename.rfind('.')
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
        if 'theodore' in QMin or 'qmmm' in QMin['template'] and QMin['template']['qmmm']:
            string += self.writeQMoutTHEODORE()
        if 'phases' in QMin:
            string += self.writeQmoutPhases()
        if 'grad' in QMin:
            if 'cobramm' in QMin['template'] and QMin['template']['cobramm']:
                string += self.writeQMoutgradcobramm()
        if 'multipolar_fit' in QMin:
            string += self.writeQMoutmultipolarfit()
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
        nprop = QMin['resources']['theodore_n']
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
            for i in QMin['resources']['theodore_prop']:
                descriptors.append('%s' % i)
                string += descriptors[-1] + '\n'
            for i in range(len(QMin['resources']['theodore_fragment'])):
                for j in range(len(QMin['resources']['theodore_fragment'])):
                    descriptors.append('Om_{%i,%i}' % (i + 1, j + 1))
                    string += descriptors[-1] + '\n'
        if QMin['template']['qmmm']:
            for label in sorted(QMin['qmmm']['MMEnergy_terms']):
                descriptors.append(label)
                string += label + '\n'

        string += '! Property Vectors (%ix%i, real)\n' % (nprop, nmstates)
        if 'theodore' in QMin:
            for i in range(QMin['resources']['theodore_n']):
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

    def writeQMoutmultipolarfit(self):
        '''Generates a string with the fitted RESP charges for each pair of states specified.

        The string starts with a ! followed by a flag specifying the type of data.
        Each line starts with the atom number (starting at 1), state i and state j.
        If i ==j: fit for single state, else fit for transition multipoles.
        One line per atom and a blank line at the end.

        Returns:
        1 string: multiline string with the Gradient vectors'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        fits = QMout['multipolar_fit']
        string = f'! 22 Atomwise multipolar density representation fits for states ({nmstates}x{nmstates}x{natom}x10)\n'

        for i, (imult, istate, ims) in zip(range(nmstates), itnmstates(states)):
            for j, (jmult, jstate, jms) in zip(range(nmstates), itnmstates(states)):
                string += f'{natom} 10 ! m1 {imult} s1 {istate} ms1 {ims: 3.1f}   m2 {jmult} s2 {jstate} ms2 {jms: 3.1f}\n'

                entry = np.zeros((natom, 10))
                if (imult, istate, jmult, jstate) in fits:
                    fit = fits[(imult, istate, jmult, jstate)]
                    entry[:, :fit.shape[1]] = fit  # cath cases where fit is not full order
                elif (jmult, jstate, imult, istate) in fits:
                    fit = fits[(jmult, jstate, imult, istate)]
                    entry[:, :fit.shape[1]] = fit  # cath cases where fit is not full order
                string += "\n".join(map(lambda x: " ".join(map(lambda y: '{: 10.8f}'.format(y), x)), entry)) + '\n'


                string += ''
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
