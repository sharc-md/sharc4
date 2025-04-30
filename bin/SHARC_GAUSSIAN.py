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


# external
import datetime
import math
import os
import shutil
import subprocess as sp
import sys
import traceback
from copy import deepcopy
from io import TextIOWrapper
from itertools import chain
from typing import Optional
import ast

import numpy as np
from constants import IAn2AName, IToMult, au2eV, au2a
from pyscf import gto

# internal
from qmin import QMin as QMin_class
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import (
    expand_path,
    build_basis_dict,
    containsstring,
    itmult,
    link,
    makecmatrix,
    makermatrix,
    mkdir,
    readfile,
    shorten_DIR,
    strip_dir,
    triangular_to_full_matrix,
    writefile,
    question
    #  number_of_bubble_swaps,
)

np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"})

__all__ = ["SHARC_GAUSSIAN"]

AUTHORS = "Sebastian Mai, Severin Polonius, Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 4, 1)
NAME = "GAUSSIAN"
DESCRIPTION = "AB INITIO interface for GAUSSIAN16 for single-reference methods (CIS, TDDFT)"

CHANGELOGSTRING = """
"""
all_features = {
    "h",
    "dm",
    "grad",
    "overlap",
    "phases",
    "ion",
    "multipolar_fit",
    "theodore",
    "point_charges",
    # raw data request
    "mol",
    "wave_functions",
    "density_matrices",
}


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
        "rtype": "cclib",
        "rfile": "GAUSSIAN.log",
        "read_binary": False,
        "jmol_orbitals": False,
        "molden_orbitals": False,
        "Om_formula": 2,
        "eh_pop": 1,
        "comp_ntos": True,
        "print_OmFrag": True,
        "output_file": "tden_summ.txt",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "keys": None,
                "basis": "6-31G",
                "functional": "PBEPBE",
                "dispersion": None,
                "scrf": None,
                "grid": None,
                "denfit": False,
                "scf": None,
                "no_tda": False,
                "td_conv": 6,
                "unrestricted_triplets": False,
                "iop": None,
                "paste_input_file": None,
                "basis_external": None,
                "noneqsolv": False,
                "state_densities": 'relaxed',
                "neglected_gradient": "zero",
            }
        )
        self.QMin.template.types.update(
            {
                "keys": str,
                "basis": str,
                "functional": str,
                "dispersion": str,
                "scrf": list,
                "grid": str,
                "denfit": bool,
                "scf": list,
                "no_tda": bool,
                "td_conv": int,
                "unrestricted_triplets": bool,
                "iop": list,
                "paste_input_file": str,
                "basis_external": str,
                "noneqsolv": bool,
                "state_densities": str,
                "neglected_gradient": str,
            }
        )

        # Add resource keys
        self.QMin.resources.update(
            {
                "groot": None,
                "numfrozcore": 0,
                "schedule_scaling": 0.9,
                "dry_run": False,
                "debug": False,
                "min_cpu": 1,
            }
        )

        self.QMin.resources.types.update(
            {
                "groot": str,
                "numfrozcore": int,
                "schedule_scaling": float,
                "dry_run": bool,
                "debug": bool,
                "min_cpu": int,
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

    @staticmethod
    def check_template(template_file):

        necessary = {"basis", "functional"}
        with open(template_file, "r") as f:
            for line in f:
                if len(necessary) == 0:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    continue
                lspt = line.split()
                if len(lspt) == 0:
                    continue
                elif line.split()[0] in necessary:
                    necessary.remove(line.split()[0])
        return not len(necessary) != 0


    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'GAUSSIAN interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        self.files = []


        self.log.info(f"{'Path to GAUSSIAN':-^60s}\n")
        tries = ['g16root', 'g09root', 'g03root']
        for i in tries:
            path = os.getenv(i)
            if path:
                break
        self.log.info('\nPlease specify path to GAUSSIAN directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n')
        self.setupINFOS['groot'] = question('Path to GAUSSIAN:', str, KEYSTROKES=KEYSTROKES, default=path)
        self.log.info('')


        # scratch
        self.log.info('{:-^60}'.format('Scratch directory') + '\n')
        self.log.info('Please specify an appropriate scratch directory. This will be used to run the GAUSSIAN calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.')
        self.setupINFOS['scratchdir'] = question('Path to scratch directory:', str, KEYSTROKES=KEYSTROKES)
        self.log.info('')


        # template file
        self.template_file = None
        self.log.info('{:-^60}'.format('GAUSSIAN input template file') + '\n')
        self.log.info('''Please specify the path to the GAUSSIAN.template file. This file must contain the following keywords:

    basis <basis>
    functional <type> <name>

    The GAUSSIAN interface will generate the appropriate GAUSSIAN input automatically.
    ''')
        if os.path.isfile('GAUSSIAN.template'):
            if SHARC_GAUSSIAN.check_template('GAUSSIAN.template'):
                self.log.info('Valid file "GAUSSIAN.template" detected. ')
                usethisone = question('Use this template file?', bool, KEYSTROKES=KEYSTROKES, default=True)
                if usethisone:
                    self.template_file = 'GAUSSIAN.template'
        if not self.template_file:
            while True:
                filename = question('Template filename:', str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(filename):
                    self.log.info('File %s does not exist!' % (filename))
                    continue
                if SHARC_GAUSSIAN.check_template(filename):
                    self.template_file = filename
                    break
        self.log.info(f"Expanding {self.template_file} to ...")
        self.template_file = expand_path(self.template_file)
        self.log.info(f"... {self.template_file}")
        
        self.log.info('')
        self.files.append(self.template_file)
        extra_file_keys = {"basis_external", "paste_input_file"}
        with open(self.template_file, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    continue
                lspt = line.split()
                if len(lspt) == 0:
                    continue
                if lspt[0] in extra_file_keys:
                    self.files.append(lspt[1])

        # initial MOs
        self.log.info('{:-^60}'.format('Initial restart: MO Guess') + '\n')
        self.log.info('''Please specify the path to an GAUSSIAN chk file containing suitable starting MOs for the GAUSSIAN calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
    ''')
        self.guess_file = None
        if question('Do you have a restart file?', bool, KEYSTROKES=KEYSTROKES, default=True):
            if True:
                while True:
                    filename = question('Restart file:', str, KEYSTROKES=KEYSTROKES, default='GAUSSIAN.chk.init')
                    if os.path.isfile(filename):
                        self.guess_file = expand_path(filename)
                        break
                    else:
                        self.log.info('Could not find file "%s"!' % (filename))

        # Resources
        # TODO
        if question("Do you have a 'GAUSSIAN.resources' file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            while True:
                resources_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="GAUSSIAN.resources")
                if os.path.isfile(resources_file):
                    break
                else:
                    self.log.info(f"file at {resources_file} does not exist!")
            self.files.append(expand_path(resources_file))
            self.make_resources = False
        else:
            self.make_resources = True
            self.log.info('{:-^60}'.format('GAUSSIAN Ressource usage') + '\n')
            self.log.info('''Please specify the number of CPUs to be used by EACH calculation.
        ''')
            self.setupINFOS['ncpu'] = abs(question('Number of CPUs:', int, KEYSTROKES=KEYSTROKES)[0])

            if self.setupINFOS['ncpu'] > 1:
                self.log.info('''Please specify how well your job will parallelize.
        A value of 0 means that running in parallel will not make the calculation faster, a value of 1 means that the speedup scales perfectly with the number of cores.
        Typical values for GAUSSIAN are 0.90-0.98.''')
                self.setupINFOS['scaling'] = min(1.0, max(0.0, question('Parallel scaling:', float, default=[0.9], KEYSTROKES=KEYSTROKES)[0]))
            else:
                self.setupINFOS['scaling'] = 0.9

            self.setupINFOS['mem'] = question('Memory (MB):', int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

            # Ionization
            # self.log.info('\n'+centerstring('Ionization probability by Dyson norms',60,'-')+'\n')
            # INFOS['ion']=question('Dyson norms?',bool,False)
            # if INFOS['ion']:
            if 'overlap' in INFOS['needed_requests']:
                self.log.info('\n' + '{:-^60}'.format('WFoverlap setup') + '\n')
                self.setupINFOS['wfoverlap'] = question('Path to wavefunction overlap executable:', str, default='$SHARC/wfoverlap.x', KEYSTROKES=KEYSTROKES)
                self.log.info('')
                self.log.info('State threshold for choosing determinants to include in the overlaps')
                self.log.info('For hybrids without TDA one should consider that the eigenvector X may have a norm larger than 1')
                self.setupINFOS['ciothres'] = question('Threshold:', float, default=[0.998], KEYSTROKES=KEYSTROKES)[0]
                self.log.info('')
                # TODO not asked: numfrozcore and numocc

                # self.log.info('Please state the number of core orbitals you wish to freeze for the overlaps (recommended to use for at least the 1s orbital and a negative number uses default values)?')
                # self.log.info('A value of -1 will use the defaults used by GAUSSIAN for a small frozen core and 0 will turn off the use of frozen cores')
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
            # INFOS['theodore']=question('TheoDORE analysis?',bool,False)
            if 'theodore' in INFOS['needed_requests']:
                self.log.info('\n' + '{:-^60}'.format('Wave function analysis by TheoDORE') + '\n')

                self.setupINFOS['theodore'] = question('Path to TheoDORE directory:', str, default='$THEODIR', KEYSTROKES=KEYSTROKES)
                self.log.info('')

                self.log.info('Please give a list of the properties to calculate by TheoDORE.\nPossible properties:')
                string = ''
                for i, p in enumerate(theodore_spelling):
                    string += '%s ' % (p)
                    if (i + 1) % 8 == 0:
                        string += '\n'
                self.log.info(string)
                line = question('TheoDORE properties:', str, default='Om  PRNTO  S_HE  Z_HE  RMSeh', KEYSTROKES=KEYSTROKES)
                if '[' in line:
                    self.setupINFOS['theodore.prop'] = ast.literal_eval(line)
                else:
                    self.setupINFOS['theodore.prop'] = line.split()
                self.log.info('')

                self.log.info('Please give a list of the fragments used for TheoDORE analysis.')
                self.log.info('You can use the list-of-lists from dens_ana.in')
                self.log.info('Alternatively, enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
                self.setupINFOS['theodore.frag'] = []
                while True:
                    line = question('TheoDORE fragment:', str, default='end', KEYSTROKES=KEYSTROKES)
                    if 'end' in line.lower():
                        break
                    if '[' in line:
                        try:
                            INFOS['theodore.frag'] = ast.literal_eval(line)
                            break
                        except ValueError:
                            continue
                    f = [int(i) for i in line.split()]
                    self.setupINFOS['theodore.frag'].append(f)
                self.setupINFOS['theodore.count'] = len(self.setupINFOS['theodore.prop']) + len(self.setupINFOS['theodore.frag'])**2

        return INFOS

    def prepare(self, INFOS: dict, workdir: str):
        """
        prepare the workdir according to dictionary

        ---
        Parameters:
        INFOS: dictionary with infos
        workdir: path to workdir
        """
        if self.make_resources:
            try:
                resources_file = open('%s/GAUSSIAN.resources' % (workdir), 'w')
            except IOError:
                self.log.error('IOError during prepareGAUSSIAN, iconddir=%s' % (workdir))
                quit(1)
#  project='GAUSSIAN'
            string = 'groot %s\nscratchdir %s/%s/\nncpu %i\nschedule_scaling %f\n' % (self.setupINFOS['groot'], self.setupINFOS['scratchdir'], workdir, self.setupINFOS['ncpu'], self.setupINFOS['scaling'])
            string += 'memory %i\n' % (self.setupINFOS['mem'])
            if 'overlap' in INFOS['needed_requests']:
                string += 'wfoverlap %s\nwfthres %f\n' % (self.setupINFOS['wfoverlap'], self.setupINFOS['ciothres'])
                # string+='numfrozcore %i\n' %(INFOS['frozcore_number'])
            if 'theodore' in INFOS['needed_requests']:
                string += 'theodir %s\n' % (self.setupINFOS['gaussian.theodore'])
                string += 'theodore_prop %s\n' % (self.setupINFOS['theodore.prop'])
                string += 'theodore_fragment %s\n' % (self.setupINFOS['theodore.frag'])
            resources_file.write(string)
            resources_file.close()

        create_file = link if INFOS["link_files"] else shutil.copy
        for file in self.files:
            self.log.info(f"Processing {file} to {workdir} as {file.split('/')[-1]}")
            create_file(file, os.path.join(workdir, file.split("/")[-1]))
        if self.guess_file is not None:
            create_file(self.guess_file, "GAUSSIAN.chk.init")


    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

    def read_resources(self, resources_file: str = "GAUSSIAN.resources", kw_whitelist: Optional[list[str]] = None) -> None:
        super().read_resources(resources_file, kw_whitelist)
        self.QMin.resources["gaussian_version"] = self.getVersion(self.QMin.resources["groot"])
        self._Gversion = self.QMin.resources["gaussian_version"]

        os.environ["groot"] = self.QMin.resources["groot"]
        os.environ["GAUSS_EXEDIR"] = self.QMin.resources["groot"]
        os.environ["GAUSS_SCRDIR"] = "."
        os.environ["PATH"] = "$GAUSS_EXEDIR:" + os.environ["PATH"]
        self.QMin.resources["GAUSS_EXEDIR"] = self.QMin.resources["groot"]
        self.QMin.resources["GAUSS_EXE"] = os.path.join(self.QMin.resources["groot"], "g%s" % self._Gversion)

        if self.QMin.requests["grad"]:
            match self.QMin.template["neglected_gradient"].lower():
                case "zero" | "gs" | "closest":
                    self.QMin.template["neglected_gradient"] = self.QMin.template["neglected_gradient"].lower()
                case _:
                    self.log.error(
                        f'Argument to "neglected_gradient" not in ["zero", "gs", "closes"]: {self.QMin.template["neglected_gradient"]}!'
                    )
                    raise ValueError()

        self._read_resources = True
        self.log.info("Detected GAUSSIAN version %s" % self._Gversion)

    def read_template(self, template_file: str = "GAUSSIAN.template") -> None:
        super().read_template(template_file)

        # Convert keys to string if list
        #  if isinstance(self.QMin.template["keys"], list):
          #  self.QMin.template["keys"] = " ".join(self.QMin.template["keys"])

        # read external basis set
        if self.QMin.template["basis_external"]:
            self.QMin.template["basis"] = "gen"
            self.QMin.template.types["basis_external"] = list
            self.QMin.template["basis_external"] = readfile(self.QMin.template["basis_external"])
        if self.QMin.template["paste_input_file"]:
            self.QMin.template.types["paste_input_file"] = list
            self.QMin.template["paste_input_file"] = readfile(self.QMin.template["paste_input_file"])
        # do logic checks
        if not self.QMin.template["unrestricted_triplets"]:
            if len(self.QMin.molecule["charge"]) >= 3 and self.QMin.molecule["charge"][0] != self.QMin.molecule["charge"][2]:
                raise RuntimeError('Charges of singlets and triplets differ. Please enable the "unrestricted_triplets" option!')

        for s in self.states:
            s.C['is_gs'] = False
            s.C['its_gs'] = None
            if s.N == 1: 
                s.C['is_gs'] = True
                if s.S == 2 and not self.QMin.template["unrestricted_triplets"]:
                    s.C['is_gs'] = False

        for s in self.states:
            if not s.C['is_gs']:
                if s.S == 2:
                    if self.QMin.template["unrestricted_triplets"]:
                        for gs in self.states:
                            if gs.C['is_gs'] and gs.M == s.M:
                                s.C['its_gs'] = gs
                                break
                    elif s.M == 0:
                        for gs in self.states:
                            if gs.C['is_gs'] and gs.S == 0:
                                s.C['its_gs'] = gs
                                break
                else:
                    for gs in self.states:
                        if gs.C['is_gs'] and gs.S == s.S and gs.M == s.M:
                            s.C['its_gs'] = gs
                            break
        #for s in self.states:
        #    print(repr(s))


    # TODO
    def setup_interface(self):
        super().setup_interface()
        # make the chargemap
        self.QMin.maps["chargemap"] = {i + 1: c for i, c in enumerate(self.QMin.molecule["charge"])}
        self._states_to_do()  # can be different in interface -> general method here with possibility to overwrite
        # make the jobs
        self._jobs()
        jobs = self.QMin.control["jobs"]
        # make the multmap (mapping between multiplicity and job)
        multmap = {}
        for ijob, job in jobs.items():
            for imult in job["mults"]:
                multmap[imult] = ijob
            multmap[-(ijob)] = job["mults"]
        multmap[1] = 1
        self.QMin.maps["multmap"] = multmap

        # get the joblist
        self.QMin.control["joblist"] = sorted(jobs.keys())
        self.QMin.control["njobs"] = len(self.QMin.control["joblist"])

        # make the gsmap
        gsmap = {}
        for i in range(self.QMin.molecule["nmstates"]):
            m1, _, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
            gs = (m1, 1, ms1)
            job = self.QMin.maps["multmap"][m1]
            if m1 == 3 and self.QMin.control["jobs"][job]["restr"]:
                gs = (1, 1, 0.0)
            for j in range(self.QMin.molecule["nmstates"]):
                m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                if (m2, s2, ms2) == gs:
                    break
            gsmap[i + 1] = j + 1
        self.QMin.maps["gsmap"] = gsmap

    def _request_logic(self) -> None:
        super()._request_logic()
        # make the ionmap
        if self.QMin.requests["ion"]:
            ionmap = []
            for m1 in itmult(self.QMin.molecule["states"]):
                job1 = self.QMin.maps["multmap"][m1]
                el1 = self.QMin.maps["chargemap"][m1]
                for m2 in itmult(self.QMin.molecule["states"]):
                    if m1 >= m2:
                        continue
                    job2 = self.QMin.maps["multmap"][m2]
                    el2 = self.QMin.maps["chargemap"][m2]
                    if abs(m1 - m2) == 1 and abs(el1 - el2) == 1:
                        ionmap.append((m1, job1, m2, job2))
            self.QMin.maps["ionmap"] = ionmap

    @staticmethod
    def getVersion(groot: str) -> str:
        tries = {"g09": "09", "g16": "16"}
        ls = os.listdir(groot)
        for key, val in tries.items():
            if key in ls:
                if os.path.isdir(os.path.join(groot,key)):
                    raise RuntimeError(f"The path $groot{key} is a directory!")
                return val
        raise RuntimeError(f"Found no executable (possible names: {list(tries)}) in $groot!")

    def _jobs(self) -> None:
        # make the jobs
        jobs = {}
        if self.QMin.control["states_to_do"][0] > 0:
            jobs[1] = {"mults": [1], "restr": True}
        if len(self.QMin.control["states_to_do"]) >= 2 and self.QMin.control["states_to_do"][1] > 0:
            jobs[2] = {"mults": [2], "restr": False}
        if len(self.QMin.control["states_to_do"]) >= 3 and self.QMin.control["states_to_do"][2] > 0:
            if not self.QMin.template["unrestricted_triplets"] and self.QMin.control["states_to_do"][0] > 0:
                # jobs[1]['mults'].append(3)
                jobs[3] = {"mults": [1, 3], "restr": True}
            else:
                jobs[3] = {"mults": [3], "restr": False}
        if len(self.QMin.control["states_to_do"]) >= 4:
            for imult, nstate in enumerate(self.QMin.control["states_to_do"][3:]):
                if nstate > 0:
                    jobs[len(jobs) + 1] = {"mults": [imult + 4], "restr": False}
        self.QMin.control["jobs"] = jobs

    def _states_to_do(self) -> None:
        # obtain the states to actually compute
        self.QMin.control["states_to_do"] = [s for s in self.QMin.molecule["states"]]
        for i in range(len(self.QMin.molecule["states"])):
            if self.QMin.control["states_to_do"][i] > 0:
                self.QMin.control["states_to_do"][i] += self.QMin["template"]["paddingstates"][i]
        if not self.QMin.template["unrestricted_triplets"]:
            if (
                len(self.QMin.molecule["states"]) >= 3
                and self.QMin.molecule["states"][2] > 0
                and self.QMin.molecule["states"][0] <= 1
            ):
                if self.QMin.requests['soc']:
                    self.QMin.control["states_to_do"][0] = 2
                else:
                    self.QMin.control["states_to_do"][0] = 1

    def _initorbs(self, qmin) -> None:
        # check for initial orbitals
        initorbs = {}
        step = qmin.save["step"]
        if qmin.save["always_guess"]:
            pass
        elif qmin.save["init"] or qmin.save["always_orb_init"]:
            for job in qmin.control["joblist"]:
                filename = os.path.join(qmin.resources["pwd"], "GAUSSIAN.chk.init")
                if os.path.isfile(filename):
                    initorbs[job] = filename
            for job in qmin.control["joblist"]:
                filename = os.path.join(qmin.resources["pwd"], f"GAUSSIAN.chk.{job}.init")
                if os.path.isfile(filename):
                    initorbs[job] = filename
            if qmin.save["always_orb_init"] and len(initorbs) < qmin.control["njobs"]:
                self.log.error("Initial orbitals missing for some jobs!")
                raise RuntimeError()
        elif qmin.save["newstep"]:
            for job in qmin.control["joblist"]:
                filename = os.path.join(qmin.save["savedir"], f"GAUSSIAN.chk.{job}.{step-1}")
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    self.log.error(f"File {filename} missing in savedir!")
                    raise RuntimeError()
        elif qmin.save["samestep"]:
            for job in qmin.control["joblist"]:
                filename = os.path.join(qmin.save["savedir"], f"GAUSSIAN.chk.{job}.{step}")
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    self.log.error(f"File {filename} missing in savedir!")
                    raise RuntimeError()
        qmin.resources["initorbs"] = initorbs
        self.log.debug(qmin.resources["initorbs"])

    def generate_joblist(self) -> None:
        # sort the gradients into the different jobs
        gradjob = {}
        for ijob in self.QMin.control["joblist"]:
            gradjob[f"master_{ijob}"] = {}
        if self.QMin.requests["grad"]:
            for m_grad, s_grad in sorted(self.QMin.maps["gradmap"], key=lambda x: x[0] * 1000 + x[1]):
                ijob = self.QMin.maps["multmap"][m_grad]
                isgs = False
                istates = self.QMin.control["states_to_do"][m_grad - 1]
                if not self.QMin.control["jobs"][ijob]["restr"]:
                    if s_grad == 1:
                        isgs = True
                else:
                    if (m_grad, s_grad) == (1, 1):
                        isgs = True
                if isgs and istates > 1:
                    gradjob[f"grad_{m_grad}_{s_grad}"] = {}
                    gradjob[f"grad_{m_grad}_{s_grad}"][(m_grad, s_grad)] = {"gs": True}
                else:
                    if len(gradjob[f"master_{ijob}"]) > 0:
                        gradjob[f"grad_{m_grad}_{s_grad}"] = {}
                        gradjob[f"grad_{m_grad}_{s_grad}"][(m_grad, s_grad)] = {"gs": False}
                    else:
                        gradjob[f"master_{ijob}"][(m_grad, s_grad)] = {"gs": False}

        # make map for states onto gradjobs
        jobgrad = {}
        for job, val in gradjob.items():
            for state in val:
                jobgrad[state] = (job, val[state]["gs"])
        self.QMin.control["jobgrad"] = jobgrad

        self.QMin.control["densjob"] = {}
        if self.QMin.requests["multipolar_fit"] or self.QMin.requests["density_matrices"]:
            for (s1,s2,spin) in self.density_recipes["read"]:
                if spin == 'tot' or spin == 'q':
                    if spin == 'tot': first = 'Total '
                    if spin == 'q': first = 'Spin '
                    if s1.C['is_gs']:
                        second = 'SCF Density'
                    else:
                        second = 'CI Density'
                    keyword = {first + second}
                    master_state = 1 if s1.S == 2 and not self.QMin.template['unrestricted_triplets'] else 2
                    if s1.N <= master_state:
                        dir = 'master_' + str(s1.S+1)
                    else:
                        dir = 'grad_' + str(s1.S + 1) + '_' + str(s1.N)
                        if dir not in gradjob:
                            dir = "dens_" + "_".join(dir.split("_")[1:])
                    self.density_recipes['read'][(s1,s2,spin)] = (dir, keyword)
                elif spin == 'aa' or spin == 'bb':
                    dir = 'master_'+str(s2.S+1)
                    if s1 is s2:
                        keyword = {"Number of ex state dens", "Excited state densities"} 
                        self.density_recipes['read'][(s1,s2,spin)] = (dir, keyword)
                    else:
                        keyword = {"Number of g2e trans dens", "G to E trans densities"} 
                        self.density_recipes['read'][(s1,s2,spin)] = (dir, keyword)

            for key, value in self.density_recipes['read'].items():
                if value[0] in self.QMin.control["densjob"]:
                    self.QMin.control["densjob"][value[0]].append(key)
                else:
                    self.QMin.control["densjob"][value[0]] = [key]

        # add the master calculations
        schedule = []
        self.QMin.control["nslots_pool"] = []
        ntasks = 0
        for i in gradjob:
            if "master" in i:
                ntasks += 1
        _, nslots, cpu_per_run = self.divide_slots(self.QMin.resources["ncpu"], ntasks, self.QMin.resources["schedule_scaling"],
                                                   min_cpu=self.QMin.resources["min_cpu"])
        self.log.debug(f"slots for master: {nslots} {cpu_per_run}")
        memory_per_core = self.QMin.resources["memory"] // self.QMin.resources["ncpu"]
        self.QMin.control["nslots_pool"].append(nslots)
        schedule.append({})
        icount = 0
        for i in sorted(gradjob):
            if "master" in i:
                QMin1 = deepcopy(self.QMin)
                del QMin1.scheduling
                QMin1.control["master"] = True
                QMin1.control["jobid"] = int(i.split("_")[1])
                QMin1.maps["gradmap"] = set(gradjob[i])
                QMin1.resources["ncpu"] = cpu_per_run[icount]
                QMin1.resources["memory"] = memory_per_core * cpu_per_run[icount]
                self.log.debug(f"gradjob: adding to schedule: {i} with {cpu_per_run[icount]} and {memory_per_core * cpu_per_run[icount]}")
                # get the rootstate for the multiplicity as the first excited state
                QMin1.control["rootstate"] = min(
                    1, self.QMin.molecule["states"][self.QMin.maps["multmap"][-QMin1.control["jobid"]][-1] - 1] - 1
                )
                if (
                    3 in self.QMin.maps["multmap"][-QMin1.control["jobid"]]
                    and self.QMin.control["jobs"][QMin1.control["jobid"]]["restr"]
                ):
                    QMin1.control["rootstate"] = 1
                    QMin1.molecule["states"][0] = 1
                    QMin1.control["states_to_do"][0] = 1
                icount += 1
                schedule[-1][i] = QMin1

        # add the gradient calculations
        ntasks = 0
        for i in gradjob:
            if "grad" in i:
                ntasks += 1
        for i in self.QMin.control["densjob"]:
            if "dens" in i:
                ntasks += 1
        if ntasks > 0:
            _, nslots, cpu_per_run = self.divide_slots(
                self.QMin.resources["ncpu"], ntasks, self.QMin.resources["schedule_scaling"], min_cpu=self.QMin.resources["min_cpu"]
            )
            self.log.debug(f"slots for grad: {nslots} {cpu_per_run}")
            self.QMin.control["nslots_pool"].append(nslots)
            schedule.append({})
            icount = 0
            for i in gradjob:
                if "grad" in i:
                    QMin1 = deepcopy(self.QMin)
                    del QMin1.scheduling
                    mult, state = (int(x) for x in i.split("_")[1:])
                    ijob = self.QMin.maps["multmap"][mult]
                    QMin1.control["jobid"] = ijob
                    gsmult = self.QMin.maps["multmap"][-ijob][0]
                    for k in ["h", "soc", "dm", "overlap", "ion"]:
                        QMin1.requests[k] = False
                    for k in ["always_guess", "always_orb_init", "init"]:
                        QMin1.save[k] = False
                    QMin1.maps["gradmap"] = set(gradjob[i])
                    QMin1.resources["ncpu"] = cpu_per_run[icount]
                    QMin1.resources["memory"] = memory_per_core * cpu_per_run[icount]
                    self.log.debug(f"gradjob: adding to schedule: {i} with {cpu_per_run[icount]} and {memory_per_core * cpu_per_run[icount]}")
                    QMin1.control["gradonly"] = True
                    QMin1.control["rootstate"] = state - 1 if gsmult == mult else state  # 1 is first excited state of mult
                    icount += 1
                    schedule[-1][i] = QMin1

            for i in self.QMin.control["densjob"]:
                if "dens" in i:
                    QMin1 = deepcopy(self.QMin)
                    del QMin1.scheduling
                    mult, state = (int(x) for x in i.split("_")[1:])
                    ijob = self.QMin.maps["multmap"][mult]
                    QMin1.control["jobid"] = ijob
                    gsmult = self.QMin.maps["multmap"][-ijob][0]
                    for k in ["h", "soc", "dm", "overlap", "ion"]:
                        QMin1.requests[k] = False
                    for k in ["always_guess", "always_orb_init", "init"]:
                        QMin1.save[k] = False
                    QMin1.resources["ncpu"] = cpu_per_run[icount]
                    self.log.debug(f"densjob: adding to schedule: {i} with {cpu_per_run[icount]} and {memory_per_core * cpu_per_run[icount]}")
                    QMin1.resources["memory"] = memory_per_core * cpu_per_run[icount]
                    QMin1.control["rootstate"] = state - 1 if gsmult == mult else state  # 1 is first excited state of mult
                    QMin1.control["densonly"] = True
                    icount += 1
                    schedule[-1][i] = QMin1
        self.QMin.scheduling["schedule"] = schedule
        return

    def _backupdir(self):
        QMin = self.QMin
        # make name for backup directory
        if QMin.requests["backup"]:
            backupdir = QMin.save["savedir"] + "/backup"
            backupdir1 = backupdir
            i = 0
            while os.path.isdir(backupdir1):
                i += 1
                if QMin.save["step"]:
                    backupdir1 = backupdir + f"/step{QMin.save['step']}_{i}"
                else:
                    backupdir1 = backupdir + "/calc_%i" % (i)
            QMin.requests["backup"] = backupdir1

    # =============================================================================================== #
    # =============================================================================================== #
    # ==================================== GAUSSIAN Job Execution =================================== #
    # =============================================================================================== #
    # =============================================================================================== #

    def run(self):
        self.generate_joblist()
        self.log.debug(f"{self.QMin.scheduling['schedule']}")

        # run all the jobs
        self.log.info(">>>>>>>>>>>>> Starting the GAUSSIAN job execution")
        if not self.QMin.resources["dry_run"]:
            self.runjobs(self.QMin.scheduling["schedule"])

        self.get_mole()
        # Create restart files and garbage collection
        self.create_restart_files()
        self.clean_savedir()    # TODO: why done here? Is in main function and in driver...

        # do all necessary overlap and Dyson calculations
        self._run_wfoverlap()

        # do all necessary Theodore calculations
        if self.QMin.requests["theodore"]:
            self._run_theodore()

    def dry_run(self):
        self.generate_joblist()
        self.log.debug(f"{self.QMin.scheduling['schedule']}")
        # do all necessary overlap and Dyson calculations
        self._run_wfoverlap()
        # do all necessary Theodore calculations
        self._run_theodore()

    def create_restart_files(self):
        self.log.print(">>>>>>>>>>>>> Saving files")
        starttime = datetime.datetime.now()
        for ijobset, jobset in enumerate(self.QMin.scheduling["schedule"]):
            if not jobset:
                continue
            for job in jobset:
                if "master" in job:
                    WORKDIR = os.path.join(self.QMin.resources["scratchdir"], job)
                    if not self.QMin.save["samestep"]:
                        self.saveFiles(WORKDIR, jobset[job])
                    if self.QMin.requests["ion"] and ijobset == 0:
                        self.saveAOmatrix(WORKDIR, self.QMin)
        self.saveGeometry(self.QMin)
        endtime = datetime.datetime.now()
        self.log.info(f"Saving Runtime: {endtime - starttime}\n")

    # ======================================================================= #

    def execute_from_qmin(self, workdir: str, qmin: QMin_class) -> tuple[int, datetime.timedelta]:
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
            exit_code = self.run_program(workdir, f"{qmin.resources['GAUSS_EXE']} < GAUSSIAN.com", "GAUSSIAN.log", "GAUSSIAN.err")
            endtime = datetime.datetime.now()
        except Exception as problem:
            self.log.info("*" * 50 + f"\nException in run_calc({workdir})!")
            self.log.info(f"{traceback.format_exc()}")
            self.log.info("*" * 50 + "\n")
            raise problem
        if strip and exit_code == 0:
            keep = ["GAUSSIAN.com", "GAUSSIAN.err", "GAUSSIAN.log", "GAUSSIAN.chk", "GAUSSIAN.fchk", "GAUSSIAN.rwf"]
            strip_dir(workdir, keep_files=keep)

        delta = str(endtime - starttime)
        return exit_code, delta

    # ======================================================================= #

    def setupWORKDIR(self, WORKDIR, QMin):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
        # then put the GAUSSIAN.com file
        self._initorbs(QMin)
        # setup the directory
        mkdir(WORKDIR)

        # write GAUSSIAN.com
        inputstring = SHARC_GAUSSIAN.writeGAUSSIANinput(QMin)
        filename = os.path.join(WORKDIR, "GAUSSIAN.com")
        writefile(filename, inputstring)
        self.log.debug(f"================== DEBUG input file for WORKDIR {shorten_DIR(WORKDIR)} =================")
        self.log.debug(inputstring)
        self.log.debug(f"GAUSSIAN input written to: {filename}")
        self.log.debug("====================================================================")

        # wf file copying
        if QMin.control["master"]:
            job = QMin.control["jobid"]
            if job in QMin.resources["initorbs"]:
                fromfile = QMin.resources["initorbs"][job]
                tofile = os.path.join(WORKDIR, "GAUSSIAN.chk")
                shutil.copy(fromfile, tofile)
        elif QMin.requests["grad"] or QMin.control["densonly"]:
            job = QMin.control["jobid"]
            fromfile = os.path.join(QMin.resources["scratchdir"], f"master_{job}", "GAUSSIAN.chk")
            tofile = os.path.join(WORKDIR, "GAUSSIAN.chk")
            shutil.copy(fromfile, tofile)

        return

    # ======================================================================= #

    @staticmethod
    def writeGAUSSIANinput(QMin):
        """

        Args:
            QMin ():

        Returns:

        """
        # general setup
        job = QMin.control["jobid"]
        gsmult = QMin.maps["multmap"][-job][0]
        restr = QMin.control["jobs"][job]["restr"]
        charge = QMin.maps["chargemap"][gsmult]

        # determine the root in case it was not determined in schedule jobs
        if "rootstate" not in QMin.control:
            QMin.control["rootstate"] = min(1, QMin.molecule["states"][QMin.maps["multmap"][-QMin.control["jobid"]][-1] - 1] - 1)
            if 3 in QMin.maps["multmap"][-QMin.control["jobid"]] and QMin.control["jobs"][QMin.control["jobid"]]["restr"]:
                QMin.control["rootstate"] = 1

        # excited states to calculate
        states_to_do = QMin.control["states_to_do"]
        for imult, _ in enumerate(states_to_do):
            if not imult + 1 in QMin.maps["multmap"][-job]:
                states_to_do[imult] = 0
        states_to_do[gsmult - 1] -= 1

        # do minimum number of states for gradient jobs
        if QMin.control["gradonly"]:
            gradmult = list(QMin.maps["gradmap"])[0][0]
            gradstat = list(QMin.maps["gradmap"])[0][1]
            for imult, _ in enumerate(states_to_do):
                if imult + 1 == gradmult:
                    states_to_do[imult] = gradstat - (gradmult == gsmult)
                else:
                    states_to_do[imult] = 0

        # number of states to calculate
        mults_td = ""
        ncalc = max(states_to_do)
        if restr:
            sing = states_to_do[0] > 0
            trip = len(states_to_do) >= 3 and states_to_do[2] > 0
            #  if sing and trip:
                #  mults_td = ",50-50"
            #  elif sing and not trip:
            if sing and not trip:
                mults_td = ",singlets"
            #elif trip and not sing:
            elif trip:
                mults_td = ",triplets"

        # gradients
        if QMin.maps["gradmap"]:
            dograd = True
            root = QMin.control["rootstate"]
        else:
            dograd = False

        dodens = False
        # TODO only activate if state is requested (i.e. is in densmap)
        if QMin.requests["multipolar_fit"] or QMin.requests["density_matrices"] or dograd:
            dodens = True
            root = QMin.control["rootstate"]

        # construct the input string TODO
        string = ""

        # link 0
        string += f"%MEM={QMin.resources['memory']}MB\n"
        string += f"%NProcShared={QMin.resources['ncpu']}\n"
        string += "%Chk=GAUSSIAN.chk\n"
        if "AOoverlap" in QMin.control or QMin.requests["ion"]:
            string += "%Rwf=GAUSSIAN.rwf\n"
            if "AOoverlap" in QMin.control:
                string += "%KJob l302\n"
        string += "\n"

        # Route section
        data = ["p", "nosym", "unit=AU"]
        if not QMin.template['functional'].lower() == 'eomccsd': data.append( QMin.template['functional'] )
        if not QMin.template["functional"].lower() == "dftba":
            data.append(QMin.template["basis"])
        if dograd:
            data.append("force")
        if "AOoverlap" in QMin.control:
            data.append("IOP(2/12=3)")
        if QMin.template["dispersion"]:
            data.append(f"EmpiricalDispersion={QMin.template['dispersion']}")
        if QMin.template["grid"]:
            data.append(f"int(grid={QMin.template['grid']})")
        if QMin.template["denfit"]:
            data.append("denfit")
        if ncalc > 0:
            if QMin.template['functional'].lower() == 'eomccsd':
                s = f"eomccsd(nstates={ncalc}{mults_td}"
            elif not QMin.template["no_tda"]:
                s = f"tda(nstates={ncalc}{mults_td}"
            else:
                s = f"td(nstates={ncalc}{mults_td}"
            #  if QMin.control["master"] or QMin.template['functional'].lower() == 'eomccsd':
                #  s += f"(nstates={ncalc}{mults_td}"
            if not QMin.control["master"] and not QMin.template['functional'].lower() == 'eomccsd':
                s += ",read"
            if dograd and root > 0:
                s += f",root={root}"
            elif dodens and root > 0:
                s += f",root={root}"
            if QMin.template["td_conv"]:
                s += f",conver={QMin.template['td_conv']}"
            if QMin.template["noneqsolv"]:
                s += ",noneqsolv"
            s += ") "
            if dodens and root > 0 and QMin.template['state_densities'] == 'relaxed': 
                s += "density=Current"
            data.append(s)
        if QMin.template["scrf"]:
            s = ",".join(QMin.template["scrf"])
            data.append("scrf(%s)" % s)
        if QMin.template["scf"]:
            s = ",".join(QMin.template["scf"])
            data.append("scf(%s)" % s)
        if QMin.template["iop"]:
            s = ",".join(QMin.template["iop"])
            data.append("iop(%s)" % s)
        if QMin.template["keys"]:
            data.extend([QMin.template["keys"]])
        if QMin.control["gradonly"]:
            data.append("Guess=read")
        if QMin.control["densonly"]:
            data.append("Guess=read")
            if QMin.template['functional'].lower() == 'eomccsd':
                data.append('force')
            else:
                data.append("pop=Regular")  # otherwise CI density will not be printed
        if QMin.requests["theodore"]:
            data.append("pop=full")
            data.append("IOP(9/40=3)")
        if QMin.requests["mol"] or QMin.requests["density_matrices"] or QMin.requests["multipolar_fit"]:
            data.append("GFINPUT")
        if QMin.molecule['point_charges']:
            data.append('charge')
            if dograd:
                data.append('prop=(field,read)')
            # TODO: also add prop=(field, read) and give the point charges a second time to get gradients

        # data.append("GFPRINT")
        string += "#"
        for i in data:
            string += i + "\n"
        # title
        string += "\nSHARC-GAUSSIAN job\n\n"

        # charge/mult and geometry
        if "AOoverlap" in QMin.control:
            string += f"{2 * charge} 1\n"
        else:
            string += f"{charge} {gsmult}\n"
        for label, coords in zip(QMin.molecule["elements"], QMin.coords["coords"]):
            string += f"{label:>4s} {coords[0]:16.15f} {coords[1]:16.15f} {coords[2]:16.15f}\n"
        string += "\n"
        if QMin.molecule['point_charges']:
            for a in range(len(QMin.coords['pccharge'])):
                pccoord = QMin.coords['pccoords'][a,:]
                pccharge = QMin.coords['pccharge'][a]
                string += f"{pccoord[0]:16.15f} {pccoord[1]:16.15f} {pccoord[2]:16.15f} {pccharge:16.15f}\n"
            string += "\n"
            if dograd:
                for a in range(len(QMin.coords['pccharge'])):
                    pccoord = QMin.coords['pccoords'][a,:]
                    string += f"{pccoord[0]*au2a:16.15f} {pccoord[1]*au2a:16.15f} {pccoord[2]*au2a:16.15f}\n"
                string += "\n"
        if QMin.template["functional"].lower() == "dftba":
            string += "@GAUSS_EXEDIR:dftba.prm\n"
        if QMin.template["basis_external"]:
            for line in QMin.template["basis_external"]:
                string += line
            string += "\n"
        if QMin.template["paste_input_file"]:
            # string += '\n'
            for line in QMin.template["paste_input_file"]:
                string += line
        string += "\n\n"

        return string

    # ======================================================================= #

    def saveGeometry(self, qmin: QMin_class) -> None:
        string = ""
        for label, atom in zip(qmin.molecule["elements"], qmin.coords["coords"]):
            string += f"{label:4s} {atom[0]:16.15f} {atom[1]:16.15f} {atom[2]:16.15f}\n"
        filename = os.path.join(qmin.save["savedir"], f'geom.dat.{qmin.save["step"]}')
        writefile(filename, string)
        self.log.print(shorten_DIR(filename))
        return

    # ======================================================================= #

    def saveFiles(self, WORKDIR, qmin: QMin_class):
        # copy the TAPE21 from master directories
        job = qmin.control["jobid"]
        step = qmin.save["step"]
        fromfile = os.path.join(WORKDIR, "GAUSSIAN.chk")
        tofile = os.path.join(qmin.save["savedir"], f"GAUSSIAN.chk.{job}.{step}")
        shutil.copy(fromfile, tofile)
        self.log.debug(shorten_DIR(tofile))

        # if necessary, extract the MOs and CI coefficients and write them to savedir
        if qmin.requests["ion"] or not qmin.requests["nooverlap"] or len(self.density_recipes["calculate"]) > 0:
            f = os.path.join(WORKDIR, "GAUSSIAN.fchk")
            self.get_fchk(WORKDIR, self.QMin.resources["groot"])
            string = SHARC_GAUSSIAN.get_MO_from_chk(f, qmin, self.QMin.molecule['Ubasis'])
            mofile = os.path.join(qmin.save["savedir"], f"mos.{job}.{step}")
            writefile(mofile, string)
            if True:
                string = SHARC_GAUSSIAN.get_MO_from_chk(f, qmin, self.QMin.molecule['Ubasis'], ignorefrozcore=True)
                mofile = os.path.join(qmin.save["savedir"], f"mos_allelec.{job}.{step}")
                writefile(mofile, string)
            self.log.debug(shorten_DIR(mofile))
            f = os.path.join(WORKDIR, "GAUSSIAN.chk")
            strings = SHARC_GAUSSIAN.get_dets_from_chk(f, qmin)
            for f in strings:
                writefile(f, strings[f])
                self.log.print(shorten_DIR(f))
            if True:
                f = os.path.join(WORKDIR, "GAUSSIAN.chk")
                strings = SHARC_GAUSSIAN.get_dets_from_chk(f, qmin, ignorefrozcore=True, filelabel='_allelec')
                for f in strings:
                    writefile(f, strings[f])

    # ======================================================================= #

    @staticmethod
    def get_rwfdump(groot, filename, number):
        WORKDIR = os.path.dirname(filename)
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        dumpname = "rwfdump.txt"
        string = f"{groot}/rwfdump {os.path.basename(filename)} {dumpname} {number}"
        try_shells = ["sh", "bash", "csh", "tcsh"]
        for shell in try_shells:
            try:
                sp.call(string, shell=True, executable=shell)
            except OSError as e:
                raise RuntimeError(f"Gaussian rwfdump has serious problems:\n {e}")
        string = readfile(dumpname)
        os.chdir(prevdir)
        return string

    # ======================================================================= #
    @staticmethod
    def get_MO_from_chk(filename, qmin: QMin_class, U: np.ndarray, ignorefrozcore: bool = False) -> str:
        job = qmin.control["jobid"]
        restr = qmin.control["jobs"][job]["restr"]
        if ignorefrozcore:
            frozcore = 0
        else:
            frozcore = qmin.molecule["frozcore"]
        MO_A = SHARC_GAUSSIAN.parse_fchk(filename, ['Alpha MO coefficients'])['Alpha MO coefficients']
        nao = int(np.sqrt(len(MO_A)))
        MO_A = np.reshape( MO_A, [nao,nao] ).T
        #  qmin.molecule.Ubasis = np.zeros((2,2))
        #print(np.shape(MO_A), np.shape(U))
        MO_A = U @ MO_A
        if not restr:
            MO_B = SHARC_GAUSSIAN.parse_fchk(filename, ['Beta MO coefficients'])['Beta MO coefficients']
            MO_B = np.reshape( MO_B, [nao,nao] ).T 
            MO_B = U @ MO_B 
        MO_A = MO_A[:,frozcore:]
        if not restr:
            MO_B = MO_B[:,frozcore:]

        if restr: 
            NMO = MO_A.shape[1]
        else:
            NMO = MO_A.shape[1] + MO_B.shape[1]
        # make string
        string = """2mocoef
    header
     1
    MO-coefficients from Gaussian
     1
     %i   %i
     a
    mocoef
    (*)
    """ % (
            MO_A.shape[0],
            NMO,
        )
        x = 0
        for imo in range(MO_A.shape[1]):
            for iao in range(MO_A.shape[0]):
                if x >= 3:
                    string += "\n"
                    x = 0
                string += "% 12.16e " % MO_A[iao,imo]
                x += 1
            if x > 0:
                string += "\n"
                x = 0
        if not restr:
            x = 0
            for imo in range(MO_B.shape[1]):
                for iao in range(MO_B.shape[0]):
                    if x >= 3:
                        string += "\n"
                        x = 0
                    string += "% 12.16e " % MO_B[iao,imo]
                    x += 1
                if x > 0:
                    string += "\n"
                    x = 0

        string += "orbocc\n(*)\n"
        n = np.shape(MO_A)[1]
        if not restr: n += np.shape(MO_B)[1]
        x = 0
        for i in range(n):
            if x >= 3:
                string += "\n"
                x = 0
            string += "% 6.12e " % (0.0)
            x += 1

        return string

    # ======================================================================= #

    @staticmethod
    def get_dets_from_chk(filename, QMin, ignorefrozcore: bool = False, filelabel: str = ''):
        # get general infos
        job = QMin.control["jobid"]
        restr = QMin.control["jobs"][job]["restr"]
        mults = QMin.control["jobs"][job]["mults"]
        if ignorefrozcore:
            frozcore = 0
        else:
            frozcore = QMin.molecule["frozcore"]
        if 3 in mults:
            mults = [3]
        gsmult = QMin.maps["multmap"][-job][0]
        nstates_to_extract = QMin.molecule["states"][:]
        for i in range(len(nstates_to_extract)):
            if not i + 1 in mults:
                nstates_to_extract[i] = 0
            elif i + 1 == gsmult:
                nstates_to_extract[i] -= 1

        # get infos from logfile
        logfile = os.path.join(os.path.dirname(filename), "GAUSSIAN.log")
        data = readfile(logfile)
        infos = {}
        for iline, line in enumerate(data):
            if "NBsUse=" in line:
                s = line.split()
                infos["nbsuse"] = int(s[1])
            if "Range of M.O.s used for correlation:" in line:
                for i in [1, 2]:
                    s = data[iline + i].replace("=", " ").split()
                    for j in range(5):
                        infos[s[2 * j]] = int(s[2 * j + 1])

        if "NOA" not in infos:
            nstates_onfile = 0
            charge = QMin.maps["chargemap"][gsmult]
            nelec = float(QMin.molecule["Atomcharge"] - charge)
            infos["NOA"] = int(nelec / 2.0 + float(gsmult - 1) / 2.0)
            infos["NOB"] = int(nelec / 2.0 - float(gsmult - 1) / 2.0)
            infos["NVA"] = infos["nbsuse"] - infos["NOA"]
            infos["NVB"] = infos["nbsuse"] - infos["NOB"]
            infos["NFC"] = 0
        else:
            # get all info from checkpoint
            data = SHARC_GAUSSIAN.get_rwfdump(QMin.resources["groot"], filename, "635R")
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
                        eigenvectors_array.append(float(i.replace("D", "E")))
                    except ValueError:
                        eigenvectors_array.append(float("NaN"))
            nstates_onfile = (len(eigenvectors_array) - 12) // (
                4 + 8 * (infos["NOA"] * infos["NVA"] + infos["NOB"] * infos["NVB"])
            )


        # get ground state configuration
        # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
        if restr:
            occ_A = [3 for i in range(infos["NFC"] + infos["NOA"])] + [0 for i in range(infos["NVA"])]
        if not restr:
            occ_A = [1 for i in range(infos["NFC"] + infos["NOA"])] + [0 for i in range(infos["NVA"])]
            occ_B = [2 for i in range(infos["NFC"] + infos["NOB"])] + [0 for i in range(infos["NVB"])]
        occ_A = tuple(occ_A)
        if not restr:
            occ_B = tuple(occ_B)

        # get infos
        nocc_A = infos["NOA"]
        nvir_A = infos["NVA"]
        nocc_B = infos["NOB"]
        nvir_B = infos["NVB"]

        # get eigenvectors
        eigenvectors = {}
        for mult in mults:
            eigenvectors[mult] = []
            if mult == gsmult:
                # add ground state
                if restr:
                    key = tuple(occ_A[frozcore :])
                else:
                    key = tuple(occ_A[frozcore :] + occ_B[frozcore :])
                eigenvectors[mult].append({key: 1.0})
            for istate in range(nstates_to_extract[mult - 1]):
                # get X+Y vector
                startindex = 12 + istate * (nvir_A * nocc_A + nvir_B * nocc_B)
                endindex = startindex + nvir_A * nocc_A + nvir_B * nocc_B
                eig = [i for i in eigenvectors_array[startindex:endindex]]
                # get X-Y vector
                startindex = (
                    12 + istate * (nvir_A * nocc_A + nvir_B * nocc_B) + 4 * nstates_onfile * (nvir_A * nocc_A + nvir_B * nocc_B)
                )
                endindex = startindex + nvir_A * nocc_A + nvir_B * nocc_B
                eigl = [i for i in eigenvectors_array[startindex:endindex]]
                # get X vector
                for i in range(len(eig)):
                    eig[i] = (eig[i] + eigl[i]) / 2.0
                # make dictionary
                dets = {}
                if restr:
                    for iocc in range(nocc_A):
                        for ivirt in range(nvir_A):
                            index = iocc * nvir_A + ivirt
                            #TOMI
                            #  key = list(occ_A)
                            #  key[infos["NFC"] + iocc] = 2
                            #  key[infos["NFC"] + nocc_A + ivirt] = 1
                            #  mos_alpha = [ i for i,mo in enumerate(key) if mo == 3 or mo == 2 ]
                            #  mos_beta = [ i for i,mo in enumerate(key) if mo == 3 or mo == 1 ]
                            #  mos = mos_alpha + mos_beta
                            #  nswap = number_of_bubble_swaps(mos)
                            #  dets[(iocc, ivirt, 1)] = (-1.)**nswap*eig[index]
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
                norm = 0.0
                for k in sorted(dets, key=lambda x: dets[x] ** 2, reverse=True):
                    if restr:
                        factor = 0.5
                    else:
                        factor = 1.0
                    if norm > factor * QMin.resources["wfthres"]:
                        del dets[k]
                        continue
                    norm += dets[k] ** 2
                # create strings and expand singlets
                dets2 = {}
                if restr:
                    for iocc, ivirt, dummy in dets:
                        # singlet
                        if mult == 1:
                            # alpha excitation
                            key = list(occ_A)
                            key[infos["NFC"] + iocc] = 2
                            key[infos["NFC"] + nocc_A + ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                            # beta excitation
                            key[infos["NFC"] + iocc] = 1
                            key[infos["NFC"] + nocc_A + ivirt] = 2
                            dets2[tuple(key)] = -dets[(iocc, ivirt, dummy)]
                        # triplet
                        elif mult == 3:
                            key = list(occ_A)
                            key[infos["NFC"] + iocc] = 1
                            key[infos["NFC"] + nocc_A + ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] * math.sqrt(2.0)
                else:
                    for iocc, ivirt, dummy in dets:
                        if dummy == 1:
                            key = list(occ_A + occ_B)
                            key[infos["NFC"] + iocc] = 0
                            key[infos["NFC"] + nocc_A + ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                        elif dummy == 2:
                            key = list(occ_A + occ_B)
                            key[2 * infos["NFC"] + nocc_A + nvir_A + iocc] = 0
                            key[2 * infos["NFC"] + nocc_A + nvir_A + nocc_B + ivirt] = 2
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                # remove frozen core
                dets3 = {}
                for key in dets2:
                    problem = False
                    if restr:
                        if any([key[i] != 3 for i in range(frozcore)]):
                            problem = True
                    else:
                        if any([key[i] != 1 for i in range(frozcore)]):
                            problem = True
                        if any(
                            [
                                key[i] != 2
                                for i in range(
                                    nocc_A + nvir_A + frozcore, nocc_A + nvir_A + 2 * frozcore
                                )
                            ]
                        ):
                            problem = True
                    if problem:
                        print("WARNING: Non-occupied orbital inside frozen core! Skipping ...")
                        continue
                        # sys.exit(70)
                    if restr:
                        key2 = key[frozcore :]
                    else:
                        key2 = (
                            key[frozcore : frozcore + nocc_A + nvir_A]
                            + key[nocc_A + nvir_A + 2 * frozcore :]
                        )
                    dets3[key2] = dets2[key]
                # append
                eigenvectors[mult].append(dets3)

        strings = {}
        step = QMin.save["step"]
        for mult in mults:
            filename = os.path.join(QMin.save["savedir"], f"dets{filelabel}.{mult}.{step}")
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

        string = "%i %i %i\n" % (nstates, norb, ndets)
        for det in sorted(alldets, reverse=True):
            for o in det:
                if o == 0:
                    string += "e"
                elif o == 1:
                    string += "a"
                elif o == 2:
                    string += "b"
                elif o == 3:
                    string += "d"
            for istate in range(len(ci_vectors)):
                if det in ci_vectors[istate]:
                    string += " %15.10E " % ci_vectors[istate][det]
                else:
                    string += " %15.10E " % 0.0
            string += "\n"
        return string

    # ======================================================================= #

    def saveAOmatrix(self, WORKDIR, QMin):
        filename = os.path.join(WORKDIR, "GAUSSIAN.rwf")
        NAO, Smat = SHARC_GAUSSIAN.get_smat(filename, QMin.resources["groot"])

        string = "%i %i\n" % (NAO, NAO)
        for irow in range(NAO):
            for icol in range(NAO):
                string += "% .15e " % (Smat[icol][irow])
            string += "\n"
        filename = os.path.join(QMin.save["savedir"], "AO_overl")
        writefile(filename, string)
        self.log.print(shorten_DIR(filename))

    # ======================================================================= #

    @staticmethod
    def get_smat(filename, groot):
        # get all info from checkpoint
        data = SHARC_GAUSSIAN.get_rwfdump(groot, filename, "514R")

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
                Smat.append(float(i.replace("D", "E")))
        NAO = int(math.sqrt(2.0 * len(Smat) + 0.25) - 0.5)

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
        filename1 = os.path.join(self.QMin.save["savedir"], f'geom.dat.{self.QMin.save["step"]-1}')
        oldgeo = SHARC_GAUSSIAN.get_geometry(filename1)

        mol_new: gto.Mole = self.QMin.molecule["mol"]
        mol_old = mol_new.copy()
        atoms = [
            [f"{g[0].upper()}{i+1}", g[1:]] for i, g in enumerate(oldgeo)
        ]
        mol_old.build(atom=atoms)

        mol_conc = gto.conc_mol(mol_old, mol_new)
        mol_conc.build()
        SAO = mol_conc.intor("int1e_ovlp")[:mol_old.nao,mol_old.nao:]
        string = "%i %i\n" % (mol_old.nao, mol_old.nao)
        string += "\n".join(map(lambda row: " ".join(map(lambda f: f"{f: .15e}", row)), SAO))
        filename = os.path.join(self.QMin.save["savedir"], "AO_overl.mixed")
        writefile(filename, string)

        # get geometries
        # filename1 = os.path.join(self.QMin.save["savedir"], f'geom.dat.{self.QMin.save["step"]-1}')
        # oldgeo = SHARC_GAUSSIAN.get_geometry(filename1)
        # filename2 = os.path.join(self.QMin.save["savedir"], f'geom.dat.{self.QMin.save["step"]}')
        # newgeo = SHARC_GAUSSIAN.get_geometry(filename2)

        # # build QMin   # TODO: always singlet for AOoverlaps
        # QMin1 = QMin_class()
        # for name in ["molecule", "coords", "requests", "save", "control", "maps", "resources", "template"]:
            # self.log.trace(f"copying {name}")
            # QMin1[name] = deepcopy(self.QMin[name])
            
        # QMin1.molecule["elements"] = [x[0] for x in chain(oldgeo, newgeo)]
        # QMin1.coords["coords"] = [x[1:] for x in chain(oldgeo, newgeo)]
        # QMin1.control["AOoverlap"] = [filename1, filename2]
        # QMin1.control["jobid"] = self.QMin.control["joblist"][0]
        # QMin1.molecule["natom"] = len(newgeo)
        # QMin1.requests.update({"nacdr":[], "grad": [], "h": False, "soc": False, "dm": False, "overlap": False, "ion":
                               # False})

        # # run the calculation
        # WORKDIR = os.path.join(self.QMin.resources["scratchdir"], "AOoverlap")
        # self.execute_from_qmin(WORKDIR, QMin1)

        # # get output
        # filename = os.path.join(WORKDIR, "GAUSSIAN.rwf")
        # NAO, Smat = SHARC_GAUSSIAN.get_smat(filename, self.QMin.resources["groot"])

        # # adjust the diagonal blocks for DFTB-A
        # if self.QMin.template["functional"] == "dftba":
            # Smat = SHARC_GAUSSIAN.adjust_DFTB_Smat(Smat, NAO, self.QMin)

        # Smat is now full matrix NAO*NAO
        # we want the lower left quarter, but transposed
        # string = "%i %i\n" % (NAO // 2, NAO // 2)
        # for irow in range(NAO // 2, NAO):
            # for icol in range(0, NAO // 2):
                # string += "% .15e " % (Smat[icol][irow])  # note the exchanged indices => transposition
            # string += "\n"
        # filename = os.path.join(self.QMin.save["savedir"], "AO_overl.mixed")
        # writefile(filename, string)
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
        nbasis = {1: ["h", "he"], 2: ["li", "be"], 4: ["b", "c", "n", "o", "f", "ne"]}
        nbs = {}
        for i in nbasis:
            for el in nbasis[i]:
                nbs[el] = i
        Nb = 0
        itot = 0
        mapping = {}
        for ii, i in enumerate(QMin.molecule["elements"]):
            try:
                Nb += nbs[i[0].lower()]
            except KeyError:
                raise RuntimeError("Error: Overlaps with DFTB need further testing!")
            for j in range(nbs[i[0].lower()]):
                mapping[itot] = ii
                itot += 1
        # make interatomic overlap blocks unit matrices
        for i in range(Nb):
            ii = mapping[i]
            for j in range(Nb):
                jj = mapping[j]
                if ii != jj:
                    continue
                if i == j:
                    Smat[i][j + Nb] = 1.0
                    Smat[i + Nb][j] = 1.0
                else:
                    Smat[i][j + Nb] = 0.0
                    Smat[i + Nb][j] = 0.0
        return Smat

    # =============================================================================================== #
    # =============================================================================================== #
    # ====================================== GAUSSIAN output parsing ================================ #
    # =============================================================================================== #
    # =============================================================================================== #

    def getQMout(self):
        self.log.print(">>>>>>>>>>>>> Reading output files")
        starttime = datetime.datetime.now()

        states = self.QMin.molecule["states"]
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]
        joblist = self.QMin.control["joblist"]
        self.QMout.allocate(
            states, natom, self.QMin.molecule["npc"], {r for r in self.QMin.requests.keys() if self.QMin.requests[r]}
        )
        self.QMout['mol'] = self.QMin.molecule['mol'].copy() # PySCF object

        # TODO:
        # excited state energies and transition moments could be read from rwfdump "770R"
        # KS orbital energies: "522R"
        # geometry SEEMS TO BE in "507R"
        # 1TDM might be in "633R"
        # Hamiltonian
        for job in joblist:
            logfile = os.path.join(self.QMin.resources["scratchdir"], "master_%i/GAUSSIAN.log" % (job))
            log_content = readfile(logfile)

            if self.QMin.requests["h"]:  # or 'soc' in self.QMin:
                if self.QMin.molecule['point_charges']:
                    pccoords = self.QMin.coords['pccoords']
                    pccharge = self.QMin.coords['pccharge']
                    npc = len(pccharge) 
                    Eexternal = sum( [ pccharge[a]*pccharge[b]/np.linalg.norm( pccoords[a,:] - pccoords[b,:] ) for a in range(npc) for b in range(a+1,npc) ] )
                else:
                    Eexternal = 0.
                logfile = os.path.join(self.QMin.resources["scratchdir"], "master_%i/GAUSSIAN.log" % (job))
                self.log.print("Energies:  " + shorten_DIR(logfile))
                energies = self.getenergy(log_content, job)
                mults = self.QMin.maps["multmap"][-job]
                if 3 in mults:
                    mults = [3]
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    if m1 not in mults:
                        continue
                    self.QMout["h"][i][i] = energies[(m1, s1)] - Eexternal

            # Dipole Moments
            if self.QMin.requests["dm"]:
                self.log.print("Dipoles:  " + shorten_DIR(logfile))
                dipoles = self.gettdm(log_content, job)
                mults = self.QMin.maps["multmap"][-job]
                mults = self.QMin.maps["multmap"][-job]
                if 3 in mults:
                    mults = [3]
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    if m1 not in self.QMin.control["jobs"][job]["mults"]:
                        continue
                    for j in range(nmstates):
                        m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                        if m2 not in self.QMin.control["jobs"][job]["mults"]:
                            continue
                        if self.QMin.requests["grad"] and i == j and (m1, s1) in self.QMin.maps["gradmap"]:
                            path, isgs = self.QMin.control["jobgrad"][(m1, s1)]
                            logfile = os.path.join(self.QMin.resources["scratchdir"], path, "GAUSSIAN.log")
                            dm = SHARC_GAUSSIAN.getdm(logfile)
                            for ixyz in range(3):
                                self.QMout["dm"][ixyz][i][j] = dm[ixyz]
                        if i == j:
                            continue
                        if not m1 == m2 == mults[0] or not ms1 == ms2:
                            continue
                        if s1 == 1:
                            for ixyz in range(3):
                                self.QMout["dm"][ixyz][i][j] = dipoles[(m2, s2)][ixyz]
                        elif s2 == 1:
                            for ixyz in range(3):
                                self.QMout["dm"][ixyz][i][j] = dipoles[(m1, s1)][ixyz]

        # Gradients
        if self.QMin.requests["grad"]:
            for grad in self.QMin.maps["gradmap"]:
                path, isgs = self.QMin.control["jobgrad"][grad]
                logfile = os.path.join(self.QMin.resources["scratchdir"], path, "GAUSSIAN.log")
                g = self.getgrad(logfile)
                if self.QMin.molecule["point_charges"]:
                    gpc = self.getgrad_pc(logfile)
                for istate in self.QMin.maps["statemap"]:
                    state = self.QMin.maps["statemap"][istate]
                    if (state[0], state[1]) == grad:
                        self.QMout["grad"][istate - 1] = g
                        if self.QMin.molecule["point_charges"]:
                            self.QMout["grad_pc"][istate - 1] = gpc
            if self.QMin.template["neglected_gradient"] != "zero":
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    if not (m1, s1) in self.QMin.maps["gradmap"]:
                        if self.QMin.template["neglected_gradient"] == "gs":
                            j = self.QMin.maps["gsmap"][i + 1] - 1
                        elif self.QMin.template["neglected_gradient"] == "closest":
                            e1 = self.QMout["h"][i][i]
                            de = 999.0
                            for grad in self.QMin.maps["gradmap"]:
                                for k in range(nmstates):
                                    m2, s2, ms2 = tuple(self.QMin.maps["statemap"][k + 1])
                                    if grad == (m2, s2):
                                        break
                                e2 = self.QMout["h"][k][k]
                                if de > abs(e1 - e2):
                                    de = abs(e1 - e2)
                                    j = k
                        self.QMout["grad"][i] = self.QMout["grad"][j]

        # Regular Overlaps
        if self.QMin.requests["overlap"]:
            for mult in itmult(self.QMin.molecule["states"]):
                job = self.QMin.maps["multmap"][mult]
                outfile = os.path.join(self.QMin.resources["scratchdir"], f"WFOVL_{mult}_{job}", "wfovl.out")
                ovlp_mat = self.parse_wfoverlap(outfile)
                self.log.print("Overlaps: " + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                        m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                        if not m1 == m2 == mult:
                            continue
                        if not ms1 == ms2:
                            continue
                        self.QMout["overlap"][i][j] = ovlp_mat[s1 - 1, s2 - 1]

        # Phases from overlaps
        if self.QMin.requests["phases"]:
            if "overlap" in self.QMout:
                for i in range(nmstates):
                    if self.QMout["overlap"][i][i].real < 0.0:
                        self.QMout["phases"][i] = complex(-1.0, 0.0)

        # Dyson norms
        if self.QMin.requests["ion"]:
            if "prop" not in self.QMout:
                self.QMout["prop"] = makecmatrix(nmstates, nmstates)
            for ion in self.QMin.maps["ionmap"]:
                outfile = os.path.join(self.QMin.resources["scratchdir"], "Dyson_%i_%i_%i_%i/wfovl.out" % ion)
                out = readfile(outfile)
                self.log.print("Dyson:    " + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                        m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                        if (ion[0], ion[2]) != (m1, m2) and (ion[0], ion[2]) != (m2, m1):
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
                            factor = (ms1 + 1.0 + (m1 - 1.0) / 2.0) / m1
                        else:
                            factor = (-ms1 + 1.0 + (m1 - 1.0) / 2.0) / m1
                        self.QMout["prop"][i][j] = SHARC_GAUSSIAN.getDyson(out, s1, s2) * factor

        # ====================== Requests that read from fchks ===============================
        # ========================== read from master ======================================
        # get the FCHK file
        masterdir = os.path.join(self.QMin.resources["scratchdir"], f"master_{joblist[0]}")
        self.get_fchk(masterdir, self.QMin.resources["groot"])
        fchk_master = os.path.join(masterdir, "GAUSSIAN.fchk")

        # Densities and multipolar fits
        if ( self.QMin.requests["density_matrices"] or self.QMin.requests["multipolar_fit"] ):
            self.get_densities()
            if self.QMin.requests["multipolar_fit"]:
                self.QMout["multipolar_fit"] = self._resp_fit_on_densities()
                self.log.debug(self.QMout["multipolar_fit"])

        # TheoDORE
        if self.QMin.requests["theodore"]:
            # theodore_arr = np.zeros(
            #     (
            #         self.QMin.molecule["nmstates"],
            #         len(self.QMin.resources["theodore_prop"]) + len(self.QMin.resources["theodore_fragment"]) ** 2,
            #     )
            # )
            nprop = len(self.QMin.resources["theodore_prop"]) + len(self.QMin.resources["theodore_fragment"]) ** 2
            labels = self.QMin.resources["theodore_prop"][:] + [ 'Om_%i_%i' % (i+1,j+1) for i in range(len(self.QMin.resources["theodore_fragment"])) for j in range(len(self.QMin.resources["theodore_fragment"])) ]
            theodore_arr = [ [labels[j], np.zeros(self.QMin.molecule["nmstates"])] for j in range(nprop)]
            for job in joblist:
                if not self.QMin.control["jobs"][job]["restr"]:
                    continue
                else:
                    mults = self.QMin.control["jobs"][job]["mults"]
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += self.QMin.molecule["states"][i - 1] - (i == gsmult)
                    if ns == 0:
                        continue
                sumfile = os.path.join(self.QMin.resources["scratchdir"], f"master_{job}", "tden_summ.txt")
                omffile = os.path.join(self.QMin.resources["scratchdir"], f"master_{job}", "OmFrag.txt")
                props = self.get_theodore(sumfile, omffile)
                # self.log.debug(f"{len(props)} {props}")
                # self.log.debug(f"{theodore_arr.shape}")
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    if (m1, s1) in props:
                        for j in range(nprop):
                            self.log.debug(f"{m1} {s1}: {i} {j} {len(props[(m1,s1)])}")
                            theodore_arr[j][1][i] = props[(m1,s1)][j]
                            # theodore_arr[i, j] = props[(m1, s1)][j]
            self.QMout["prop1d"].extend(theodore_arr)
            self.log.info(self.QMout["prop1d"])

        endtime = datetime.datetime.now()
        self.log.print(f"Readout Runtime: {endtime - starttime}")

        if self.QMin.resources["debug"]:
            copydir = os.path.join(self.QMin.save["savedir"], "debug_GAUSSIAN_stdout")
            if not os.path.isdir(copydir):
                mkdir(copydir)
            for job in joblist:
                outfile = os.path.join(self.QMin.resources["scratchdir"], f"master_{job}", "GAUSSIAN.log")
                shutil.copy(outfile, os.path.join(copydir, f"GAUSSIAN_{job}.log"))
                if self.QMin.control["jobs"][job]["restr"] and self.QMin.requests["theodore"]:
                    outfile = os.path.join(self.QMin.resources["scratchdir"], f"master_{job}", "tden_summ.txt")
                    try:
                        shutil.copy(outfile, os.path.join(copydir, f"THEO_{job}.out"))
                    except IOError:
                        pass
                    outfile = os.path.join(self.QMin.resources["scratchdir"], f"master_{job}", "OmFrag.txt")
                    try:
                        shutil.copy(outfile, os.path.join(copydir, f"THEO_OMF_{job}.out"))
                    except IOError:
                        pass
            if self.QMin.requests["grad"]:
                for grad in self.QMin.maps["gradmap"]:
                    path, isgs = self.QMin.control["jobgrad"][grad]
                    outfile = os.path.join(self.QMin.resources["scratchdir"], path, "GAUSSIAN.log")
                    shutil.copy(outfile, os.path.join(copydir, "GAUSSIAN_GRAD_%i_%i.log" % grad))
            if self.QMin.requests["overlap"]:
                for mult in itmult(self.QMin.molecule["states"]):
                    job = self.QMin.maps["multmap"][mult]
                    outfile = os.path.join(self.QMin.resources["scratchdir"], "WFOVL_%i_%i/wfovl.out" % (mult, job))
                    shutil.copy(outfile, os.path.join(copydir, "WFOVL_%i_%i.out" % (mult, job)))
            if self.QMin.requests["ion"]:
                for ion in self.QMin.maps["ionmap"]:
                    outfile = os.path.join(self.QMin.resources["scratchdir"], "Dyson_%i_%i_%i_%i/wfovl.out" % ion)
                    shutil.copy(outfile, os.path.join(copydir, "Dyson_%i_%i_%i_%i.out" % ion))

        del self.QMin.molecule['mol']
        return self.QMout

    # ======================================================================= #

    def getenergy(self, log_content, ijob):
        # read ground state
        for line in log_content:
            if " SCF Done:" in line:
                gsenergy = float(line.split()[4])

        # figure out the excited state settings
        mults = self.QMin.control["jobs"][ijob]["mults"]
        restr = self.QMin.control["jobs"][ijob]["restr"]
        gsmult = mults[0]
        estates_to_extract = deepcopy(self.QMin.molecule["states"])
        estates_to_extract[gsmult - 1] -= 1
        for imult, _ in enumerate(estates_to_extract):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0
        for imult, _ in enumerate(estates_to_extract):
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
                    if "Excited State" in line:
                        if restr:
                            if not IToMult[imult] in line:
                                continue
                        energies[(imult, istate + 1 + (gsmult == imult))] = float(line.split()[4]) / au2eV + gsenergy
                        istate += 1
                        if istate >= nstates:
                            break
        return energies

    # ======================================================================= #
    def gettdm(self, log_content, ijob):
        # figure out the excited state settings
        mults = self.QMin.control["jobs"][ijob]["mults"]
        if 3 in mults:
            mults = [3]
        restr = self.QMin.control["jobs"][ijob]["restr"]
        gsmult = mults[0]
        estates_to_extract = self.QMin.molecule["states"][:]
        estates_to_extract[gsmult - 1] -= 1
        for imult, _ in enumerate(estates_to_extract):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0

        # get ordering of states in Gaussian output
        istate = [int(i + 1 == gsmult) for i in range(len(self.QMin.molecule["states"]))]
        index = 0
        gaustatemap = {}
        for iline, line in enumerate(log_content):
            if "Excited State" in line:
                if restr:
                    s = line.replace("-", " ").split()
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
                    if "Ground to excited state transition electric dipole moments " in line:
                        for istate in range(nstates):
                            shift = gaustatemap[(imult, istate + 1 + (gsmult == imult))]
                            s = log_content[iline + 2 + shift].split()
                            dipoles[(imult, istate + 1 + (gsmult == imult))] = [float(i) for i in s[1:4]]
        return dipoles

    # ======================================================================= #
    @staticmethod
    def get_fchk(workdir, groot=""):
        #  if os.path.isfile(os.path.join(workdir, 'GAUSSIAN.fchk')):
            #  return
        prevdir = os.getcwd()
        os.chdir(workdir)
        string = os.path.join(groot, "formchk") + " GAUSSIAN.chk"
        outfile = open(os.path.join(workdir, "fchk.out"), "w", encoding="utf-8")
        errfile = open(os.path.join(workdir, "fchk.err"), "w", encoding="utf-8")
        try:
            sp.call(string, shell=True, stdout=outfile, stderr=errfile)
        except OSError as e:
            raise RuntimeError("Call have had some serious problems:", e)
        os.chdir(prevdir)

    @staticmethod
    def parse_fchk(fchkfile: str, keywords: set) -> dict[str,]:
        """
        Parse some keywords from an fchkfile raw

            fchkfile: str  name of the FCHK file
            properties: list[str]  list of keywords you want to parse
        Returns:
        ---
            dict[str,]  dictionary with properties as keys and values
        """
        types = {"I": int, "R": float, "C": str}
        res = {k: None for k in keywords}

        with open(fchkfile, "r", encoding="utf-8") as f:
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
                    if k in line[:len(k)]:
                        llst = line[len(k):].split()
                        cast = types[llst[0]]
                        n = parse_num(llst, cast)
                        if llst[1] == "N=":
                            res[k] = parse_array(int(llst[-1]), cast, f)
                        else:
                            res[k] = n
                # ----------------------------
                line = f.readline()
        return res

    @staticmethod
    def prepare_basis(properties: dict[str,]):
        """
        prepares the basis object from raw FCHK properties
        necessary properties:
        ---
            'Atomic numbers'
            'Number of basis functions'
            'Pure/Cartesian d shells'
            'Pure/Cartesian f shells'
            'Shell types'
            'Number of primitives per shell'
            'Shell to atom map'
            'Primitive exponents'
            'Contraction coefficients'
            'P(S=P) Contraction coefficients'
        """

        p_eq_s = "P(S=P) Contraction coefficients" in properties and properties["P(S=P) Contraction coefficients"] is not None
        if p_eq_s:
            properties["P(S=P) Contraction coefficients"] = properties["P(S=P) Contraction coefficients"].tolist()
        atom_symbols = [IAn2AName[x] for x in properties["Atomic numbers"]]
        return (
            build_basis_dict(
                atom_symbols,
                properties["Shell types"].tolist(),
                properties["Number of primitives per shell"].tolist(),
                properties["Shell to atom map"].tolist(),
                properties["Primitive exponents"].tolist(),
                properties["Contraction coefficients"].tolist(),
                properties["P(S=P) Contraction coefficients"],
            ),
            properties["Number of basis functions"],
            properties["Pure/Cartesian d shells"],
            properties["Pure/Cartesian f shells"],
            p_eq_s,
        )

    @staticmethod
    def get_pyscf_basis_order(atom_symbols, basis_dict, cartesian_d=False, cartesian_f=False, p_eq_s=False):
        """
        Generates the reorder list to reorder atomic orbitals (from GAUSSIAN) to pyscf.

        Sources:
        GAUSSIAN: https://gaussian.com/interfacing/
        pyscf:  https://pyscf.org/user/gto.html#ordering-of-basis-function

        Parameters
        ----------
        atom_symbols : list[str]
            list of element symbols for all atoms (same order as AOs)
        basis_dict : dict[str, list]
            basis set for each atom in pyscf format
        cartesian_d : bool
            whether the d-orbitals are cartesian
        cartesian_f : bool
            whether the f-orbitals are cartesian
        """
        #  return matrix

        # in the case of P(S=P) coefficients the order is 1S, 2S, 2Px, 2Py, 2Pz, 3S in gaussian and pyscf

        # if there are any d-orbitals they need to be swapped!!!
        if cartesian_d:
            # in the case of a cartesian basis the ordering is
            # gauss order:     xx, yy, zz, xy, xz, yz
            # pyscf order:     xx, xy, xz, yy, yz, zz
            d_order = [0, 3, 4, 1, 5, 2]
            #  d_order = [0, 1, 2, 3, 4, 5]
            nd = 6
        else:
            # from gauss order: z2, xz, yz, x2-y2, xy
            # to   pyscf order: xy, yz, z2, xz, x2-y2
            d_order = [4, 2, 0, 1, 3]
            nd = 5

        if cartesian_f:
            # F shells cartesian:
            # gauss order: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
            # pyscf order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
            f_order = [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]
            nf = 10
        else:
            # F shells spherical:
            # gauss order: zzz, xzz, yzz, xxz-yyz, xyz, xxx-xyy, xxy
            # pyscf order: xxy, xyz, yzz, zzz, xzz, xxz-yyz, xxx-xyy
            f_order = [6, 4, 2, 0, 1, 3, 5]
            nf = 7

        # G shells cartesian, not needed anyway
        # pyscf order: xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,xzzz,yyyy,yyyz,yyzz,yzzz,zzzz
        g_order = [8, 6, 4, 2, 0, 1, 3, 5, 7]
        ng = 9

        # H shells cartesian coordinates, not needed anyway
        # pyscf order: xxxxx,xxxxy,xxxxz,xxxyy,xxxyz,xxxzz,xxyyy,xxyyz,xxyzz,xxzzz,xyyyy,xyyyz,xyyzz,xyzzz,xzzzz,yyyyy,yyyyz,yyyzz,yyzzz,yzzzz,zzzzz
        h_order = [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]
        nh = 11

        # I shells cartesian coordinates, not needed anyway
        # pyscf order: xxxxxx,xxxxxy,xxxxxz,xxxxyy,xxxxyz,xxxxzz,xxxyyy,xxxyyz,xxxyzz,xxxzzz,xxyyyy,xxyyyz,xxyyzz,xxyzzz,xxzzzz,xyyyyy,xyyyyz,xyyyzz,xyyzzz,xyzzzz,xzzzzz,yyyyyy,yyyyyz,yyyyzz,yyyzzz,yyzzzz,yzzzzz,zzzzzz
        i_order = [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11]
        ni = 13

        # compile the new_order for the whole matrix
        new_order = []
        it = 0
        for i, a in enumerate(atom_symbols):
            key = f"{a.upper()}{i+1}"
            #       s  p  d  f  g  h  i
            n_bf = [0, 0, 0, 0, 0, 0, 0]

            # count the shells for each angular momentun
            for shell in basis_dict[key]:
                n_bf[shell[0]] += 1

            if p_eq_s:
                s, p = n_bf[0:2]
                if s == p:
                    s_order = [4 * n for n in range(s)]
                    sp_order = s_order + [n for n in range(1, p * 3 + s) if (n) % 4 != 0]
                elif p == 0:
                    s_order = [x for x in range(s)]
                    sp_order = s_order
                else:
                    s_order = [0] + [1 + 4 * n for n in range(s - 1)]
                    sp_order = s_order + [n for n in range(2, p * 3 + s) if (n - 1) % 4 != 0]
                # offset new_order with iterator
                new_order.extend([it + n for n in sp_order])
            else:
                s, p = n_bf[0:2]
                new_order.extend([it + n for n in range(s + p * 3)])

            it += s + p * 3

            # do d shells
            for x in range(n_bf[2]):
                new_order.extend([it + n for n in d_order])
                it += nd

            # do f shells
            for x in range(n_bf[3]):
                new_order.extend([it + n for n in f_order])
                it += nf
            # do g shells
            for x in range(n_bf[4]):
                new_order.extend([it + n for n in g_order])
                it += ng

            # do h shells
            for x in range(n_bf[5]):
                new_order.extend([it + n for n in h_order])
                it += nh

            # do i shells
            for x in range(n_bf[6]):
                new_order.extend([it + n for n in i_order])
                it += ni

            assert it == len(new_order)

        return new_order

    @staticmethod
    def prepare_ecp(props: dict[str,]):
        """
        Prepares ECP from raw parsed FCHK properties
        needed props:
        ---
            'Number of atoms'
            'Atomic numbers'
            'ECP-MaxLECP'
            'ECP-KFirst'
            'ECP-KLast'
            'ECP-LMax'
            'ECP-LPSkip'
            'ECP-RNFroz'
            'ECP-NLP'
            'ECP-CLP1'
            'ECP-ZLP'
        """

        # ++++++++++++++++++ Start making things
        natom = props["Number of atoms"]
        if props["ECP-NLP"] is None:
            return {}
        skips = props["ECP-LPSkip"] == 0
        kfirst = props["ECP-KFirst"].reshape((-1, natom))[:, skips]
        klast = props["ECP-KLast"].reshape((-1, natom))[:, skips]
        lmax = props["ECP-LMax"][skips]
        froz = props["ECP-RNFroz"][skips].astype(int)
        nlp = props["ECP-NLP"]
        clp1 = props["ECP-CLP1"]
        zlp = props["ECP-ZLP"]

        atom_ids = np.where(skips)[0]
        symbols = [IAn2AName[props["Atomic numbers"][x]] for x in atom_ids]
        fun_sym = "SPDFGHIJKLMNOTU"

        ECPs = {}
        # loop over all atoms
        for (i, a), s, lm in zip(enumerate(atom_ids), symbols, lmax):
            ecp_string = f"{s} nelec {froz[i]: d}\n"

            # build the momentum list (with highest momentum first labeled as u1)
            funs = [fun_sym[x] for x in reversed(range(lm))]
            funs.append("ul")

            # loop over all angular momentums
            for fi, la, fun in zip(kfirst[:, i], klast[:, i], reversed(funs)):
                ecp_string += f"{s} {fun}\n"
                for y in range(fi - 1, la):
                    ecp_string += f"{nlp[y]:2d}    {zlp[y]: 12.7f}       {clp1[y]: 12.7f}\n"
            ECPs[a] = ecp_string

        return ECPs

    #Start TOMI
    def get_mole(self):
        job = self.QMin.control["joblist"][0]
        masterdir = os.path.join(self.QMin.resources["scratchdir"], f"master_{job}")
        self.get_fchk(masterdir, self.QMin.resources["groot"])
        fchk_master = os.path.join(masterdir, "GAUSSIAN.fchk")

        # collect properties to read
        keywords_from_master = set()
        get_basis = (
            self.QMin.requests["mol"] or self.QMin.requests["density_matrices"] or self.QMin.requests["multipolar_fit"]
        )
        get_ecp = self.QMin.requests["density_matrices"] or self.QMin.requests["multipolar_fit"]

        keywords_from_master.update(
                {
                    "Atomic numbers",
                    "Number of basis functions",
                    "Pure/Cartesian d shells",
                    "Pure/Cartesian f shells",
                    "Shell types",
                    "Number of primitives per shell",
                    "Shell to atom map",
                    "Primitive exponents",
                    "Contraction coefficients",
                    "P(S=P) Contraction coefficients",
                }
            )
        keywords_from_master.update(
                {
                    "Number of atoms",
                    "Atomic numbers",
                    "ECP-MaxLECP",
                    "ECP-KFirst",
                    "ECP-KLast",
                    "ECP-LMax",
                    "ECP-LPSkip",
                    "ECP-RNFroz",
                    "ECP-NLP",
                    "ECP-CLP1",
                    "ECP-ZLP",
                }
            )

        raw_properties_from_master = SHARC_GAUSSIAN.parse_fchk(fchk_master, keywords_from_master)

        #if get_basis:
        basis, n_bf, cartesian_d, cartesian_f, p_eq_s_shell = SHARC_GAUSSIAN.prepare_basis(raw_properties_from_master)
        self.log.debug(f"{basis}")
        self.log.debug(f"basis information: P(S=P):{p_eq_s_shell} cartesian d:{cartesian_d}, cartesian_f {cartesian_f}")
        #if get_ecp:
        ECPs = SHARC_GAUSSIAN.prepare_ecp(raw_properties_from_master)
        self.log.debug(f"{'ECP:':=^80}\n{ECPs}")
        if len(ECPs) == 0:
            self.log.debug("No ECPs found")
        #gsmult = self.QMin.maps["statemap"][1][0]
        #charge = self.QMin.maps["chargemap"][gsmult]
        atoms = [
            [f"{s.upper()}{j+1}", c.tolist()]
            for j, s, c in zip(range(self.QMin.molecule["natom"]), self.QMin.molecule["elements"], self.QMin.coords["coords"])
        ]
        self.QMin.molecule['mol'] = gto.Mole(
            atom=atoms,
            basis=basis,
            unit="AU",
            #spin=gsmult - 1,
            #charge=charge,
            charge=0,
            spin=0,
            symmetry=False,
            cart=cartesian_d,
            ecp={f'{self.QMin.molecule["elements"][n]}{n+1}': ecp_string for n, ecp_string in ECPs.items()},
        )
        try:
            self.QMin.molecule["mol"].build()
        except:
            self.QMin.molecule["mol"].spin = 1
            self.QMin.molecule["mol"].build()

        self.QMin.molecule["SAO"] = self.QMin.molecule['mol'].intor("int1e_ovlp")
        #self.QMin.molecule.mu = self.molecule.mol.intor("int1e_r")
        new_order = SHARC_GAUSSIAN.get_pyscf_basis_order(
            self.QMin.molecule["elements"], basis, cartesian_d=cartesian_d, cartesian_f=cartesian_f
        )
        self.QMin.molecule["Ubasis"] = np.identity(self.QMin.molecule['mol'].nao)[new_order,:]/np.sqrt(np.diag(self.QMin.molecule['SAO']))
        self.log.debug('Matrix that rotates GAUSSIAN basis set to PySCF basis set:')
        if self.QMin.resources['debug']:
            for i in range(self.QMin.molecule['mol'].nao):
                self.log.print(' '.join([ f"{self.QMin.molecule['Ubasis'][i,j]: 11.8f}" for j in range(self.QMin.molecule['mol'].nao) ] ) )
        return

    def get_readable_densities(self):
        densities = {}
        for s1 in self.states:
            for s2 in self.states:
                if s1 is s2 and s1.C['is_gs']: # Total (and maybe spin) SCF density
                    densities[(s1, s2, 'tot')] = {'how':'read'}
                    if s1.S % 2 == 1 and s1.S == s1.M:
                        densities[(s1, s2,'q')] = {'how':'read'}
                elif s1 is s2 and not s1.C['is_gs']: # Total/spin or aa/bb CIS-like state densities
                    match self.QMin.template['state_densities']:
                        case 'relaxed':
                            densities[(s1, s2, 'tot')] = {'how':'read'}
                            if s1.S % 2 == 1 and s1.S == s1.M:
                                densities[(s1, s2,'q')] = {'how':'read'}
                        case 'unrelaxed':
                            if s1.S == 2:
                                if self.QMin.template['unrestricted_triplets']: 
                                    if s1.M == 2:
                                        densities[(s1, s2, 'aa')] = {'how':'read'}
                                        densities[(s1, s2, 'bb')] = {'how':'read'}
                                else:
                                    if s1.M == 0:
                                        densities[(s1, s2, 'aa')] = {'how':'read'}
                                        densities[(s1, s2, 'bb')] = {'how':'read'}
                            elif s1.S == s1.M:
                                densities[(s1, s2, 'aa')] = {'how':'read'}
                                densities[(s1, s2, 'bb')] = {'how':'read'}
                elif s1.C['is_gs'] and not s2.C['is_gs'] and s1.M == s2.M and s1 is s2.C['its_gs']:
                    densities[(s1,s2,'aa')] = {'how':'read'} 
                    densities[(s1,s2,'bb')] = {'how':'read'} 


        return densities

    def read_and_append_densities(self): #-> dict[(electronic_state,electronic_state,str), np.ndarray]:
        """
        builds a dictionary of densities out of the information in 'density_matrices' and 'densjob'

        Args:
            n_bf: int   number of basis functions
            Sao: overlap matrix of atomic orbitals
        """
        # retrieving densities
        constructed_matrices = {}
        for job, keys in self.QMin.control["densjob"].items():
            # ===================== PARSING BLOCK ===============================
            # create all necessary FCHKs
            workdir = os.path.join(self.QMin.resources["scratchdir"], job)
            fchkfile = os.path.join(workdir, "GAUSSIAN.fchk")
            #  if not os.path.isfile(fchkfile):
                #  self.get_fchk(workdir, self.QMin.resources["groot"])
            self.get_fchk(workdir, self.QMin.resources["groot"])

            keywords = set()
            for key in keys:
                # density_matrices stores a tuple (jobname, set(keywords from fchk))
                keywords.update(self.density_recipes['read'][key][1])
            # get properties!
            self.log.debug(f"Parsing {fchkfile} -> {keywords}")
            raw_matrices = SHARC_GAUSSIAN.parse_fchk(fchkfile, keywords)
            parsed_matrices = {}

            for keyword, raw_data in raw_matrices.items():
                self.log.debug(f"{keyword} is {'found' if raw_data is not None else None}")
                match (keyword, raw_data):
                    case (_, None):
                        self.log.warning(f"'{keyword}' not found in:\n\t {fchkfile}")
                    case ("Total SCF Density" | "Total CI Density" | "Spin SCF Density" | "Spin CI Density", _):
                        parsed_matrices[keyword] = triangular_to_full_matrix(raw_data, self.QMin.molecule['mol'].nao)
                    case ("G to E trans densities" | _) if raw_matrices["Number of g2e trans dens"] is not None:
                        parsed_matrices[keyword] = raw_matrices["G to E trans densities"].reshape(
                            raw_matrices["Number of g2e trans dens"], 2, self.QMin.molecule['mol'].nao, self.QMin.molecule['mol'].nao 
                        ) / math.sqrt(2)
                        parsed_matrices[keyword] = np.einsum("abcd->bacd", parsed_matrices[keyword])
                    #  case("Excited state densities" | _) if raw_matrices["Number of ex state dens"] is not None:
                        #  self.log.debug(' Tuuuuu sam:')
                        #  self.log.debug(str(raw_matrices['Excited state densities']))
                        #  exit()
                        #  parsed_matrices[keyword] = raw_matrices["Excited state densities"].reshape( 
                            #  raw_matrices["Number of ex state dens"], 2, self.QMin.molecule['mol'].nao*(self.QMin.molecule['mol'].nao-1)//2+self.QMin.molecule['mol'].nao 
                        #  )
                        #  parsed_matrices[keyword] = np.einsum("abc->bac", parsed_matrices[keyword])
                        #  tmp = np.empty( ( 2, raw_matrices["Number of ex state dens"], self.QMin.molecule['mol'].nao, self.QMin.molecule['mol'].nao ) )
                        #  for i in range(raw_matrices["Number of ex state dens"]):
                            #  for j in range(2):
                                #  tmp[j,i,:,:] = triangular_to_full_matrix( parsed_matrices[keyword][i,j,:], self.QMin.molecule['mol'].nao )
                        #  parsed_matrices[keyword] = tmp.copy()
                    case _:
                        pass
                if keyword == 'Excited state densities' and raw_matrices["Number of ex state dens"] is not None:
                    self.log.debug(' Tuuuuu sam:')
                    self.log.debug(str(raw_matrices['Excited state densities']))
                    parsed_matrices[keyword] = raw_matrices["Excited state densities"].reshape( 
                        raw_matrices["Number of ex state dens"], 2, self.QMin.molecule['mol'].nao*(self.QMin.molecule['mol'].nao-1)//2+self.QMin.molecule['mol'].nao 
                    )
                    parsed_matrices[keyword] = np.einsum("abc->bac", parsed_matrices[keyword])
                    tmp = np.empty( ( 2, raw_matrices["Number of ex state dens"], self.QMin.molecule['mol'].nao, self.QMin.molecule['mol'].nao ) )
                    for i in range(raw_matrices["Number of ex state dens"]):
                        for j in range(2):
                            tmp[j,i,:,:] = triangular_to_full_matrix( parsed_matrices[keyword][j,i,:], self.QMin.molecule['mol'].nao )
                    parsed_matrices[keyword] = tmp.copy()
            for key in keys:
                s1, s2, mat = key
                dens_type = list(self.density_recipes['read'][key][1])
                # also means mat == 'aa' |'bb'
                if any( [ k in dens_type for k in [ "Total CI Density", "Total SCF Density", "Spin CI Density", "Spin SCF Density" ] ] ):
                    self.QMout['density_matrices'][key] = parsed_matrices[dens_type[0]]
                    #  print(' parsed rho = ', self.QMout['density_matrices'][key][0,0], id(self))
                elif "G to E trans densities" in dens_type: 
                    ab = {"aa": 0, "bb": 1}
                    if s2.S == 2 and not self.QMin.template['unrestricted_triplets']: 
                        delta = 1
                    else:
                        delta = 2
                    self.QMout['density_matrices'][key] = parsed_matrices[dens_type[0]][ab[mat], s2.N - delta, ...]
                elif "Excited state densities" in dens_type: 
                    ab = {"aa": 0, "bb": 1}
                    sign = 1.
                    if s2.S == 2 and not self.QMin.template['unrestricted_triplets']: 
                        delta = 1
                        if mat == 'bb': sign = -1.
                    else:
                        delta = 2
                    self.QMout['density_matrices'][key] = sign*parsed_matrices[dens_type[0]][ab[mat], s2.N - delta, ...]

        for density, rho in self.QMout['density_matrices'].items():
            self.QMout['density_matrices'][density] = self.QMin.molecule['Ubasis'] @ rho @ self.QMin.molecule['Ubasis'].T 
        return
    # End TOMI

    # ======================================================================= #

    @staticmethod
    def getdm(logfile):
        # open file
        f = readfile(logfile)

        for iline, line in enumerate(f):
            if "Forces (Hartrees/Bohr)" in line:
                s = f[iline - 2].split("=")[1].replace("D", "E")
                dmx = float(s[0:15])
                dmy = float(s[15:30])
                dmz = float(s[30:45])
                dm = [dmx, dmy, dmz]
                return dm

    # ======================================================================= #

    def getgrad(self, logfile):
        # read file
        out = readfile(logfile)
        self.log.print("Gradient: " + shorten_DIR(logfile))

        # initialize
        natom = self.QMin.molecule["natom"]
        g = [[0.0 for i in range(3)] for j in range(natom)]

        # get gradient
        string = "Forces (Hartrees/Bohr)"
        shift = 3
        for iline, line in enumerate(out):
            if string in line:
                for iatom in range(natom):
                    s = out[iline + shift + iatom].split()
                    for i in range(3):
                        g[iatom][i] = -float(s[2 + i])

        return g
    
    # ======================================================================= #

    def getgrad_pc(self, logfile):
        # read file
        out = readfile(logfile)
        self.log.print("PC Gradient: " + shorten_DIR(logfile))

        # initialize
        natom = self.QMin.molecule["natom"]
        npc = self.QMin.molecule["npc"]
        g = [[0.0 for i in range(3)] for j in range(npc)]

        # get gradient
        string = "Center     Electric         -------- Electric Field --------"
        shift = 3 + natom
        for iline, line in enumerate(out):
            if string in line:
                for iatom in range(npc):
                    s = out[iline + shift + iatom].split()
                    for i in range(3):
                        g[iatom][i] =  -float(s[2 + i]) * self.QMin.coords["pccharge"][iatom]
        return g

    # ======================================================================= #

    @staticmethod
    def getDyson(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                raise RuntimeError("Dyson norm of states %i - %i not found!" % (s1, s2))
            if containsstring("Dyson norm matrix <PsiA_i|PsiB_j>", out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    # ======================================================================= #

    #  def dyson_orbitals_with_other(self, other):
        #  pass


if __name__ == "__main__":
    SHARC_GAUSSIAN().main()
