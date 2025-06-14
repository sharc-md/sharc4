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


import datetime
import os
import re
import shutil
import subprocess as sp
from copy import deepcopy
from functools import cmp_to_key
from io import TextIOWrapper
from itertools import product
from math import ceil
from typing import Any

import h5py
import numpy as np
from constants import au2a
from pyscf import tools
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from sympy.physics.wigner import wigner_3j
from utils import convert_list, expand_path, link, mkdir, question, writefile

__all__ = ["SHARC_MOLCAS"]

AUTHORS = "Sascha Mausenberger, Lorenz Grünewald, Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "MOLCAS"
DESCRIPTION = "AB INITIO interface for OpenMolcas (>v23) for multireference calculations (CASSCF/RASSCF, MS/XMS-CASPT2, CMS-PDFT)"

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "dm",
        "soc",
        "nacdr",
        "grad",
        "ion",
        "overlap",
        "phases",
        "multipolar_fit",
        "molden",
        "theodore",
        "point_charges",
        "grad_pc",
        "nacdr_pc",
        # raw data request
        "mol",
        "wave_functions",
        "density_matrices",
    ]
)


class SHARC_MOLCAS(SHARC_ABINITIO):
    """
    SHARC interface for MOLCAS
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Setup
        self._template_file = None
        self._resource_file = None

        # Features of MOLCAS installation
        self._hdf5 = False
        self._wfa = False
        self._mpi = False

        # Save sort order for H matrix
        self._h_sort = None

        # Add resource keys
        self.QMin.resources.update(
            {"molcas": None, "mpi_parallel": False, "delay": 0.0, "dry_run": False, "driver": None, "wfoverlap": ""}
        )

        self.QMin.resources.types.update({"molcas": str, "mpi_parallel": bool, "delay": float, "dry_run": bool, "driver": str})

        # Add template keys
        self.QMin.template.update(
            {
                "basis": None,
                "baslib": None,
                "nactel": None,
                "ras1": None,
                "ras2": None,
                "ras3": None,
                "inactive": None,
                "roots": list(range(8)),
                "method": "casscf",
                "functional": "t:pbe",
                "ipea": 0.25,
                "cort_or_dort": "CORT",
                "imaginary": 0.0,
                "frozen": None,
                "gradaccudefault": 1e-4,
                "gradaccumax": 1e-3,
                "pcmset": None,
                "pcmstate": None,
                "iterations": [200, 100],
                "cholesky_accu": 1e-4,
                "rasscf_thrs": [1e-8, 1e-4, 1e-4],
                "density_calculation_methods": [],
            }
        )

        self.QMin.template.types.update(
            {
                "basis": str,
                "baslib": str,
                "nactel": list,
                "ras1": int,
                "ras2": int,
                "ras3": int,
                "inactive": int,
                "roots": list,
                "method": str,
                "functional": str,
                "ipea": float,
                "cort_or_dort": str,
                "imaginary": float,
                "frozen": int,
                "gradaccudefault": float,
                "gradaccumax": float,
                "pcmset": (list, dict),
                "pcmstate": list,
                "iterations": list,
                "cholesky_accu": float,
                "rasscf_thrs": list,  # e, rot egrd
            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_MOLCAS._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_MOLCAS._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_MOLCAS._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_MOLCAS._authors

    @staticmethod
    def name() -> str:
        return SHARC_MOLCAS._name

    @staticmethod
    def description() -> str:
        return SHARC_MOLCAS._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_MOLCAS._name}\n{SHARC_MOLCAS._description}"

    @staticmethod
    def check_template(template_file):
        necessary = {"basis", "ras2", "nactel", "inactive"}
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

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'MOLCAS interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.log.info("\nSpecify path to MOLCAS.")
        self.setupINFOS["molcas"] = question("Path to MOLCAS:", str, default="$MOLCAS", KEYSTROKES=KEYSTROKES)

        self.log.info("\n\nSpecify a scratch directory. The scratch directory will be used to run the calculations.")
        self.setupINFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
        self.setupINFOS["scratchdir"] += '/$$/'

        if os.path.isfile("MOLCAS.template"):
            self.log.info("Found MOLCAS.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "MOLCAS.template"
        else:
            self.log.info("Specify a path to a MOLCAS template file.")
            while not os.path.isfile(template_file := question("Template path:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file

        self.log.info("Specify the number of CPUs to be used.")
        self.setupINFOS["ncpu"] = question("Number of CPUs:", int, default=[1], KEYSTROKES=KEYSTROKES)[0]

        self.log.info("Specify the amount of RAM to be used.")
        self.setupINFOS["memory"] = question("Memory (MB):", int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

        self.log.info("Initial wavefunction: MO Guess\n")
        self.log.info(
            "Please specify the path to a MOLCAS JobIph file containing suitable starting MOs for the CASSCF calculation."
        )
        self.log.info(
            "Please note that this script cannot check whether the wavefunction file and the Input template are consistent!"
        )
        self.setupINFOS["molcas.guess"] = {}
        string = "Do you have initial wavefunction files for multiplicit"
        mults = []
        for mult, state in enumerate(INFOS["states"]):
            if state <= 0:
                continue
            mults.append(mult)
        if len(mults) > 1:
            string += "ies "
        else:
            string += "y "
        string += " ".join([str(i + 1) for i in mults]) + "?"
        if question(string, bool, KEYSTROKES=KEYSTROKES, default=True):
            while (jobiph_or_rasorb := question("JobIph files (1) or RasOrb files (2)?", int, KEYSTROKES=KEYSTROKES)[0]) not in (
                1,
                2,
            ):
                self.log.info(f"{jobiph_or_rasorb} invalid option!")
            self.setupINFOS["molcas.jobiph_or_rasorb"] = jobiph_or_rasorb
            for mult, state in enumerate(INFOS["states"]):
                if state <= 0:
                    continue
                guess_file = f"MOLCAS.{mult + 1}.{'JobIph' if jobiph_or_rasorb == 1 else 'RasOrb'}.init"
                while not os.path.isfile(
                    filename := question(
                        f"Initial wavefunction file for multiplicity {mult + 1}:", str, default=guess_file, KEYSTROKES=KEYSTROKES
                    )
                ):
                    self.log.info("File not found!")
                self.setupINFOS["molcas.guess"][mult + 1] = filename
        else:
            self.log.warning(
                "Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs."
            )

        # TheoDORE
        theodore_spelling = [
            "Om",
            "POSi",
            "POSf",
            "POS",
            "CT",
            "CT2",
            "CTnt",
            "PRi",
            "PRf",
            "PR",
            "PRh",
            "DEL",
            "COH",
            "COHh",
            "MC",
            "LC",
            "MLCT",
            "LMCT",
            "LLCT",
        ]
        # INFOS['theodore']=question('TheoDORE analysis?',bool,False)
        if "theodore" in INFOS["needed_requests"]:
            self.log.info(f"\n{'Wave function analysis by TheoDORE':-^60}\n")

            # self.setupINFOS["theodir"] = question("Path to TheoDORE directory:", str, default="$THEODIR", KEYSTROKES=KEYSTROKES)
            # self.log.info("")

            self.log.info("Please give a list of the properties to calculate by TheoDORE.\nPossible properties:")
            string = ""
            for i, p in enumerate(theodore_spelling):
                string += "%s " % (p)
                if (i + 1) % 8 == 0:
                    string += "\n"
            self.log.info(string)
            line = question("TheoDORE properties:", str, default="Om PR POS COH", KEYSTROKES=KEYSTROKES)
            self.setupINFOS["theodore_prop"] = line.split()
            self.log.info("")

            self.log.info("Please give a list of the fragments used for TheoDORE analysis.")
            # self.log.info("You can use the list-of-lists from dens_ana.in")
            self.log.info('Enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
            self.log.info("Atom numbering starts at 1 for TheoDORE.")
            self.setupINFOS["theodore_fragment"] = []
            while True:
                line = question("TheoDORE fragment:", str, default="end", KEYSTROKES=KEYSTROKES)
                if "end" in line.lower():
                    break
                f = [int(i) for i in line.split()]
                self.setupINFOS["theodore_fragment"].append(f)
            self.setupINFOS["theodore_count"] = len(self.setupINFOS["theodore_prop"]) + len(self.setupINFOS["theodore_fragment"]) ** 2

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resource_file:
            with open(os.path.join(dir_path, "MOLCAS.resources"), "w", encoding="utf-8") as file:
                for key in ("molcas", "scratchdir", "ncpu", "memory", 
                    # "theodir",
                    "theodore_prop",
                    "theodore_fragment"):
                    if key in self.setupINFOS:
                        file.write(f"{key} {self.setupINFOS[key]}\n")
        else:
            create_file(expand_path(self._resource_file), os.path.join(dir_path, "MOLCAS.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "MOLCAS.template"))

        for key, val in self.setupINFOS["molcas.guess"].items():
            val = os.path.abspath(val)
            dest = os.path.join(
                dir_path, f"MOLCAS.{key}.{'JobIph' if self.setupINFOS['molcas.jobiph_or_rasorb'] == 1 else 'RasOrb'}.init"
            )
            create_file(val, dest)

    def read_resources(self, resources_file: str = "MOLCAS.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, ["theodore_fragment"])

        # Path to MOLCAS
        if not self.QMin.resources["molcas"]:
            self.log.error(f"molcas key not found in {resources_file}")
            raise ValueError()

        self.QMin.resources["molcas"] = expand_path(self.QMin.resources["molcas"])
        os.environ["MOLCAS"] = self.QMin.resources["molcas"]

        # Get MOLCAS version and features
        if self.get_molcas_version(self.QMin.resources["molcas"]) < (23, 0):
            self.log.error("This version of SHARC-MOLCAS is only compatible with MOLCAS 23 or higher!")
            raise ValueError()
        self._get_molcas_features()

        if self.QMin.resources["mpi_parallel"] and not self._mpi:
            self.log.warning("MPI requested but MOLCAS version does not support MPI. Run without MPI")
            self.QMin.resources["mpi_parallel"] = False

        # MOLCAS driver
        for p in os.walk(self.QMin.resources["molcas"]):
            if "pymolcas" in p[2]:
                self.QMin.resources.update({"driver": os.path.join(p[0], "pymolcas")})
                break

        if not os.path.isfile(self.QMin.resources["driver"]):
            self.log.error(f"No driver found in {self.QMin.resources['molcas']}")
            raise ValueError()

        # Check orb init and guess
        if self.QMin.save["always_guess"] and self.QMin.save["always_orb_init"]:
            self.log.error("always_guess and always_orb_init cannot be used together!")
            raise ValueError()

        # Set RAM limit
        os.environ["MOLCASMEM"] = str(self.QMin.resources["memory"])
        os.environ["MOLCAS_MEM"] = str(self.QMin.resources["memory"])

        # Theodore descriptor whitelist
        allowed_descriptors = (
            "Om",
            "POSi",
            "POSf",
            "POS",
            "CT",
            "CT2",
            "CTnt",
            "PRi",
            "PRf",
            "PR",
            "PRh",
            "DEL",
            "COH",
            "COHh",
            "MC",
            "LC",
            "MLCT",
            "LMCT",
            "LLCT",
        )
        if self.QMin.resources["theodore_prop"]:
            for prop in self.QMin.resources["theodore_prop"]:
                if prop not in allowed_descriptors:
                    self.log.error(f"Theodore descriptor {prop} is not allowed!")
                    self.log.error(f"Allowed descriptors are {allowed_descriptors}.")
                    raise ValueError()
        if self.QMin.resources["theodore_fragment"]:
            self.QMin.resources["theodore_fragment"] = convert_list(self.QMin.resources["theodore_fragment"], str)

    def read_template(self, template_file: str = "MOLCAS.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Roots
        self.QMin.template["roots"] = convert_list(self.QMin.template["roots"])

        # RASSCF_thresholds
        self.QMin.template["rasscf_thrs"] = convert_list(self.QMin.template["rasscf_thrs"], float)

        if not all(map(lambda x: x >= 0, self.QMin.template["roots"])):
            self.log.error("roots must contain positive integers.")
            raise ValueError()

        for idx, val in enumerate(self.QMin.molecule["states"]):
            if val > self.QMin.template["roots"][idx]:
                self.log.error(f"Too few states in state-averaging in multiplicity {idx+1}!")
                raise ValueError()

        if self.QMin.template["roots"][-1] == 0:
            for idx, val in enumerate(reversed(self.QMin.template["roots"])):
                if val != 0:
                    self.QMin.template["roots"] = self.QMin.template["roots"][: -1 * idx]
                    break

        # Check gradaccu
        if self.QMin.template["gradaccudefault"] > self.QMin.template["gradaccumax"]:
            self.log.error("Template key gradaccudefault cannot be higher than gradaccumax")
            raise ValueError()

        # Path to baslib
        if self.QMin.template["baslib"]:
            self.QMin.template["baslib"] = expand_path(self.QMin.template["baslib"])

        # Iterations
        match len(self.QMin.template["iterations"]):
            case 2:
                self.QMin.template["iterations"] = convert_list(self.QMin.template["iterations"])
            case 1:
                self.QMin.template["iterations"] = [int(self.QMin.template["iterations"][0]), 100]
            case _:
                self.log.error(f"{self.QMin.template['iterations']} is not a valid iteration value!")
                raise ValueError()

        # PCM
        if isinstance(self.QMin.template["pcmset"], list):
            if len(self.QMin.template["pcmset"][0]) != 3:
                self.log.error("pcmset must contain three parameter!")
                raise ValueError()

            self.QMin.template["pcmset"] = {
                "solvent": self.QMin.template["pcmset"][0][0],
                "aare": float(self.QMin.template["pcmset"][0][1]),
                "r-min": float(self.QMin.template["pcmset"][0][2]),
            }
        if self.QMin.template["pcmstate"]:
            self.QMin.template["pcmstate"] = convert_list(self.QMin.template["pcmstate"])

        # Check for basis and cas settings
        for i in ["basis", "nactel", "ras2", "inactive"]:
            if not self.QMin.template[i]:
                self.log.error(f"Key {i} is missing in template file!")
                raise ValueError()

        # Check nactel
        match len(self.QMin.template["nactel"]):
            case 1:
                self.QMin.template["nactel"] = [int(*self.QMin.template["nactel"]), 0, 0]
            case 3:
                self.QMin.template["nactel"] = convert_list(self.QMin.template["nactel"])
            case _:
                self.log.error("nactel must contain either 1 or 3 numbers!")
                raise ValueError()

        for idx, charge in enumerate(self.QMin.molecule["charge"], 1):
            nactel = self.QMin.template["nactel"][0]
            if (
                (nactel - idx - charge) % 2 == 0
                or nactel - charge < 1
                or nactel - charge >= self.QMin.template["ras2"] * 2
            ) and self.QMin.molecule["states"][idx-1] > 0:
                self.log.error(f"Charge {charge} not compatible with multiplicity {idx}")
                self.log.error("Please provide the nactel as if the charge was neutral!")
                raise ValueError()

        # Validate method
        self.QMin.template["method"] = self.QMin.template["method"].lower()
        if self.QMin.template["method"] not in [
            "casscf",
            "caspt2",
            "ms-caspt2",
            "cms-pdft",
            "xms-caspt2",
        ]:
            self.log.error(f"{self.QMin.template['method']} is not a valid method!")
            raise ValueError()

        # Validate functional
        if self.QMin.template["method"] == "cms-pdft" and self.QMin.template["functional"] not in [
            "t:pbe",
            "ft:pbe",
            "t:blyp",
            "ft:blyp",
            "t:revPBE",
            "ft:revPBE",
            "t:LSDA",
            "ft:LSDA",
        ]:
            self.log.error(f"No analytical gradients for cms-pdft and {self.QMin.template['functional']}.")
            raise ValueError()

        # States must be > 1 for xms-caspt2
        if self.QMin.template["method"] == "xms-caspt2" and any((i == 1 for i in self.QMin.molecule["states"])):
            self.log.error("All states in XMS-CASPT2 must be > 1!")
            raise ValueError()

    def setup_interface(self) -> None:
        """
        Setup MOLCAS interface
        """
        super().setup_interface()

        # Setup mults for reordering H matrix
        mults = {}
        for i, _ in enumerate(self.QMin.molecule["states"]):
            # Create list with possible ms values, e.g. duplet [0.5,-0.5], triplet [1,0,-1], ...
            mults[i] = sorted(list({sum(x) for x in map(list, product([0.5, -0.5], repeat=i))}), reverse=True)

        counter = 0
        self.QMin.maps["multmap"] = {}
        for mult, state in enumerate(self.QMin.molecule["states"]):
            for s in range(state):
                for m in mults[mult]:
                    self.QMin.maps["multmap"][counter] = (mult, m, s)
                    counter += 1

        self._h_sort = sorted(list(range(self.QMin.molecule["nmstates"])), key=cmp_to_key(self._sort_mults))

    def _sort_mults(self, state1, state2):
        """
        Sort states (mult, ms, state) in MOLCAS fashion,
        first by mult then by ms then by state
        """
        s1 = self.QMin.maps["multmap"][state1]
        s2 = self.QMin.maps["multmap"][state2]
        return (s1[0] - s2[0]) * 1000 + (s1[1] - s2[1]) * 100 + (s1[2] - s2[2])

    def run(self) -> None:
        starttime = datetime.datetime.now()

        # Generate schedule and run jobs
        self.log.debug("Generate schedule")
        self.QMin.scheduling["schedule"] = self._generate_schedule()

        self.log.debug("Execute schedule")
        if not self.QMin.resources["dry_run"]:
            self.runjobs(self.QMin.scheduling["schedule"])

            # Save Jobiphs and/or molden files
            re_jobiph = re.compile(r"^MOLCAS\.\d+\.JobIph")
            re_molden = re.compile(r"^MOLCAS\.\d+\.molden")
            for file in os.listdir(os.path.join(self.QMin.resources["scratchdir"], "master")):
                if re_jobiph.match(file) or (self.QMin.requests["molden"] and re_molden.match(file)):
                    self.log.debug(f"Copy {file} from scratch to savedir")
                    shutil.copy(
                        os.path.join(self.QMin.resources["scratchdir"], "master", file),
                        os.path.join(self.QMin.save["savedir"], f"{file}.{self.QMin.save['step']}"),
                    )
            self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def _generate_schedule(self) -> list[dict[str, QMin]]:
        """
        Generate schedule, one main job and n grad/nac jobs
        """
        schedule = [{"master": deepcopy(self.QMin)}]
        self.QMin.control["nslots_pool"].append(1)

        ## Setup master job
        schedule[0]["master"].control["master"] = True
        if not schedule[0]["master"].resources["mpi_parallel"]:
            schedule[0]["master"].resources["ncpu"] = 1

        ## Setup grad and nac jobs
        # Get number of tasks
        ntasks = 0
        if self.QMin.requests["grad"]:
            ntasks += len(self.QMin.maps["gradmap"])
        if self.QMin.requests["nacdr"]:
            ntasks += len(self.QMin.maps["nacmap"])
        if ntasks == 0:
            return schedule

        # Get number of slots for grads/nacs
        nslots, cpu_per_run = 1, self.QMin.resources["ncpu"]
        if not self.QMin.resources["mpi_parallel"]:
            nslots, cpu_per_run = self.QMin.resources["ncpu"], 1
        self.QMin.control["nslots_pool"].append(nslots)

        jobs = {}

        # Create gradjobs
        if self.QMin.requests["grad"]:
            for grad in self.QMin.maps["gradmap"]:
                gradjob = deepcopy(self.QMin)
                gradjob.save.update({"init": False, "always_guess": False, "always_orb_init": False, "samestep": True})
                gradjob.requests.update({"h": False, "soc": False, "dm": False, "overlap": False, "ion": False})
                gradjob.resources["ncpu"] = cpu_per_run
                gradjob.maps["gradmap"] = {(grad)}
                gradjob.maps["nacmap"] = set()
                jobs[f"grad_{'_'.join(str(g) for g in grad)}"] = gradjob

        # Create nacjobs
        if self.QMin.requests["nacdr"]:
            for nac in self.QMin.maps["nacmap"]:
                nacjob = deepcopy(self.QMin)
                nacjob.save.update({"init": False, "always_guess": False, "always_orb_init": False, "samestep": True})
                nacjob.requests.update({"h": False, "soc": False, "dm": False, "overlap": True, "ion": False})
                nacjob.resources["ncpu"] = cpu_per_run
                nacjob.maps["gradmap"] = set()
                nacjob.maps["nacmap"] = {(nac)}
                jobs[f"nacdr_{'_'.join(str(n) for n in nac)}"] = nacjob

        schedule.append(jobs)
        return schedule

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Setup workdir, write input file, copy initial guess, execute
        """
        os.environ["WorkDir"] = workdir
        os.environ["MOLCAS_NPROCS"] = str(qmin.resources["ncpu"])
        self.log.debug(f"Create workdir {workdir}")
        mkdir(workdir)

        # Write files
        self.log.debug(f"Writing input files to {workdir}")
        writefile(os.path.join(workdir, "MOLCAS.xyz"), self._write_geom(qmin.molecule["elements"], qmin.coords["coords"]))
        writefile(os.path.join(workdir, "MOLCAS.input"), self._write_input(self._gen_tasklist(qmin), qmin))

        # Set env variable for master path
        os.environ["master_path"] = os.path.join(qmin.resources["scratchdir"], "master")
        # Copy JobIphs if not master job
        if not qmin.control["master"]:
            self._copy_run_files(workdir)

        # Make subdirs
        if qmin.resources["mpi_parallel"]:
            for i in range(qmin.resources["ncpu"]):
                self.log.debug(f"Create subdir tmp_{i+1}")
                mkdir(os.path.join(workdir, f"tmp_{i+1}"))

        # Execute MOLCAS
        starttime = datetime.datetime.now()
        while qmin.template["gradaccudefault"] < qmin.template["gradaccumax"]:
            if (code := self.run_program(workdir, f"{qmin.resources['driver']} MOLCAS.input", "MOLCAS.out", "MOLCAS.err")) != 96:
                break
            qmin.template["gradaccudefault"] *= 10

        # Generate MO and det file for dyson
        if qmin.control["master"] and self._hdf5 and code == 0:
            self._get_dets(workdir)
            self._get_mos(workdir)

        return code, datetime.datetime.now() - starttime

    def _get_mos(self, workdir: str) -> None:
        """
        Get MO vectors from rasscf.h5, reorder to pySCF format
        and write them to savedir in wfoverlap format
        """
        for mult, states in enumerate(self.QMin.molecule["states"], 1):
            if states < 1:
                continue
            with h5py.File(os.path.join(workdir, f"MOLCAS.{mult}.rasscf.h5"), "r") as f:
                mos = f["MO_VECTORS"][:]
                mos = mos.reshape(int(mos.shape[0] ** 0.5), -1)

                # Sort MOs
                with h5py.File(os.path.join(workdir, "MOLCAS.rassi.h5"), "r") as f:
                    ao_order = np.asarray(f["BASIS_FUNCTION_IDS"][:])

                # MOLCAS orders by ml -> e.g. 2px, 3px, 4px, 2py, ...
                def sort_ao(key1, key2):  # reorder to 2px, 2py, 2pz, 3py, ...
                    ao1 = ao_order[key1]
                    ao2 = ao_order[key2]
                    return (ao1[0] - ao2[0]) * 1000 + (ao1[2] - ao2[2]) * 100 + (ao1[1] * ao1[2] - ao2[1] * ao2[2])

                ao_sorted = sorted(list(range(ao_order.shape[0])), key=cmp_to_key(sort_ao))
                mos = mos[:, ao_sorted]

                mo_string = (
                    f"2mocoef\nheader\n1\nMO-coefficients from MOLCAS\n1\n{mos.shape[0]}   {mos.shape[0]}\na\nmocoef\n(*)\n"
                )
                for mat in mos:
                    for idx, i in enumerate(mat):
                        if idx > 0 and idx % 3 == 0:
                            mo_string += "\n"
                        mo_string += f"{i: 6.12e} "
                    if mos.shape[0] - 1 % 3 != 0:
                        mo_string += "\n"

                mo_string += "orbocc\n(*)"
                for i in range(mos.shape[0]):
                    if i % 3 == 0:
                        mo_string += "\n"
                    mo_string += f"{0.0: 6.12e} "

                writefile(os.path.join(self.QMin.save["savedir"], f"mos.{mult}.{self.QMin.save['step']}"), mo_string)

    def _get_dets(self, workdir: str) -> None:
        """
        Parse determinants from VecDet files for all mults
        and write them to savedir in wfoverlap format
        """
        for mult, states in enumerate(self.QMin.molecule["states"], 1):
            eigenvectors = []
            if states < 1:
                continue
            for state in range(1, states + 1):
                # Parse VecDet for every state
                tmp = {}
                with open(os.path.join(workdir, f"MOLCAS.{mult}.VecDet.{state}"), "r", encoding="utf-8") as f:
                    inactive = int(next(f).split()[0])  # First line, first number = inactive electrons
                    dets = re.findall(r"(-?\d\.\d+E[-|+]\d{2})\s+([20ab]+)", f.read())
                    for det in dets:
                        tmp[
                            tuple(
                                [3] * inactive
                                + [int(i) for i in det[1].replace("2", "3").replace("a", "1").replace("b", "2")]
                                + [0] * 29
                            )
                        ] = float(det[0])
                    self.log.debug(f"Determinant norm {mult} {state:4d} {np.linalg.norm(list(tmp.values())):.5f}")

                self.trim_civecs(tmp)
                eigenvectors.append(tmp)
            writefile(
                os.path.join(self.QMin.save["savedir"], f"dets.{mult}.{self.QMin.save['step']}"),
                self.format_ci_vectors(eigenvectors),
            )

    def _copy_run_files(self, workdir: str) -> None:
        """
        Copy files from master to grad/nac folder
        """
        self.log.debug("Copy run files from master")
        re_runfiles = ("ChVec", "QVec", "ChRed", "ChDiag", "ChRst", "ChMap", "RunFile", "GRIDFILE", "NqGrid", "Rotate.txt")
        try:
            for file in os.listdir(os.path.join(self.QMin.resources["scratchdir"], "master")):
                if any(i in file for i in re_runfiles):
                    shutil.copy(os.path.join(self.QMin.resources["scratchdir"], "master", file), os.path.join(workdir, file))
                shutil.copy(
                    os.path.join(self.QMin.resources["scratchdir"], "master/MOLCAS.OneInt"), os.path.join(workdir, "ONEINT")
                )
        except FileNotFoundError:
            self.log.error("Copy file(s) failed! This is most likely due to an error in master job!")

    def get_readable_densities(self) -> dict[str, str]:
        densities = {}
        for s1 in self.states:
            for s2 in self.states:
                for spin in ["tot", "q"]:
                    if self.density_logic(s1, s2, spin):
                        densities[(s1, s2, spin)] = {"how": "read"}
        return densities

    def _gen_tasklist(self, qmin: QMin) -> list[list[Any]]:
        """
        Generate tasklist
        """

        # Check if master or grad/nac job
        if not qmin.control["master"]:
            self.log.debug("Generating tasklist for grad/nac job.")
            return self._gen_grad_tasks(qmin)

        # Master job
        self.log.debug("Generating tasklist for master job.")
        tasks = [["gateway"], ["seward"]]

        # Mult loop
        list_to_do = list(enumerate(qmin.molecule["states"]))
        for mult, states in list_to_do:
            if states == 0:
                continue

            # Check if initorbs or samestep, copy guess orbitals
            is_jobiph = False
            is_rasorb = False
            match qmin.save:
                case {"always_guess": True}:
                    pass
                case {"always_orb_init": True} | {"init": True}:
                    if os.path.isfile((init := os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.JobIph.init"))):
                        tasks.append(["copy", init, "JOBOLD"])
                        is_jobiph = True
                    elif os.path.isfile((init := os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.RasOrb.init"))):
                        tasks.append(["copy", init, "INPORB"])
                        is_rasorb = True
                case {"samestep": True}:
                    tasks.append(
                        ["copy", os.path.join(qmin.save["savedir"], f"MOLCAS.{mult+1}.JobIph.{qmin.save['step']}"), "JOBOLD"]
                    )
                    is_jobiph = True
                case _:
                    tasks.append(
                        [
                            "copy",
                            os.path.join(qmin.save["savedir"], f"MOLCAS.{mult+1}.JobIph.{qmin.save['step']-1}"),
                            "JOBOLD",
                        ]
                    )
                    is_jobiph = True

            # RASSCF block
            tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], is_jobiph, is_rasorb])

            # Save VecDet files and rasscf.h5
            if self._hdf5:
                for state in range(1, states + 1):
                    tasks.append(["copy", f"MOLCAS.VecDet.{state}", f"MOLCAS.{mult+1}.VecDet.{state}"])
                tasks.append(["copy", "MOLCAS.rasscf.h5", f"MOLCAS.{mult+1}.rasscf.h5"])

            # MOLDEN request
            if qmin.requests["molden"]:
                tasks.append(["copy", "MOLCAS.rasscf.molden", f"MOLCAS.{mult+1}.molden"])

            # Generate DFT task
            match qmin.template["method"]:
                case "casscf":
                    tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
                case "cms-pdft":
                    keys = [f"KSDFT={qmin.template['functional']}"]
                    keys.append("noGrad")
                    keys += ["MSPDFT", "WJOB", "CMSS=Do_Rotate.txt", "CMTH=1.0d-10"]
                    tasks.append(["mcpdft", keys])
                    tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
                case _:  # PT2 methods
                    if not qmin.save["samestep"]:
                        tasks.append(["caspt2", mult + 1, states, qmin.template["method"]])
                        tasks.append(["copy", "MOLCAS.JobMix", f"MOLCAS.{mult+1}.JobIph"])

        # Do first dp then overlap tasks
        for mult, states in list_to_do:
            if states == 0:
                continue
            tasks += self._gen_dp_task(qmin, mult, states)

        for mult, states in list_to_do:
            if states == 0:
                continue
            tasks += self._gen_ovlp_task(qmin, mult, states)

        roots = []
        i = 0
        for mult, states in enumerate(qmin.molecule["states"], 1):
            if states == 0:
                continue
            i += 1
            roots.append(states)
            tasks.append(["link", f"MOLCAS.{mult}.JobIph", f"JOB{i:03d}"])
        tasks.append(["rassi", "soc" if qmin.requests["soc"] else "", roots])

        if qmin.requests["theodore"] or qmin.requests["multipolar_fit"] or qmin.requests["density_matrices"]:
            if self._hdf5:
                tasks.append(["link", "MOLCAS.rassi.h5", "MOLCAS.rassi.h5.bak"])
            all_states = qmin.molecule["states"][:]  # Copy of states, to modify in loop
            for mult, states in enumerate(qmin.molecule["states"], 1):
                if states > 0:
                    if len(all_states) >= mult + 2 and all_states[mult + 1] > 0:
                        tasks.append(["link", f"MOLCAS.{mult}.JobIph", "JOB001"])
                        tasks.append(["link", f"MOLCAS.{mult+2}.JobIph", "JOB002"])
                        tasks.append(["rassi", "theodore", [states, qmin.molecule["states"][mult + 1]]])
                        all_states[mult + 1] = 1  # Do not do rassi for same mult twice
                        if qmin.requests["theodore"]:
                            tasks.append(["theodore"])
                        if qmin.requests["multipolar_fit"] or qmin.requests["density_matrices"]:
                            tasks.append(["link", "MOLCAS.rassi.h5", f"MOLCAS.rassi.trd{mult}.h5"])
                            tasks.append(["link", "MOLCAS.rassi.h5.bak", "MOLCAS.rassi.h5"])
                    elif all_states[mult - 1] > 1:
                        tasks.append(["link", f"MOLCAS.{mult}.JobIph", "JOB001"])
                        tasks.append(["rassi", "theodore", [states]])
                        if qmin.requests["theodore"]:
                            tasks.append(["theodore"])
            if self._hdf5:
                tasks.append(["link", "MOLCAS.rassi.h5.bak", "MOLCAS.rassi.h5"])
        self.log.debug("\n".join(str(i) for i in tasks))
        return tasks

    def _gen_ovlp_task(self, qmin: QMin, mult: int, states: int) -> list[list[Any]]:
        """
        Generate tasklist for overlap
        """
        tasks = []
        if qmin.requests["overlap"]:
            if qmin.control["master"]:
                tasks.append(
                    ["copy", os.path.join(qmin.save["savedir"], f'MOLCAS.{mult+1}.JobIph.{qmin.save["step"]-1}'), "JOB001"]
                )
                tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOB002"])
            else:
                tasks.append(["link", f"MOLCAS.{mult + 1}.JobIph", "JOB001"])
                tasks.append(["link", "MOLCAS.JobIph", "JOB002"])
            tasks.append(["rassi", "overlap", [states, states]])
            if self._hdf5:
                tasks.append(["copy", "MOLCAS.rassi.h5", f"MOLCAS.rassi.ovlp.{mult+1}.h5"])
        return tasks

    def _gen_dp_task(self, qmin: QMin, mult: int, states: int) -> list[list[Any]]:
        """
        Generate tasklist for dipoles, ion
        """
        tasks = []
        if qmin.requests["dm"] or qmin.requests["multipolar_fit"]:
            tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOB001"])
            tasks.append(["rassi", "dm", [states]])
            if self._hdf5:
                tasks.append(["copy", "MOLCAS.rassi.h5", f"MOLCAS.rassi.{mult+1}.h5"])
        return tasks

    def _gen_grad_tasks(self, qmin: QMin) -> list[list[Any]]:
        """
        Generate tasklist for nac/grad jobs
        """
        tasks = []

        if qmin.maps["gradmap"]:
            grad = list(qmin.maps["gradmap"])[0]
            mult = grad[0] - 1
            states = qmin.molecule["states"][mult]
            tasks.append(["copy", f"$master_path/MOLCAS.{mult+1}.JobIph", "JOBOLD"])
            if qmin.template["roots"][mult] == 1:
                tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                if qmin.template["method"] in ("ms-caspt2", "xms-caspt2"):
                    self.log.error("Single state gradient with MS/XMS-CASPT2")
                    raise ValueError()
                tasks.append(["alaska"])
            else:
                match qmin.template["method"]:
                    case "cms-pdft":
                        tasks.append(
                            [
                                "rasscf",
                                mult + 1,
                                qmin.template["roots"][mult],
                                True,
                                False,
                                [f"RLXROOT={grad[1]}"],
                            ]
                        )
                        tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD", "MSPDFT"]])
                        tasks.append(["alaska", grad[1]])
                    case "casscf":
                        tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                        tasks.append(["mclr", qmin.template["gradaccudefault"], f"sala={grad[1]}"])
                        tasks.append(["alaska"])
                    case "ms-caspt2" | "xms-caspt2" | "caspt2":
                        tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                        if qmin.template['ipea'] != 0:
                            cort_or_dort = qmin.template["cort_or_dort"] + "\n"
                        else:
                            cort_or_dort = ""
                        tasks.append(["caspt2", mult + 1, states, qmin.template["method"], f"GRDT\n{cort_or_dort}rlxroot = {grad[1]}"])
                        tasks.append(["mclr", qmin.template["gradaccudefault"]])
                        tasks.append(["alaska"])

        if qmin.maps["nacmap"]:
            nac = list(qmin.maps["nacmap"])[0]
            mult = nac[0] - 1
            states = qmin.molecule["states"][mult]
            tasks.append(["copy", f"$master_path/MOLCAS.{mult + 1}.JobIph", f"MOLCAS.{mult + 1}.JobIph"])
            tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOBOLD"])
            match qmin.template["method"]:
                case "casscf":
                    tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False])
                    tasks.append(["mclr", qmin["template"]["gradaccudefault"], f"nac={nac[1]} {nac[3]}"])
                    tasks.append(["alaska"])
                case "cms-pdft":
                    tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False])
                    tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD", "MSPDFT", f"nac={nac[1]} {nac[3]}"]])
                    tasks.append(["mclr", qmin.template["gradaccudefault"], f"nac={nac[1]} {nac[3]}"])
                    tasks.append(["alaska"])
                case "ms-caspt2" | "xms-caspt2" | "caspt2":
                    tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False])
                    if qmin.template['ipea'] != 0:
                        cort_or_dort = qmin.template["cort_or_dort"] + "\n"
                    else:
                        cort_or_dort = ""
                    tasks.append(["caspt2", mult + 1, states, qmin.template["method"], f"GRDT\n{cort_or_dort}nac = {nac[1]} {nac[3]}"])
                    tasks.append(["alaska", nac[1], nac[3]])

            tasks += self._gen_ovlp_task(qmin, mult, states)

        self.log.debug("\n".join(str(i) for i in tasks))
        return tasks

    def _write_input(self, tasks: list[list[str]], qmin: QMin) -> str:
        """
        Write MOLCAS input file

        tasks:  Contains information about the requested tasks
        qmin:   QMin object containing information about the current task
        """
        input_str = ""

        for task in tasks:
            match task[0]:
                case "gateway":
                    input_str += self._write_gateway(qmin)
                case "seward":
                    input_str += self._write_seward(qmin)
                case "link":
                    input_str += f">> COPY {os.path.basename(task[1])} {task[2]}\n\n"
                case "copy":
                    input_str += f">> COPY {task[1]} {task[2]}\n\n"
                case "rm":
                    input_str += f">> RM {task[1]}\n\n"
                case "rasscf":
                    input_str += self._write_rasscf(qmin, task)
                case "caspt2":
                    input_str += self._write_caspt2(qmin, task)
                case "mcpdft":
                    input_str += "&MCPDFT\n" + "\n".join(task[1]) + "\n\n"
                case "rassi":
                    input_str += self._write_rassi(qmin, task)
                case "mclr":
                    input_str += f"&MCLR\nTHRESHOLD={task[1]}\n"
                    input_str += f"{task[2]}\n" if len(task) > 2 else "\n"
                case "alaska":
                    input_str += "&ALASKA"
                    if len(task) == 2:
                        input_str += f"\nroot={task[1]}\n"
                    elif len(task) == 3:
                        input_str += f"\nnac={task[1]} {task[2]}\n"
                    input_str += "\n"
                case "theodore":
                    input_str += self._write_wfa()
        return input_str

    def _write_wfa(self) -> str:
        """
        Write WFA input for theodore
        """
        input_str = "&WFA\nH5file = MOLCAS.rassi.h5\nDoCTnumbers\nATLISTS\n"
        input_str += f"{len(self.QMin.resources['theodore_fragment'])}\n"
        for frag in self.QMin.resources["theodore_fragment"]:
            input_str += " ".join(str(i) for i in frag)
            input_str += " *\n"
        input_str += f"PROP {' '.join(i for i in self.QMin.resources['theodore_prop'])} *\n"
        input_str += "LOWDIN\nNXO\nEXCITON\nWFALEVEL 4\n\n"
        return input_str

    def _write_rassi(self, qmin: QMin, task: list[Any]) -> str:
        """
        Write RASSI part of MOLCAS input string
        """
        input_str = f"&RASSI\nNROFJOBIPHS\n{len(task[2])} "
        input_str += " ".join(convert_list(task[2], str)) + "\n"
        for i in task[2]:
            input_str += " ".join([str(j) for j in range(1, i + 1)]) + "\n"
        input_str += "MEIN\n"
        if qmin.template["method"] != "casscf":
            input_str += "EJOB\n"
        if task[1] == "dm" and (qmin.requests["multipolar_fit"] or qmin.requests["density_matrices"]):
            input_str += "TRD1\n"
        if task[1] == "soc":
            input_str += "SPINORBIT\nSOCOUPLING=0.0d0\nEJOB\n"
        if task[1] == "overlap":
            input_str += "STOVERLAPS\nOVERLAPS\n"
            if qmin.control["master"] and (qmin.requests["multipolar_fit"] or qmin.requests["density_matrices"]):
                input_str += "TRD1\n"
        if task[1] == "theodore":
            input_str += "TRD1\n"
        if task[1] in ("", "soc") and qmin.requests["ion"]:
            input_str += "CIPR\nTHRS=0.000005d0\n"
            input_str += "DYSON\n"
            input_str += f"DYSEXPORT\n{sum(qmin.molecule['states'])} 0\n"
        input_str += "\n"
        return input_str

    def _write_caspt2(self, qmin: QMin, task: list[Any]) -> str:
        """
        Write CASPT2 part of MOLCAS input string
        """
        input_str = f"&CASPT2\nSHIFT=0.0\nIMAGINARY={qmin.template['imaginary']: 5.3f}\n"
        input_str += f"IPEASHIFT={qmin.template['ipea'] : 4.2f}\nMAXITER=120\n"
        if qmin.template["frozen"]:
            input_str += f"FROZEN={qmin.template['frozen']}\n"
        match qmin.template["method"]:
            case "caspt2":
                input_str += "NOMULT\n"
            case "ms-caspt2":
                input_str += f"MULTISTATE= {task[2]} "
            case "xms-caspt2":
                input_str += f"XMULTISTATE= {task[2]} "
        if qmin.template["method"] in ("ms-caspt2", "xms-caspt2"):
            input_str += " ".join(str(i + 1) for i in range(task[2]))
        input_str += "\nOUTPUT=BRIEF\nPRWF=0.1\n"
        if qmin.template["pcmset"]:
            input_str += "RFPERT\n"
        if len(task) == 5:
            input_str += task[4]
        input_str += "\n"
        return input_str

    def _write_rasscf(self, qmin: QMin, task: list[Any]) -> str:
        """
        Write RASSCF part of MOLCAS input string
        """
        nactel = qmin.template["nactel"][:]
        nactel[0] -= qmin.molecule["charge"][task[1] - 1]
        input_str = f"&RASSCF\nSPIN={task[1]}\nNACTEL={' '.join(str(n) for n in nactel)}\n"
        input_str += f"INACTIVE={qmin.template['inactive']}\nRAS2={qmin.template['ras2']}\n"
        input_str += f"ITERATIONS={qmin.template['iterations'][0]},{qmin.template['iterations'][1]}\n"
        if qmin.template["ras1"]:
            input_str += f"RAS1={qmin.template['ras1']}\n"
        if qmin.template["ras3"]:
            input_str += f"RAS3={qmin.template['ras3']}\n"
        input_str += f"CIROOT={task[2]} {task[2]} 1\n"
        if qmin.template["method"] not in ("ms-caspt2", "xms-caspt2"):
            input_str += "ORBLISTING=NOTHING\nPRWF=1.0e-12\n"
        if qmin.template["method"] == "cms-pdft":
            input_str += "CMSInter\n"
        if qmin.maps["gradmap"] and len(qmin.maps["gradmap"]) > 0:
            input_str += "THRS=1.0e-10 1.0e-06 1.0e-06\n"
        else:
            input_str += "THRS=" + " ".join(f"{i:14.12f}" for i in qmin.template["rasscf_thrs"]) + "\n"
        if task[3]:
            input_str += "JOBIPH\n"
            if qmin.save["samestep"]:
                input_str += "CIRESTART\n"
        if task[4]:
            input_str += "LUMORB\n"
        if len(task) > 5:
            input_str += "\n".join(str(a) for a in task[5]) + "\n"
        if qmin.template["pcmset"]:
            if task[1] == qmin.template["pcmstate"][0]:
                input_str += f"RFROOT = {qmin.template['pcmstate'][1]}\n"
            else:
                input_str += "NONEQUILIBRIUM\n"
        input_str += "PRSD\n"
        input_str += "\n"
        return input_str

    def _write_seward(self, qmin: QMin) -> str:
        """
        Write SEWARD part of MOLCAS input string
        """
        input_str = "&SEWARD\n"
        if qmin.template["method"] == "cms-pdft":
            input_str += "GRID INPUT\nNOSC\nEND OF GRID INPUT\n"
        input_str += "\n"
        return input_str

    def _write_gateway(self, qmin: QMin) -> str:
        """
        Write GATEWAY part of MOLCAS input string
        """
        input_str = f"&GATEWAY\nCOORD=MOLCAS.xyz\nGROUP=NOSYM\nBASIS={qmin.template['basis']}\n"
        if qmin.molecule["point_charges"]:
            input_str = "&GATEWAY\n"
            for idx, (charge, coord) in enumerate(zip(qmin.molecule["elements"], qmin.coords["coords"]), 1):
                input_str += f"basis set\n{charge}.{qmin.template['basis']}....\n"
                input_str += f"{charge}{idx} {coord[0]*au2a: >10.15f} {coord[1]*au2a: >10.15f} {coord[2]*au2a: >10.15f}"
                input_str += " /Angstrom\nend of basis\n\n"
            for idx, (charge, coord) in enumerate(zip(qmin.coords["pccharge"], qmin.coords["pccoords"]), 1):
                input_str += (
                    f"basis set\nX...0s.0s.\nX{idx} {coord[0]*au2a: >10.15f} {coord[1]*au2a: >10.15f} {coord[2]*au2a: >10.15f} /Angstrom\n"
                )
                input_str += f"Charge = {charge}\nend of basis\n\n"
        input_str += "\n"

        if qmin.requests["soc"]:
            input_str += "AMFI\nangmom\n0 0 0\n"
        if qmin.template["baslib"]:
            input_str += f"BASLIB\n{qmin.template['baslib']}\n\n"
        input_str += "RICD\n"
        if qmin.template["method"] != "cms-pdft":
            input_str += f"CDTHreshold={qmin.template['cholesky_accu']}\n"
        if qmin.template["pcmset"]:
            input_str += f"RF-INPUT\nPCM-MODEL\nSOLVENT = {qmin.template['pcmset']['solvent']}\n"
            input_str += f"AARE = {qmin.template['pcmset']['aare']}\nR-MIN = {qmin.template['pcmset']['r-min']}\nEND of RF-INPUT\n\n"
        return input_str

    def _write_geom(self, atoms: list[str], coords: list[list[float]] | np.ndarray) -> str:
        """
        Generate xyz file from coords
        """
        geom_str = f"{len(atoms)}\n\n"
        for idx, (at, crd) in enumerate(zip(atoms, coords)):
            geom_str += f"{at}{idx+1}  {crd[0]*au2a:6f} {crd[1]*au2a:6f} {crd[2]*au2a:6f}\n"
        return geom_str

    def _create_aoovl(self) -> None:
        pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        if self.QMin.requests["nacdr"] or self.QMin.requests["grad"]:
            if self.QMin.template["method"] == "caspt2":
                self.log.error("NACs/Gradients are not possible with caspt2")
                raise ValueError()
            elif self.QMin.template["method"] in ("xms-caspt2", "ms-caspt2"):
                for mult, (state, root) in enumerate(zip(self.QMin.molecule["states"], self.QMin.template["roots"]), 1):
                    if state != root:
                        self.log.error(f"*pt2 with NACs/grad. Number of roots does not equal number of states in mult {mult}.")
                        raise ValueError

        # if (
        #     self.QMin.template["method"] in ("ms-caspt2", "xms-caspt2")
        #     and self.QMin.template["ipea"] > 0
        #     and (self.QMin.requests["grad"] or self.QMin.requests["nacdr"])
        # ):
        #     self.log.error("Analytical gradients/NACs are not possible with pt2 methods and ipea shift!")
        #     raise ValueError()

        if (
            self.QMin.template["pcmset"]
            and (self.QMin.requests["grad"] or self.QMin.requests["nacdr"])
        ):
            self.log.error("Analytical gradients/NACs are not possible with PCM!")
            raise ValueError()

        if self.QMin.requests["theodore"]:
            if not self._wfa or not self._hdf5:
                self.log.error("Theodore not possible without WFA or HDF5!")
                raise ValueError()
            if not self.QMin.resources["theodore_prop"] or not self.QMin.resources["theodore_fragment"]:
                self.log.error("theodore_prop and theodore_frag have to be set in resources!")
                raise ValueError()

        if self.QMin.requests["multipolar_fit"] or self.QMin.requests["density_matrices"] or self.QMin.requests["mol"]:
            if not self._hdf5:
                self.log.error("Densities, basis_set and multipolar_fit request require HDF5 support!")
                raise ValueError()
        if self.QMin.requests["multipolar_fit"] and self.QMin.molecule["point_charges"]:
            self.log.error("Multipolar fit not compatible with point charges!")
            raise ValueError()
        if self.QMin.requests["phases"]:
            self.QMin.requests["overlap"] = True

    def _get_molcas_features(self) -> None:
        """
        Get information about the MOLCAS installation (HDF5, WFA, MPI)
        """
        if os.path.isfile(os.path.join(self.QMin.resources["molcas"], "bin/wfa.exe")):
            self.log.debug("MOLCAS version supports WFA")
            self._wfa = True

        try:
            with sp.Popen(["ldd", os.path.join(self.QMin.resources["molcas"], "bin/rassi.exe")], stdout=sp.PIPE) as proc:
                modules = proc.stdout.read().decode()
                if re.search(r"libhdf5\.so", modules):
                    self.log.debug("MOLCAS version supports HDF5")
                    self._hdf5 = True

                if re.search(r"libmpi\.so", modules):
                    self.log.debug("MOLCAS version supports MPI")
                    self._mpi = True
        except FileNotFoundError:
            self.log.warning("ldd not found on this machine, feature check failed. Disable hdf5 and MPI support.")

    @staticmethod
    def get_molcas_version(path: str) -> tuple[int, int]:
        """
        Get version number of MOLCAS

        path:   Path to MOLCAS directory
        """

        with open(os.path.join(path, ".molcasversion"), "r", encoding="utf-8") as version_file:
            version = re.match(r"v(\d+)\.(\d+)", version_file.read())
            if not version:
                raise ValueError(f"No MOLCAS version found in {os.path.join(path, '.molcasversion')}")
        return int(version.group(1)), int(version.group(2))

    def getQMout(self) -> dict[str, np.ndarray]:
        """
        Parse MOLCAS output files and return requested properties
        """
        # Allocate matrices
        requests = set()
        for key, val in self.QMin.requests.items():
            if not val:
                continue
            requests.add(key)

        self.log.debug("Allocate space in QMout object")
        self.QMout.allocate(
            states=self.QMin.molecule["states"],
            natom=self.QMin.molecule["natom"],
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )

        # Fill QMout
        scratchdir = self.QMin.resources["scratchdir"]
        states = self.QMin.molecule["states"]

        if not self._hdf5:
            with open(os.path.join(scratchdir, "master/MOLCAS.out"), "r", encoding="utf-8") as f:
                master_out = f.read()
        else:
            master_out = h5py.File(os.path.join(scratchdir, "master/MOLCAS.rassi.h5"), "r+")

        if self.QMin.requests["theodore"] or self.QMin.requests["ion"]:
            ascii_out = master_out
            if not isinstance(ascii_out, str):
                with open(os.path.join(scratchdir, "master/MOLCAS.out"), "r", encoding="utf-8") as f:
                    ascii_out = f.read()

            if self.QMin.requests["theodore"]:
                # Get theodore output (nstates)
                theodore_raw = self._get_theodore(ascii_out)

                # Create final prop list with nmstates
                theodore_list = list(
                    zip(
                        self.QMin.resources["theodore_prop"],
                        np.zeros((len(self.QMin.resources["theodore_prop"]), self.QMin.molecule["nmstates"])),
                    )
                )
                # Expand raw output from nstates to nmstates
                s_cnt = 0
                for m, s in enumerate(states, 1):
                    for idx, (_, val) in enumerate(theodore_raw):
                        theodore_list[idx][1][s_cnt : s_cnt + s * m] = np.tile(val[sum(states[: m - 1]) : sum(states[:m])], m)
                    s_cnt += s * m
                self.QMout["prop1d"].extend(theodore_list)

            if self.QMin.requests["ion"]:
                dyson_mat = self._get_dyson(ascii_out)
                self.QMout["prop2d"].append(("ion", dyson_mat))

        if self.QMin.requests["soc"]:
            self.QMout["h"] = self._get_socs(master_out)

        if self.QMin.requests["h"]:
            np.einsum("ii->i", self.QMout["h"])[:] = self._get_energy(master_out)

        if self.QMin.requests["grad"]:
            for grad in self.QMin.maps["gradmap"]:
                with open(os.path.join(scratchdir, f"grad_{grad[0]}_{grad[1]}/MOLCAS.out"), "r", encoding="utf-8") as grad_file:
                    grad_out = self._get_grad(grad_file.read())
                    for key, val in self.QMin.maps["statemap"].items():
                        if (val[0], val[1]) == grad:
                            self.QMout["grad"][key - 1] = grad_out[: self.QMin.molecule["natom"], :]  # Filter MM
                            if self.QMin.molecule["point_charges"]:
                                self.QMout["grad_pc"][key - 1] = grad_out[self.QMin.molecule["natom"] :, :]  # Filter QM

        if self.QMin.requests["nacdr"]:
            for nac in self.QMin.maps["nacmap"]:
                with open(
                    os.path.join(scratchdir, f"nacdr_{nac[0]}_{nac[1]}_{nac[2]}_{nac[3]}/MOLCAS.out"), "r", encoding="utf-8"
                ) as nac_file:
                    nac_out = nac_file.read()
                    istate = None
                    jstate = None
                    for key, val in self.QMin.maps["statemap"].items():
                        if (val[0], val[1]) == (nac[0], nac[1]):
                            istate = key - 1
                        if (val[0], val[1]) == (nac[2], nac[3]):
                            jstate = key - 1
                    nacdr = self._get_nacdr(nac_out)
                    self.QMout["nacdr"][istate, jstate] = self.QMout["nacdr"][jstate, istate] = nacdr[
                        : self.QMin.molecule["natom"], :
                    ]  # Filter MM
                    self.QMout["nacdr"][jstate, istate] *= -1

                    if self.QMin.molecule["point_charges"]:
                        self.QMout["nacdr_pc"][istate, jstate] = self.QMout["nacdr_pc"][jstate, istate] = nacdr[
                            self.QMin.molecule["natom"] :, :
                        ]  # Filter QM
                        self.QMout["nacdr_pc"][jstate, istate] *= -1

        if self.QMin.requests["dm"]:
            # Full DM matrix in ascii file, sub matrices of mult in h5 files
            if isinstance(master_out, str):
                self.QMout["dm"] = self._get_dipoles(master_out)
            else:
                s_cnt = 0
                for m, s in enumerate(states, 1):
                    if s > 0:
                        with h5py.File(os.path.join(scratchdir, f"master/MOLCAS.rassi.{m}.h5"), "r") as dp:
                            for _ in range(m):
                                self.QMout["dm"][:, s_cnt : s_cnt + s, s_cnt : s_cnt + s] = dp["SFS_EDIPMOM"][:]
                                s_cnt += s

        if self.QMin.requests["overlap"]:
            # Full overlap in ascii file, sub matrices of mult in h5 files
            ovlp = None
            if isinstance(master_out, str):
                ovlp = self._get_overlaps(master_out)

            s_cnt, o_cnt = 0, 0
            for m, s in enumerate(states, 1):
                if s > 0:
                    if not isinstance(master_out, str):
                        with h5py.File(os.path.join(scratchdir, f"master/MOLCAS.rassi.ovlp.{m}.h5")) as f:
                            ovlp = self._get_overlaps(f)
                            for _ in range(m):
                                self.QMout["overlap"][s_cnt : s_cnt + s, s_cnt : s_cnt + s] = ovlp
                                s_cnt += s
                    else:
                        for _ in range(m):
                            self.QMout["overlap"][s_cnt : s_cnt + s, s_cnt : s_cnt + s] = ovlp[
                                o_cnt : o_cnt + s, o_cnt : o_cnt + s
                            ]
                            s_cnt += s
                        o_cnt += s
            if self.QMin.requests["phases"]:
                self.QMout["phases"] = deepcopy(np.einsum("ii->i", self.QMout["overlap"]))
                self.QMout["phases"][self.QMout["phases"] > 0] = 1
                self.QMout["phases"][self.QMout["phases"] < 0] = -1

        if self.QMin.requests["mol"] or self.QMin.requests["density_matrices"] or self.QMin.requests["multipolar_fit"]:
            # Parse basis
            mol, _, _, _, _, _ = tools.molden.load(os.path.join(scratchdir, "master/MOLCAS.rasscf.molden"))
            mol.basis = mol._basis
            self.QMout["mol"] = mol
            if self.QMin.requests["density_matrices"] or self.QMin.requests["multipolar_fit"]:
                self._get_densities(master_out["BASIS_FUNCTION_IDS"][:])
                self.get_densities()
                if self.QMin.requests["multipolar_fit"]:
                    self.QMout["multipolar_fit"] = self._resp_fit_on_densities()

        return self.QMout

    def read_and_append_densities(self):
        pass

    def _get_densities(self, ao_order: np.ndarray) -> None:
        """
        Parse densities from h5 files
        """
        states = self.QMin.molecule["states"]

        dens_one_mult = {}  # Transition densities between same mult
        dens_one_mult_spin = {}  # Spin transition densities between same mult
        trans_dens = {}  # Transition densities between mult and mult + 1
        trans_dens_spin = {}  # Spin transition densities between mult and mult + 1

        # MOLCAS orders by ml -> e.g. 2px, 3px, 4px, 2py, ...
        def sort_ao(key1, key2):  # reorder to 2px, 2py, 2pz, 3py, ...
            ao1 = ao_order[key1]
            ao2 = ao_order[key2]
            return (ao1[0] - ao2[0]) * 1000 + (ao1[2] - ao2[2]) * 100 + (ao1[1] * ao1[2] - ao2[1] * ao2[2])

        ao_sorted = sorted(list(range(ao_order.shape[0])), key=cmp_to_key(sort_ao))

        for m, s in enumerate(states, 1):
            if s < 1:  # skip 0 states
                continue
            # Densities between same mults
            with h5py.File(os.path.join(self.QMin.resources["scratchdir"], f"master/MOLCAS.rassi.{m}.h5"), "r+") as f:
                dens_one_mult[m] = f["SFS_TRANSITION_DENSITIES"][:]
                dens_one_mult_spin[m] = f["SFS_TRANSITION_SPIN_DENSITIES"][:]
            if not os.path.isfile(os.path.join(self.QMin.resources["scratchdir"], f"master/MOLCAS.rassi.trd{m}.h5")):
                continue
            # Densities between mult and mult + 1
            with h5py.File(os.path.join(self.QMin.resources["scratchdir"], f"master/MOLCAS.rassi.trd{m}.h5"), "r+") as f:
                trans_dens[m] = f["SFS_TRANSITION_DENSITIES"][:]
                trans_dens_spin[m] = f["SFS_TRANSITION_SPIN_DENSITIES"][:]

        # Matrix in h5 flattened, calc dimension for squared matrix
        dim = int(next(iter(dens_one_mult.values())).shape[2] ** 0.5)

        for s1, s2, spin in self.density_recipes["read"].keys():
            if spin == "tot":
                if s1.S == s2.S and s1.M == s2.M:
                    mult = s1.S
                    x, y = min(s1.N - 1, s2.N - 1), max(s1.N - 1, s2.N - 1) # matrices in h5 are triangle
                    dens = dens_one_mult
                else:
                    mult = min(s1.S, s2.S)
                    x, y = s1.N - 1 + states[mult], s2.N - 1
                    if s1.S < s2.S:
                        x, y = s1.N - 1, s2.N - 1 + states[mult]
                    dens = trans_dens
            elif spin == "q":
                if s1.S == s2.S and s1.M == s2.M:
                    mult = s1.S
                    x, y = min(s1.N - 1, s2.N - 1), max(s1.N - 1, s2.N - 1) # matrices in h5 are triangle
                    dens = dens_one_mult_spin
                else:
                    mult = min(s1.S, s2.S)
                    x, y = s1.N - 1 + states[mult], s2.N - 1
                    if s1.S < s2.S:
                        x, y = s1.N - 1, s2.N - 1 + states[mult]
                    dens = trans_dens_spin
            self.QMout["density_matrices"][(s1, s2, spin)] = dens[mult + 1][x, y, :].reshape(dim, -1)[
                        np.ix_(ao_sorted, ao_sorted)
                    ]


    def _get_energy(self, output_file: str | h5py.File) -> np.ndarray:
        """
        Extract energies from outputfile
        """
        energies = None
        if isinstance(output_file, str):
            match self.QMin.template["method"]:
                case "casscf":
                    energies = re.findall(r"RASSCF root number.*Total energy:\s+(.*)\n", output_file)
                case "xms-caspt2" | "caspt2":
                    energies = re.findall(r"CASPT2 Root.*Total energy:\s+(.*)\n", output_file)
                case "ms-caspt2":
                    energies = re.findall(r"MS-CASPT2 Root.*Total energy:\s+(.*)\n", output_file)
                case "cms-pdft":
                    energies = re.findall(r"CMS-PDFT Root.*Total energy:\s+(.*)\n", output_file)
            # Remove extra roots
            if self.QMin.template["method"] in ("casscf", "cms-pdft", "caspt2"):
                s_cnt = 0
                for m, s in enumerate(self.QMin.molecule["states"]):
                    if self.QMin.template["roots"][m] > s > 0:
                        for _ in range(self.QMin.template["roots"][m] - s):
                            del energies[s_cnt + s]
                    s_cnt += s

        else:
            energies = output_file["SFS_ENERGIES"][:].tolist()

        if energies is None:
            self.log.error("No energies found in output file!")
            raise ValueError()

        # Expand energy list by multiplicity
        expandend_energies = []
        for i in range(len((states := self.QMin.molecule["states"]))):
            expandend_energies += energies[sum(states[:i]) : sum(states[: i + 1])] * (i + 1)
        return np.asarray(expandend_energies, dtype=np.complex128)

    def _get_socs(self, output_file: str | h5py.File) -> np.ndarray:
        """
        Extract SOCs from outputfile
        """
        soc_mat = None
        if isinstance(output_file, str):
            socs = re.search(r"Real part\s+Imag part\s+Absolute\n(.*) -{70}\n", output_file, re.DOTALL)
            if socs is None:
                self.log.error("No SOCs found in output file!")
                raise ValueError()

            socs = np.asarray(convert_list(socs.group(1).split(), float)).reshape(-1, 9)[:, 6:8]
            socs_complex = np.zeros((socs.shape[0]), dtype=np.complex128)
            for idx, val in enumerate(socs):
                socs_complex[idx] = complex(val[0], val[1])
            idx = np.tril_indices(self.QMin.molecule["nmstates"])
            soc_mat = np.zeros((self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]), dtype=np.complex128)
            soc_mat[idx] = socs_complex
            soc_mat += soc_mat.T.conj()
            soc_mat = soc_mat.conj()
            soc_mat *= 4.556335e-6
        else:
            soc_mat = output_file["HSO_MATRIX_REAL"][:] + 1j * output_file["HSO_MATRIX_IMAG"][:]

        # Reorder multiplicities
        return soc_mat[np.ix_(self._h_sort, self._h_sort)]

    def _get_grad(self, output_file: str) -> np.ndarray:
        """
        Extract gradients from outputfile
        """
        if not (grad_block := re.search(r"X\s+Y\s+Z\s+-{90}\n(.*) -{90}", output_file, re.DOTALL)):
            self.log.error("No gradients in output file!")
            raise ValueError()

        grad = re.findall(r"(-?\d+\.\d+E[-|+]\d{2,3})", grad_block.group(1), re.DOTALL)
        return np.asarray(grad, dtype=float).reshape(-1, 3)

    def _get_nacdr(self, output_file: str) -> np.ndarray:
        """
        Extract NACs from outputfile
        """

        # Get NACs from output file
        nac_block = re.search(
            r"Total derivative coupling\W*[\w* ]+:[\s|\w]+-{90}\s+X\s+Y\s+Z\n -{90}\n([A-Z|\s|\.|\-{1}|\d|+]+) -{90}",
            output_file,
            re.DOTALL,
        )
        if not nac_block:
            self.log.error("No NACs in output file!")
            raise ValueError()
        nac = re.findall(r"(-?\d+\.\d+E[-|+]\d{2,3})", nac_block.group(1), re.DOTALL)

        # get NAC states and get overlaps
        state_i, state_j = re.findall(r"nac=(\d+) (\d+)", output_file)[0]
        ovlp_mat = self._get_overlaps(output_file)

        # Adapt NAC to phase
        return (
            np.asarray(nac, dtype=float).reshape(-1, 3)
            * ovlp_mat[int(state_i) - 1, int(state_i) - 1]
            * ovlp_mat[int(state_j) - 1, int(state_j) - 1]
        )

    def _get_overlaps(self, output_file: str | h5py.File) -> np.ndarray:
        """
        Extract overlaps from outputfile
        Return full matrix for ascii file, sub matrices for hdf5
        """
        ovlp = None
        if isinstance(output_file, str):
            mults = re.findall(r"SPIN=(\d+)", output_file)
            nmstates = sum(self.QMin.molecule["states"][int(m) - 1] for m in mults)
            ovlp = np.zeros((nmstates, nmstates))
            all_ovlps = re.findall(r"OVERLAP MATRIX FOR THE ORIGINAL STATES:\n([\s|\n|\d|\.|-]+)\+\+", output_file, re.DOTALL)
            s_cnt = 0
            for i, m in enumerate(mults):
                sub_mat = re.sub(r"\n", "\\t", all_ovlps[i])
                sub_mat = np.asarray(re.findall(r"(-?\d{1}\.\d{8})", sub_mat), dtype=float)

                states = self.QMin.molecule["states"][int(m) - 1]
                idx = np.tril_indices(states * 2)

                tmp_mat = -0.5 * np.eye(states * 2)
                tmp_mat[idx] += sub_mat
                tmp_mat += tmp_mat.T
                ovlp[s_cnt : s_cnt + states, s_cnt : s_cnt + states] = tmp_mat[states:, :states]
                s_cnt += states
        else:
            ovlp = output_file["ORIGINAL_OVERLAPS"][:]
            ovlp = ovlp[ovlp.shape[0] // 2 :, : ovlp.shape[0] // 2]

        return np.asarray(ovlp, dtype=float)

    def _get_dipoles(self, output_file: str) -> np.ndarray:
        """
        Extract (transition) dipole moments from outputfile
        """
        dipole_mat = np.zeros((3, self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]))

        # Find all occurences of dipole sub matrices
        all_dp = iter(
            re.findall(r"PROPERTY: MLTPL\s+1\d?\D+[1-3]\n[^\n]+\n[^\n]+\n([\s|\d|\.|E|\+|\-|\n]+)", output_file, re.DOTALL)
        )

        s_cnt = 0
        for mult, state in enumerate(self.QMin.molecule["states"], 1):
            dipoles = np.zeros((3, state, state), dtype=float)
            n_block = ceil(state / 4)  # Matrices are in blocks states*4
            for i in range(3):
                for block in range(n_block):
                    dipoles[i, :, block * 4 : block * 4 + 4] = np.asarray(
                        re.findall(r"(-?\d+\.\d+[E|D]?[\+|-]?\d{0,2})", next(all_dp))
                    ).reshape(state, -1)

            for _ in range(mult):
                dipole_mat[:, s_cnt : s_cnt + state, s_cnt : s_cnt + state] = dipoles
                s_cnt += state

        return dipole_mat

    def _get_theodore(self, output_file: str) -> list[tuple[str, np.ndarray]]:
        """
        Extract theodore props from outputfile
        """
        # Get all outputs from WFA
        if not (find_theo := re.findall(r"TheoDORE analysis of CT numbers \(Lowdin\)=+\n([^=]*)", output_file, re.DOTALL)):
            self.log.error("No theodore output found!")
            raise ValueError()

        # Extract values from submatrices
        find_theo = [
            np.array(re.findall(r"(-?\d+\.\d{6})", i), dtype=float).reshape(-1, len(self.QMin.resources["theodore_prop"]) + 2)
            for i in find_theo
        ]

        sub_it = iter(find_theo)
        tmp_states = self.QMin.molecule["states"][:]
        states = self.QMin.molecule["states"][:]
        theo_mat = np.zeros((sum(tmp_states), len(self.QMin.resources["theodore_prop"])))

        s_cnt = 0
        for mult, state in enumerate(tmp_states, 1):
            # Filter states with 0 contribution
            if state == 0 or (state == 1 and (len(tmp_states) < mult + 2 or tmp_states[mult + 1] == 0)):
                continue

            sub_mat = next(sub_it)[:, 2:]  # Skip dE and f
            if (s := state) > 1:
                for i, _ in enumerate(self.QMin.resources["theodore_prop"]):
                    theo_mat[s_cnt + 1 : s_cnt + s, i] = sub_mat[: s - 1, i]
                s -= 1

            if len(tmp_states) >= mult + 2 and tmp_states[mult + 1] > 0:
                for i, _ in enumerate(self.QMin.resources["theodore_prop"]):
                    theo_mat[
                        s_cnt + states[mult - 1] + states[mult] : s_cnt + states[mult - 1] + states[mult] + states[mult + 1], i
                    ] = sub_mat[s:, i]
                    tmp_states[mult + 1] = 1  # Avoid double counting

            s_cnt += states[mult - 1]

        return list(zip(self.QMin.resources["theodore_prop"], theo_mat.T))

    def _get_dyson(self, output_file: str) -> np.ndarray:
        """
        Extract dyson norms from outputfile
        """
        if not (find_dyson := re.search(r"\+\+ Dyson amplitudes Biorth.*?intensity([^\*]*)", output_file, re.DOTALL)):
            self.log.error("No dyson norms found in output!")

        # Extract s1, s2, val tuples
        dyson_val = re.findall(r"^\s+(\d+)\s+(\d+)\s+[\d\.]+\s+(\d+\.\d{5}E\+?\-?\d{2})", find_dyson.group(1), re.MULTILINE)
        dyson_raw = [(int(i[0]), int(i[1]), float(i[2])) for i in dyson_val]

        # Fill states*states array with values
        dyson_mat = np.zeros((self.QMin.molecule["nstates"], self.QMin.molecule["nstates"]))
        for i, j, v in dyson_raw:
            dyson_mat[i - 1, j - 1] = dyson_mat[j - 1, i - 1] = v

        # Expand state*state array to nmstates*nmstates
        dyson_nmat = np.zeros((self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]))
        for i in range(dyson_nmat.shape[0]):
            for j in range(i, dyson_nmat.shape[0]):
                m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                if not abs(ms1 - ms2) == 0.5:
                    continue
                if m1 > m2:
                    s1, s2, m1, m2, ms1, ms2 = s2, s1, m2, m1, ms2, ms1
                if ms1 < ms2:
                    factor = (ms1 + 1 + (m1 - 1) / 2) / m1
                else:
                    factor = (-ms1 + 1 + (m1 - 1) / 2) / m1
                x = sum(self.QMin.molecule["states"][: m1 - 1]) + s1 - 1
                y = sum(self.QMin.molecule["states"][: m2 - 1]) + s2 - 1
                dyson_nmat[i, j] = dyson_nmat[j, i] = dyson_mat[x, y] * factor
        return dyson_nmat

    def dyson_orbitals_with_other2(self, other, workdir, ncpu, mem):
        if self.get_molcas_version(self.QMin.resources["molcas"]) < (23, 10):
            self.log.error("Dyson orbital calculation requires MOLCAS version 23.10 or higher!")
            raise ValueError()

        mkdir(os.path.join(workdir, "dyson"))

        qmin1 = self.QMin
        qmin2 = other.QMin
        save1 = qmin1.save["savedir"]
        save2 = qmin2.save["savedir"]
        step1 = qmin1.save["step"]
        step2 = qmin2.save["step"]
        nstates1 = qmin1.molecule["states"]
        nstates2 = qmin2.molecule["states"]

        # Getting all non-zero DOs
        dyson_orbitals_from_rassi = {}
        dyson_orbitals_from_wigner_eckart = []
        for s1 in self.states:
            for s2 in other.states:
                for spin in ["a", "b"]:
                    if self.dyson_logic(s1, s2, spin):
                        if s1.M == s1.S and s2.M == s2.S:
                            if (s1.S, s2.S) in dyson_orbitals_from_rassi:
                                dyson_orbitals_from_rassi[(s1.S, s2.S)].append((s1, s2, spin))
                            else:
                                dyson_orbitals_from_rassi[(s1.S, s2.S)] = [(s1, s2, spin)]
                        else:
                            dyson_orbitals_from_wigner_eckart.append((s1, s2, spin))

        phi = {}
        # Calling RASSI for each pair of multiplicities
        for (dyson_s1, dyson_s2), dos in dyson_orbitals_from_rassi.items():
            phi_work = np.zeros(((n_s1 := nstates1[dyson_s1]), (n_s2 := nstates2[dyson_s2]), self.QMout["mol"].nao))
            # Fetch JOBIPH files
            shutil.copy(os.path.join(save1, f"MOLCAS.{dyson_s1+1}.JobIph.{step1}"), os.path.join(workdir, "dyson/JOB001"))
            shutil.copy(os.path.join(save2, f"MOLCAS.{dyson_s2+1}.JobIph.{step2}"), os.path.join(workdir, "dyson/JOB002"))

            # Make RASSI.input
            input_str = self._write_gateway(self.QMin)
            input_str += self._write_seward(self.QMin)
            input_str += "&RASSI\n"
            input_str += f"NROFJOBIPHS\n2 {n_s1} {n_s2}\n"
            input_str += f"{' '.join([str(j) for j in range(1, n_s1 + 1)])}\n{' '.join([str(j) for j in range(1, n_s2 + 1)])}\n"
            input_str += "MEIN\n"
            input_str += "CIPR\nTHRS=0.000005d0\n"
            input_str += "DYSON\n"
            input_str += f"DYSEXPORT\n{n_s1 + n_s2} {n_s1 + n_s2}\n\n"
            input_str += f"*{nstates1}, {nstates2}\n"
            input_str += f"*{nstates1[dyson_s1]}, {nstates2[dyson_s2]}\n"
            with open(os.path.join(workdir, "dyson/RASSI.input"), "w", encoding="utf-8") as f:
                f.write(input_str)

            writefile(
                os.path.join(workdir, "dyson/MOLCAS.xyz"),
                self._write_geom(self.QMin.molecule["elements"], self.QMin.coords["coords"]),
            )
            # Call pymolcas
            if (
                code := self.run_program(
                    os.path.join(workdir, "dyson"),
                    self.QMin.resources["driver"] + " RASSI.input",
                    "RASSI.out",
                    "RASSI.err",
                    {
                        "MOLCASMEM": str(mem),
                        "MOLCAS_MEM": str(mem),
                        "WorkDir": os.path.join(workdir, "dyson"),
                        "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH"),
                    },
                )
            ) != 0:
                self.log.error(f"Dyson orbital calculation failed with exit code {code}")
                raise ValueError()

            with h5py.File(os.path.join(workdir, "dyson/RASSI.rassi.h5"), "r") as f:
                ao_order = np.asarray(f["BASIS_FUNCTION_IDS"][:])

            # MOLCAS orders by ml -> e.g. 2px, 3px, 4px, 2py, ...
            def sort_ao(key1, key2):  # reorder to 2px, 2py, 2pz, 3py, ...
                ao1 = ao_order[key1]
                ao2 = ao_order[key2]
                return (ao1[0] - ao2[0]) * 1000 + (ao1[2] - ao2[2]) * 100 + (ao1[1] * ao1[2] - ao2[1] * ao2[2])

            ao_sorted = sorted(list(range(ao_order.shape[0])), key=cmp_to_key(sort_ao))
            # Parse
            for i in range(n_s1):
                with open(os.path.join(workdir, f"dyson/RASSI.DysOrb.SF.{i+1}"), "r", encoding="utf-8") as f:
                    orbitals = re.findall(r"ORBITAL\s+\d+\s+\d+\n([\s+-?\d+\.\d{14}E+|\-\d{2}]*)", f.read(), re.DOTALL)
                    for j, orbital in enumerate(orbitals):
                        phi_work[i, j, :] = np.asarray(re.findall(r"\-?\d+\.\d{14}E[\-|+]\d+", orbital), dtype=float)[
                            np.ix_(ao_sorted)
                        ]

            for s1, s2, spin in dos:
                phi[(s1, s2, spin)] = phi_work[s1.N - 1, s2.N - 1, :]

        all_done = False
        while not all_done:
            for thes1, thes2, thespin in dyson_orbitals_from_wigner_eckart:
                for (s1, s2, spin), phi_work in phi.items():
                    to_append = {}
                    if s1 // thes1 and s2 // thes2:
                        if thespin == "a":
                            numerator = wigner_3j(
                                thes1.S / 2.0, 1.0 / 2.0, thes2.S / 2.0, thes1.M / 2.0, -1.0 / 2.0, -thes2.M / 2.0
                            )
                        elif thespin == "b":
                            numerator = wigner_3j(
                                thes1.S / 2.0, 1.0 / 2.0, thes2.S / 2.0, thes1.M / 2.0, 1.0 / 2.0, -thes2.M / 2.0
                            )
                        if spin == "a":
                            denominator = wigner_3j(s1.S / 2.0, 1.0 / 2.0, s2.S / 2.0, s1.M / 2.0, -1.0 / 2.0, -s2.M / 2.0)
                        elif spin == "b":
                            denominator = wigner_3j(s1.S / 2.0, 1.0 / 2.0, s2.S / 2.0, s1.M / 2.0, 1.0 / 2.0, -s2.M / 2.0)
                        if denominator != 0:
                            to_append[(thes1, thes2, thespin)] = (
                                (-1.0) ** (thes2.M / 2.0 - s2.M / 2.0)
                                * float(numerator.evalf())
                                / float(denominator.evalf())
                                * phi_work
                            )
                        break
                for do, phi_work in to_append.items():
                    phi[do] = phi_work
            all_done = all(do in phi for do in dyson_orbitals_from_wigner_eckart)
        return phi


if __name__ == "__main__":
    SHARC_MOLCAS().main()
