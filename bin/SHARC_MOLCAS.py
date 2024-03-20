import datetime
import os
import re
import shutil
import subprocess as sp
from copy import deepcopy
from functools import cmp_to_key
from io import TextIOWrapper
from itertools import product
from typing import Any

import h5py
import numpy as np
from constants import au2a
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import convert_list, expand_path, mkdir, writefile

__all__ = ["SHARC_MOLCAS"]

AUTHORS = ""
VERSION = ""
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "MOLCAS"
DESCRIPTION = ""

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
        # raw data request
        "basis_set",
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

        # Features of MOLCAS installation
        self._hdf5 = False
        self._wfa = False
        self._mpi = False

        # Save sort order for H matrix
        self._h_sort = None

        # Add resource keys
        self.QMin.resources.update({"molcas": None, "mpi_parallel": False, "delay": 0.0, "dry_run": False, "driver": None})

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
                "rootpad": list(range(8)),
                "method": "casscf",
                "functional": "t:pbe",
                "ipea": 0.25,
                "imaginary": 0.0,
                "frozen": None,
                "gradaccudefault": 1e-4,
                "gradaccumax": 1e-2,
                "pcmset": None,
                "pcmstate": None,
                "iterations": [200, 100],
                "cholesky_accu": 1e-4,
                "rasscf_thrs": [1e-8, 1e-4, 1e-4],
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
                "rootpad": list,
                "method": str,
                "functional": str,
                "ipea": float,
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
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        pass

    def create_restart_files(self) -> None:
        pass

    def read_resources(self, resources_file: str = "MOLCAS.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

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
        driver = None
        for p in os.walk(self.QMin.resources["molcas"]):
            if "pymolcas" in p[2]:
                driver = os.path.join(p[0], "pymolcas")
                break

        if os.path.isfile(driver):
            self.QMin.resources.update({"driver": driver})


        if not self.QMin.resources["driver"]:
            self.log.error(f"No driver found in {self.QMin.resources['molcas']}")
            raise ValueError()

        # WFOVERLAP
        if self.QMin.resources["wfoverlap"]:
            self.QMin.resources["wfoverlap"] = expand_path(self.QMin.resources["wfoverlap"])

        # Check orb init and guess
        if self.QMin.save["always_guess"] and self.QMin.save["always_orb_init"]:
            self.log.error("always_guess and always_orb_init cannot be used together!")
            raise ValueError()

        # Set RAM limit
        os.environ["MOLCASMEM"] = str(self.QMin.resources["memory"])
        os.environ["MOLCAS_MEM"] = str(self.QMin.resources["memory"])

    def read_template(self, template_file: str = "MOLCAS.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Roots
        self.QMin.template["roots"] = convert_list(self.QMin.template["roots"])
        self.QMin.template["rootpad"] = convert_list(self.QMin.template["rootpad"])

        # RASSCF_thresholds
        self.QMin.template["rasscf_thrs"] = convert_list(self.QMin.template["rasscf_thrs"], float)

        if not all(map(lambda x: x >= 0, [*self.QMin.template["roots"], *self.QMin.template["rootpad"]])):
            self.log.error("roots and rootpad must contain positive integers.")
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

        self.QMin.template["rootpad"] = self.QMin.template["rootpad"][: len(self.QMin.template["roots"])]

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

        # Validate method
        if self.QMin.template["method"].lower() not in [
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

            # TODO: wfoverlap, theodore?

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
            self.log.debug("All hobs finished successful")

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
            exit_code = self.run_program(workdir, f"{qmin.resources['driver']} MOLCAS.input", "MOLCAS.out", "MOLCAS.err")
            if exit_code != 96:
                break
            qmin.template["gradaccudefault"] *= 10
        endtime = datetime.datetime.now()

        return exit_code, endtime - starttime

    def _copy_run_files(self, workdir: str) -> None:
        """
        Copy files from master to grad/nac folder
        """
        self.log.debug("Copy run files from master")
        re_runfiles = ("ChVec", "QVec", "ChRed", "ChDiag", "ChRst", "ChMap", "RunFile", "GRIDFILE", "NqGrid", "Rotate.txt")
        for file in os.listdir(os.path.join(self.QMin.resources["scratchdir"], "master")):
            if any(i in file for i in re_runfiles):
                shutil.copy(os.path.join(self.QMin.resources["scratchdir"], "master", file), os.path.join(workdir, file))
        shutil.copy(os.path.join(self.QMin.resources["scratchdir"], "master/MOLCAS.OneInt"), os.path.join(workdir, "ONEINT"))

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
        list_to_do = list((i, j) for i, j in enumerate(qmin.molecule["states"]))
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
                    if os.path.isfile(os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.JobIph.init")):
                        tasks.append(["copy", os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.JobIph.init"), "JOBOLD"])
                        is_jobiph = True
                    elif os.path.isfile(os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.RasOrb.init")):
                        tasks.append(["copy", os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.RasOrb.init"), "INPORB"])
                        is_rasorb = True
                case {"samestep": True}:
                    tasks.append(
                        ["link", os.path.join(qmin.save["savedir"], f"MOLCAS.{mult+1}.JobIph.{qmin.save['step']}"), "JOBOLD"]
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

            # Copy Do_Rotate.txt for cms-pdft
            if qmin.template["method"] == "cms-pdft" and not qmin.save["init"]:
                # TODO: CIRESTART
                tasks.append(["copy", os.path.join(qmin.save["savedir"], f"Do_Rotate.{mult+1}.txt"), "Do_Rotate.txt"])

            # RASSCF block
            tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], is_jobiph, is_rasorb])
            if qmin.template["method"] == "cms-pdft":
                tasks[-1].append(["CMSI"])

            # ION and MOLDEN requests
            if qmin.requests["ion"]:
                tasks.append(["copy", "MOLCAS.RasOrb", f"MOLCAS.{mult+1}.RasOrb"])
            if qmin.requests["molden"]:
                tasks.append(["copy", "MOLCAS.rasscf.molden", f"MOLCAS.{mult+1}.molden"])

            # Generate DFT task
            match qmin.template["method"]:
                case "casscf":
                    tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
                case "cms-pdft":
                    keys = [f"KSDFT={qmin.template['functional']}"]
                    keys.append("noGrad")
                    keys += ["MSPDFT", "WJOB", "CMMI=0", "CMSS=Do_Rotate.txt", "CMTH=1.0d-10"]
                    tasks.append(["mcpdft", keys])
                    tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
                case _:  # PT2 methods
                    if not qmin.save["samestep"]:
                        tasks.append(["caspt2", mult + 1, states, qmin.template["method"]])
                        tasks.append(["copy", "MOLCAS.JobMix", f"MOLCAS.{mult+1}.JobIph"])

        # Do property jobs last
        for mult, states in list_to_do:
            if states == 0:
                continue
            tasks += self._gen_property_tasks(qmin, mult, states)

        if qmin.requests["soc"]:
            i = 0
            roots = []
            for mult, states in enumerate(qmin.molecule["states"]):
                if states == 0:
                    continue
                i += 1
                roots.append(states)
                tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", f"JOB{i:03d}"])
            tasks.append(["rassi", "soc", roots])

        return tasks

    def _gen_property_tasks(self, qmin: QMin, mult: int, states: int) -> list[list[Any]]:
        """
        Generate tasklist for properties
        """
        tasks = []
        if qmin.requests["overlap"]:
            if qmin.control["master"]:
                tasks.append(
                    ["link", os.path.join(qmin.save["savedir"], f'MOLCAS.{mult+1}.JobIph.{qmin.save["step"]-1}'), "JOB001"]
                )
                tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOB002"])
            else:
                tasks.append(["link", f"MOLCAS.{mult + 1}.JobIph", "JOB001"])
                tasks.append(["link", "MOLCAS.JobIph", "JOB002"])
            tasks.append(["rassi", "overlap", [states, states]])
            if qmin.requests["multipolar_fit"]:
                for i in range(states):
                    for j in range(i + 1):
                        tasks.append(["copy", f"TRD2_{i+states+1:03d}_{j+states+1:03d}", f"TRD_{mult+1}_{i+1:03d}_{j+1:03d}"])

        elif qmin.requests["dm"] or qmin.requests["ion"] or qmin.requests["multipolar_fit"]:
            tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOB001"])
            tasks.append(["rassi", "dm", [states]])
            if qmin.requests["multipolar_fit"]:
                for i in range(states):
                    for j in range(i + 1):
                        tasks.append(["copy", f"TRD2_{i+1:03d}_{j+1:03d}", f"TRD_{mult+1}_{i+1:03d}_{j+1:03d}"])

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
            if qmin.template["roots"][mult] == 1:
                tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                if qmin.template["method"] in ("ms-caspt2", "xms-caspt2"):
                    self.log.error("Single state gradient with MS/XMS-CASPT2")
                    raise ValueError()
                tasks.append(["alaska"])
            else:
                tasks.append(["copy", f"$master_path/MOLCAS.{mult+1}.JobIph", "JOBOLD"])
                match qmin.template["method"]:
                    case "cms-pdft":
                        # tasks.append(["copy", os.path.join(qmin.save["savedir"], f"Do_Rotate.{mult+1}.txt"), "Do_Rotate.txt"])
                        tasks.append(
                            [
                                "rasscf",
                                mult + 1,
                                qmin.template["roots"][mult],
                                True,
                                False,
                                [f"RLXROOT={grad[1]}", "CMSI"],
                            ]
                        )
                        tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD", "MSPDFT", "WJOB"]])
                        tasks.append(["alaska", grad[1]])
                    case "casscf":  # TODO: CIRESTART
                        tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                        tasks.append(["mclr", qmin.template["gradaccudefault"], f"sala={grad[1]}"])
                        tasks.append(["alaska"])
                    case "ms-caspt2" | "xms-caspt2" | "caspt2":
                        tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                        tasks.append(["caspt2", mult + 1, states, qmin.template["method"], f"GRDT\nrlxroot = {grad[1]}"])
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
                    tasks.append(["copy", os.path.join(qmin.save["savedir"], f"Do_Rotate.{mult+1}.txt"), "Do_Rotate.txt"])
                    tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False, ["CMSI"]])
                    tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD", "MSPDFT", "WJOB"]])
                    tasks.append(["mclr", qmin.template["gradaccudefault"], f"nac={nac[1]} {nac[3]}"])
                    tasks.append(["alaska"])
                case "ms-caspt2" | "xms-caspt2":
                    tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False])
                    tasks.append(["caspt2", mult + 1, states, qmin.template["method"], f"GRDT\nnac = {nac[1]} {nac[3]}"])
                    tasks.append(["alaska", nac[1], nac[3]])

            tasks += self._gen_property_tasks(qmin, mult, states)
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
        if task[1] != "soc" and qmin.control["master"] and qmin.requests["ion"]:
            input_str += "CIPR\nTHRS=0.000005d0\n"
        if task[1] == "dm" and qmin.requests["multipolar_fit"]:
            input_str += "TRD1\n"
        if task[1] == "soc":
            input_str += "SPINORBIT\nSOCOUPLING=0.0d0\nEJOB\n"
        elif task[1] == "overlap":
            input_str += "OVERLAPS\n"
            if qmin.control["master"] and qmin.requests["multipolar_fit"]:
                input_str += "TRD1\n"
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
        if (nactel[0] - task[1]) % 2 == 0:
            nactel[0] -= 1
        input_str = f"&RASSCF\nSPIN={task[1]}\nNACTEL={' '.join(str(n) for n in nactel)}\n"
        input_str += f"INACTIVE={qmin.template['inactive']}\nRAS2={qmin.template['ras2']}\n"
        input_str += f"ITERATIONS={qmin.template['iterations'][0]},{qmin.template['iterations'][1]}\n"
        if qmin.template["ras1"]:
            input_str += f"RAS1={qmin.template['ras1']}\n"
        if qmin.template["ras3"]:
            input_str += f"RAS3={qmin.template['ras3']}\n"
        if qmin.template["rootpad"][task[1] - 1] == 0:
            input_str += f"CIROOT={task[2]} {task[2]} 1\n"
        else:
            input_str += f"CIROOT={task[2]} {task[2]+ qmin.template['rootpad'][task[1] - 1]}; "
            input_str += " ".join(str(i + 1) for i in range(task[2]))
            input_str += " ;"
            input_str += " ".join("1" for _ in range(task[2]))
            input_str += " \n"

        if qmin.template["method"] not in ("ms-caspt2", "xms-caspt2"):
            input_str += "ORBLISTING=NOTHING\nPRWF=0.1\n"
        if qmin.template["method"] == "cms-pdft":
            input_str += "CMSInter\n"
        if qmin.maps["gradmap"] and len(qmin.maps["gradmap"]) > 0:
            input_str += "THRS=1.0e-10 1.0e-06 1.0e-06\n"
        else:
            input_str += "THRS=" + " ".join(f"{i:14.12f}" for i in qmin.template["rasscf_thrs"]) + "\n"
        if task[3]:
            input_str += "JOBIPH\n"
        if task[4]:
            input_str += "LUMORB\n"
        if len(task) > 5:
            input_str += "\n".join(str(a) for a in task[5]) + "\n"
        if qmin.template["pcmset"]:
            if task[1] == qmin.template["pcmstate"][0]:
                input_str += f"RFROOT = {qmin.template['pcmstate'][1]}\n"
            else:
                input_str += "NONEQUILIBRIUM\n"
        input_str += "\n"
        return input_str

    def _write_seward(self, qmin: QMin) -> str:
        """
        Write SEWARD part of MOLCAS input string
        """
        input_str = "&SEWARD\n"  # DOANA\n"
        if qmin.template["method"] == "cms-pdft":
            input_str += "GRID INPUT\nNORO\nNOSC\nEND OF GRID INPUT\n"
        input_str += "\n"
        return input_str

    def _write_gateway(self, qmin: QMin) -> str:
        """
        Write GATEWAY part of MOLCAS input string
        """
        # TODO: qmmm
        input_str = f"&GATEWAY\nCOORD=MOLCAS.xyz\nGROUP=NOSYM\nBASIS={qmin.template['basis']}\n"
        if qmin.requests["soc"]:
            input_str += "AMFI\nangmom\n0 0 0\n"
        if qmin.template["baslib"]:
            input_str += f"BASLIB\n{qmin.template['baslib']}\n\n"
        input_str += f"RICD\nCDTHreshold={qmin.template['cholesky_accu']}\n"
        if qmin.template["pcmset"]:
            input_str += f"TF-INPUT\nPCM-MODEL\nSOLVENT = {qmin.template['pcmset']['solvent']}\n"
            input_str += f"AARE = {qmin.template['pcmset']['aare']}\nR-MIN = {qmin.template['pcmset']['r-min']}"
        input_str += "\n"
        return input_str

    def _write_geom(self, atoms: list[str], coords: list[list[float]] | np.ndarray) -> str:
        """
        Generate xyz file from coords
        """
        # TODO: qmmm
        geom_str = f"{len(atoms)}\n\n"
        for idx, (at, crd) in enumerate(zip(atoms, coords)):
            geom_str += f"{at}{idx+1}  {crd[0]*au2a:6f} {crd[1]*au2a:6f} {crd[2]*au2a:6f}\n"
        return geom_str

    def _create_aoovl(self) -> None:
        pass

    def dyson_orbitals_with_other(self, other) -> None:
        pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                self.log.error(f"Found unsupported request {req}.")
                raise ValueError()

        if self.QMin.template["method"] not in ["casscf", "ms-caspt2", "xms-caspt2"] and self.QMin.requests["nacdr"]:
            self.log.error("NACs are only possible with casscf, ms/xms-capt2!")
            raise ValueError()

        if self.QMin.template["method"] != "casscf":
            if self.QMin.requests["grad"] or self.QMin.requests["nacdr"]:
                if sum(self.QMin.template["rootpad"]) > 0:
                    self.log.error("Gradients/NACs with rootpad only possible with CASSCF!")
                    raise ValueError()

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
        if self.QMin.requests["theodore"]:
            theodore_arr = np.zeros(
                (
                    self.QMin.molecule["nmstates"],
                    len(self.QMin.resources["theodore_prop"]) + len(self.QMin.resources["theodore_fragment"]) ** 2,
                )
            )

        # Fill QMout
        scratchdir = self.QMin.resources["scratchdir"]

        if not self._hdf5:
            with open(os.path.join(scratchdir, "master/MOLCAS.out"), "r", encoding="utf-8") as f:
                master_out = f.read()
        else:
            master_out = h5py.File(os.path.join(scratchdir, "master/MOLCAS.rassi.h5"), "r+")

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
                            self.QMout["grad"][key - 1] = grad_out
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
                    self.QMout["nacdr"][istate, jstate] = self._get_nacdr(nac_out)
        if self.QMin.requests["dm"]:
            self.QMout["dm"] = self._get_dipoles(master_out)

    def _get_energy(self, output_file: str | h5py.File) -> np.ndarray:
        """
        Extract energies from outputfile
        """
        energies = None
        if isinstance(output_file, str):
            match self.QMin.template["method"]:
                case "casscf":
                    energies = re.findall(r"RASSCF root number\s+\d+\s+Total energy:\s+(.*)\n", output_file)
                    del energies[self.QMin.molecule["states"][0]]  # Delete extra singlet
                case "xms-caspt2" | "caspt2":
                    energies = re.findall(r"CASPT2 Root\s+\d+\s+Total energy:\s+(.*)\n", output_file)
                case "ms-caspt2":
                    energies = re.findall(r"MS-CASPT2 Root\s+\d+\s+Total energy:\s+(.*)\n", output_file)
                case "cms-pdft":
                    energies = re.findall(r"CMS-PDFT Root\s+\d+\s+Total energy:\s+(.*)\n", output_file)
                    del energies[self.QMin.molecule["states"][0]]  # Delete extra singlet
        else:
            energies = output_file["SFS_ENERGIES"][:].tolist()

        if energies is None:
            self.log.error("No energies found in output file!")
            raise ValueError()

        # Expand energy list by multiplicity
        states = self.QMin.molecule["states"]
        expandend_energies = []
        for i in range(len(states)):
            expandend_energies += energies[sum(states[:i]) : sum(states[: i + 1])] * (i + 1)
        return np.asarray(expandend_energies, dtype=np.complex128)

    def _get_socs(self, output_file: str | h5py.File) -> np.ndarray:
        """
        Extract SOCs from outputfile
        """
        soc_mat = None
        if isinstance(output_file, str):
            socs = re.search(r"Real part\s+Imag part\s+Absolute\n(.*) -{70}\n", output_file, re.DOTALL)
            socs = np.asarray(convert_list(socs.group(1).split(), float)).reshape(-1, 9)[:, 6:8]
            socs_complex = np.zeros((socs.shape[0]), dtype=np.complex128)
            for idx, val in enumerate(socs):
                socs_complex[idx] = complex(val[0], val[1])
            idx = np.tril_indices(self.QMin.molecule["nmstates"])
            soc_mat = np.zeros((self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]), dtype=np.complex128)
            soc_mat[idx] = socs_complex.reshape(-1)
            soc_mat += soc_mat.T.conj()
            soc_mat = soc_mat.conj()
            soc_mat *= 4.556335e-6
        else:
            soc_mat = output_file["HSO_MATRIX_REAL"][:] + 1j * output_file["HSO_MATRIX_IMAG"][:]

        # Reorder multiplicities
        soc_mat = soc_mat[np.ix_(self._h_sort, self._h_sort)]

        if soc_mat is None:
            self.log.error("No SOCs found in output file!")
            raise ValueError()
        return soc_mat

    def _get_grad(self, output_file: str) -> np.ndarray:
        """
        Extract gradients from outputfile
        """
        grad_block = re.search(r"X\s+Y\s+Z\s+-{90}\n(.*) -{90}", output_file, re.DOTALL)
        if not grad_block:
            self.log.error("No gradients in output file!")
            raise ValueError()

        grad = re.findall(r"(-{0,1}\d+\.\d+E[-|+]\d{2,3})", grad_block.group(1), re.DOTALL)
        return np.asarray(grad, dtype=float).reshape(-1, 3)

    def _get_nacdr(self, output_file: str) -> np.ndarray:
        """
        Extract NACs from outputfile
        """
        nac_block = re.search(
            r"Total derivative coupling\W*[\w* ]+:[\s|\w]+-{90}\s+X\s+Y\s+Z\n -{90}\n([A-Z|\s|\.|\-{1}|\d|+]+) -{90}",
            output_file,
            re.DOTALL,
        )
        if not nac_block:
            self.log.error("No NACs in output file!")
            raise ValueError()
        nac = re.findall(r"(-{0,1}\d+\.\d+E[-|+]\d{2,3})", nac_block.group(1), re.DOTALL)
        return np.einsum("ij->ji", np.asarray(nac, dtype=float).reshape(-1, 3))

    def _get_dipoles(self, output_file: str | h5py.File) -> np.ndarray:
        """
        Extract (transition) dipole moments from outputfile
        """
        dipoles = None
        if isinstance(output_file, str):
            pass
        else:
            dipoles = output_file["SFS_EDIPMOM"][:]

        if dipoles is None:
            self.log.error("No gradients in output file!")
            raise ValueError()
        return np.asarray(dipoles, dtype=float)


if __name__ == "__main__":
    SHARC_MOLCAS().main()
