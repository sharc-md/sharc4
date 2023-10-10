import datetime
import math
import os
import subprocess as sp
import time
import re
from abc import abstractmethod
from datetime import date
from io import TextIOWrapper
from multiprocessing import Pool
from typing import Optional

import numpy as np
from qmin import QMin
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import containsstring, readfile, safe_cast

all_features = {
    "h",
    "soc",
    "dm",
    "grad",
    "nacdr",
    "overlap",
    "phases",
    "ion",
    "dmdr",
    "socdr",
    "multipolar_fit",
    "theodore",
    "point_charges",
    # raw data request
    "basis_set",
    "wave_functions",
    "density_matrices",
}


class SHARC_ABINITIO(SHARC_INTERFACE):
    """
    Abstract base class for ab-initio interfaces
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add ab-initio specific keywords to template
        self.QMin.template.update({"charge": None, "paddingstates": None})

        self.QMin.template.types.update({"charge": list, "paddingstates": list})
        # Add ab-initio specific keywords to resources
        self.QMin.resources["delay"] = 0.0

        self.QMin.resources.types["delay"] = float

        # Add list of slots per pool to control
        self.QMin.control["nslots_pool"] = []

        self.QMin.control.types["nsplots_pool"] = list

    @staticmethod
    @abstractmethod
    def authors() -> str:
        return "Severin Polonius, Sebastian Mai"

    @staticmethod
    @abstractmethod
    def version() -> str:
        return "3.0"

    @staticmethod
    @abstractmethod
    def versiondate() -> date:
        return date(2021, 7, 15)

    @staticmethod
    @abstractmethod
    def name() -> str:
        return "base"

    @staticmethod
    @abstractmethod
    def description() -> str:
        return "Abstract base class for SHARC interfaces."

    @staticmethod
    @abstractmethod
    def changelogstring() -> str:
        return "This is the changelog string"

    @staticmethod
    @abstractmethod
    def about() -> str:
        return "Name and description of the interface"

    @abstractmethod
    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    @abstractmethod
    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    @abstractmethod
    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        return

    @abstractmethod
    def print_qmin(self) -> None:
        pass

    @abstractmethod
    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen, stripping)

        returns (exit code, runtime)
        """

    @abstractmethod
    def read_template(self, template_file: str) -> None:
        super().read_template(template_file)

        # Check if charge in template and autoexpand if needed
        if self.QMin.template["charge"]:
            if len(self.QMin.template["charge"]) == 1:
                charge = int(self.QMin.template["charge"][0])
                if (self.QMin.molecule["Atomcharge"] + charge) % 2 == 1 and len(self.QMin.molecule["states"]) > 1:
                    self.log.info("HINT: Charge shifted by -1 to be compatible with multiplicities.")
                    charge -= 1
                self.QMin.template["charge"] = [i % 2 + charge for i in range(len(self.QMin.molecule["states"]))]
                self.log.info(
                    f'HINT: total charge per multiplicity automatically assigned, please check ({self.QMin.template["charge"]}).'
                )
                self.log.info('You can set the charge in the template manually for each multiplicity ("charge 0 +1 0 ...")')
            elif len(self.QMin.template["charge"]) >= len(self.QMin.molecule["states"]):
                self.QMin.template["charge"] = [
                    int(self.QMin.template["charge"][i]) for i in range(len(self.QMin.molecule["states"]))
                ]
                compatible = True
                for imult, cha in enumerate(self.QMin.template["charge"]):
                    if not (self.QMin.molecule["Atomcharge"] + cha + imult) % 2 == 0:
                        compatible = False
                if not compatible:
                    self.log.warning(
                        "Charges from template not compatible with multiplicities!  (this is probably OK if you use QM/MM)"
                    )
            else:
                raise ValueError('Length of "charge" does not match length of "states"!')
        else:
            self.QMin.template["charge"] = [i % 2 for i in range(len(self.QMin.molecule["states"]))]

    @abstractmethod
    def read_resources(self, resources_file: str, kw_whitelist: Optional[list[str]] = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

    @abstractmethod
    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

    @abstractmethod
    def printQMout(self) -> None:
        super().writeQMout()

    @abstractmethod
    def setup_interface(self) -> None:
        """
        Create maps from QMin object
        """
        self.log.debug("Setup interface -> building maps")

        # Setup gradmap
        if self.QMin.requests["grad"]:
            self.log.debug("Building gradmap")
            self.QMin.maps["gradmap"] = set({tuple(self.QMin.maps["statemap"][i][0:2]) for i in self.QMin.requests["grad"]})

        # Setup densmap
        if self.QMin.requests["multipolar_fit"]:
            self.log.debug("Building densmap")
            self.QMin.maps["densmap"] = set(
                {tuple(self.QMin.maps["statemap"][i][0:2]) for i in self.QMin.requests["multipolar_fit"]}
            )

        # Setup nacmap
        if self.QMin.requests["nacdr"]:
            if self.QMin.requests["nacdr"] == ["all"]:
                mat = [
                    (i + 1, j + 1) for i in range(self.QMin.molecule["nmstates"]) for j in range(self.QMin.molecule["nmstates"])
                ]
                # self.QMin.requests["nacdr"] = mat
            else:
                mat = self.QMin.requests["nacdr"]
            self.log.debug("Building nacmap")
            self.QMin.maps["nacmap"] = set()
            for i in mat:
                m1, s1, ms1 = self.QMin.maps["statemap"][int(i[0])]
                m2, s2, ms2 = self.QMin.maps["statemap"][int(i[1])]
                if m1 != m2 or i[0] == i[1] or ms1 != ms2 or s1 > s2:
                    continue
                self.QMin.maps["nacmap"].add(tuple([m1, s1, m2, s2]))

        # Setup charge and paddingstates
        if not self.QMin.template["charge"]:
            self.QMin.template["charge"] = [i % 2 for i in range(len(self.QMin.molecule["states"]))]
            self.log.info(f"charge not specified setting default, {self.QMin.template['charge']}")

        if not self.QMin.template["paddingstates"]:
            self.QMin.template["paddingstates"] = [0 for _ in self.QMin.molecule["states"]]
            self.log.info(
                f"paddingstates not specified setting default, {self.QMin.template['paddingstates']}",
            )

        # Setup chargemap
        self.log.debug("Building chargemap")
        self.QMin.maps["chargemap"] = {idx + 1: int(chrg) for (idx, chrg) in enumerate(self.QMin.template["charge"])}

        # Setup jobs
        self.QMin.control["states_to_do"] = [
            v + int(self.QMin.template["paddingstates"][i]) if v > 0 else v for i, v in enumerate(self.QMin.molecule["states"])
        ]

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
    def create_restart_files(self):
        pass

    # TODO: move to SHARC_INTERFACE, unclear yet where to call it (end of run? separately?)
    @abstractmethod
    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """

    def run_program(self, workdir: str, cmd: str, out: str, err: str) -> int:
        """
        Runs a ab-initio programm and returns the exit_code

        workdir:    Path of the working directory
        cmd:        Contains path and arguments for execution of ab-initio program
        out:        Name of the output file
        err:        Name of the error file (optional)
        """
        current_dir = os.getcwd()
        os.chdir(workdir)
        self.log.debug(f"Working directory of ab-initio call {workdir}")

        with open(out, "w", encoding="utf-8") as outfile, open(err, "w", encoding="utf-8") as errfile:
            try:
                exit_code = sp.call(cmd, shell=True, stdout=outfile, stderr=errfile)
            except OSError as error:
                self.log.error(f"Execution of {cmd} failed!")
                raise OSError from error

        os.chdir(current_dir)
        return exit_code

    def runjobs(self, schedule: list[dict[str, QMin]]) -> dict[str, int]:
        """
        Runs all jobs in the schedule in a parallel queue

        schedule:   List of jobs (dictionary with jobnames and QMin objects)
                    First entry is a list with number of threads for the pool
                    for each job.
        """
        self.log.info("Starting job execution")
        error_codes = {}

        # Submit jobs to queue
        self.log.debug("Submit jobs to pool")
        for job_idx, jobset in enumerate(schedule):
            self.log.debug(f"Processing jobset number {job_idx} from schedule list")
            if not jobset:
                continue
            with Pool(processes=self.QMin.control["nslots_pool"][job_idx]) as pool:
                for job, qmin in jobset.items():
                    self.log.debug(f"Adding job: {job}")
                    workdir = os.path.join(self.QMin.resources["scratchdir"], job)
                    error_codes[job] = pool.apply_async(self.execute_from_qmin, args=(workdir, qmin))
                    time.sleep(self.QMin.resources["delay"])
                pool.close()
                pool.join()

        # Processing error codes
        error_string = "All jobs finished:\n"
        for job, code in error_codes.items():
            error_string += f"job: {job:<10s} code: {code.get()[0]:<4d} runtime: {code.get()[1]}\n"
        self.log.info(f"{error_string}")

        if any(map(lambda x: x.get()[0] != 0, error_codes.values())):
            raise RuntimeError("Some subprocesses did not finish successfully!")

        # Create restart files and garbage collection
        self.create_restart_files()
        self.remove_old_restart_files()

        return error_codes

    @staticmethod
    def divide_slots(ncpu: int, ntasks: int, scaling: float) -> tuple[int, int, list[int]]:
        """
        This routine figures out the optimal distribution of the tasks over the CPU cores
        returns the number of rounds (how many jobs each CPU core will contribute to),
        the number of slots which should be set in the Pool,
        and the number of cores for each job.
        """
        ntasks_per_round = min(ncpu, ntasks)
        optimal = {}
        for i in range(1, 1 + ntasks_per_round):
            nrounds = int(math.ceil(ntasks / i))
            ncores = ncpu // i
            optimal[i] = nrounds / 1.0 / ((1 - scaling) + scaling / ncores)
        best = min(optimal, key=optimal.get)
        nrounds = int(math.ceil(float(ntasks) // best))
        ncores = ncpu // best

        cpu_per_run = [0] * ntasks
        if nrounds == 1:
            itask = 0
            for _ in range(ncpu):
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
    def clean_savedir(path: str, retain: int, step: int) -> None:
        """
        Remove older files than step-retain

        path:       Path to savedir
        retain:     Number of timesteps to keep (-1 = all)
        step:       Current step
        """
        if retain < 0:
            return

        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory!")

        files = os.listdir(path)
        for file in files:
            ext = os.path.splitext(file)[1].replace(".", "")
            if not re.match(r"^\d+$", ext):  # Skip if extension is not a number
                continue
            if int(ext) < step - retain:
                os.remove(os.path.join(path, file))

    @staticmethod
    def parse_wfoverlap(overlap_file: str) -> np.ndarray:
        """
        Parse (Dyson) overlap matrix from wfoverlap output

        overlap_file: path to wfovlp.out
        """
        overlap_mat = []
        with open(overlap_file, "r", encoding="utf-8") as wffile:
            overlap_mat = []
            while True:
                line = next(wffile, False)
                if not line or containsstring("matrix <PsiA_i|PsiB_j>", line):
                    dim = -1 if not line else len(next(wffile).split()) // 2
                    break

            for line in wffile:
                if containsstring("<PsiA", line):
                    overlap_mat.append([float(x) for x in line.split()[2:]])
                else:
                    break

            if len(overlap_mat) != dim:
                raise ValueError(f"File {overlap_file} does not contain an overlap matrix!")
        return np.asarray(overlap_mat)

    @staticmethod
    def format_ci_vectors(ci_vectors: list[dict[tuple[int, ...], float]]) -> str:
        """
        Converts a list of ci vectors from (list[int],float) to str
        """
        alldets = set()
        for dets in ci_vectors:
            for key in dets:
                alldets.add((key))
        trans_table = str.maketrans({"0": "e", "1": "a", "2": "b", "3": "d"})
        string = f"{len(ci_vectors)} {len(next(iter(alldets)))} {len(alldets)}\n"
        for det in sorted(alldets, reverse=True):
            string += "".join(str(x) for x in det).translate(trans_table)
            for ci_vec in ci_vectors:
                if det in ci_vec:
                    string += f" {ci_vec[det]: 11.7f} "
                else:
                    string += f" {0: 11.7f} "
            string += "\n"
        return string

    @staticmethod
    def get_theodore(sumfile: str, omffile: str) -> dict[tuple[int], list[float]]:
        """
        Read and parse theodore output
        """
        out = readfile(sumfile)

        props = {}
        for line in out[2:]:
            s = line.replace("(", " ").replace(")", " ").split()
            if len(s) == 0:
                continue
            n = int(s[0])
            m = int(s[1])
            props[(m, n + (m == 1))] = [safe_cast(i, float, 0.0) for i in s[5:]]

        out = readfile(omffile)

        for line in out[1:]:
            s = line.replace("(", " ").replace(")", " ").split()
            if len(s) == 0:
                continue
            n = int(s[0])
            m = int(s[1])
            props[(m, n + (m == 1))].extend([safe_cast(i, float, 0.0) for i in s[4:]])
        return props

    # also add staticmethod
    # routine to read wfoverlap output

    @abstractmethod
    def run(self) -> None:
        """
        request & other logic
            requestmaps anlegen -> DONE IN SETUP_INTERFACE
            pfade für verschiedene orbital restart files
        make schedule
        runjobs()
        run_wfoverlap (braucht input files)
        run_theodore
        save directory handling
        """
