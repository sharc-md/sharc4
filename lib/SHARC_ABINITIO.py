import math
import os
import sys
import time
from abc import abstractmethod
from datetime import date
from io import TextIOWrapper
from multiprocessing import Pool
from typing import Dict, List, Tuple

from logger import log as logging
from qmin import QMin
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import itmult

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
        self.QMin.template["charge"] = None
        self.QMin.template["paddingstates"] = None

        self.QMin.template.types["charge"] = list
        self.QMin.template.types["paddingstates"] = list

        # Add ab-initio specific keywords to resources
        self.QMin.resources["delay"] = 0

        self.QMin.resources.types["delay"] = int

    @abstractmethod
    def authors(self) -> str:
        return "Severin Polonius, Sebastian Mai"

    @abstractmethod
    def version(self) -> str:
        return "3.0"

    @abstractmethod
    def versiondate(self) -> date:
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
    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    @abstractmethod
    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    @abstractmethod
    def prepare(self, INFOS: dict, dir: str):
        "setup the calculation in directory 'dir'"
        return

    @abstractmethod
    def print_qmin(self) -> None:
        pass

    @abstractmethod
    def execute_from_qmin(self, workdir: str, qmin: QMin):
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen)
        """

    @abstractmethod
    def read_template(self, template_file: str) -> None:
        super().read_template(template_file)

        # Check if charge in template and autoexpand if needed
        if self.QMin.template["charge"]:
            if len(self.QMin.template["charge"]) == 1:
                charge = int(self.QMin.template["charge"][0])
                if (self.QMin.molecule["Atomcharge"] + charge) % 2 == 1 and len(
                    self.QMin.molecule["states"]
                ) > 1:
                    logging.info(
                        "HINT: Charge shifted by -1 to be compatible with multiplicities."
                    )
                    charge -= 1
                self.QMin.template["charge"] = [
                    i % 2 + charge for i in range(len(self.QMin.molecule["states"]))
                ]
                logging.info(
                    f'HINT: total charge per multiplicity automatically assigned, please check ({self.QMin.template["charge"]}).'
                )
                logging.info(
                    'You can set the charge in the template manually for each multiplicity ("charge 0 +1 0 ...")'
                )
            elif len(self.QMin.template["charge"]) >= len(self.QMin.molecule["states"]):
                self.QMin.template["charge"] = [
                    int(self.QMin.template["charge"][i])
                    for i in range(len(self.QMin.molecule["states"]))
                ]
                compatible = True
                for imult, cha in enumerate(self.QMin.template["charge"]):
                    if not (self.QMin.molecule["Atomcharge"] + cha + imult) % 2 == 0:
                        compatible = False
                if not compatible:
                    logging.warning(
                        "Charges from template not compatible with multiplicities!  (this is probably OK if you use QM/MM)"
                    )
            else:
                logging.error('Length of "charge" does not match length of "states"!')
                sys.exit(54)

    @abstractmethod
    def read_resources(self, resources_file: str, kw_whitelist: List[str] = []) -> None:
        super().read_resources(resources_file, kw_whitelist)

    @abstractmethod
    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

    @abstractmethod
    def write_step_file(self) -> None:
        super().write_step_file()

    @abstractmethod
    def printQMout(self):
        super().writeQMout()

    @abstractmethod
    def setup_interface(self):
        """
        Create maps from QMin object
        """
        logging.debug("Setup interface -> building maps")

        # Setup gradmap
        if self.QMin.requests["grad"]:
            logging.debug("Building gradmap")
            self.QMin.maps["gradmap"] = set(
                {
                    tuple(self.QMin.maps["statemap"][i][0:2])
                    for i in self.QMin.requests["grad"]
                }
            )

        # Setup densmap
        if self.QMin.requests["multipolar_fit"]:
            logging.debug("Building densmap")
            self.QMin.maps["densmap"] = set(
                {
                    tuple(self.QMin.maps["statemap"][i][0:2])
                    for i in self.QMin.requests["multipolar_fit"]
                }
            )

        # Setup nacmap
        if (
            self.QMin.requests["nacdr"]
            and len(self.QMin.requests["nacdr"]) > 0
            and self.QMin.requests["nacdr"][0] != "all"
        ):
            logging.debug("Building nacmap")
            self.QMin.maps["nacmap"] = set()
            for i in self.QMin.requests["nacdr"]:
                s1 = self.QMin.maps["statemap"][int(i[0])]
                s2 = self.QMin.maps["statemap"][int(i[1])]
                if s1[0] != s2[0] or s1 == s2:
                    continue
                if s1[1] > s2[1]:
                    continue
                self.QMin.maps["nacmap"].add(tuple(s1 + s2))

        # Setup charge and paddingstates
        if not self.QMin.template["charge"]:
            self.QMin.template["charge"] = [
                i % 2 for i in range(len(self.QMin.molecule["states"]))
            ]
            logging.info(
                f"charge not specified setting default, {self.QMin.template['charge']}"
            )

        if not self.QMin.template["paddingstates"]:
            self.QMin.template["paddingstates"] = [
                0 for _ in self.QMin.molecule["states"]
            ]
            logging.info(
                f"paddingstates not specified setting default, {self.QMin.template['paddingstates']}",
            )

        # Setup chargemap
        logging.debug("Building chargemap")
        self.QMin.maps["chargemap"] = {
            idx + 1: int(chrg)
            for (idx, chrg) in enumerate(self.QMin.template["charge"])
        }

        # Setup jobs
        self.QMin.control["states_to_do"] = [
            v + int(self.QMin.template["paddingstates"][i]) if v > 0 else v
            for i, v in enumerate(self.QMin.molecule["states"])
        ]

        if "unrestricted_triplets" not in self.QMin.template.keys():
            if (
                len(self.QMin.molecule["states"]) >= 3
                and self.QMin.molecule["states"][2] > 0
            ):
                self.QMin.control["states_to_do"][0] = max(
                    self.QMin.molecule["states"][0], 1
                )
                req = max(
                    self.QMin.molecule["states"][0] - 1, self.QMin.molecule["states"][2]
                )
                self.QMin.control["states_to_do"][0] = req + 1
                self.QMin.control["states_to_do"][2] = req

        jobs = {}
        if self.QMin.control["states_to_do"][0] > 0:
            jobs[1] = {"mults": [1], "restr": True}
        if (
            len(self.QMin.control["states_to_do"]) >= 2
            and self.QMin.control["states_to_do"][1] > 0
        ):
            jobs[2] = {"mults": [2], "restr": False}
        if (
            len(self.QMin.control["states_to_do"]) >= 3
            and self.QMin.control["states_to_do"][2] > 0
        ):
            if (
                "unrestricted_triplets" not in self.QMin.template.keys()
                and self.QMin.control["states_to_do"][0] > 0
            ):
                jobs[1]["mults"].append(3)
            else:
                jobs[3] = {"mults": [3], "restr": False}

        if len(self.QMin.control["states_to_do"]) >= 4:
            for imult, nstate in enumerate(self.QMin.control["states_to_do"][3:]):
                if nstate > 0:
                    jobs[imult + 4] = {"mults": [imult + 4], "restr": False}

        logging.debug("Building mults")
        self.QMin.maps["mults"] = set(jobs)

        # Setup multmap
        logging.debug("Building multmap")
        self.QMin.maps["multmap"] = {}
        for ijob, job in jobs.items():
            for imult in job["mults"]:
                self.QMin.maps["multmap"][imult] = ijob
            self.QMin.maps["multmap"][-(ijob)] = job["mults"]
        self.QMin.maps["multmap"][1] = 1

        # Setup ionmap
        if self.QMin.requests["ion"]:
            logging.debug("Building ionmap")
            self.QMin.maps["ionmap"] = []
            for m1 in itmult(self.QMin.molecule["states"]):
                job1 = self.QMin.maps["multmap"][m1]
                el1 = self.QMin.maps["chargemap"][m1]
                for m2 in itmult(self.QMin.molecule["states"]):
                    if m1 >= m2:
                        continue
                    job2 = self.QMin.maps["multmap"][m2]
                    el2 = self.QMin.maps["chargemap"][m2]
                    if abs(m1 - m2) == 1 and abs(el1 - el2) == 1:
                        self.QMin.maps["ionmap"].append((m1, job1, m2, job2))

        # Setup gsmap
        logging.debug("Building gsmap")
        self.QMin.maps["gsmap"] = {}
        for i in range(self.QMin.molecule["nmstates"]):
            m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
            gs = (m1, 1, ms1)
            job = self.QMin.maps["multmap"][m1]
            if m1 == 3 and jobs[job]["restr"]:
                gs = (1, 1, 0.0)
            for j in range(self.QMin.molecule["nmstates"]):
                m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                if (m2, s2, ms2) == gs:
                    break
            self.QMin.maps["gsmap"][i + 1] = j + 1

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
    def create_restart_files(self):
        pass

    @abstractmethod
    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """

    def run_program(
        self, workdir: str, cmd: str, out: str, err: str, keepfiles: List[str]
    ) -> int:
        return error_code

    def runjobs(self, schedule: List[Dict[str, QMin]]) -> Dict[int]:
        """
        Runs all jobs in the schedule in a parallel queue

        schedule:   List of jobs (dictionary with jobnames and QMin objects)
        """
        logging.info("Starting job execution")
        error_codes = {}

        # Submit jobs to queue
        for job_idx, jobset in enumerate(schedule):
            logging.debug(f"Processing jobset number {job_idx} from schedule list")
            if not jobset:
                continue
            with Pool(processes=jobset["npool"][job_idx]) as pool:
                logging.debug("Submit jobs to pool")
                for job, qmin in jobset.items():
                    logging.debug(f"Adding job: {job}")
                    workdir = os.path.join(self.QMin.resources["scratchdir"], job)
                    error_codes[job] = pool.apply_async(
                        self.execute_from_qmin, args=(workdir, qmin)
                    ).get()
                    time.sleep(self.QMin.resources["delay"])

        # Processing error codes
        logging.debug("All jobs finished")

        error_string = "Error Codes:\n"
        for idx, (job, code) in enumerate(error_codes.items()):
            error_string += f"\t{job + ' ' * (10 - len(job))}\t{code}"
            if (idx + 1) % 4 == 0:
                error_string += "\n"
        logging.info(f"{error_string}")

        if any(lambda x: x != 0, error_codes.values()):
            logging.error("Some subprocesses did not finish successfully!")
            sys.exit(101)

        # Create restart files and garbage collection
        self.create_restart_files()
        self.remove_old_restart_files()

        return error_codes

    @staticmethod
    def divide_slots(ncpu: int, ntasks: int, scaling: float) -> Tuple[int]:
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

    @abstractmethod
    def run(self):
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
