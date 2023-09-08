import datetime
import math
import os
import re
import struct
import subprocess as sp
from copy import deepcopy
from io import TextIOWrapper
from typing import Optional

from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import expand_path, itmult

AUTHORS = ""
VERSION = ""
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "ORCA"
DESCRIPTION = ""

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "dm",
        "soc",
        "theodore",
        "grad",
        "ion",
        "overlap",
        "phases",
        # raw data request
        "basis_set",
        "wave_functions",
        "density_matrices",
    ]
)


class SHARC_ORCA(SHARC_ABINITIO):
    """
    SHARC interface for ORCA
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources["orcadir"] = None
        self.QMin.resources["schedule_scaling"] = 0.9
        self.QMin.resources["orcadir"] = None
        self.QMin.resources["orcaversion"] = None
        self.QMin.resources["wfoverlap"] = None
        self.QMin.resources["wfthres"] = None
        self.QMin.resources["numfrozcore"] = 0
        self.QMin.resources["numocc"] = None

        self.QMin.resources.types["orcadir"] = str
        self.QMin.resources.types["orcaversion"] = tuple
        self.QMin.resources.types["wfoverlap"] = str
        self.QMin.resources.types["wfthres"] = float
        self.QMin.resources.types["numfrozcore"] = int
        self.QMin.resources.types["numocc"] = int
        self.QMin.resources.types["schedule_scaling"] = float

        # Add template keys
        self.QMin.template["no_tda"] = False
        self.QMin.template["unrestricted_triplets"] = False
        self.QMin.template["picture_change"] = False
        self.QMin.template["basis"] = "6-31G"
        self.QMin.template["auxbasis"] = None
        self.QMin.template["functional"] = "PBE"
        self.QMin.template["dispersion"] = None
        self.QMin.template["grid"] = None
        self.QMin.template["gridx"] = None
        self.QMin.template["gridxc"] = None
        self.QMin.template["ri"] = None
        self.QMin.template["scf"] = None
        self.QMin.template["keys"] = None
        self.QMin.template["paste_input_file"] = None
        self.QMin.template["frozen"] = -1
        self.QMin.template["maxiter"] = 700
        self.QMin.template["hfexchange"] = -1.0
        self.QMin.template["intacc"] = -1.0

        self.QMin.template.types["no_tda"] = bool
        self.QMin.template.types["unrestricted_triplets"] = bool
        self.QMin.template.types["picture_change"] = bool
        self.QMin.template.types["basis"] = str
        self.QMin.template.types["auxbasis"] = str
        self.QMin.template.types["functional"] = str
        self.QMin.template.types["dispersion"] = str
        self.QMin.template.types["grid"] = str
        self.QMin.template.types["gridx"] = str
        self.QMin.template.types["gridxc"] = str
        self.QMin.template.types["ri"] = str
        self.QMin.template.types["scf"] = str
        self.QMin.template.types["keys"] = (str, list)
        self.QMin.template.types["paste_input_file"] = str
        self.QMin.template.types["frozen"] = int
        self.QMin.template.types["maxiter"] = int
        self.QMin.template.types["hfexchange"] = float
        self.QMin.template.types["intacc"] = float
        # no range_sep_settings, can be done with paste_input_file

    @staticmethod
    def version() -> str:
        return SHARC_ORCA._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_ORCA._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_ORCA._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_ORCA._authors

    @staticmethod
    def name() -> str:
        return SHARC_ORCA._name

    @staticmethod
    def description() -> str:
        return SHARC_ORCA._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_ORCA._name}\n{SHARC_ORCA._description}"

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

    def create_restart_files(self):
        pass

    def execute_from_qmin(self, workdir: str, qmin: QMin):
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen, stripping)
        """

    def getQMout(self):
        pass

    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        return

    def printQMout(self) -> None:
        super().writeQMout()

    def print_qmin(self) -> None:
        pass

    def read_resources(self, resources_file: str, kw_whitelist: Optional[list[str]] = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        # LD PATH???
        if "orcadir" not in self.QMin.resources:
            raise ValueError("orcadir has to be set in resource file!")

        self.QMin.resources["orcadir"] = expand_path(self.QMin.resources["orcadir"])
        self.log.debug(f'orcadir set to {self.QMin.resources["orcadir"]}')

        self.QMin.resources["orcaversion"] = SHARC_ORCA.get_orca_version(self.QMin.resources["orcadir"])
        self.log.info(f'Detected ORCA version {".".join(str(i) for i in self.QMin.resources["orcaversion"])}')

        if self.QMin.resources["orcaversion"] < (5, 0):
            raise ValueError("This version of the SHARC-ORCA interface is only compatible to Orca 5.0 or higher!")

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

    def read_template(self, template_file: str) -> None:
        super().read_template(template_file)

    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """

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

    def setup_interface(self) -> None:
        """
        Setup remaining maps (ionmap, gsmap) and build jobs dict
        """
        super().setup_interface()

        if (
            not self.QMin.template["unrestricted_triplets"]
            and len(self.QMin.molecule["states"]) >= 3
            and self.QMin.molecule["states"][2] > 0
        ):
            self.log.debug("Unrestricted triplets requested, setup states_to_do")
            self.QMin.control["states_to_do"][0] = max(self.QMin.molecule["states"][0], 1)
            req = max(self.QMin.molecule["states"][0] - 1, self.QMin.molecule["states"][2])
            self.QMin.control["states_to_do"][0] = req + 1
            self.QMin.control["states_to_do"][2] = req

        self._build_jobs()
        # Setup multmap
        self.log.debug("Building multmap")
        self.QMin.maps["multmap"] = {}
        for ijob, job in self.QMin.control["jobs"].items():
            for imult in job["mults"]:
                self.QMin.maps["multmap"][imult] = ijob
            self.QMin.maps["multmap"][-(ijob)] = job["mults"]
        self.QMin.maps["multmap"][1] = 1

        # Setup ionmap
        if self.QMin.requests["ion"]:
            self.log.debug("Building ionmap")
            self.QMin.maps["ionmap"] = []
            for mult1 in itmult(self.QMin.molecule["states"]):
                job1 = self.QMin.maps["multmap"][mult1]
                el1 = self.QMin.maps["chargemap"][mult1]
                for mult2 in itmult(self.QMin.molecule["states"]):
                    if mult1 >= mult2:
                        continue
                    job2 = self.QMin.maps["multmap"][mult2]
                    el2 = self.QMin.maps["chargemap"][mult2]
                    if abs(mult1 - mult2) == 1 and abs(el1 - el2) == 1:
                        self.QMin.maps["ionmap"].append((mult1, job1, mult2, job2))

        # Setup gsmap
        self.log.debug("Building gsmap")
        self.QMin.maps["gsmap"] = {}
        for i in range(self.QMin.molecule["nmstates"]):
            mult1, _, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
            ground_state = (mult1, 1, ms1)
            job = self.QMin.maps["multmap"][mult1]
            if mult1 == 3 and self.QMin.control["jobs"][job]["restr"]:
                ground_state = (1, 1, 0.0)
            for j in range(self.QMin.molecule["nmstates"]):
                if tuple(self.QMin.maps["statemap"][j + 1]) == ground_state:
                    break
                self.QMin.maps["gsmap"][i + 1] = j + 1

    def _build_jobs(self) -> None:
        """
        Build job dictionary from states_to_do
        """
        self.log.debug("Building job map.")
        jobs = {}
        for idx, val in enumerate(self.QMin.control["states_to_do"]):
            if val == 0:
                continue
            match idx:
                case 0:
                    jobs[1] = {"mults": [1], "restr": True}
                case 1:
                    jobs[2] = {"mults": [2], "restr": False}
                case 2:
                    if self.QMin.template["unrestricted_triplets"] and 1 in jobs:
                        jobs[1]["mults"].append(3)
                    else:
                        jobs[3] = {"mults": [3], "restr": False}
                case _:
                    jobs[idx + 5] = {"mults": [idx + 5], "restr": False}

        self.QMin.control["jobs"] = jobs

    def write_step_file(self) -> None:
        super().write_step_file()

    def get_dets_from_cis(self, cis_path: str) -> dict[str, str]:
        """
        Parse ORCA.cis file from WORKDIR
        """
        # Set variables
        cis_path = cis_path if os.path.isfile(cis_path) else os.path.join(cis_path, "ORCA.cis")
        self.QMin.control["jobid"] = 1  # TODO: REMOVE AFTER DEBUGGING!!!
        jobid = self.QMin.control["jobid"]
        restricted = self.QMin.control["jobs"][jobid]["restr"]
        mults = self.QMin.control["jobs"][jobid]["mults"]
        gsmult = self.QMin.maps["multmap"][-jobid]
        frozcore = self.QMin.resources["numfrozcore"]
        states_extract = deepcopy(self.QMin.molecule["states"])
        states_skip = [self.QMin.control["states_to_do"][i] - states_extract[i] for i in range(len(states_extract))]
        for i, _ in enumerate(states_extract):
            if not i + 1 in mults:
                states_extract[i] = 0
                states_skip[i] = 0
            elif i + 1 == gsmult:
                states_extract[i] -= 1

        # Parse file
        with open(cis_path, "rb") as cis_file:
            cis_file.read(4)
            header = [struct.unpack("i", cis_file.read(4))[0] for i in range(8)]

            # Extract information from header
            # Number occupied A/B and number virtual A/B
            nfc = header[0]
            noa = header[1] - header[0] + 1
            nva = header[3] - header[2] + 1
            nob = header[5] - header[4] + 1 if not restricted else noa
            nvb = header[7] - header[6] + 1 if not restricted else nva
            self.log.debug(f"CIS file header, NOA: {noa}, NVA: {nva}, NOB: {nob}, NVB: {nvb}, NFC: {nfc}")

            # ground state configuration
            # 0: empty, 1: alpha, 2: beta, 3: double oppupied
            if restricted:
                occ_a = [3] * (nfc + noa) + [0] * nva
                occ_b = []
            else:
                occ_a = [1] * (nfc + noa) + [0] * nva
                occ_b = [2] * (nfc + nob) + [0] * nvb

            # Iterate over multiplicities and parse determinants
            eigenvectors = {}
            for mult in mults:
                eigenvectors[mult] = []

                if mult == gsmult:
                    key = occ_a[frozcore:] + occ_b[frozcore:]
                    eigenvectors[mult].append({tuple(key): 1.0})

                for _ in range(states_extract[mult - 1]):
                    cis_file.read(40)
                    dets = {}
                    for occ in range(nfc, header[1] + 1):
                        for virt in range(header[2], header[3] + 1):
                            dets[(occ, virt, 1)] = struct.unpack("d", cis_file.read(8))[0]

                    if not restricted:
                        for occ in range(header[4], header[5] + 1):
                            for virt in range(header[6], header[7] + 1):
                                dets[(occ, virt, 2)] = struct.unpack("d", cis_file.read(8))[0]

                    if self.QMin.template["no_tda"]:
                        cis_file.read(40)
                        for occ in range(nfc, header[1] + 1):
                            for virt in range(header[2], header[3] + 1):
                                dets[(occ, virt, 1)] += struct.unpack("d", cis_file.read(8))[0]
                                dets[(occ, virt, 1)] /= 2

                        if not restricted:
                            for occ in range(header[4], header[5] + 1):
                                for virt in range(header[6], header[7] + 1):
                                    dets[(occ, virt, 2)] += struct.unpack("d", cis_file.read(8))[0]
                                    dets[(occ, virt, 2)] /= 2

                    # Truncate determinants with contribution under threshold
                    norm = 0
                    for k in sorted(dets, key=lambda x: dets[x] ** 2, reverse=True):
                        if norm > self.QMin.resources["wfthres"]:
                            del dets[k]
                            continue
                        norm += dets[k] ** 2

                    dets_exp = {}
                    for occ, virt, dummy in dets:
                        if restricted:
                            key = deepcopy(occ_a)
                            match mult:
                                case 1:
                                    key[occ], key[virt] = 2, 1
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)] * math.sqrt(0.5)
                                    key[occ], key[virt] = 1, 2
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)] * math.sqrt(0.5)
                                case 3:
                                    key[occ], key[virt] = 1, 1
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)]
                        else:
                            key = occ_a + occ_b
                            match dummy:
                                case 1:
                                    key[occ], key[virt] = 0, 1
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)]
                                case 2:
                                    key[nfc + noa + nva + occ] = 0
                                    key[nfc + noa + nva + virt] = 2
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)]

                    # Remove frozen core
                    dets_nofroz = {}
                    for key, val in dets_exp.items():
                        if frozcore == 0:
                            dets_nofroz = dets_exp
                            break
                        if restricted:
                            if any(map(lambda x: x != 3, key[:frozcore])):
                                self.log.warning("Non-occupied orbital inside frozen core! Skipping ...")
                                continue
                            dets_nofroz[key[frozcore:]] = val
                            continue
                        if any(map(lambda x: x != 1, key[:frozcore])) or any(
                            map(lambda x: x != 2, key[frozcore + noa + nva : noa + nva + 2 * frozcore])
                        ):
                            self.log.warning("Non-occupied orbital inside frozen core! Skipping ...")
                            continue
                        dets_nofroz[key[frozcore : frozcore + noa + nva] + key[noa + nva + 2 * frozcore]] = val
                    eigenvectors[mult].append(dets_nofroz)

                    # Skip extra roots
                    skip = 40 + noa * nva * 8
                    if not restricted:
                        skip += nob * nvb * 8
                    if self.QMin.template["no_tda"]:
                        skip += 40 + noa * nva * 8
                        if not restricted:
                            skip += nob * nvb * 8
                    skip *= states_skip[mult - 1]
                    cis_file.read(skip)

            # Convert determinant lists to strins
            strings = {}
            for mult in mults:
                filename = os.path.join(self.QMin.save["savedir"], f"dets.{mult}")
                strings[filename] = self.format_ci_vectors(eigenvectors[mult])
            print(strings)
            return strings

    @staticmethod
    def get_orca_version(path: str) -> tuple[int, ...]:
        """
        Get ORCA version number of given path
        """
        string = os.path.join(path, "orca") + " nonexisting"
        with sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE) as proc:
            comm = proc.communicate()[0].decode()
            if not comm:
                raise ValueError("ORCA version not found!")
            version = re.findall(r"Program Version (\d.\d.\d)", comm)[0].split(".")
            return tuple(int(i) for i in version)


if __name__ == "__main__":
    test = SHARC_ORCA(loglevel=10)
    test.setup_mol("QM.in")
    test.read_resources("ORCA.resources", kw_whitelist=["theodore_prop", "theodore_fragment"])
    test.read_template("ORCA.template")
    test.read_requests("QM.in")
    test.setup_interface()
    test.get_dets_from_cis(
        # "/user/mai/Documents/CoWorkers/FelixProche/full/orca.cis"
        "/user/mai/Documents/CoWorkers/Anna/test2/orca.cis"
    )
    # print(test.QMin)
