import datetime
import math
import os
import re
import shutil
import struct
import subprocess as sp
from copy import deepcopy
from io import TextIOWrapper
from itertools import pairwise
from typing import Optional

from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import expand_path, itmult, mkdir, writefile, readfile

__all__ = ["SHARC_ORCA"]

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
        self.QMin.resources.update(
            {
                "orcadir": None,
                "orcaversion": None,
                "wfoverlap": None,
                "wfthres": None,
                "numfrozcore": 0,
                "numocc": None,
                "schedule_scaling": 0.9,
            }
        )
        self.QMin.resources.types.update(
            {
                "orcadir": str,
                "orcaversion": tuple,
                "wfoverlap": str,
                "wfthres": float,
                "numfrozcore": int,
                "numocc": int,
                "schedule_scaling": float,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "no_tda": False,
                "unrestricted_triplets": False,
                "picture_change": False,
                "basis": "6-31G",
                "auxbasis": None,
                "functional": "PBE",
                "dispersion": None,
                "grid": None,
                "gridx": None,
                "gridxc": None,
                "ri": None,
                "scf": None,
                "keys": None,
                "paste_input_file": None,
                "frozen": -1,
                "maxiter": 700,
                "hfexchange": -1.0,
                "intacc": -1.0,
            }
        )
        self.QMin.template.types.update(
            {
                "no_tda": bool,
                "unrestricted_triplets": bool,
                "picture_change": bool,
                "basis": str,
                "auxbasis": str,
                "functional": str,
                "dispersion": str,
                "grid": str,
                "gridx": str,
                "gridxc": str,
                "ri": str,
                "scf": str,
                "keys": (str, list),
                "paste_input_file": str,
                "frozen": int,
                "maxiter": int,
                "hfexchange": float,
                "intacc": float,
            }
        )
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

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> int:
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen, stripping)
        """

        # Setup workdir
        mkdir(workdir)

        # Write ORCA input
        input_str = self.generate_inputstr(qmin)
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "ORCA.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        # Write point charges
        # TODO: QMMM

        # Copy wf files
        jobid = qmin.control["jobid"]
        if qmin.control["master"] and jobid in qmin.control["initorbs"]:
            self.log.debug("Copy ORCA.gbw to work directory")
            shutil.copy(qmin.control["initorbs"][jobid], os.path.join(workdir, "ORCA.gbw"))
        elif qmin.control["gradonly"]:
            self.log.debug(f"Copy ORCA.gbw from master_{jobid}")
            shutil.copy(
                os.path.join(qmin.resources["scratchdir"], f"master_{jobid}", "ORCA.gbw"), os.path.join(workdir, "ORCA.gbw")
            )

        # Setup ORCA
        prevdir = os.getcwd()
        os.chdir(workdir)

        exec_str = f"{os.path.join(qmin.resources['orcadir'],'orca')} ORCA.inp"
        stdoutfile = open(os.path.join(workdir, "ORCA.log"), "w", encoding="utf-8")
        stderrfile = open(os.path.join(workdir, "ORCA.err"), "w", encoding="utf-8")
        try:
            self.log.debug("Executing ORCA")
            exit_code = sp.call(exec_str, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError as err:
            self.log.error("Execution of ORCA failed!")
            raise OSError from err
        finally:
            stdoutfile.close()
            stderrfile.close()

        # TODO: postprocessing

        os.chdir(prevdir)
        return exit_code

    def getQMout(self) -> None:
        """
        Parse ORCA output files
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

        # Get contents of output file(s)
        log_files = {}
        for job in self.QMin.control["joblist"]:
            # with open(os.path.join(self.QMin.resources["scratchdir"], f"master_{job}", "ORCA.log")) as file:
            with open(os.path.join(self.QMin.resources["scratchdir"], "ORCA.log"), "r", encoding="utf-8") as file:
                log_files[job] = file.read()
        print(self._get_energy(log_files[1], self.QMin.control["jobs"][1]["mults"]))

    def _get_energy(self, output: str, mults: list[int]) -> dict[tuple[int, int], float]:
        """
        Extract energies from ORCA outfile

        output:     Content of outfile as string
        mult:       Multiplicity
        restr:      Restricted or unrestricted
        """

        # Define variables
        gsmult = mults[0]
        states_extract = deepcopy(self.QMin.molecule["states"])
        states_extract[gsmult - 1] -= 1

        states_extract = [0 if idx + 1 not in mults else val for idx, val in enumerate(states_extract)]
        states_extract = [max(states_extract) if idx + 1 in mults else val for idx, val in enumerate(states_extract)]

        # Find ground state energy and apply dispersion correction
        find_energy = re.search(r"Total Energy[\s:]+([-\d\.]+)", output)
        if not find_energy:
            self.log.error("No energy in ORCA outfile found!")
            raise ValueError()

        gs_energy = float(find_energy.group(1))
        dispersion = re.search(r"Dispersion correction\s+([-\d\.]+)", output)
        if dispersion:
            gs_energy += float(dispersion.group(1))

        energies = {(gsmult, int(1)): gs_energy}

        # Find excited states e.g. 2 sing + 2 trip: [(1, en1), (2, en2), (1,en_trip1), (2,en_trip2)
        exc_states = re.findall(r"STATE\s+(\d+):[A-Z\s=]+([-\d\.]+)\s+au", output)

        iter_states = iter(exc_states)
        for imult in mults:
            nstates = states_extract[imult - 1]
            for state, energy in iter_states:
                if int(state) <= self.QMin.molecule["states"][imult - 1]:  # Skip extra states
                    energies[(imult, int(state) + (gsmult == imult))] = gs_energy + float(energy)
                if int(state) == nstates:
                    break

        return energies

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

        # Convert keys to string if list
        if isinstance(self.QMin.template["keys"], list):
            self.QMin.template["keys"] = " ".join(self.QMin.template["keys"])

    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """

    def run(self) -> None:
        """
        request & other logic
            requestmaps anlegen -> DONE IN SETUP_INTERFACE
            pfade für verschiedene orbital restart files -> DONE IN SETUP_INTERFACE
        make schedule
        runjobs()
        run_wfoverlap (braucht input files)
        run_theodore
        save directory handling
        """

        # Generate schedule and run jobs
        self.log.debug("Generating schedule")
        self._gen_schedule()

        self.log.debug("Execute schedule")
        err_codes = self.runjobs(self.QMin.scheduling["schedule"])

        if any(map(lambda x: x != 0, err_codes)):
            self.log.error(f"Some jobs failed! {err_codes}")
            raise OSError()
        self.log.debug("All jobs finished successful")
        # TODO: wfoverlap and theodore

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
            self.log.debug("Setup states_to_do")
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

        # Populate initial orbitals dict
        self.QMin.control["initorbs"] = self._get_initorbs()  # TODO: control?

    def _build_jobs(self) -> None:
        """
        Build job dictionary from states_to_do
        """
        self.log.debug("Building job map.")
        jobs = {}
        if self.QMin.control["states_to_do"][0] > 0:
            jobs[1] = {"mults": [1], "restr": True}
        if len(self.QMin.control["states_to_do"]) >= 2 and self.QMin.control["states_to_do"][1] > 0:
            jobs[2] = {"mults": [2], "restr": False}
        if len(self.QMin.control["states_to_do"]) >= 3 and self.QMin.control["states_to_do"][2] > 0:
            if not self.QMin.template["unrestricted_triplets"] and self.QMin.control["states_to_do"][0] > 0:
                jobs[1]["mults"].append(3)
            else:
                jobs[3] = {"mults": [3], "restr": False}
        if len(self.QMin.control["states_to_do"]) >= 4:
            for imult, nstate in enumerate(self.QMin.control["states_to_do"][3:]):
                if nstate > 0:
                    # jobs[len(jobs)+1]={'mults':[imult+4],'restr':False}
                    jobs[imult + 4] = {"mults": [imult + 4], "restr": False}
        self.QMin.control["jobs"] = jobs
        self.QMin.control["joblist"] = sorted(set(jobs))

    def write_step_file(self) -> None:
        super().write_step_file()

    def get_dets_from_cis(self, cis_path: str) -> dict[str, str]:
        """
        Parse ORCA.cis file from WORKDIR
        """
        # Set variables
        cis_path = cis_path if os.path.isfile(cis_path) else os.path.join(cis_path, "ORCA.cis")
        jobid = self.QMin.control["jobid"]
        restricted = self.QMin.control["jobs"][jobid]["restr"]
        mults = self.QMin.control["jobs"][jobid]["mults"]
        gsmult = self.QMin.maps["multmap"][-int(jobid)]
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
            header = struct.unpack("8i", cis_file.read(32))

            # Extract information from header
            # Number occupied A/B and number virtual A/B
            nfc = header[0]
            noa = header[1] - header[0] + 1
            nva = header[3] - header[2] + 1
            nob = header[5] - header[4] + 1 if not restricted else noa
            nvb = header[7] - header[6] + 1 if not restricted else nva
            self.log.debug(f"CIS file header, NOA: {noa}, NVA: {nva}, NOB: {nob}, NVB: {nvb}, NFC: {nfc}")

            buffsize = (header[1] + 1 - nfc) * (header[3] + 1 - header[2])  # size of det
            self.log.debug(f"CIS determinant buffer size {buffsize*8} byte")
            self.log.debug(header)

            # ground state configuration
            # 0: empty, 1: alpha, 2: beta, 3: double oppupied
            if restricted:
                occ_a = [3] * (nfc + noa) + [0] * nva
                occ_b = []
            else:
                buffsize += (header[5] + 1 - header[4]) * (header[7] + 1 - header[6])
                occ_a = [1] * (nfc + noa) + [0] * nva
                occ_b = [2] * (nfc + nob) + [0] * nvb

            # Iterate over multiplicities and parse determinants
            eigenvectors = {}
            for mult in mults:
                eigenvectors[mult] = []

                if mult == gsmult:
                    key = occ_a[frozcore:] + occ_b[frozcore:]
                    eigenvectors[mult].append({tuple(key): 1.0})

                for _ in range(1, states_extract[mult - 1]):
                    cis_file.read(40)
                    dets = {}

                    buffer = iter(struct.unpack(f"{buffsize}d", cis_file.read(buffsize * 8)))
                    for occ in range(nfc, header[1] + 1):
                        for virt in range(header[2], header[3] + 1):
                            dets[(occ, virt, 1)] = next(buffer)

                    if not restricted:
                        for occ in range(header[4], header[5] + 1):
                            for virt in range(header[6], header[7] + 1):
                                dets[(occ, virt, 2)] = next(buffer)

                    if self.QMin.template["no_tda"]:
                        cis_file.read(40)
                        buffer = iter(struct.unpack(f"{buffsize}d", cis_file.read(buffsize * 8)))
                        for occ in range(nfc, header[1] + 1):
                            for virt in range(header[2], header[3] + 1):
                                dets[(occ, virt, 1)] += next(buffer)
                                dets[(occ, virt, 1)] /= 2

                        if not restricted:
                            for occ in range(header[4], header[5] + 1):
                                for virt in range(header[6], header[7] + 1):
                                    dets[(occ, virt, 2)] += next(buffer)
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
            return strings

    def _get_initorbs(self) -> dict[int, str]:
        """
        Generate initial orbitals
        """
        initorbs = {}

        if self.QMin.save["init"] or self.QMin.save["always_orb_init"]:
            self.log.debug("Found init or always_orb_init")
            for job in self.QMin.control["joblist"]:
                # Add ORCA.gbw.init if exists
                file = os.path.join(self.QMin.resources["pwd"], "ORCA.gbw.init")
                if os.path.isfile(os.path.join(file)):
                    initorbs[job] = file

                # Add ORCA.gbw.init for step if exists
                file = os.path.join(self.QMin.resources["pwd"], f"ORCA.gbw.{job}.init")
                if os.path.isfile(os.path.join(file)):
                    initorbs[job] = file
            # Check if some initials missing
            if self.QMin.save["always_orb_init"] and len(initorbs) < len(self.QMin.control["joblist"]):
                self.log.error("Initial orbitals missing for some jobs!")
                raise ValueError()

        elif self.QMin.save["newstep"] or self.QMin.save["samestep"]:
            for job in self.QMin.control["joblist"]:
                file = os.path.join(self.QMin.save["savedir"], f"ORCA.gbw.{job}")
                if not os.path.isfile(file):
                    self.log.error(f"File {file} missing in savedir!")
                    raise FileNotFoundError()
                initorbs[job] = file + ".old" if self.QMin.save["newstep"] else file

        # TODO: restart in new interface?
        return initorbs

    def _gen_schedule(self) -> None:
        """
        Generates scheduling from joblist
        """

        # sort the gradients into the different jobs
        gradjob = {f"master_{job}": {} for job in self.QMin.control["joblist"]}
        if self.QMin.maps["gradmap"]:
            for grad in self.QMin.maps["gradmap"]:
                ijob = self.QMin.maps["multmap"][grad[0]]
                gradjob[f"master_{ijob}"][grad] = {
                    "gs": bool((not self.QMin.control["jobs"][ijob]["restr"] and grad[1] == 1) or grad == (1, 1))
                }

        # make map for states onto gradjobs
        jobgrad = {}
        for job in gradjob:
            for state in gradjob[job]:
                jobgrad[state] = (job, gradjob[job][state]["gs"])
        self.QMin.control["jobgrad"] = jobgrad  # TODO: control? what is it used?

        schedule = [{}]

        # add the master calculations
        ntasks = len([1 for g in gradjob if "master" in g])
        _, nslots, cpu_per_run = self.divide_slots(self.QMin.resources["ncpu"], ntasks, self.QMin.resources["schedule_scaling"])
        self.QMin.control["nslots_pool"] = [nslots]

        for idx, job in enumerate(sorted(gradjob)):
            if not "master" in job:
                continue
            qmin = self.QMin.copy()
            qmin.control["master"] = True
            qmin.control["jobid"] = int(job.split("_")[1])
            qmin.resources["ncpu"] = cpu_per_run[idx]
            qmin.maps["gradmap"] = set(gradjob[job])
            schedule[-1][job] = qmin

        # add the gradient calculations
        ntasks = len([1 for g in gradjob if "grad" in g])
        if ntasks > 0:
            self.QMin.control["nslots_pool"].append(nslots)
            schedule.append({})
            for idx, job in enumerate(sorted(gradjob)):
                if not "grad" in job:
                    continue
                qmin = self.QMin.copy()
                qmin.control["jobid"] = qmin.maps["multmap"][list(gradjob[job])[0][0]]
                qmin.resources["ncpu"] = cpu_per_run[idx]
                qmin.maps["gradmap"] = set(gradjob[job])
                qmin.control["gradonly"] = True
                for i in ["h", "soc", "dm", "overlap", "ion"]:
                    qmin.requests[i] = False
                for i in ["always_guess", "always_orb_init", "init"]:
                    qmin.save[i] = False
                schedule[-1][job] = qmin

        self.QMin.scheduling["schedule"] = schedule

    @staticmethod
    def generate_inputstr(qmin: QMin) -> str:
        """
        Generate ORCA input file string from QMin object
        """
        job = qmin.control["jobid"]
        gsmult = qmin.maps["multmap"][-job][0]
        restr = qmin.control["jobs"][job]["restr"]
        charge = qmin.maps["chargemap"][gsmult]

        # excited states to calculate
        states_to_do = qmin.control["states_to_do"]
        for imult, _ in enumerate(states_to_do):
            if not imult + 1 in qmin.maps["multmap"][-job]:
                states_to_do[imult] = 0
        states_to_do[gsmult - 1] -= 1

        # do minimum number of states for gradient jobs
        if qmin.control["gradonly"]:
            gradmult = qmin.maps["gradmap"][0][0]
            gradstat = qmin.maps["gradmap"][0][1]
            for imult, _ in enumerate(states_to_do):
                if imult + 1 == gradmult:
                    states_to_do[imult] = gradstat - (gradmult == gsmult)
                else:
                    states_to_do[imult] = 0

        # number of states to calculate
        trip = bool(restr and len(states_to_do) >= 3 and states_to_do[2] > 0)

        # gradients
        do_grad = False
        egrad = ()
        if qmin.requests["grad"] and qmin.maps["gradmap"]:
            do_grad = True
            for grad in qmin.maps["gradmap"]:
                if not (gsmult, 1) == grad:
                    egrad = grad
        singgrad = []
        tripgrad = []
        for grad in qmin.maps["gradmap"]:
            if grad[0] == gsmult:
                singgrad.append(grad[1] - 1)
            if grad[0] == 3 and restr:
                tripgrad.append(grad[1])

        # Add header
        string = "! "
        keys = ["basis", "auxbasis", "functional", "dispersion", "ri", "keys"]
        string += " ".join(qmin.template[x] for x in keys if qmin.template[x] is not None)
        string += " nousesym "
        string += "engrad\n" if do_grad else "\n"

        # TODO: Whats AOoverlap?
        # CPU cores
        if qmin.resources["ncpu"] > 1:
            string += f"%pal\n\tnprocs {qmin.resources['ncpu']}\nend\n\n"
        string += f"%maxcore {qmin.resources['memory']}\n\n"

        # Basis sets + ECP basis set
        if "basis_per_element" in qmin.template:
            string += "%basis\n"
            # basis_per_element key is list, need to iterate pairwise
            for elem, basis in pairwise(qmin.template["basis_per_element"]):
                string += f'\tnewgto {elem} "{basis}" end\n'
            if "ecp_per_element" in qmin.template:
                for elem, basis in pairwise(qmin.template["ecp_per_element"]):
                    string += f'\tnewECP {elem} "{basis}" end\n'
            string += "end\n\n"

        # Frozen core
        string += f"%method\n\tfrozencore {-2*qmin.molecule['frozcore'] if qmin.molecule['frozcore'] >0 else 'FC_NONE'}\nend\n\n"

        # HF exchange
        if qmin.template["hfexchange"] > 0:
            string += f"%method\n\tScalHFX = {qmin.template['hfexchange']}\nend\n\n"

        # Range separation
        # TODO

        # Intacc
        if qmin.template["intacc"] > 0:
            string += f"%method\n\tintacc {qmin.template['intacc']:3.1f}\nend\n\n"

        # Gaussian point charges
        if "cpcm" in qmin.template["keys"]:
            string += "%cpcm\n\tsurfacetype vdw_gaussian\nend\n\n"

        # Excited states
        if max(states_to_do) > 0:
            string += f"%tddft\n\ttda {'false' if qmin.template['no_tda'] else 'true'}\n"
            # TODO: Theodore
            if restr and trip:
                string += "\ttriplets true\n"
            string += f"\tnroots {max(states_to_do)}\n"
            if restr and qmin.requests["soc"]:
                string += "\tdosoc true\n\tprintlevel 3\n"

            if do_grad and egrad:
                string += f"\tiroot {egrad[1] - (gsmult == egrad[0])}\n"
            string += "end\n\n"

        # Output
        # TODO: AOoverlap?
        string += "%output\n"
        if qmin.requests["ion"] or qmin.requests["theodore"]:
            string += "\tPrint[ P_Overlap ] 1\n"
        if qmin.control["master"] or qmin.requests["theodore"]:
            string += "\tPrint[ P_MOs ] 1\n"
        string += "end\n\n"

        # SCF
        string += f"%scf\n\tmaxiter {qmin.template['maxiter']}\nend\n\n"

        # Charge mult geom
        string += "%coords\n\tCtyp xyz\n\tunits bohrs\n"
        string += f"\tcharge {charge}\n"
        string += "\tcoords\n"
        for iatom, (label, coords) in enumerate(zip(qmin.molecule["elements"], qmin.coords["coords"])):
            string += f"\t{label:4s} {coords[0]:16.9f} {coords[1]:16.9f} {coords[2]:16.9f}"
            if "basis_per_atom" in qmin.template and str(iatom) in qmin.template["basis_per_atom"]:
                idx = qmin.template["basis_per_atom"].index(str(iatom))
                string += f"\tnewgto \"{qmin.template['basis_per_atom'][idx+1]}\" end"
            string += "\n"
        string += "end\nend\n\n"  # TODO: 2 ends on purpose?

        # Point charges
        # TODO
        return string

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
    # test.QMin.control["jobid"] = 1
    test._gen_schedule()
    # cidets = test.get_dets_from_cis(
    #   # "/user/mai/Documents/CoWorkers/FelixProche/full/orca.cis"
    #   # "/user/mai/Documents/CoWorkers/Anna/test2/orca.cis"
    #   # "/user/mai/Documents/CoWorkers/AnnaMW/ORCA_wfoverlap/real_test/A/ORCA.cis"
    #   "/user/sascha/development/eci/sharc_main/TEST/ORCA.cis"
    # )
    # print(cidets["./SAVEDIR/dets.1"][:5000])
    test.set_coords("QM.in")
    test.QMin.scheduling["schedule"][0]["master_1"].coords = test.QMin.coords
    test.getQMout()
    # print(test.generate_inputstr(test.QMin.scheduling["schedule"][0]["master_1"]))
    # code = test.execute_from_qmin(
    # os.path.join(test.QMin.resources["pwd"], "TEST"), test.QMin.scheduling["schedule"][0]["master_1"]
    # )
    # print(code)
