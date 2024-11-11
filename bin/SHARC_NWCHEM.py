#!/usr/bin/env python3
import datetime
import os
import re
import shutil
from copy import deepcopy
from functools import cmp_to_key
from io import TextIOWrapper

import numpy as np
from pyscf import gto
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import convert_list, expand_path, is_exec, link, mkdir, question, writefile

__all__ = ["SHARC_NWCHEM"]

AUTHORS = "Sascha Mausenberger, Sebastian Mai"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2024, 3, 25)
NAME = "NWCHEM"
DESCRIPTION = "AB INITIO interface for NWChem (TDDFT)"

CHANGELOGSTRING = """
"""

all_features = set(["h", "dm", "grad", "molden", "overlap", "phases"])


class SHARC_NWCHEM(SHARC_ABINITIO):
    """
    SHARC interface for NWCHEM
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update({"nwchem": None, "dry_run": False, "ncpu": 2})

        self.QMin.resources.types.update({"nwchem": str, "dry_run": bool})

        # Add template keys
        self.QMin.template.update(
            {
                "tda": False,  # Tamm-Dancoff
                "maxiter": None,  # Max SCF iterations
                "basis": "def2-svp",  # Basis set
                "functional": "b3lyp",  # DFT functional
                "cam": None,  # For CAM functionals
                "dispersion": None,  # Dispersion correction
                "grid": None,  # Integration grid
                "nooverlap": False,  # Do not calculate overlaps
                "spherical": False,  # Spherical or cartesian coordinates
                "forcecartesian": False,  # Force cartesian overlaps
                "cosmo": None,  # Dielectric constant
                "library_path": None,
            }
        )

        self.QMin.template.types.update(
            {
                "tda": bool,
                "maxiter": int,
                "basis": str,
                "functional": str,
                "cam": str,
                "dispersion": str,
                "grid": str,
                "nooverlap": bool,
                "spherical": bool,
                "forcecartesian": bool,
                "cosmo": float,
                "library_path": str,
            }
        )

        self._basis = None  # Needed for overlaps
        self._ao_labels = None  # PySCF AO labels

        # Setup stuff
        self._template_file = None
        self._resource_file = None

    @staticmethod
    def version() -> str:
        return SHARC_NWCHEM._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_NWCHEM._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_NWCHEM._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_NWCHEM._authors

    @staticmethod
    def name() -> str:
        return SHARC_NWCHEM._name

    @staticmethod
    def description() -> str:
        return SHARC_NWCHEM._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_NWCHEM._name}\n{SHARC_NWCHEM._description}"

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
        self.log.info(f"||{'NWChem interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.log.info("\nSpecify path to NWChem binary.")
        INFOS["nwchem"] = question("Path to NWChem:", str, KEYSTROKES=KEYSTROKES)

        self.log.info("\n\nSpecify a scratch directory. The scratch directory will be used to run the calculations.")
        INFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)

        if os.path.isfile("NWCHEM.template"):
            self.log.info("Found NWCHEM.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "NWCHEM.template"
        else:
            self.log.info("Specify a path to a NWCHEM template file.")
            while True:
                template_file = question("Template path:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(template_file):
                    self.log.info(f"File {template_file} does not exist!")
                    continue
                self._template_file = template_file
                break

        if question("Do you have a NWCHEM.resources file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            self._resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES)
            while not os.path.isfile(self._resource_file):
                self.log.info(f"{self._resource_file} does not exist!")
                self._resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES)
        else:
            self.log.info("Specify the number of CPUs to be used.")
            INFOS["ncpu"] = question("Number of CPUs (at least 2):", int, default=[2], KEYSTROKES=KEYSTROKES)[0]

            self.log.info("Specify the amount of RAM to be used.")
            INFOS["memory"] = question("Memory (MB):", int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

            if "overlap" in INFOS["needed_requests"]:
                INFOS["wfoverlap"] = question(
                    "Path to wavefunction overlap executable:", str, default="$SHARC/wfoverlap.x", KEYSTROKES=KEYSTROKES
                )
                self.log.info("State threshold for choosing determinants to include in the overlaps")
                self.log.info("For hybrids without TDA one should consider that the eigenvector X may have a norm larger than 1")
                INFOS["wfthres"] = question("Threshold:", float, default=[0.998], KEYSTROKES=KEYSTROKES)[0]

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resource_file:
            with open(os.path.join(dir_path, "NWCHEM.resources"), "w", encoding="utf-8") as file:
                for key in ("nwchem", "scratchdir", "ncpu", "memory", "wfoverlap", "wfthres"):
                    if key in INFOS:
                        file.write(f"{key} {INFOS[key]}\n")
        else:
            create_file(expand_path(self._resource_file), os.path.join(dir_path, "NWCHEM.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "NWCHEM.template"))


    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Do NWChem QM calculation
        """
        jobid = qmin.control["jobid"]
        step = qmin.save["step"]
        savedir = qmin.save["savedir"]
        # Setup workdir
        mkdir(workdir)

        self._copy_files(qmin, workdir)

        # Write input, and xyz
        writefile(os.path.join(workdir, "nwchem.inp"), self._write_input(qmin))
        xyz_str = f"{self.QMin.molecule['natom']}\n\n"
        for label, coords in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"]):
            xyz_str += f"{label:4s} {coords[0]:16.9f} {coords[1]:16.9f} {coords[2]:16.9f}\n"
        writefile(os.path.join(workdir, "input.xyz"), xyz_str)

        # Run NWChem
        starttime = datetime.datetime.now()
        exec_str = f"mpirun -np {self.QMin.resources['ncpu']} {os.path.join(qmin.resources['nwchem'])} nwchem.inp"
        exit_code = self.run_program(workdir, exec_str, os.path.join(workdir, "nwchem.log"), os.path.join(workdir, "nwchem.err"))
        endtime = datetime.datetime.now()

        if exit_code != 0:
            with open(os.path.join(workdir, "nwchem.err"), "r", encoding="utf-8") as f:
                self.log.error(f.read())
        elif exit_code == 0 and not self.QMin.save["samestep"]:
            # Save files
            if self.QMin.requests["molden"]:
                shutil.copy(os.path.join(workdir, "nwchem.molden"), os.path.join(savedir, f"nwchem.molden.{jobid}.{step}"))
            shutil.copy(os.path.join(workdir, "nwchem.movecs"), os.path.join(savedir, f"nwchem.movecs.{jobid}.{step}"))
            if not os.path.isfile(os.path.join(savedir, f"input.xyz.{step}")):
                writefile(os.path.join(savedir, f"input.xyz.{step}"), xyz_str)

            if not self.QMin.template["nooverlap"]:
                self._write_mos(qmin, workdir)
                if qmin.molecule["states"][jobid - 1] > 1:
                    _, _, det = self._dets_from_civec(
                        os.path.join(workdir, "nwchem.civecs" if jobid != 1 else "nwchem.civecs_singlet"),
                        jobid == 3,
                    )
                    writefile(os.path.join(savedir, f"dets.{jobid}.{step}"), self.format_ci_vectors(det))
                else:
                    occ, _ = self._mo_from_movec(os.path.join(workdir, "nwchem.movecs"))
                    if len(occ) == 1:
                        occ = [3 if i == 2 else 0 for i in occ[0]]
                    else:
                        occ = occ[0] + [2 if i == 1 else 0 for i in occ[1]]

                    writefile(os.path.join(savedir, f"dets.{jobid}.{step}"), self.format_ci_vectors([{tuple(occ): 1.0}]))

        return exit_code, endtime - starttime

    def _write_mos(self, qmin: QMin, workdir: str) -> None:
        """
        Write MO file for overlaps

        qmin:       QMin object of job
        workdir:    Working directory
        """
        _, movecs = self._mo_from_movec(os.path.join(workdir, "nwchem.movecs"))

        n_ao = movecs.shape[1]
        n_mo = movecs.shape[0]

        mo_string = f"2mocoef\nheader\n1\nMO-coefficients from NWChem\n1\n{n_ao}   {n_mo}\na\nmocoef\n(*)\n"

        for mat in movecs:
            for idx, i in enumerate(mat):
                if idx > 0 and idx % 3 == 0:
                    mo_string += "\n"
                mo_string += f"{i: 6.12e} "
            if n_ao - 1 % 3 != 0:
                mo_string += "\n"

        mo_string += "orbocc\n(*)"
        for i in range(n_ao):
            if i % 3 == 0:
                mo_string += "\n"
            mo_string += f"{0.0: 6.12e} "

        writefile(os.path.join(qmin.save["savedir"], f"mos.{qmin.control['jobid']}.{qmin.save['step']}"), mo_string)

    def _copy_files(self, qmin: QMin, workdir: str) -> None:
        """
        Copy initial guess files

        qmin:       QMin object of current job
        workdir:    Working directory
        """
        savedir = self.QMin.save["savedir"]
        jobid = qmin.control["jobid"]
        step = qmin.save["step"] - 1

        if not qmin.save["always_guess"]:
            if os.path.isfile(os.path.join(savedir, f"nwchem.movecs.{jobid}.{step}")):
                shutil.copy(os.path.join(savedir, f"nwchem.movecs.{jobid}.{step}"), os.path.join(workdir, "nwchem.movecs"))

    def read_template(self, template_file: str = "NWCHEM.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        if isinstance(self.QMin.template["functional"], list):
            self.QMin.template["functional"] = " ".join(self.QMin.template["functional"])
        if isinstance(self.QMin.template["dispersion"], list):
            self.QMin.template["dispersion"] = " ".join(self.QMin.template["dispersion"])
        if isinstance(self.QMin.template["cam"], list):
            self.QMin.template["cam"] = " ".join(self.QMin.template["cam"])

        if self.QMin.template["spherical"] and self.QMin.template["forcecartesian"]:
            self.log.warning("Both spherical and forcecartesian defined in template. Using cartesian!")
            self.QMin.template["spherical"] = False

        if not self.QMin.template["library_path"]:
            self.log.error("library_path has to be set in the template file!")
            raise ValueError()
        self.QMin.template["library_path"] = expand_path(self.QMin.template["library_path"])

    def read_resources(self, resources_file: str = "NWCHEM.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        if self.QMin.resources["ncpu"] < 2:
            self.log.error("NWChem needs at least 2 CPU cores!")
            raise ValueError()
        if not self.QMin.resources["nwchem"]:
            self.log.error("nmwchem has to be set in the resource file!")
            raise ValueError()
        self.QMin.resources["nwchem"] = expand_path(self.QMin.resources["nwchem"])
        if not is_exec(self.QMin.resources["nwchem"]):
            self.log.error(f"{self.QMin.resources['nwchem']} is not an executable!")
            raise ValueError()

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        if not self.QMin.template["spherical"] and self.QMin.requests["overlap"]:
            if not self.QMin.template["forcecartesian"]:
                self.log.error(
                    "Overlaps are only compatible with spherical basis! Use forcecartesian to force cartesian overlaps."
                )
                raise ValueError()
            self.log.warning("Using cartesian overlaps, results may be wrong!")

    def setup_interface(self) -> None:
        super().setup_interface()
        self._build_jobs()

        self.log.info(f"Scratchdir: {self.QMin.resources['scratchdir']}")
        self.log.info(f"Savedir: {self.QMin.save['savedir']}")

        # Setup multmap
        self.log.debug("Building multmap")
        self.QMin.maps["multmap"] = {}
        for ijob, job in self.QMin.control["jobs"].items():
            for imult in job["mults"]:
                self.QMin.maps["multmap"][imult] = ijob
            self.QMin.maps["multmap"][-(ijob)] = job["mults"]
        self.QMin.maps["multmap"][1] = 1

        # Check if basisset in library
        if self.QMin.template["nooverlap"]:
            return

        basis_path = os.path.join(self.QMin.template["library_path"], self.QMin.template["basis"])
        if not os.path.isfile(basis_path):
            self.log.error(f"Basis {self.QMin.template['basis']} not in library path!")
            raise ValueError()
        self._basis = self._load_basis(basis_path, self.QMin.molecule["elements"])

    def _load_basis(self, basis_file: str, elements: list[str]) -> dict[str, str]:
        """
        Parse basis functions from basis file and return functions for requested elements

        basis_file: Path to NWChem basis file
        elements:   List of elements
        """
        basis = {}
        with open(basis_file, "r", encoding="utf-8") as file:
            data = re.findall(r"basis \".*?\n(.*?)end", file.read(), re.DOTALL)
        for i in data:
            if i.split()[0] in elements:
                basis[i.split()[0]] = i
        return basis

    def _build_jobs(self) -> None:
        """
        Generate joblist for schedule generation
        """
        self.log.debug("Building job map")
        jobs = {}

        for idx, state in enumerate(self.QMin.control["states_to_do"], 1):
            if state > 0:
                jobs[idx] = {"mults": [idx]}

        self.QMin.control["jobs"] = jobs
        self.QMin.control["joblist"] = sorted(set(jobs))

    def _build_schedule(self) -> None:
        """
        Generate schedule from joblist
        """
        # sort the gradients into the different jobs
        gradjob = {f"master_{job}": {} for job in self.QMin.control["joblist"]}
        if self.QMin.maps["gradmap"]:
            for grad in self.QMin.maps["gradmap"]:
                ijob = self.QMin.maps["multmap"][grad[0]]
                gradjob[f"master_{ijob}"][grad] = {"gs": grad == (1, 1)}

        schedule = [{}]

        # add the master calculations
        self.QMin.control["nslots_pool"].append(1)

        for job in sorted(gradjob):
            qmin = deepcopy(self.QMin)
            qmin.control["master"] = True
            qmin.control["jobid"] = int(job.split("_")[1])
            qmin.maps["gradmap"] = set(gradjob[job])
            schedule[-1][job] = qmin

        self.QMin.scheduling["schedule"] = schedule

    def getQMout(self) -> None:
        """
        Parse output file(s) and populate QMout
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

        for job in self.QMin.control["joblist"]:
            with open(os.path.join(self.QMin.resources["scratchdir"], f"master_{job}/nwchem.log"), "r", encoding="utf-8") as f:
                output = f.read()

            # Get energies and S**2
            civec = None
            if self.QMin.molecule["states"][job - 1] > 1:
                civec = os.path.join(
                    self.QMin.resources["scratchdir"], f"master_{job}", "nwchem.civecs" if job != 1 else "nwchem.civecs_singlet"
                )
            energies, s2 = self._get_energies(output, civec, job)
            start = sum(s * m for (m, s) in enumerate(self.QMin.molecule["states"][: job - 1], 1))
            stop = sum(s * m for (m, s) in enumerate(self.QMin.molecule["states"][:job], 1))
            np.einsum("ii->i", self.QMout["h"])[start:stop] = np.tile(energies, job)
            self.QMout["notes"][f"S**2 mult {job}"] = s2

            # Get gradients
            if self.QMin.requests["grad"]:
                for state, grad in self._get_gradients(output):
                    for key, val in self.QMin.maps["statemap"].items():
                        if (val[0], val[1]) == (job, state):
                            self.QMout["grad"][key - 1] = grad

            # Get Dipoles
            if self.QMin.requests["dm"]:
                dipoles = np.einsum("ij->ji", self._get_dipoles(output, self.QMin.molecule["states"][job - 1]))
                dipole_sub = np.zeros((3, dipoles.shape[1], dipoles.shape[1]))
                dipole_sub[:, 0, :] = dipole_sub[:, :, 0] = dipoles
                for i in range(1, job + 1):
                    self.QMout["dm"][
                        :,
                        start + (i - 1) * dipoles.shape[1] : start + i * dipoles.shape[1],
                        start + (i - 1) * dipoles.shape[1] : start + i * dipoles.shape[1],
                    ] = dipole_sub

            # Get overlaps
            if self.QMin.requests["overlap"]:
                ovlp_mat = self.parse_wfoverlap(
                    os.path.join(self.QMin.resources["scratchdir"], f"WFOVL_{job}_{job}", "wfovl.out")
                )
                for i in range(self.QMin.molecule["nmstates"]):
                    for j in range(self.QMin.molecule["nmstates"]):
                        m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                        m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                        if not m1 == m2 == job:
                            continue
                        if not ms1 == ms2:
                            continue
                        self.QMout["overlap"][i, j] = ovlp_mat[s1 - 1, s2 - 1]

        # Get phases
        if self.QMin.requests["phases"]:
            self.QMout["phases"] = deepcopy(np.einsum("ii->i", self.QMout["overlap"]))
            self.QMout["phases"][self.QMout["phases"] > 0] = 1
            self.QMout["phases"][self.QMout["phases"] < 0] = -1

        return self.QMout

    def _get_dipoles(self, nw_out: str, states: int) -> np.ndarray:
        """
        Get dipoles from logfile

        nw_out: NWChem logfile output
        states: number of states
        """
        dm = re.findall(r"Dipole moment.*A\.U\.\n\s+DMX\s+(-?\d+\.\d+).*\n\s+DMY\s+(-?\d+\.\d+).*\n\s+DMZ\s+(-?\d+\.\d+)", nw_out)
        tdm = re.findall(r"Transition Moments\s+X\s+(-?\d+\.\d{5})\s+Y\s+(-?\d+\.\d{5})\s+Z\s+(-?\d+\.\d{5})", nw_out)
        if not dm:
            self.log.error("No dipoles found in output!")
            raise ValueError()
        return np.asarray(dm[:] + tdm[: states - 1], dtype=float)

    def _get_gradients(self, nw_out: str) -> list[tuple[int, np.ndarray]]:
        """
        Get gradients from logfile

        nw_out: NWChem logfile output
        """
        gradients = []
        grad = re.findall(r"[^D]DFT ENERGY GRADIENTS(.*?)Time", nw_out, re.DOTALL)
        if grad:  # Ground state
            grad = re.findall(r"(-?\d+\.\d+)", grad[0])
            gradients.append((1, np.asarray(grad, dtype=float).reshape(-1, 6)[:, 3:]))

        # Root gradients
        grad = re.findall(r"Root\s+(\d+)\n\n\s+TDDFT ENERGY GRADIENTS(.*?)TDDFT", nw_out, re.DOTALL)
        for i, g in grad:
            gradients.append((int(i) + 1, np.asarray(re.findall(r"(-?\d+\.\d+)", g), dtype=float).reshape(-1, 6)[:, 3:]))
        return gradients

    def _get_energies(
        self, nw_out: str, nw_civec: str | None = None, jobid: int | None = None
    ) -> tuple[list[float], list[float]]:
        """
        Get energies from logfile and/or civecs
        Save mo file if nw_civec specified

        jobid:    Job id
        nw_out:   NWChem logfile output
        nw_civec: NWChem civec file path
        """
        s2 = 0.0
        if jobid != 1:
            s2 = re.search(r"<S2> =\s+(\d+\.\d+)", nw_out)
            if not s2:
                self.log.error("S**2 not found in logfile!")
                raise ValueError()
            s2 = float(s2.group(1))

        if not nw_civec:
            energy = re.search(r"Total DFT energy\s+=\s+(\-?\d+\.\d+)", nw_out)
            if not energy:
                self.log.error("Energy not found in logfile!")
                raise ValueError()
            return [float(energy.group(1))], [s2]

        # Find ground state energy from nwchem.log
        gs_energy = re.search(r"Ground state energy =\s+(\-?\d+\.\d+)", nw_out)
        if not gs_energy:
            self.log.error("No ground state energy found in output file!")
            raise ValueError()
        gs_energy = float(gs_energy.group(1))
        self.log.debug(f"Ground state energy job {jobid} {gs_energy:.7f} S**2 {s2}")

        # Get root energies, S**2, save dets
        roots, s2_root, _ = self._dets_from_civec(
            os.path.join(self.QMin.resources["scratchdir"], f"master_{jobid}", nw_civec), jobid == 3
        )
        return [gs_energy] + [gs_energy + i for i in roots], [s2] + s2_root

    def _read_value(self, f, t: str = "i", l: int = 8) -> np.ndarray:
        """
        Parse value(s) from Fortran file, remove padding before and after
        Fortran padding: [len(byte) | value(s) | len(byte)]
        Automatically detects how many values should be read

        f:  File handle to binary file
        t:  Type of value (numpy dtype)
        l:  Length of a single value (byte)
        """
        n = np.fromfile(f, dtype=np.dtype("i4"), count=1)[0] // l
        vals = np.fromfile(f, dtype=np.dtype(f"{t}{l}"), count=n)
        f.read(4)  # Skip padding
        return vals

    def _dets_from_civec(
        self, civec_path: str, trip: bool = False
    ) -> tuple[list[float], list[float], list[dict[tuple[int, ...], float]]]:
        """
        Parse CI vectors from civec file
        https://github.com/nwchemgit/nwchem/blob/hotfix/release-7-2-0/src/nwdft/lr_tddft/tddft_analysis.F#L1851
        Return energies, S**2 (basically for free, easy to parse) and civecs

        civec_path: Path to civec binary file
        trip:       Is triplet state
        """

        with open(civec_path, "rb") as f:
            self.log.debug(f"Parsing {civec_path}")

            # Get infos from header
            tda = bool(self._read_value(f)[0])
            ipol = self._read_value(f)[0]  # 1 = restricted, 2 = unrestricted
            nroots = self._read_value(f)[0]
            nocc, nmo, _, _, nov = (
                self._read_value(f).tolist(),  # nocc a, b electrons
                self._read_value(f).tolist(),  # nmo a, b orbitals
                self._read_value(f).tolist(),  # nfc a, b frozen core, always 0 in interface
                self._read_value(f).tolist(),  # nfv a, b frozen virtuals, always 0 in interface
                self._read_value(f).tolist(),  # nov a, b occupied virtual pairs
            )
            f.read(8)  # Skip empty line

            self.log.debug(f"TDA {tda}, restr {ipol}, roots {nroots}, nocc {nocc}, nmo {nmo}, nov {nov}")
            self.log.debug(f"CIvec {(nov[0]+nov[1])*(int(tda)+1)} floats, {(nov[0]+nov[1])*(int(tda)+1)*8} bytes per root")

            # Parse civecs, get energies and S**2 for free
            occ_str = [3 if ipol == 1 else 1] * nocc[0] + [0] * (nmo[0] - nocc[0]) + [2] * nocc[1] + [0] * (nmo[1] - nocc[1])

            skip_beta = False
            if ipol == 2 and nocc[1] == 0:
                self.log.warning("NWChem bug, unrestricted calculation, but only alpha coefficients given, skipping beta.")
                skip_beta = True

            eigenvectors = [{tuple(occ_str): 1.0}]
            energies = []
            s2 = []
            for i in range(nroots):
                energies.append(self._read_value(f, "f")[0])
                s2.append(self._read_value(f, "f")[0])
                self.log.debug(f"Root {i+1}, energy {energies[-1]:.7f}, S**2 {s2[-1]:.5f}")

                civecs = []
                for _ in range(ipol):
                    civec = self._read_value(f, "f")
                    if not tda:
                        civec += self._read_value(f, "f")
                        civec /= 2
                    civecs.extend(civec)
                    if skip_beta:
                        break

                # Generate determinant tuple and assign value
                tmp = {}
                it_civec = iter(civecs)
                for occ in range(nocc[0]):
                    for virt in range(nocc[0], nmo[0]):
                        val = next(it_civec)
                        val *= np.sqrt(0.5) if ipol == 1 else 1  # Normalization factor

                        key = occ_str[:]
                        key[occ], key[virt] = (2, 1) if ipol == 1 else (0, 1)
                        tmp[tuple(key)] = val

                        if ipol == 1:
                            key[occ], key[virt] = 1, 2
                            tmp[tuple(key)] = val if trip else -val

                # Unrestricted
                for occ in range(nmo[0], nmo[0] + nocc[1]):
                    for virt in range(nmo[0] + nocc[1], sum(nmo)):
                        key = occ_str[:]
                        key[occ], key[virt] = 0, 2
                        tmp[tuple(key)] = next(it_civec)
                self.log.debug(f"Determinants norm: {np.linalg.norm(list(tmp.values())):.5f}")

                # Filter out dets with lowest contribution
                self.trim_civecs(tmp)
                eigenvectors.append(tmp)
            return energies, s2, eigenvectors

    def _mo_from_movec(self, movec_path: str) -> tuple[list[int], list[np.ndarray]]:
        """
        Parse coefficients and occupations from movec file

        movec_path: Path to movec file
        """
        with open(movec_path, "rb") as file:
            self.log.debug("--- Start of movec header ---")
            for _ in range(6):  # Skip header
                val = self._read_value(file, "u", 1)
                self.log.debug("".join([chr(i) for i in val]))  # Header is mostly text
            self.log.debug("--- End of movec header ---")

            restr = self._read_value(file)[0]
            # Number of AO/MO, or the other way around, source not available
            nao, nmo = self._read_value(file)[0], self._read_value(file)[0]
            self.log.debug(f"AO {nao}, MO {nmo}, {'restricted' if restr == 1 else 'unrestricted'}")

            # List of occupations and MO energies
            occ, energies = [], []
            occ.append(self._read_value(file, "f").tolist())
            occ = convert_list(occ)  # Float to integer
            energies.append(self._read_value(file, "f"))

            self.log.debug(f"Occupations {'alpha' if restr == 2 else ''} {occ[0]}")
            self.log.debug(f"MO energies {'alpha' if restr == 2 else ''} {' '.join([f'{i:.4f}' for i in energies[0]])}")

            # Get coefficients and scale AO
            coeff = np.zeros((nmo * restr, nmo))
            mol = gto.Mole()
            mol.basis = self._basis
            mol.unit = "Bohr"
            mol.charge = 0 if not self.QMin.molecule["charge"] else self.QMin.molecule["charge"][0]
            mol.atom = [[e, c] for e, c in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"].tolist())]
            mol.cart = not self.QMin.template["spherical"]
            mol.build()

            for i in range(nmo):
                coeff[i, :] = self._read_value(file, "f")

            if restr == 2:
                occ.append(convert_list(self._read_value(file, "f").tolist()))
                energies.append(self._read_value(file, "f"))

                self.log.debug(f"Occupations beta {occ[1]}")
                self.log.debug(f"MO energies beta {' '.join([f'{i:.4f}' for i in energies[1]])}")

                for i in range(nmo):
                    coeff[nmo + i, :] = self._read_value(file, "f")

            if self.QMin.template["spherical"]:
                for idx, label in enumerate(mol.ao_labels()):
                    if "dxz" in label or "f+1" in label or "f+3" in label or "g-3" in label or "g+1" in label:
                        coeff[:, idx] *= -1

        return occ, coeff

    def _create_aoovl(self) -> None:
        """
        Calculate AO overlaps between old and new geometry with PySCF
        """
        mol = gto.Mole()

        # Set basis, unit and coords
        mol.basis = self._basis
        mol.unit = "Bohr"
        with open(
            os.path.join(self.QMin.save["savedir"], f"input.xyz.{self.QMin.save['step']-1}"), "r", encoding="utf-8"
        ) as file:
            atoms = file.readlines()[2:]  # Skip the first two lines
            atom_coords = [line.strip() for line in atoms]

        mol.atom = atom_coords
        mol.atom.extend([[e, c] for e, c in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"].tolist())])
        mol.cart = not self.QMin.template["spherical"]
        mol.build()

        self._ao_labels = [i.split()[::2] for i in mol.ao_labels()[: mol.nao // 2]]

        ovlp = mol.intor("int1e_ovlp")[: mol.nao // 2, mol.nao // 2 :]
        order = sorted(list(range(mol.nao // 2)), key=cmp_to_key(self._sort_ao))

        self.log.debug(f"PySCF AO order:\n{self._print_ao(self._ao_labels)}")
        if "SP" in str(self._basis.values()):  # Reorder basis if SP shell
            self.log.debug(f"NWChem AO order:\n{self._print_ao([self._ao_labels[i] for i in order])}")

            ovlp = ovlp[np.ix_(order, order)]

        # Write overlap file
        ovlp_str = f"{mol.nao//2} {mol.nao//2}\n"
        ovlp_str += "\n".join(" ".join(f"{elem: .15e}" for elem in row) for row in ovlp)
        writefile(os.path.join(self.QMin.save["savedir"], "AO_overl.mixed"), ovlp_str)

    def _print_ao(self, ao_labels: list[list[str]]) -> str:
        """
        Make PySCF AO label string better readable

        ao_labels:  List of AO labels
        """
        ao_str = ""
        for label in ao_labels:
            ao_str += f"{self.QMin.molecule['elements'][int(label[0])]:4s} {label[1]:10s}\n"
        return ao_str

    def _sort_ao(self, ao1: int, ao2: int) -> int:
        """
        PySCF AO label order to NWChem AO label order
        """
        first = self._ao_labels[ao1]
        second = self._ao_labels[ao2]

        label_to_int = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5, "h": 6, "i": 7}
        return (
            (int(first[0]) - int(second[0])) * 100
            + (int(first[1][0]) - int(second[1][0])) * 10
            + (label_to_int[first[1][1]] - label_to_int[second[1][1]])
        )

    def run(self) -> None:
        starttime = datetime.datetime.now()

        self._build_schedule()

        if not self.QMin.resources["dry_run"]:
            self.runjobs(self.QMin.scheduling["schedule"])

        if self.QMin.requests["overlap"]:
            self._run_wfoverlap()

        self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def _write_input(self, qmin: QMin) -> str:
        """
        Write input file for NWChem
        """
        job = qmin.control["jobid"]
        # Total memory, charge and geometry
        input_str = f"memory total {self.QMin.resources['memory']} mb\n"
        input_str += f"charge {self.QMin.molecule['charge'][job-1]}\n"
        input_str += "geometry units bohr noautosym nocenter\n load format xyz input.xyz\nend\n\n"

        # Basis set
        input_str += (
            f"basis{' spherical' if self.QMin.template['spherical'] else ''}\n * library {self.QMin.template['basis']}\nend\n\n"
        )

        # COSMO
        if self.QMin.template["cosmo"]:
            input_str += f"cosmo\n minbem 3\n ificos 1\n dielec {self.QMin.template['cosmo']}\nend\n\n"

        # DFT section, functional, grid, ...
        input_str += f"dft\n xc {self.QMin.template['functional']}\n"
        if self.QMin.template["cam"]:
            input_str += f" cam {self.QMin.template['cam']}\n"
        if self.QMin.template["dispersion"]:
            input_str += f" disp {self.QMin.template['dispersion']}\n"
        if self.QMin.template["grid"]:
            input_str += f" grid {self.QMin.template['grid']}\n"
        if self.QMin.template["maxiter"]:
            input_str += f" maxiter {self.QMin.template['maxiter']}\n"
        if not qmin.save["init"] and not qmin.save["always_guess"]:
            input_str += " vectors nwchem.movecs\n"
        input_str += f" mult {job}\nend\n\n"

        if (job, 1) in qmin.maps["gradmap"]:
            input_str += "task dft gradient\n\n"
        elif qmin.molecule["states"][job - 1] < 2:
            input_str += "task dft energy\n\n"

        # TDDFT section
        if not qmin.maps["gradmap"] or qmin.maps["gradmap"] == {(job, 1)}:
            input_str += self._generate_tddft_input(job)
        else:
            for _, root in qmin.maps["gradmap"]:
                if root > 1:
                    input_str += self._generate_tddft_input(job, root - 1)

        if self.QMin.requests["molden"] or self.QMin.requests["dm"]:
            input_str += "property\n"
            if self.QMin.requests["molden"]:
                input_str += " moldenfile\n molden_norm nwchem\n"
            if self.QMin.requests["dm"]:
                input_str += " dipole\n"
            input_str += " vectors nwchem.movecs\n"
            input_str += "end\ntask dft property\n"

        return input_str

    def _generate_tddft_input(self, job: int, root: int | None = None) -> str:
        """
        Genreate TDDFT input block

        job:    Job id
        root:   Gradient root number
        """
        if self.QMin.molecule["states"][job - 1] < 2:
            return ""
        tddft_str = f"tddft\n nroots {self.QMin.molecule['states'][job-1]-1}\n target 1\n civecs"
        if self.QMin.template["maxiter"]:
            tddft_str += f"\n maxiter {self.QMin.template['maxiter']}"
        if self.QMin.template["tda"]:
            tddft_str += "\n cis"
        if job == 1:
            tddft_str += "\n notriplet"
        if root:
            tddft_str += f"\n grad\n  root {root}\n end"
        tddft_str += f"\nend\ntask tddft {'gradient' if root else 'energy'}\n\n"
        return tddft_str


if __name__ == "__main__":
    SHARC_NWCHEM().main()
