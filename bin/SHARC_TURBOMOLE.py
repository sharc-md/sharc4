#!/usr/bin/env python3
import datetime
import os
import re
import shutil
import subprocess as sp
from copy import deepcopy
from functools import cmp_to_key
from io import TextIOWrapper

import numpy as np
from constants import NUMBERS, rcm_to_Eh, au2a
from pyscf import gto, tools
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from SHARC_ORCA import SHARC_ORCA
from utils import expand_path, itmult, mkdir, writefile

__all__ = ["SHARC_TURBOMOLE"]

AUTHORS = "Sascha Mausenberger, Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 5, 29)
NAME = "TURBOMOLE"
DESCRIPTION = "AB INITIO interface for TURBOMOLE (RICC2/ADC2)"

CHANGELOGSTRING = """
"""

BASISSETS = (
    "SV",
    "SVP",
    "SV(P)",
    "def-SVP",
    "def2-SVP",
    "dhf-SVP",
    "dhf-SVP-2c",
    "def-SV(P)",
    "def2-SV(P)",
    "dhf-SV(P)",
    "dhf-SV(P)-2c",
    "DZ",
    "DZP",
    "TZ",
    "TZP",
    "TZV",
    "TZVP",
    "def-TZVP",
    "TZVE",
    "TZVEP",
    "TZVPP",
    "def-TZVPP",
    "def2-TZVP",
    "dhf-TZVP",
    "dhf-TZVP-2c",
    "def2-TZVPP",
    "dhf-TZVPP",
    "dhf-TZVPP-2c",
    "TZVPPP",
    "QZV",
    "def-QZV",
    "def2-QZV",
    "QZVP",
    "def-QZVP",
    "def2-QZVP",
    "dhf-QZVP",
    "dhf-QZVP-2c",
    "QZVPP",
    "def-QZVPP",
    "def2-QZVPP",
    "dhf-QZVPP",
    "dhf-QZVPP-2c",
    "minix",
    "sto-3ghondo",
    "4-31ghondo",
    "6-31ghondo",
    "3-21ghondo",
    "dzphondo",
    "tzphondo",
    "6-31G",
    "6-31G*",
    "6-31G**",
    "6-311G",
    "6-311G*",
    "6-311G**",
    "6-311++G**",
    "cc-pVDZ",
    "cc-pV(D+d)Z",
    "aug-cc-pVDZ",
    "aug-cc-pV(D+d)Z",
    "YP-aug-cc-pVDZ",
    "cc-pwCVDZ",
    "aug-cc-pwCVDZ",
    "cc-pVDZ-sp",
    "cc-pVTZ",
    "cc-pV(T+d)Z",
    "aug-cc-pVTZ",
    "aug-cc-pV(T+d)Z",
    "YP-aug-cc-pVTZ",
    "cc-pwCVTZ",
    "aug-cc-pwCVTZ",
    "cc-pVTZ-sp",
    "cc-pVQZ",
    "cc-pV(Q+d)Z",
    "aug-cc-pVQZ",
    "aug-cc-pV(Q+d)Z",
    "YP-aug-cc-pVQZ",
    "cc-pwCVQZ",
    "aug-cc-pwCVQZ",
    "cc-pVQZ-sp",
    "cc-pV5Z",
    "cc-pV(5+d)Z",
    "aug-cc-pV5Z",
    "aug-cc-pV(5+d)Z",
    "YP-aug-cc-pV5Z",
    "cc-pwCV5Z",
    "aug-cc-pwCV5Z",
    "cc-pV5Z-large-s",
    "cc-pV6Z",
    "cc-pV(6+d)Z",
    "aug-cc-pV6Z",
    "aug-cc-pV(6+d)Z",
    "cc-pV6Z-sp",
    "cc-pVDZ-F12",
    "cc-pVTZ-F12",
    "cc-pVQZ-F12",
    "def2-SVPD",
    "def2-TZVPPD",
    "def2-TZVPD",
    "def2-QZVPPD",
    "def2-QZVPD",
)

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
        "molden",
        "point_charges",
        "grad_pc",
        # raw data request
        "mol",
        "wave_functions",
        "density_matrices",
    ]
)


class SHARC_TURBOMOLE(SHARC_ABINITIO):
    """
    SHARC interface for TURBOMOLE
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION
    _theodore_settings = {
        "rtype": "ricc2",
        "rfile": "ricc2.out",
        "mo_file": "molden.input",
        "read_binary": True,
        "jmol_orbitals": False,
        "molden_orbitals": True,
        "Om_formula": 2,
        "eh_pop": 1,
        "comp_ntos": True,
        "print_OmFrag": True,
        "output_file": "tden_summ.txt",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "frozen": -1,
                "basis": None,
                "auxbasis": None,
                "method": "adc2",
                "scf": "dscf",
                "spin-scaling": None,
                "basislib": None,
                "douglas-kroll": False,
                "dipolelevel": 0,
            }
        )

        self.QMin.template.types.update(
            {
                "frozen": int,
                "basis": str,
                "auxbasis": str,
                "method": str,
                "scf": str,
                "spin-scaling": str,
                "basislib": str,
                "douglas-kroll": bool,
                "dipolelevel": int,
            }
        )

        # Add resource keys
        self.QMin.resources.update(
            {"turbodir": None, "orcadir": None, "neglected_gradient": "zero", "schedule_scaling": 0.1, "dry_run": False}
        )
        self.QMin.resources.types.update(
            {"turbodir": str, "orcadir": str, "neglected_gradient": str, "schedule_scaling": float, "dry_run": bool}
        )

        self._ao_labels = None

    @staticmethod
    def version() -> str:
        return SHARC_TURBOMOLE._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_TURBOMOLE._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_TURBOMOLE._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_TURBOMOLE._authors

    @staticmethod
    def name() -> str:
        return SHARC_TURBOMOLE._name

    @staticmethod
    def description() -> str:
        return SHARC_TURBOMOLE._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_TURBOMOLE._name}\n{SHARC_TURBOMOLE._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        pass

    def prepare(self, INFOS: dict, dir_path: str):
        pass

    def read_template(self, template_file: str = "TURBOMOLE.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Check if basis available and correct casing
        if not isinstance(self.QMin.template["basis"], str):
            self.log.error("No basis set defined in template!")
            raise ValueError()
        for i in BASISSETS:
            if self.QMin.template["basis"].casefold() == i.casefold():
                self.QMin.template["basis"] = i
                break
        else:
            self.log.error(f"{self.QMin.template['basis']} is not a valid basisset!")
            raise ValueError()

        # Do same for auxbasis if specified
        if self.QMin.template["auxbasis"]:
            if self.QMin.template["basislib"]:
                self.log.error("basislib and auxbasis keyword cannot be used together!")
                raise ValueError()
            for i in BASISSETS:
                if self.QMin.template["auxbasis"].casefold() == i.casefold():
                    self.QMin.template["auxbasis"] = i
                    break
            else:
                self.log.error(f"{self.QMin.template['auxbasis']} is not a valid auxbasisset!")
                raise ValueError()

        # Check method
        if self.QMin.template["method"] not in ("cc2", "adc2", "adc(2)"):
            self.log.error(f"Method {self.QMin.template['method']} invalid. Valid methods are cc2 and adc2")
            raise ValueError()
        if self.QMin.template["method"] == "adc2":
            self.QMin.template["method"] = "adc(2)"

        # Check dipolelevel
        if self.QMin.template["dipolelevel"] < 0 or self.QMin.template["dipolelevel"] > 2:
            self.log.error("Dipolelevel must be an integer between 0 and 2.")
            raise ValueError()

        # Check spin-scaling
        if (scaling := self.QMin.template["spin-scaling"]) and scaling not in ("scs", "sos", "lt-sos"):
            self.log.error(f"spin-scaling {scaling} invalid. Use scs, sos or lt-sos.")
            raise ValueError()

        if self.QMin.template["spin-scaling"] == "lt-sos" and self.QMin.template["dipolelevel"] > 1:
            self.log.error("lt-sos not compatible with dipolelevel 2. Use dipole level 0 or 1.")
            raise ValueError()

        # Check scf
        if self.QMin.template["scf"] not in ("dscf", "ridft"):
            self.log.error(f"scf {self.QMin.template['scf']} invalid. Use dscf or ridft.")
            raise ValueError()

    def read_resources(self, resources_file: str = "TURBOMOLE.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        if not self.QMin.resources["turbodir"]:
            self.log.error("turbodir has to be set in resources.")
            raise ValueError()
        self.QMin.resources["turbodir"] = expand_path(self.QMin.resources["turbodir"])

        if orcadir := self.QMin.resources["orcadir"]:
            self.QMin.resources["orcadir"] = expand_path(orcadir)
            os.environ["PATH"] += f":{orcadir}"
            os.environ["LD_LIBRARY_PATH"] += f":{orcadir}"

        # Setup environment
        os.environ["TURBODIR"] = (turbodir := self.QMin.resources["turbodir"])

        if (ncpu := self.QMin.resources["ncpu"]) > 1:
            os.environ["PARA_ARCH"] = "SMP"
            os.environ["PARANODES"] = str(ncpu)
        os.environ["OMP_NUM_THREADS"] = str(ncpu)

        arch = sp.Popen([os.path.join(turbodir, "scripts", "sysname")], stdout=sp.PIPE).communicate()[0].decode().strip()
        os.environ["PATH"] = f"{turbodir}/scripts:{turbodir}/bin/{arch}:" + os.environ["PATH"]

    def setup_interface(self) -> None:
        super().setup_interface()

        if len(self.QMin.molecule["states"]) > 2 and self.QMin.molecule["states"][0] < 1:
            self.log.error("Due to a TURBOMOLE bug at least two singlets are required if triplet states are requested!")
            raise ValueError

        if self.QMin.resources["ncpu"] > 1 and self.QMin.template["spin-scaling"] == "lt-sos":
            self.log.warning("lt-sos is not fully SMP parallelized.")

        # Build job map
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

    def _build_jobs(self) -> None:
        """
        Build dictionary with master jobs
        """
        self.log.debug("Building job map")
        jobs = {}
        for idx, state in enumerate(self.QMin.molecule["states"]):
            if state > 0 and idx != 2:
                jobs[idx + 1] = {"mults": [idx + 1], "restr": idx % 2 == 0}
            if state > 0 and idx == 2:
                jobs[1]["mults"].append(3)

        self.QMin.control["jobs"] = jobs
        self.QMin.control["joblist"] = sorted(set(jobs))

    def read_requests(self, requests_file: str | dict = "QM.in") -> None:
        super().read_requests(requests_file)

        # Check incompabilities with socs
        if self.QMin.requests["soc"]:
            if self.QMin.template["method"] == "cc2":
                self.log.error("SOCs and CC2 are not compatible. Use ADC2 instead.")
                raise ValueError()
            if any(NUMBERS[i] > 36 for i in self.QMin.molecule["elements"]):
                self.log.error("SOCs are not possible for elements beyond Kr.")
                raise ValueError()
            if self.QMin.template["spin-scaling"] == "lt-sos":
                self.log.error("SOCs are not possible with lt-sos.")
                raise ValueError()
            if not self.QMin.resources["orcadir"]:
                self.log.error("orcadir has to be specified in resources for SOCs.")
                raise ValueError()
            if (orca := SHARC_ORCA.get_orca_version(self.QMin.resources["orcadir"])) >= (5,):
                self.log.error(
                    f"SOCs are only compatible with ORCA version <= 4.x, found version {'.'.join(str(i) for i in orca)}"
                )
                raise ValueError()
            if len(states := self.QMin.molecule["states"]) < 3 or (states[0] == 0 or states[2] == 0):
                self.log.warning("SOCs require S+T states, disable SOCs!")
                self.QMin.requests["soc"] = False

    def _create_aoovl(self) -> None:
        pass

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        # Create workdir
        self.log.debug(f"Create workdir {workdir}")
        mkdir(workdir)

        # Write input files
        self._writegeom(workdir)

        # Collect all return codes
        codes = []

        starttime = datetime.datetime.now()
        # Master job related things
        if qmin.control["master"]:
            codes.append(self._run_define(workdir, (jobid := qmin.control["jobid"])))

            # Modify control file
            add_lines = ["$excitations\n"]
            add_section = [("$ricc2", "maxiter 45\n")]
            if self.QMin.requests["soc"] and jobid == 1:
                add_lines.append("$mkl\n")
            if self.QMin.template["douglas-kroll"]:
                add_lines.append("$rdkh\n")
            match self.QMin.template["spin-scaling"]:
                case "scs":
                    add_section.append(("$ricc2", "scs\n"))
                case "sos" | "lt-sos":
                    add_section.append(("$ricc2", "sos cos= 1.20000 css= 0.33333\n"))
            if self.QMin.template["spin-scaling"] == "lt-sos":
                add_lines.append("$laplace\nconv=5\n")
            add_lines.append("$scfiterlimit 100\n")
            if self.QMin.molecule["point_charges"]:
                add_lines.append("$point_charges file=pc\n")
                add_lines.append("$point_charge_gradients file=pc_grad\n")
                add_section.append(("$drvopt", "point charges\n"))

            # Add states
            for mult in qmin.control["jobs"][jobid]["mults"]:
                nst = qmin.control["states_to_do"][mult - 1]
                add_section.append(("$excitations", f"irrep=a multiplicity={mult} nexc={nst} npre={nst+1}, nstart={nst+1}\n"))

            # Always calc both sides with CC2
            if qmin.template["method"] == "cc2":
                add_section.append(("$excitations", "bothsides\n"))

            self._modify_file((control := os.path.join(workdir, "control")), add_lines, ["$scfiterlimit"], add_section)

            # Run dscf/ridft, backup control file
            shutil.copy(control, os.path.join(workdir, "control.bak"))

            # Copy initial guess
            match qmin.save:
                case {"always_guess": True}:
                    pass
                case {"always_orb_init": True} | {"init": True}:
                    try:
                        if jobid == 1:
                            shutil.copy(os.path.join(qmin.resources["pwd"], "mos.init"), os.path.join(workdir, "mos"))
                        else:
                            shutil.copy(
                                os.path.join(qmin.resources["pwd"], f"alpha.{jobid}.init"), os.path.join(workdir, "alpha")
                            )
                            shutil.copy(os.path.join(qmin.resources["pwd"], f"beta.{jobid}.init"), os.path.join(workdir, "beta"))
                    except FileNotFoundError as exc:
                        if qmin.save["always_orb_init"]:
                            self.log.error("mos.init file not found!")
                            raise exc
                case {"samestep": True}:
                    if jobid == 1:
                        shutil.copy(
                            os.path.join(qmin.save["savedir"], f"mos.{jobid}.{qmin.save['step']}"), os.path.join(workdir, "mos")
                        )
                    else:
                        shutil.copy(
                            os.path.join(qmin.save["savedir"], f"alpha.{jobid}.{qmin.save['step']}"),
                            os.path.join(workdir, "alpha"),
                        )
                        shutil.copy(
                            os.path.join(qmin.save["savedir"], f"beta.{jobid}.{qmin.save['step']}"), os.path.join(workdir, "beta")
                        )
                case _:
                    if jobid == 1:
                        shutil.copy(
                            os.path.join(qmin.save["savedir"], f"mos.{jobid}.{qmin.save['step']-1}"), os.path.join(workdir, "mos")
                        )
                    else:
                        shutil.copy(
                            os.path.join(qmin.save["savedir"], f"alpha.{jobid}.{qmin.save['step']-1}"),
                            os.path.join(workdir, "alpha"),
                        )
                        shutil.copy(
                            os.path.join(qmin.save["savedir"], f"beta.{jobid}.{qmin.save['step']-1}"),
                            os.path.join(workdir, "beta"),
                        )

            if qmin.template["scf"] == "ridft":
                codes.append(self._run_ridft(workdir, qmin))
                self.log.debug(f"ridft exited with code {codes[-1]}")
            codes.append(self.run_program(workdir, "dscf", "dscf.out", "dscf.err"))
            self.log.debug(f"dscf exited with code {codes[-1]}")

            # TODO: dm
            if qmin.requests["soc"] and jobid == 1:
                self._modify_file(control, None, None, [("$excitations", "tmexc istates=all fstates=all operators=soc\n")])
            # self._modify_file(control, None, None, [("$excitations", "exprop states=all relaxed operators=diplen\n")])
            # self._modify_file(control, None, None, [("$excitations", "static relaxed operators=diplen\n")])
            # self._modify_file(control, None, None, [("$excitations", "spectrum states=all operators=diplen\n")])

            codes.append(self._run_ricc2(workdir, qmin.resources["ncpu"]))

            # Generate molden file
            self._generate_molden(workdir)

            # Run orca_soc if soc requested
            if qmin.requests["soc"] and jobid == 1:
                codes.append(self._run_orca_soc(workdir))
                self.log.debug(f"orca_mkl/orca_soc exited with code {codes[-1]}")

            # Save files
            if jobid == 1:
                shutil.copy(os.path.join(workdir, "mos"), os.path.join(qmin.save["savedir"], f"mos.{jobid}.{qmin.save['step']}"))
            else:
                # Alpha and beta files for restarting unrestricted jobs
                shutil.copy(
                    os.path.join(workdir, "alpha"), os.path.join(qmin.save["savedir"], f"alpha.{jobid}.{qmin.save['step']}")
                )
                shutil.copy(
                    os.path.join(workdir, "beta"), os.path.join(qmin.save["savedir"], f"beta.{jobid}.{qmin.save['step']}")
                )
                # Concat alpha and beta file to mos file
                with open(os.path.join(qmin.save["savedir"], f"mos.{jobid}.{qmin.save['step']}"), "w", encoding="utf-8") as f:
                    with open(os.path.join(workdir, "alpha"), "r", encoding="utf-8") as alpha:
                        for line in alpha:
                            if "$end" in line:
                                continue
                            f.write(line)
                    with open(os.path.join(workdir, "beta"), "r", encoding="utf-8") as beta:
                        for _ in range(5):  # Skip header
                            next(beta)
                        for line in beta:
                            # Add total number of AO for wfoverlap not complaining
                            if nsao := re.findall(r"\s+(\d+).*nsaos\=(\d+)", line):
                                line = re.sub(r"\d+", str(int(nsao[0][0]) + int(nsao[0][1])), line, count=1)
                            f.write(line)

        else:
            grad = list(qmin.maps["gradmap"])[0]
            self._copy_from_master(
                workdir, master_dir=os.path.join(qmin.resources["scratchdir"], f"master_{grad[0] if grad[0] != 3 else 1}")
            )
            if grad[0] != 3 and grad[1] == 1:  # (n,1) groundstate, except (3,1) -> excited state
                self._modify_file(os.path.join(workdir, "control"), ["$response\n gradient\n"])
            else:
                exgrad = f"xgrad states=(a{{{grad[0] if grad[0] == 3 else 1}}} {grad[1] - (grad[0] != 3)})"
                self._modify_file(os.path.join(workdir, "control"), None, None, [("$excitations", f"{exgrad}\n")])
            codes.append(self._run_ricc2(workdir, qmin.resources["ncpu"]))
        return max(codes), datetime.datetime.now() - starttime

    def _get_dets(self, workdir: str, mult: int, side: str = "R") -> list[dict[tuple[int, int], float]]:
        """
        Parse determinants from CC* binary files
        """
        # Get nAO, nOCC and frozens from control
        n_mo, n_occ, n_froz = 0, [0, 0], 0
        restr = False
        with open(os.path.join(workdir, "control"), "r", encoding="utf-8") as f:
            while line := f.readline():
                if "nbf(AO)" in line:
                    n_mo = int(line.split("=")[-1])
                if "$closed shells" in line:
                    n_occ[0] = int(f.readline().split()[1].split("-")[-1])
                    restr = True
                if "$alpha shells" in line:
                    n_occ[0] = int(f.readline().split()[1].split("-")[-1])
                if "$beta shells" in line:
                    n_occ[1] = int(f.readline().split()[1].split("-")[-1])
                if "implicit core" in line:
                    n_froz = int(line.split()[2])
        n_virt = [n_mo - n_occ[0], (n_mo if not restr else 0) - n_occ[1]]

        self.log.debug(f"Found nMO {n_mo} nOCC {n_occ} nFROZ {n_froz} and nVIRT {n_virt} in control file.")

        occ_str = [3 if restr else 1] * n_occ[0] + [0] * n_virt[0] + [2] * n_occ[1] + [0] * n_virt[1]

        # Parse CC files
        eigenvectors = [{tuple(occ_str): 1.0}] if mult != 3 else []
        for state in range(1, self.QMin.molecule["states"][mult - 1] + (1 if mult == 3 else 0)):
            with open(
                fname := os.path.join(workdir, f"CC{side}E0-1--{mult if mult == 3 else 1}{state:4d}".replace(" ", "-")), "rb"
            ) as f:
                self.log.debug(f"Parsing file {fname}")
                self._read_value(f, "u", 1)  # Skip header
                coeffs = self._read_value(f, "f")
                self.log.debug(f"Parsed {len(coeffs)} values")
                tmp = {}
                it_coeffs = iter(coeffs)
                for occ in range(n_froz, n_occ[0]):
                    for virt in range(n_occ[0], n_mo):
                        key = occ_str[:]
                        key[occ], key[virt] = (1, 2) if restr else (0, 1)
                        val = next(it_coeffs) * (np.sqrt(0.5) if restr else 1.0)
                        tmp[tuple(key)] = val
                        if restr:
                            key[occ], key[virt] = 2, 1
                            tmp[tuple(key)] = val if mult == 3 else -val

                # Unrestricted
                if not restr:
                    coeffs = self._read_value(f, "f")
                    self.log.debug(f"Parsed {len(coeffs)} beta values")
                    it_coeffs = iter(coeffs)

                for occ in range(n_mo + n_froz, n_mo + n_occ[1]):
                    for virt in range(n_mo + n_occ[1], n_mo * 2):
                        key = occ_str[:]
                        key[occ], key[virt] = 0, 2
                        tmp[tuple(key)] = next(it_coeffs)

                norm = np.linalg.norm(list(tmp.values()))
                self.log.debug(f"Determinant {mult} {state} norm {norm:.5f}")

                # Renormalize (needed for SOCs)
                tmp = {k: v / norm for k, v in tmp.items()}

                # Filter out dets with lowest contribution
                self.trim_civecs(tmp)
                eigenvectors.append(tmp)
        return eigenvectors

    def _generate_molden(self, workdir: str) -> None:
        """
        Generate molden file, copy to savedir if requested
        """
        # Write tm2molden input file
        writefile(os.path.join(workdir, "tm2molden.input"), "molden.input\ny\n")

        # Execute tm2molden
        if code := self.run_program(workdir, "tm2molden < tm2molden.input", "tm2molden.out", "tm2molden.err") != 0:
            self.log.error(f"tm2molden failed with exit code {code}")
            raise RuntimeError()

        # Copy to savedir
        if self.QMin.requests["molden"]:
            shutil.copy(
                os.path.join(workdir, "molden.input"),
                os.path.join(
                    self.QMin.save["savedir"], f"molden.{os.path.split(workdir)[-1].split('_')[-1]}.{self.QMin.save['step']}"
                ),
            )

    def _get_aoovl(self, dyson: bool = False) -> None:
        """
        Calculate overlap matrix between previous and current geom
        """
        # Load molden file
        mol2, _, _, _, _, _ = tools.molden.load(
            os.path.join(self.QMin.resources["scratchdir"], f"master_{self.QMin.control['joblist'][0]}/molden.input")
        )

        # Create Mole object, assign basis from molden
        mol = gto.Mole()
        mol.basis = {}
        for atom, basis in mol2._basis.items():
            mol.basis[re.split(r"\d+", atom)[0]] = basis

        # Take prev geom and append current
        if not dyson:
            with open(
                os.path.join(self.QMin.save["savedir"], f"input.xyz.{self.QMin.save['step']-1}"), "r", encoding="utf-8"
            ) as f:
                mol.atom = [line.strip() for line in f.readlines()[2:]]
        mol.atom.extend([[e, c] for e, c in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"].tolist())])
        mol.cart = False
        mol.unit = "Bohr"
        mol.build()
        n_ao = mol.nao if dyson else (mol.nao // 2)
        # Sort functions from pySCF to Turbomole order
        self._ao_labels = [i.split()[::2] for i in mol.ao_labels()[:n_ao]]
        self.log.debug(f"PySCF AO order:\n{self._print_ao(self._ao_labels)}")
        ovlp = mol.intor("int1e_ovlp")
        if not dyson:
            ovlp = ovlp[:n_ao, n_ao:]
        order = sorted(list(range(n_ao)), key=cmp_to_key(self._sort_ao))
        for i, l in enumerate(self._ao_labels):  # Some functions have opposite sign
            if "f-3" in l[1] or "g+2" in l[1] or "g-3" in l[1]:
                ovlp[i, :] *= -1
                ovlp[:, i] *= -1
        self.log.debug(f"TURBOMOLE AO order:\n{self._print_ao([self._ao_labels[i] for i in order])}")
        ovlp = ovlp[np.ix_(order, order)]
        ovlp_str = f"{n_ao} {n_ao}\n"
        ovlp_str += "\n".join(" ".join(f"{elem: .15e}" for elem in row) for row in ovlp)
        writefile(os.path.join(self.QMin.save["savedir"], f"AO_overl{'.mixed' if not dyson else ''}"), ovlp_str)

    def _sort_ao(self, ao1: int, ao2: int) -> int:
        """
        PySCF AO label order to NWChem AO label order
        """
        first = self._ao_labels[ao1]
        second = self._ao_labels[ao2]

        label_to_int = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5}
        angular_to_int = {
            "s": 1,
            "px": 1,
            "py": 1,
            "pz": 1,
            "dz^2": 1,
            "dxz": 2,
            "dyz": 3,
            "dx2-y2": 5,
            "dxy": 4,
            "f+0": 1,
            "f+1": 2,
            "f-1": 3,
            "f+2": 5,
            "f-2": 4,
            "f+3": 6,
            "f-3": 7,
            "g+0": 1,
            "g+1": 2,
            "g-1": 3,
            "g-2": 4,
            "g+2": 5,
            "g+3": 6,
            "g-3": 7,
            "g-4": 8,
            "g+4": 9,
        }
        return (
            (int(first[0]) - int(second[0])) * 1000  # Atom
            + (int(first[1][0]) - int(second[1][0])) * 10  # Shell
            + (label_to_int[first[1][1]] - label_to_int[second[1][1]]) * 100  # Function
            + (angular_to_int[first[1][1:]] - angular_to_int[second[1][1:]]) * 1  # Angular
        )

    def _print_ao(self, ao_labels: list[list[str]]) -> str:
        """
        Make PySCF AO label string better readable

        ao_labels:  List of AO labels
        """
        ao_str = ""
        for label in ao_labels:
            ao_str += f"{self.QMin.molecule['elements'][int(label[0])]:4s} {label[1]:10s}\n"
        return ao_str

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

    @staticmethod
    def _setup_env(ncpu: int) -> dict[str, str]:
        """
        Generate an env dict for given number of CPUs
        """
        ricc2_env = deepcopy(os.environ)
        if hasattr(ricc2_env, "PARA_ARCH"):
            del ricc2_env["PARA_ARCH"]
        if hasattr(ricc2_env, "PARANODES"):
            del ricc2_env["PARANODES"]
        if ncpu > 1:
            ricc2_env["PARA_ARCH"] = "SMP"
            ricc2_env["PARANODES"] = str(ncpu)
        ricc2_env["OMP_NUM_THREADS"] = str(ncpu)
        return ricc2_env

    def _run_ricc2(self, workdir: str, ncpu: int = 1) -> int:
        """
        Run RICC2 module
        """
        shift_mask = iter([(+1, +1), (-2, -1), (+1, +1), (+1, +1), (+1, +1), (+1, +1)])

        # Do first ricc2 call
        ricc2_bin = "ricc2" if ncpu < 2 else "ricc2_omp"
        if (code := self.run_program(workdir, ricc2_bin, "ricc2.out", "ricc2.err")) != 0:
            return code

        # Check if converged
        while not self._check_ricc2(workdir):
            self.log.debug(f"Not converged {workdir}")
            # Try again with different parameters
            try:
                pre_shift, start_shift = next(shift_mask)
            except StopIteration as exc:
                self.log.error("No convergence in RICC2")
                raise ValueError() from exc

            excitations = []
            if (nst := self.QMin.molecule["states"][0] - 1) > 0:
                excitations.append(
                    ("$excitations", f"irrep=a multiplicity=1 nexc={nst} npre={nst+1+pre_shift}, nstart={nst+1+start_shift}\n")
                )
            if len(self.QMin.molecule["states"]) > 2 and (nst := self.QMin.molecule["states"][2]) > 0:
                excitations.append(
                    ("$excitations", f"irrep=a multiplicity=3 nexc={nst} npre={nst+1+pre_shift}, nstart={nst+1+start_shift}\n")
                )
            self._modify_file(os.path.join(workdir, "control"), None, ["irrep=a"], excitations)
            # Run RICC2 again
            if (code := self.run_program(workdir, ricc2_bin, "ricc2.out", "ricc2.err", self._setup_env(ncpu))) != 0:
                return code
        return code

    def _check_ricc2(self, workdir: str) -> bool:
        """
        Check if ricc2 calculation converged
        """
        with open(os.path.join(workdir, "ricc2.out"), "r", encoding="utf-8") as f:
            while line := f.readline():
                if "NO CONVERGENCE" in line:
                    return False
        return True

    def _run_orca_soc(self, workdir: str) -> int:
        """
        Convert soc.mkl to ORCA format and run orca_soc
        """
        # convert mkl to gbw
        if (code := self.run_program(workdir, "orca_2mkl soc -gbw", "orca_2mkl.out", "orca_2mkl.err")) != 0:
            return code
        self.log.debug(f"orca_2mkl exited with code {code}")

        # write orca_soc input, execute orca_soc
        orca_soc = "soc.gbw\nsoc.psoc\nsoc.soc\n3\n1 2 3 0 4 0 0 4\n0\n"
        writefile(os.path.join(workdir, "soc.socinp"), orca_soc)
        return self.run_program(workdir, "orca_soc soc.socinp -gbw", "orca_soc.out", "orca_soc.err")

    def _run_ridft(self, workdir: str, qmin: QMin) -> int:
        """
        Modify control, run ridft, revert control
        """
        self._modify_file(
            os.path.join(workdir, "control"),
            [
                f"$maxcor {(mem := qmin.resources['memory'])*0.6}\n$ricore {mem*0.4}\n",
                "$jkbas file=auxbasis\n",
                "$rij\n",
                "$rik\n",
            ],
            ["$maxcor"],
        )
        code = self.run_program(
            workdir,
            "ridft_omp" if (ncpu := qmin.resources["ncpu"]) > 1 else "ridft",
            "ridft.out",
            "ridft.err",
            self._setup_env(ncpu),
        )
        shutil.copy(os.path.join(workdir, "control.bak"), os.path.join(workdir, "control"))
        return code

    def _copy_from_master(self, workdir: str, master_dir) -> None:
        """
        Copy run files from master job
        """
        self.log.debug("Copy run files from master job")
        files = ("auxbasis", "basis", "mos", "alpha", "beta", "restart.cc")
        for f in files:
            try:
                shutil.copy(os.path.join(master_dir, f), os.path.join(workdir, f))
            except FileNotFoundError as e:
                if f not in ("mos", "alpha", "beta"):
                    self.log.error(f"File {f} not found in {master_dir}")
                    raise FileNotFoundError() from e
        shutil.copy(os.path.join(master_dir, "control.bak"), os.path.join(workdir, "control"))

    def getQMout(self) -> dict[str, np.ndarray]:

        states = self.QMin.molecule["states"]
        scratchdir = self.QMin.resources["scratchdir"]

        # Allocate matrices
        requests = set()
        for key, val in self.QMin.requests.items():
            if not val:
                continue
            requests.add(key)

        self.log.debug("Allocate space in QMout object")
        self.QMout.allocate(
            states=states,
            natom=self.QMin.molecule["natom"],
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )

        # Prepare theodore properties
        if self.QMin.requests["theodore"]:
            nprop = len(self.QMin.resources["theodore_prop"]) + (nfrag := len(self.QMin.resources["theodore_fragment"])) ** 2
            labels = self.QMin.resources["theodore_prop"][:] + [f"Om_{i}_{j}" for i in range(nfrag) for j in range(nfrag)]
            theodore_arr = [[labels[j], np.zeros(self.QMin.molecule["nmstates"])] for j in range(nprop)]

        # Open master output file
        for mult, job_dict in self.QMin.control["jobs"].items():
            with open(os.path.join(scratchdir, f"master_{mult}/ricc2.out"), "r", encoding="utf-8") as f:
                ricc2_out = f.read()

            # Get D1 diagnostics
            if diagnostic := re.findall(r"D1 diagnostic\s+:\s+(\d+\.\d+)", ricc2_out):
                self.log.debug(f"D1 job {mult}: {diagnostic[-1]}")
                self.QMout["notes"][f"D1 job {mult}"] = diagnostic[-1]
            else:
                self.log.warning(f"No D1 value for job {mult} found!")

            # Get SCF iterations
            with open(os.path.join(scratchdir, f"master_{mult}/dscf.out"), "r", encoding="utf-8") as f:
                for line in f:
                    if iterations := re.findall(r"(\d+)\s+iterations", line):
                        self.log.debug(f"Job {mult} converged after {iterations[0]} cycles.")
                        self.QMout["notes"][f"iterations job {mult}"] = iterations[0]
                        break
                else:
                    self.log.error(f"No iteration count found for job {mult}")

            # SOCs, only possible between S1-T
            if mult == 1 and self.QMin.requests["soc"]:
                socs = self._get_socs(ricc2_out)[: states[0] - 1, states[0] - 1 :, :]

                skip = states[0] + 2 * states[1]
                self.QMout["h"][1 : states[0], skip : skip + states[2]] = socs[:, :, 0]
                self.QMout["h"][1 : states[0], states[2] + skip : skip + (2 * states[2])] = socs[:, :, 1]
                self.QMout["h"][1 : states[0], (2 * states[2]) + skip : skip + (3 * states[2])] = socs[:, :, 2]
                self.QMout["h"] += self.QMout["h"].T

            # Energies
            if self.QMin.requests["h"]:
                energies, t2_err = self._get_energies(ricc2_out)
                self.log.debug(f"%T2 job {mult} {t2_err}")
                self.QMout["notes"][f"%T2 job {mult}"] = t2_err
                s_cnt = 0
                for i in self.QMin.control["jobs"][mult]["mults"]:
                    np.einsum("ii->i", self.QMout["h"])[
                        sum(s * m for m, s in enumerate(states[: i - 1], 1)) : sum(s * m for m, s in enumerate(states[:i], 1))
                    ] = np.tile(energies[s_cnt : s_cnt + states[i - 1]], i)
                    s_cnt += states[i - 1]

            # Theodore
            if self.QMin.requests["theodore"]:
                if self.QMin.control["jobs"][mult]["restr"]:
                    ns = 0
                    for i in job_dict["mults"]:
                        ns += states[i - 2] - (i == job_dict["mults"][0])
                    if ns != 0:
                        props = self.get_theodore(
                            os.path.join(scratchdir, f"master_{mult}", "tden_summ.txt"),
                            os.path.join(scratchdir, f"master_{mult}", "OmFrag.txt"),
                        )
                        for i in range(self.QMin.molecule["nmstates"]):
                            m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                            if (m1, s1) in props:
                                for j in range(
                                    len(self.QMin.resources["theodore_prop"]) + len(self.QMin.resources["theodore_fragment"]) ** 2
                                ):
                                    theodore_arr[j][1][i] = props[(m1, s1)][j]

        if self.QMin.requests["theodore"]:
            self.QMout["prop1d"].extend(theodore_arr)

        # Overlaps
        if self.QMin.requests["overlap"]:
            for mult in itmult(self.QMin.molecule["states"]):
                job = self.QMin.maps["multmap"][mult]
                ovlp_mat = self.parse_wfoverlap(os.path.join(scratchdir, f"WFOVL_{mult}_{job}", "wfovl.out"))
                for i in range(self.QMin.molecule["nmstates"]):
                    for j in range(self.QMin.molecule["nmstates"]):
                        m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                        m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                        if not m1 == m2 == mult:
                            continue
                        if not ms1 == ms2:
                            continue
                        self.QMout["overlap"][i, j] = ovlp_mat[s1 - 1, s2 - 1]
            # Phases
            if self.QMin.requests["phases"]:
                for i in range(self.QMin.molecule["nmstates"]):
                    self.QMout["phases"][i] = -1 if self.QMout["overlap"][i, i] < 0 else 1

        # Dyson norms
        if self.QMin.requests["ion"]:
            ion_mat = np.zeros((self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]))

            for ion in self.QMin.maps["ionmap"]:
                dyson_mat = self.get_dyson(os.path.join(scratchdir, f"Dyson_{'_'.join(str(i) for i in ion)}", "wfovl.out"))
                for i in range(self.QMin.molecule["nmstates"]):
                    for j in range(self.QMin.molecule["nmstates"]):
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
                        ion_mat[i, j] = dyson_mat[s1 - 1, s2 - 1] * factor
            self.QMout["prop2d"].append(("ion", ion_mat))

        # Gradients
        if self.QMin.requests["grad"]:
            for grad in self.QMin.maps["gradmap"]:
                with open(os.path.join(scratchdir, f"grad_{grad[0]}_{grad[1]}/ricc2.out"), "r", encoding="utf-8") as grad_file:
                    self.log.debug(f"Parsing gradient {grad[0]}_{grad[1]}")
                    grads = self._get_gradients(grad_file.read())
                    for key, val in self.QMin.maps["statemap"].items():
                        if (val[0], val[1]) == grad:
                            self.QMout["grad"][key - 1] = grads
                            if self.QMin.molecule["point_charges"]:
                                with open(
                                    os.path.join(scratchdir, f"grad_{grad[0]}_{grad[1]}/pc_grad"), "r", encoding="utf-8"
                                ) as pc:
                                    point_charges = pc.read()
                                    point_charges = point_charges.replace("D", "E")
                                    point_charges = point_charges.split("\n")[1:-2]
                                    point_charges = [c.split() for c in point_charges]
                                    self.QMout["grad_pc"][key - 1] = np.asarray(point_charges, dtype=float)

        return self.QMout

    def _get_gradients(self, ricc2_out: str) -> np.ndarray:
        """
        Parse forces from ASCII ricc2 output, reshape to atoms*xyz
        """
        if not (raw_gradients := re.findall(r"dE\/d[x|y|z](.*)", re.sub("D", "E", ricc2_out))):
            self.log.error("No gradients found in ricc2.out")
            raise ValueError()

        gradients = ["", "", ""]
        for i in range(len(raw_gradients) // 3):
            for j in range(3):
                gradients[j] += raw_gradients[j + i * 3]
        gradients = [re.findall(r"-?\d+\.\d{7}E[-|+]\d{2}", grad) for grad in gradients]
        return np.asarray(gradients, dtype=float).T

    def _get_socs(self, ricc2_out: str) -> np.ndarray:
        """
        Parse socs S1-T from ASCII ricc2 output
        """
        states = self.QMin.molecule["states"]
        # Get SOC xyz values (upper triangular arrangement)
        if not (raw_socs := re.findall(r"\s+soc\s+\|\s+-?\d+\.\d+(.*)", ricc2_out)):
            self.log.error("No SOC values found in ricc2.out.")
            raise ValueError()

        # Filter values -> states*xyz
        socs = [re.findall(r"-?\d+\.\d{2}", i) for i in raw_socs]

        # Create soc matrix -> states*states*xyz, populate upper triangle with socs
        soc_mat = np.zeros((states[0] - 1 + states[2], states[0] - 1 + states[2], 3), dtype=float)
        idx = np.triu_indices(states[0] - 1 + states[2], 1)
        soc_mat[idx] = np.asarray(socs, dtype=float) * rcm_to_Eh
        return soc_mat

    def _get_energies(self, ricc2_out: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract energies and %t2 from ASCII ricc2 output
        """

        # Get GS energy
        if not (gs_energy := re.findall(r"Final \w+ energy\s+:\s+(-?\d+\.\d+)", ricc2_out)):
            self.log.error("Ground state energy not found in ricc2.out")
            raise ValueError()

        energies = ["0.0"]
        # Get energy table
        if not (
            raw_energies := re.findall(r"excitation energies\s+\|\s+\%t1\s+\|\s+\%t2(.*)?=\+\n\n\s+Energy", ricc2_out, re.DOTALL)
        ):
            # No excitation energies found
            return np.asarray([gs_energy[-1]], dtype=float), np.zeros(1)

        # Extract energy values
        energies += re.findall(r"-?\d+\.\d{7}", raw_energies[0])
        return (
            np.asarray(energies, dtype=np.complex128) + float(gs_energy[-1]),
            re.findall(r"(\d+\.\d{2})\s", raw_energies[0])[1::2],
        )

    def run(self) -> None:
        starttime = datetime.datetime.now()

        # Generate schedule
        self.log.debug("Generate schedule")
        self._generate_schedule()
        for idx, job in enumerate(self.QMin.scheduling["schedule"], 1):
            self.log.info(f"Schedule {idx}: {list(job.keys())}")

        self.log.debug("Execute schedule")
        if not self.QMin.resources["dry_run"]:
            self.runjobs(self.QMin.scheduling["schedule"])

        # Generate dets
        for jobid, mults in self.QMin.control["jobs"].items():
            for mult in mults["mults"]:
                dets = self._get_dets(os.path.join(self.QMin.resources["scratchdir"], f"master_{jobid}"), mult)
                writefile(
                    os.path.join(self.QMin.save["savedir"], f"dets.{mult}.{self.QMin.save['step']}"), self.format_ci_vectors(dets)
                )
                if self.QMin.template["method"] == "cc2":
                    dets = self._get_dets(os.path.join(self.QMin.resources["scratchdir"], f"master_{jobid}"), mult, "L")
                    writefile(
                        os.path.join(self.QMin.save["savedir"], f"dets_left.{mult}.{self.QMin.save['step']}"),
                        self.format_ci_vectors(dets),
                    )

        # Save copy of current geometry
        xyz_str = f"{self.QMin.molecule['natom']}\n\n"
        for label, coords in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"]):
            xyz_str += f"{label:4s} {coords[0]:16.9f} {coords[1]:16.9f} {coords[2]:16.9f}\n"
        writefile(os.path.join(self.QMin.save["savedir"], f"input.xyz.{self.QMin.save['step']}"), xyz_str)
        if self.QMin.requests["overlap"]:
            self._get_aoovl()
        if self.QMin.requests["ion"]:
            self._get_aoovl(True)
        self._run_wfoverlap(mo_read=2, left=self.QMin.template["method"] == "cc2")

        # Run theodore
        if self.QMin.requests["theodore"]:
            self._run_theodore()

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def _generate_schedule(self) -> None:
        """
        Generate schedule, one master job and n jobs for gradients
        """
        # Add master job and assign all available cores
        _, nslots, cpu_per_run = self.divide_slots(
            self.QMin.resources["ncpu"], len(self.QMin.control["jobs"]), self.QMin.resources["schedule_scaling"]
        )
        master = {}
        self.log.debug(f"Master jobs executed in {nslots} slots.")
        for idx, job in enumerate(self.QMin.control["jobs"]):
            master_job = deepcopy(self.QMin)
            master_job.resources["ncpu"] = cpu_per_run[idx]
            master_job.control["master"] = True
            master_job.control["jobid"] = job
            master_job.control["states_to_do"][job - 1] -= 1
            if job != 1:
                master_job.requests["soc"] = False
            master[f"master_{job}"] = master_job
            self.log.debug(f"Job master_{job} CPU: {cpu_per_run[idx]}")
        self.QMin.control["nslots_pool"].append(nslots)
        schedule = [master]

        # Add gradient jobs
        gradjobs = {}
        if self.QMin.requests["grad"]:
            # Distribute available cores
            _, nslots, cpu_per_run = self.divide_slots(
                self.QMin.resources["ncpu"], len(self.QMin.requests["grad"]), self.QMin.resources["schedule_scaling"]
            )
            self.log.debug(f"Gradient jobs executed in {nslots} slots.")
            self.QMin.control["nslots_pool"].append(nslots)
            # Add job for each gradient
            for idx, grad in enumerate(self.QMin.maps["gradmap"]):
                job = deepcopy(self.QMin)
                job.resources["ncpu"] = cpu_per_run[idx]
                job.maps["gradmap"] = {(grad)}
                job.control["gradonly"] = True
                gradjobs[f"grad_{'_'.join(str(g) for g in grad)}"] = job
                self.log.debug(f"Job grad_{'_'.join(str(g) for g in grad)} CPU: {cpu_per_run[idx]}")
            schedule.append(gradjobs)

        self.QMin.scheduling["schedule"] = schedule

    def _writegeom(self, workdir: str) -> None:
        """
        Write xyz file
        """
        # Save coords in turbomole format
        self.log.debug("Write geom file")
        xyz_str = f"$coord  natoms={self.QMin.molecule['natom']:>6d}\n"
        for label, coords in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"]):
            xyz_str += f"{coords[0]:>20.14f} {coords[1]:>21.14f} {coords[2]:>21.14f}      {label.lower()}\n"
        xyz_str += "$user-defined bonds\n$end\n"
        writefile(os.path.join(workdir, "coord"), xyz_str)

        # Write point charges
        if self.QMin.molecule["point_charges"]:
            self.log.debug("Write point charge file")
            pc_str = "$point_charges nocheck\n"
            for charge, coord in zip(self.QMin.coords["pccharge"], self.QMin.coords["pccoords"]):
                pc_str += f"{coord[0]/au2a:16.12f} {coord[1]/au2a:16.12f} {coord[2]/au2a:16.12f} {charge:12.9f}\n"
            pc_str += "$end\n"
            writefile(os.path.join(workdir, "pc"), pc_str)

    def _run_define(self, workdir: str, mult: int, cc2: bool = True) -> int:
        """
        Run define command
        """
        self.log.debug("Run define")
        # Generate input string
        if baslib := self.QMin.template["basislib"]:
            writefile(os.path.join(workdir, ".definerc"), f"basis={baslib}/basen\nbasis={baslib}/cbasen\n")

        string = "\ntitle: SHARC-TURBOMOLE run\na coord\n*\nno\n"
        if self.QMin.template["basislib"]:
            string += "lib\n3"
        string += f"b\nall {self.QMin.template['basis']}\n*\neht\ny\n{self.QMin.molecule['charge'][mult-1]}\n"
        if mult - 1 % 2 == 0:
            string += "y\n"
        else:
            string += f"n\nu {mult-1}\n*\nn\n"

        if cc2:
            match (frozen := self.QMin.template["frozen"]):
                case 0:
                    string += "cc\n"
                case frozen if frozen < 0:
                    string += "cc\nfreeze\n*\n"
                case frozen if frozen > 0:
                    string += f"cc\nfreeze\ncore {frozen}\n*\n"

            if not self.QMin.template["auxbasis"] and self.QMin.template["basislib"]:
                string += "cbas\n"
                string += "\n\n" * len(set(self.QMin.molecule["elements"]))
                string += f"lib\n4\nb\nall {self.QMin.template['basis']}\n*\n"
            elif not self.QMin.template["auxbasis"]:
                string += "cbas\n*\n"
            else:
                string += f"cbas\nb\nall {self.QMin.template['auxbasis']}\n*\n"

            string += f"memory {self.QMin.resources['memory']}\nricc2\n{self.QMin.template['method']}\n"
            string += f"mxdiis = {max(10, 5*max(self.QMin.molecule['states']))}\n*\n*\n"
        string += "*\n"

        # Execute define
        writefile(os.path.join(workdir, "define.input"), string)
        if (code := self.run_program(workdir, "define < define.input", "define.output", "define.err")) != 0:
            self.log.error(f"define failed with code {code}")
            with open(os.path.join(workdir, "define.err"), "r", encoding="utf-8") as f:
                self.log.error(f.read())
            raise RuntimeError()
        return code

    def _modify_file(
        self,
        file_name: str,
        add_lines: list[str] | None = None,
        remove_lines: list[str] | None = None,
        add_sec: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Read content of file, add/remove things and overwrite file

        file_name:      Path to file
        add_lines:      Lines to add to file
        remove_lines:   Lines to be removed from file
        add_sec:        Add line to section
        """

        # Read content of file
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines.pop()  # Remove $end

        # Remove lines
        if remove_lines:
            lines = [line for line in lines if line.split()[0] not in remove_lines]

        # Add lines
        if add_lines:
            lines += add_lines

        # Add to section
        if add_sec:
            for sec, insert in add_sec:
                for idx, line in enumerate(lines, 1):
                    if sec in line:
                        lines.insert(idx, insert)
                        break

        writefile(file_name, lines + ["$end"])


if __name__ == "__main__":
    SHARC_TURBOMOLE().main()
