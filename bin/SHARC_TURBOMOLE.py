#!/usr/bin/env python3
import datetime
import os
import re
import shutil
import subprocess as sp
from copy import deepcopy
from io import TextIOWrapper

import numpy as np
from constants import NUMBERS, rcm_to_Eh
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from SHARC_ORCA import SHARC_ORCA
from utils import expand_path, mkdir, writefile

__all__ = ["SHARC_TURBOMOLE"]

AUTHORS = "Sascha Mausenberger, Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 5, 29)
NAME = "TURBOMOLE"
DESCRIPTION = "SHARC interface for TURBOMOLE (RICC2/ADC2)"

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

        # TODO: grad jobs need different values
        if (ncpu := self.QMin.resources["ncpu"]) > 1:
            os.environ["PARA_ARCH"] = "SMP"
            os.environ["PARANODES"] = str(ncpu)
        os.environ["OMP_NUM_THREADS"] = str(ncpu)

        arch = (
            sp.Popen([os.path.join(self.QMin.resources["turbodir"], "scripts", "sysname")], stdout=sp.PIPE)
            .communicate()[0]
            .decode()
            .strip()
        )
        os.environ["PATH"] = f"{turbodir}/scripts:{turbodir}/bin/{arch}:" + os.environ["PATH"]

    def setup_interface(self) -> None:
        super().setup_interface()

        if self.QMin.resources["ncpu"] > 1 and self.QMin.template["spin-scaling"] == "lt-sos":
            self.log.warning("lt-sos is not fully SMP parallelized.")

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

    def _create_aoovl(self) -> None:
        pass

    def create_restart_files(self) -> None:
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
            codes.append(self._run_define(workdir))

            # Modify control file
            add_lines = ["$excitations\n"]
            add_section = [("$ricc2", "maxiter 45\n")]
            if self.QMin.requests["soc"]:
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
            if (nst := qmin.molecule["states"][0] - 1) > 0:
                add_section.append(("$excitations", f"irrep=a multiplicity=1 nexc={nst} npre={nst+1}, nstart={nst+1}\n"))
            if len(qmin.molecule["states"]) > 2 and (nst := qmin.molecule["states"][2]) > 0:
                add_section.append(("$excitations", f"irrep=a multiplicity=3 nexc={nst} npre={nst+1}, nstart={nst+1}\n"))

            self._modify_file((control := os.path.join(workdir, "control")), add_lines, ["$scfiterlimit"], add_section)

            # Run dscf/ridft, backup control file
            shutil.copy(control, os.path.join(workdir, "control.bak"))

            # Copy initial guess
            match qmin.save:
                case {"always_guess": True}:
                    pass
                case {"always_orb_init": True} | {"init": True}:
                    try:
                        shutil.copy(os.path.join(qmin.resources["pwd"], "mos.init"), os.path.join(workdir, "mos"))
                    except FileNotFoundError as exc:
                        if qmin.save["always_orb_init"]:
                            self.log.error("mos.init file not found!")
                            raise exc
                case {"samestep": True}:
                    shutil.copy(os.path.join(qmin.save["savedir"], f"mos.{qmin.save['step']}"), os.path.join(workdir, "mos"))
                case _:
                    shutil.copy(os.path.join(qmin.save["savedir"], f"mos.{qmin.save['step']-1}"), os.path.join(workdir, "mos"))

            if qmin.template["scf"] == "ridft":
                codes.append(self._run_ridft(workdir, qmin))
                self.log.debug(f"ridft exited with code {codes[-1]}")
            codes.append(self.run_program(workdir, "dscf", "dscf.out", "dscf.err"))
            self.log.debug(f"dscf exited with code {codes[-1]}")

            # TODO: molden (theodore)

            # TODO: e, socs, dm
            if qmin.requests["soc"]:
                self._modify_file(control, None, None, [("$excitations", "tmexc istates=all fstates=all operators=soc\n")])

            codes.append(self._run_ricc2(workdir))

            # Run orca_soc if soc requested
            if qmin.requests["soc"]:
                codes.append(self._run_orca_soc(workdir))
                self.log.debug(f"orca_mkl/orca_soc exited with code {codes[-1]}")

            # Save files
            shutil.copy(os.path.join(workdir, "mos"), os.path.join(qmin.save["savedir"], f"mos.{qmin.save['step']}"))

        else:
            self._copy_from_master(workdir)
            if (grad := list(qmin.maps["gradmap"])[0]) == (1, 1):
                self._modify_file(os.path.join(workdir, "control"), ["$response\n gradient\n"])
            else:
                exgrad = f"xgrad states=(a{{{grad[0]}}} {grad[1] - (grad[0] == 1)})"
                self._modify_file(os.path.join(workdir, "control"), None, None, [("$excitations", f"{exgrad}\n")])
            codes.append(self._run_ricc2(workdir))
        return max(codes), datetime.datetime.now() - starttime

    def _run_ricc2(self, workdir: str) -> int:
        """
        Run RICC2 module
        """
        shift_mask = iter([(+1, +1), (-2, -1), (+1, +1), (+1, +1), (+1, +1), (+1, +1)])

        # Do first ricc2 call
        ricc2_bin = "ricc2" if self.QMin.resources["ncpu"] < 2 else "ricc2_omp"
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
            self._modify_file(os.path.join(workdir, "control"), None, ["irrep"], excitations)
            # Run RICC2 again
            if (code := self.run_program(workdir, ricc2_bin, "ricc2.out", "ricc2.err")) != 0:
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
        code = self.run_program(workdir, "ridft_omp" if qmin.resources["ncpu"] > 1 else "ridft", "ridft.out", "ridft.err")
        shutil.copy(os.path.join(workdir, "control.bak"), os.path.join(workdir, "control"))
        return code

    def _copy_from_master(self, workdir: str) -> None:
        """
        Copy run files from master job
        """
        self.log.debug("Copy run files from master job")
        master_dir = os.path.join(self.QMin.resources["scratchdir"], "master")
        files = ("auxbasis", "basis", "mos", "restart.cc")
        for f in files:
            try:
                shutil.copy(os.path.join(master_dir, f), os.path.join(workdir, f))
            except Exception as e:
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

        # Open master output file
        with open(os.path.join(scratchdir, "master/ricc2.out"), "r", encoding="utf-8") as f:
            ricc2_out = f.read()

        # SOCs, only possible between S1-T
        if self.QMin.requests["soc"]:
            # TODO: cleanup
            socs = self._get_socs(ricc2_out)[: states[0] - 1, states[0] - 1 :, :]

            self.QMout["h"][1 : states[0], states[0] : states[0] + states[2]] = socs[:, :, 0]
            self.QMout["h"][1 : states[0], states[2] + states[0] : states[0] + (2 * states[2])] = socs[:, :, 1]
            self.QMout["h"][1 : states[0], (2 * states[2]) + states[0] : states[0] + (3 * states[2])] = socs[:, :, 2]
            self.QMout["h"] += self.QMout["h"].T

        # Energies
        if self.QMin.requests["h"]:
            np.einsum("ii->i", self.QMout["h"])[:] = self._get_energies(ricc2_out)

        # Gradients
        if self.QMin.requests["grad"]:
            for grad in self.QMin.maps["gradmap"]:
                with open(os.path.join(scratchdir, f"grad_{grad[0]}_{grad[1]}/ricc2.out"), "r", encoding="utf-8") as grad_file:
                    grads = self._get_gradients(grad_file.read())
                    for key, val in self.QMin.maps["statemap"].items():
                        if (val[0], val[1]) == grad:
                            self.QMout["grad"][key - 1] = grads

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

    def _get_energies(self, ricc2_out: str) -> np.ndarray:
        """
        Extract energies from ASCII ricc2 output
        """

        # Get GS energy
        if not (gs_energy := re.findall(r"Final \w+ energy\s+:\s+(-?\d+\.\d+)", ricc2_out)):
            self.log.error("Ground state energy not found in ricc2.out")
            raise ValueError()

        energies = ["0.0"]
        # Get energy table
        if sum(self.QMin.molecule["states"]) > 1:
            if not (
                raw_energies := re.findall(
                    r"excitation energies\s+\|\s+\%t1\s+\|\s+\%t2(.*)?=\+\n\n\s+Energy", ricc2_out, re.DOTALL
                )
            ):
                self.log.error("No energies found in ricc2.out")
                raise ValueError()

            # Extract energy values
            energies += re.findall(r"\d+\.\d{7}", raw_energies[0])

        # Expand by multiplicity
        expandend_energies = []
        for i in range(len((states := self.QMin.molecule["states"]))):
            expandend_energies += energies[sum(states[:i]) : sum(states[: i + 1])] * (i + 1)
        return np.asarray(expandend_energies, dtype=np.complex128) + float(gs_energy[0])

    def run(self) -> None:
        starttime = datetime.datetime.now()

        # Generate schedule
        self.log.debug("Generate schedule")
        self._generate_schedule()

        self.log.debug("Execute schedule")
        if not self.QMin.resources["dry_run"]:
            self.runjobs(self.QMin.scheduling["schedule"])

            # Copy mos to save
            shutil.copy(
                os.path.join(self.QMin.resources["scratchdir"], "master/mos"),
                os.path.join(self.QMin.save["savedir"], f"mos.{self.QMin.save['step']}"),
            )

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def _generate_schedule(self) -> None:
        """
        Generate schedule, one master job and n jobs for gradients
        """
        # Add master job and assign all available cores
        schedule = [{"master": deepcopy(self.QMin)}]
        self.QMin.control["nslots_pool"].append(1)
        schedule[0]["master"].control["master"] = True

        # Add gradient jobs
        gradjobs = {}
        if self.QMin.requests["grad"]:
            # Distribute available cores
            _, nslots, cpu_per_run = self.divide_slots(
                self.QMin.resources["ncpu"], len(self.QMin.requests["grad"]), self.QMin.resources["schedule_scaling"]
            )
            self.QMin.control["nslots_pool"].append(nslots)
            self.log.debug(f"Schedule with {nslots} slots, cpu distribution {cpu_per_run}")
            # Add job for each gradient
            for idx, grad in enumerate(self.QMin.maps["gradmap"]):
                job = deepcopy(self.QMin)
                job.resources["ncpu"] = cpu_per_run[idx]
                job.maps["gradmap"] = {(grad)}
                gradjobs[f"grad_{'_'.join(str(g) for g in grad)}"] = job
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
                coord += self.QMin.molecule["factor"]
                pc_str += f"{coord[0]:16.12f} {coord[1]:16.12f} {coord[2]:16.12f} {charge:12.9f}\n"
            pc_str += "$end\n"
            writefile(os.path.join(workdir, "pc"), pc_str)

    def _run_define(self, workdir: str, cc2: bool = True) -> int:
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
        string += f"b\nall {self.QMin.template['basis']}\n*\neht\ny\n{self.QMin.template['charge'][0]}\ny\n"

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
