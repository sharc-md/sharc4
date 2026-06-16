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
import json
import math
import os
import re
import shutil
import struct
import subprocess as sp
from copy import deepcopy
from functools import cmp_to_key
from io import TextIOWrapper
from itertools import chain, count

import numpy as np
from constants import IToMult, au2a
from pyscf import gto, tools
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import batched, convert_list, expand_path, itmult, link, mkdir, question, readfile, writefile

__all__ = ["SHARC_ORCA"]

AUTHORS = "Sascha Mausenberger and Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "ORCA"
DESCRIPTION = "AB INITIO interface for ORCA v5-6 (CIS/TDDFT)"

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
        "molden",
        "point_charges",
        "multipolar_fit",
        "density_matrices",
        # "grad_pc",
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
    _theodore_settings = {
        "rtype": "orca",
        "rfile": "ORCA.log",
        "mo_file": "ORCA.molden.input",
        "read_binary": True,
        "jmol_orbitals": False,
        "molden_orbitals": False,
        "Om_formula": 2,
        "eh_pop": 1,
        "comp_ntos": True,
        "print_OmFrag": True,
        "output_file": "tden_summ.txt",
        "link_files": [("ORCA.cis", "orca.cis")],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.template_file = None
        self.resources_file = None
        self.order = None
        self.ao_labels = None
        self.ao_sign = None

        # Add resource keys
        self.QMin.resources.update(
            {"orcadir": None, "orcaversion": None, "numfrozcore": -1, "schedule_scaling": 0.9, "dry_run": False}
        )
        self.QMin.resources.types.update(
            {"orcadir": str, "orcaversion": tuple, "numfrozcore": int, "schedule_scaling": float, "dry_run": bool}
        )

        # Add template keys
        self.QMin.template.update(
            {
                "no_tda": False,
                "basis": "def2-SVP",
                "auxbasis": None,
                "functional": "PBE",
                "dispersion": None,
                "ri": None,
                "keys": None,
                "paste_input_file": None,
                # "frozen": -1,           # TODO: currently has no effect
                "maxiter": 700,
                "hfexchange": -1.0,
                "intacc": -1.0,
                "unrestricted_triplets": False,
                "neglected_gradient": "zero",
                "basis_per_element": None,
                "basis_per_atom": None,
                "ecp_per_element": None,
                "alltrans": None,
            }
        )
        self.QMin.template.types.update(
            {
                "no_tda": bool,
                "basis": str,
                "auxbasis": str,
                "functional": str,
                "dispersion": str,
                "ri": str,
                "keys": str,
                "paste_input_file": str,
                # "frozen": int,
                "maxiter": int,
                "hfexchange": float,
                "intacc": float,
                "unrestricted_triplets": bool,
                "neglected_gradient": str,
                "basis_per_element": list,
                "basis_per_atom": list,
                "ecp_per_element": list,
                "alltrans": bool,
            }
        )

        # List of deprecated keys
        self._deprecated = ["range_sep_settings", "grid", "gridx", "gridxc", "picture_change", "qmmm"]

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
        self.log.info(f"||{'ORCA interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.log.info(
            "\nPlease specify path to ORCA directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n"
        )
        self.setupINFOS["orcadir"] = question("Path to ORCA:", str, KEYSTROKES=KEYSTROKES)
        self.log.info("")

        # scratch
        self.log.info(f"{'Scratch directory':-^60}\n")
        self.log.info(
            "Please specify an appropriate scratch directory. This will be used to run the ORCA calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script."
        )
        self.setupINFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
        self.setupINFOS["scratchdir"] += "/$$/"
        self.log.info("")

        self.log.info(f"{'ORCA input template file':-^60}\n")

        if os.path.isfile("ORCA.template"):
            usethisone = question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                self.template_file = "ORCA.template"
        else:
            while True:
                self.template_file = question("Template filename:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(self.template_file):
                    self.log.info(f"File {self.template_file} does not exist!")
                    continue
                break
        self.log.info("")

        # Resources
        if question("Do you have a 'ORCA.resources' file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            while not os.path.isfile(
                (resources_file := question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="ORCA.resources"))
            ):
                self.log.info(f"file at {resources_file} does not exist!")
            self.resources_file = resources_file
        else:
            self.log.info(f"{'ORCA Ressource usage':-^60}\n")
            self.log.info("""Please specify the number of CPUs to be used by EACH calculation.
        """)
            self.setupINFOS["ncpu"] = abs(question("Number of CPUs:", int, default=[1], KEYSTROKES=KEYSTROKES)[0])

            if self.setupINFOS["ncpu"] > 1:
                self.log.info("""Please specify how well your job will parallelize.
        A value of 0 means that running in parallel will not make the calculation faster, a value of 1 means that the speedup scales perfectly with the number of cores.
        Typical values for ORCA are 0.90-0.98.""")
                self.setupINFOS["scaling"] = max(
                    0.0, question("Parallel scaling:", float, default=[0.9], KEYSTROKES=KEYSTROKES)[0]
                )
            else:
                self.setupINFOS["scaling"] = 0.9

            self.setupINFOS["memory"] = question("Memory (MB):", int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

            if "overlap" in INFOS["needed_requests"]:
                self.log.info(f"\n{'WFoverlap setup':-^60}\n")
                self.setupINFOS["wfoverlap"] = question(
                    "Path to wavefunction overlap executable:", str, default="$SHARC/wfoverlap.x", KEYSTROKES=KEYSTROKES
                )
                self.log.info("")
                self.log.info("State threshold for choosing determinants to include in the overlaps")
                self.log.info("For hybrids without TDA one should consider that the eigenvector X may have a norm larger than 1")
                self.setupINFOS["wfthres"] = question("Threshold:", float, default=[0.998], KEYSTROKES=KEYSTROKES)[0]
                self.log.info("")

            # TheoDORE
            theodore_spelling = [
                "Om",
                "PRNTO",
                "Z_HE",
                "S_HE",
                "RMSeh",
                "POSi",
                "POSf",
                "POS",
                "PRi",
                "PRf",
                "PR",
                "PRh",
                "CT",
                "CT2",
                "CTnt",
                "MC",
                "LC",
                "MLCT",
                "LMCT",
                "LLCT",
                "DEL",
                "COH",
                "COHh",
            ]
            # INFOS['theodore']=question('TheoDORE analysis?',bool,False)
            if "theodore" in INFOS["needed_requests"]:
                self.log.info(f"\n{'Wave function analysis by TheoDORE':-^60}\n")

                self.setupINFOS["theodir"] = question(
                    "Path to TheoDORE directory:", str, default="$THEODIR", KEYSTROKES=KEYSTROKES
                )
                self.log.info("")

                self.log.info("Please give a list of the properties to calculate by TheoDORE.\nPossible properties:")
                string = ""
                for i, p in enumerate(theodore_spelling):
                    string += f"{p} "
                    if (i + 1) % 8 == 0:
                        string += "\n"
                self.log.info(string)
                line = question("TheoDORE properties:", str, default="Om  PRNTO  S_HE  Z_HE  RMSeh", KEYSTROKES=KEYSTROKES)
                self.setupINFOS["theodore_prop"] = line.split()
                self.log.info("")

                self.log.info("Please give a list of the fragments used for TheoDORE analysis.")
                self.log.info('Enter all atom numbers for one fragment in one line. After defining all fragments, type "end".')
                self.log.info("Atom numbering starts at 1 for TheoDORE.")
                self.setupINFOS["theodore_fragment"] = []
                while True:
                    line = question("TheoDORE fragment:", str, default="end", KEYSTROKES=KEYSTROKES)
                    if "end" in line.lower():
                        break
                    f = [int(i) for i in line.split()]
                    self.setupINFOS["theodore_fragment"].append(f)
                self.setupINFOS["theodore_count"] = (
                    len(self.setupINFOS["theodore_prop"]) + len(self.setupINFOS["theodore_fragment"]) ** 2
                )

            if "multipolar_fit" in INFOS["needed_requests"]:
                self.setupINFOS["resp"] = {}
                defaults = {
                    "resp_target": "zero",
                    "resp_layers": [4],
                    "resp_first_layer": [1.4],
                    "resp_density": [10.0],
                    "resp_fit_order": [2],
                    "resp_grid": "lebedev",
                    "resp_betas": [0.0005, 0.0015, 0.003],
                    "resp_mk_radii": False,  # use radii for original Merz-Kollmann-Singh scheme for HCNOSP
                    "resp_vdw_radii": [],
                    "resp_vdw_radii_symbol": {},
                    "resp_block_size": [5000],
                    "resp_nuke_ram": False,  # Old, memory heavy fitting
                }
                self.log.info('\n' + '{:-^60}'.format('Multipolar RESP fit setup') + '\n')
                self.log.info("Please set the options for the RESP multipolar fit.")
                # ---
                valid = ["zero", "mulliken", "lowdin"]
                while True:
                    answer = question("RESP restraint target:", str, default = defaults["resp_target"], autocomplete = False, KEYSTROKES=KEYSTROKES)
                    if answer in valid:
                        self.setupINFOS["resp"]["resp_target"] = answer
                        break
                    self.log.info("Must be 'zero', 'mulliken', or 'lowdin'!")
                # ---
                while True:
                    answer = question("RESP layer count:", int, default = defaults["resp_layers"], KEYSTROKES=KEYSTROKES)[0]
                    if answer >= 1:
                        self.setupINFOS["resp"]["resp_layers"] = answer
                        break
                    self.log.info("Must be positive!")
                # ---
                answer = question("RESP first layer:", float, default = defaults["resp_first_layer"], KEYSTROKES=KEYSTROKES)[0]
                self.setupINFOS["resp"]["resp_first_layer"] = answer
                # ---
                answer = question("RESP point density:", float, default = defaults["resp_density"], KEYSTROKES=KEYSTROKES)[0]
                self.setupINFOS["resp"]["resp_density"] = answer
                # ---
                while True:
                    answer = question("RESP fitting order:", int, default = defaults["resp_fit_order"], KEYSTROKES=KEYSTROKES)[0]
                    if 0<=answer<=2:
                        self.setupINFOS["resp"]["resp_fit_order"] = answer
                        break
                    self.log.info("Must be 0, 1, or 2!")
                # ---
                valid = ["lebedev", "random", "golden_spiral", "gamess", "marcus_deserno"]
                while True:
                    answer = question("RESP spherical grid:", str, default = defaults["resp_grid"], autocomplete = False, KEYSTROKES=KEYSTROKES)
                    if answer in valid:
                        self.setupINFOS["resp"]["resp_grid"] = answer
                        break
                    self.log.info('Must be "lebedev", "random", "golden_spiral", "gamess", or "marcus_deserno"!')
                # ---
                while True:
                    answer = question("RESP constraint parameters (betas):", int, default = defaults["resp_betas"], KEYSTROKES=KEYSTROKES)[0:3]
                    if all( i>0. for i in answer ):
                        self.setupINFOS["resp"]["resp_betas"] = answer
                        break
                    self.log.info("All must be positive!")
                # ---
                self.log.info("WARNING: This setup routine does not handle manual adjustments of VdW radii.")
                self.log.info("Please manually adjust VdW radii in the resource file if necessary!")
                # ---
                while True:
                    answer = question("RESP block size:", int, default = defaults["resp_block_size"], KEYSTROKES=KEYSTROKES)[0]
                    if answer >= 1:
                        self.setupINFOS["resp"]["resp_block_size"] = answer
                        break
                    self.log.info("Must be positive!")

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self.resources_file:
            with open(os.path.join(dir_path, "ORCA.resources"), "w", encoding="utf-8") as file:
                for key in (
                    "orcadir",
                    # "scratchdir",
                    "ncpu",
                    "memory",
                    "scaling",
                    "theodir",
                    "theodore_prop",
                    "theodore_fragment",
                    "wfoverlap",
                    "wfthres",
                ):
                    if key in self.setupINFOS:
                        file.write(f"{key} {self.setupINFOS[key]}\n")
                if "scratchdir" in self.setupINFOS:
                    file.write(f"scratchdir {os.path.join(self.setupINFOS['scratchdir'], dir_path)}\n")
                if "multipolar_fit" in INFOS['needed_requests']:
                    string = "resp_target %s\n" % (self.setupINFOS['resp']["resp_target"])
                    string += "resp_layers %i\n" % (self.setupINFOS['resp']["resp_layers"])
                    string += "resp_first_layer %f\n" % (self.setupINFOS['resp']["resp_first_layer"])
                    string += "resp_density %f\n" % (self.setupINFOS['resp']["resp_density"])
                    string += "resp_fit_order %i\n" % (self.setupINFOS['resp']["resp_fit_order"])
                    string += "resp_grid %s\n" % (self.setupINFOS['resp']["resp_grid"])
                    string += "resp_betas %i %i %i\n" % tuple(self.setupINFOS['resp']["resp_betas"])
                    string += "resp_block_size %i\n" % (self.setupINFOS['resp']["resp_block_size"])
                    string += "# resp_mk_radii False\n"
                    string += "# resp_vdw_radii {}\n"
                    string += "# resp_vdw_radii_symbols {}\n"
                    file.write(string)
        else:
            create_file(expand_path(self.resources_file), os.path.join(dir_path, "ORCA.resources"))
        create_file(expand_path(self.template_file), os.path.join(dir_path, "ORCA.template"))

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
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
        gradmaps = [sorted(qmin.maps["gradmap"])[i : i + 255] for i in range(0, len(qmin.maps["gradmap"]), 255)]
        input_str = ""
        if len(gradmaps) > 1:
            for ichunk, chunk in enumerate(gradmaps):
                if ichunk >= 1:
                    input_str += '\n\n$new_job\n\n%base "ORCA"\n\n'
                qmin.maps["gradmap"] = set(chunk)
                input_str += self.generate_inputstr(qmin)
        else:
            input_str += self.generate_inputstr(qmin)
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "ORCA.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        # Write point charges
        if self.QMin.molecule["point_charges"]:
            pc_str = f"{self.QMin.molecule['npc']}\n"
            for atom, coords in zip(self.QMin.coords["pccharge"], self.QMin.coords["pccoords"]):
                pc_str += f"{atom} {' '.join(str(i*au2a) for i in coords)}\n"
            writefile(os.path.join(workdir, "ORCA.pc"), pc_str)

        # Copy wf files
        self._copy_gbw(qmin, workdir)

        # Setup ORCA
        starttime = datetime.datetime.now()
        exec_str = f"{os.path.join(qmin.resources['orcadir'],'orca')} ORCA.inp"
        exit_code = self.run_program(workdir, exec_str, os.path.join(workdir, "ORCA.log"), os.path.join(workdir, "ORCA.err"))
        endtime = datetime.datetime.now()

        if exit_code == 0:
            # make molden files
            exec_str = f"{os.path.join(self.QMin.resources['orcadir'],'orca_2mkl')} ORCA -molden"
            molden_out = os.path.join(workdir, "orca_2mkl.out")
            molden_err = os.path.join(workdir, "orca_2mkl.err")
            self.run_program(workdir, exec_str, molden_out, molden_err)
            # Save files
            if not qmin.save["samestep"]:
                self._save_files(workdir, qmin.control["jobid"])
            if self.QMin.requests["ion"] and qmin.control["jobid"] == 1:
                writefile(os.path.join(self.QMin.save["savedir"], "AO_overl"), self._get_ao_matrix(workdir))

            if self.QMin.requests["density_matrices"] or self.QMin.requests["multipolar_fit"]:
                jsonstr = '{\n"Densities": ["all"]\n}'
                writefile(os.path.join(workdir, "ORCA.json.conf"), jsonstr)
                gbwfile = "ORCA.gbw" if self.QMin.resources["orcaversion"] >= (6, 0) else "ORCA"
                self.run_program(
                    workdir,
                    f"{os.path.join(self.QMin.resources['orcadir'],'orca_2json')} {gbwfile}",
                    os.path.join(workdir, "json.log"),
                    os.path.join(workdir, "json.err"),
                )

        return exit_code, endtime - starttime

    def _copy_gbw(self, qmin: QMin, workdir: str) -> None:
        """
        Copy gbw file from last/current time step

        jobid:      Job ID
        qmin:       QMin object
        workdir:    Current working directory
        """
        if not qmin.save["always_guess"]:
            self.log.debug("Copy ORCA.gbw to work directory")
            gbw_file = None
            if qmin.control["jobid"] in qmin.control["initorbs"]:
                gbw_file = qmin.control["initorbs"][qmin.control["jobid"]]
            elif not qmin.save["always_orb_init"]:
                gbw_file = os.path.join(qmin.save["savedir"], f"ORCA.gbw.{qmin.control['jobid']}.{qmin.save['step']}")
                if not os.path.isfile(gbw_file):
                    gbw_file = os.path.join(qmin.save["savedir"], f"ORCA.gbw.{qmin.control['jobid']}.{qmin.save['step']-1}")
            if gbw_file and os.path.isfile(gbw_file):
                shutil.copy(gbw_file, os.path.join(workdir, "ORCA.gbw"))

    def _get_ao_matrix(
        self, workdir: str, gbw_first: str = "ORCA.gbw", gbw_second: str = "ORCA.gbw", decimals: int = 7, trans: bool = False
    ) -> str:
        """
        Call orca_fragovl and extract ao matrix

        workdir:    Path of working directory
        gbw_first:  Name of first wave function file
        gbw_second: Name of second wave function file
        decimals:   Number of decimal places
        trans:      Transpose matrix
        """
        self.log.debug(f"Extracting AO matrix from {gbw_first} and {gbw_second} from orca_fragovl")
        gbw_first = os.path.join(workdir, gbw_first)
        gbw_second = os.path.join(workdir, gbw_second)

        match = re.fullmatch(r"(?:STO|[\d+]+(?:-[\d+]+)?[Gg](?:\*{0,2}|\([^)]+\))?)", self.QMin.template["basis"])
        if match:
            self.log.error("Detected a Pople basis set. These are not compatible with wfoverlap.x!")
            raise ValueError("Detected a Pople basis set. These are not compatible with wfoverlap.x!")

        # run orca_fragovl
        string = f"{os.path.join(self.QMin.resources['orcadir'],'orca_fragovl')} {gbw_first} {gbw_second}"
        self.run_program(workdir, string, "fragovlp.out", "fragovlp.err")

        with open(os.path.join(workdir, "fragovlp.out"), "r", encoding="utf-8") as file:
            fragovlp = file.read()

            # Get number of atomic orbitals
            n_ao = re.findall(r"\s{3,}(\d+)\s{5}", fragovlp)
            n_ao = max(list(map(int, n_ao))) + 1

            # Parse wfovlp.out and convert to n_ao*n_ao matrix
            find_mat = re.search(r"OVERLAP MATRIX\n-{32}\n([\s\d+\.-]*)\n\n", fragovlp)
            if not find_mat:
                raise ValueError
            ovlp_mat = list(map(float, re.findall(r"-?\d+\.\d{12}", find_mat.group(1))))
            ovlp_mat = self._matrix_from_output(ovlp_mat, n_ao)
            if trans:
                ovlp_mat = ovlp_mat.T

            # Convert matrix to string
            ao_mat = f"{n_ao} {n_ao}\n"
            for i in ovlp_mat:
                ao_mat += "".join(f"{j: .{decimals}e} " for j in i) + "\n"
            return ao_mat

    def _matrix_from_output(self, raw_matrix: list[float | complex], dim: int, orca_col: int = 6) -> np.ndarray:
        """
        Create a dim*dim numpy array from a raw orca matrix.
        The input array must only contain the actual data without row/col numbers

        raw_matrix: List of float or complex values parsed from ORCA output
        dim:        Dimension of the final matrix, dim*dim
        orca_col:   Number of columns per line, default 6
        """
        # Check if the array need to be padded
        padding = dim % orca_col
        padding_array = []
        if padding > 0:
            last_elems = raw_matrix[-(padding * dim) :]
            raw_matrix += [0] * (dim * (6 - padding))
            for i in range(dim):
                padding_array += last_elems[i * padding : i * padding + padding] + [0] * (6 - padding)
            # Add padding
            raw_matrix = np.asarray(raw_matrix)
            raw_matrix[-len(padding_array) :] = padding_array
        else:
            # No padding is needed
            raw_matrix = np.asarray(raw_matrix)

        return np.hstack(raw_matrix.reshape(-1, dim, orca_col))[:, :dim]

    def _save_files(self, workdir: str, jobid: int) -> None:
        """
        Save files (molden, gbw, mos) to savedir
        Naming convention: file.job.step
        """
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]
        self.log.debug("Copying files to savedir")

        # Generate molden file
        if self.QMin.requests["molden"] or self.QMin.requests["theodore"]:
            self.log.debug("Save molden file to savedir")
            shutil.copy(
                os.path.join(workdir, "ORCA.molden.input"),
                os.path.join(savedir, f"ORCA.molden.{jobid}.{step}"),
            )

        # Save gbw and dets from cis
        if self.QMin.requests["ion"] or not self.QMin.requests["nooverlap"]:
            self.log.debug("Write MO coefficients to savedir")
            writefile(os.path.join(savedir, f"mos.{jobid}.{step}"), self._get_mos(workdir, jobid))
            writefile(os.path.join(savedir, f"mos_allelec.{jobid}.{step}"), self._get_mos(workdir, jobid, ignore_frozcore=True))
            if self.QMin.molecule["states"][jobid - 1] > 1 and os.path.isfile(os.path.join(workdir, "ORCA.cis")):
                self.log.debug("Write CIS determinants to savedir")
                cis_dets = self.get_dets_from_cis(os.path.join(workdir, "ORCA.cis"), jobid)
                for det_file, cis_det in cis_dets.items():
                    writefile(os.path.join(savedir, f"{det_file}.{step}"), cis_det)
                cis_dets = self.get_dets_from_cis(
                    os.path.join(workdir, "ORCA.cis"), jobid, ignore_frozencore=True, filelabel="_allelec"
                )
                for det_file, cis_det in cis_dets.items():
                    writefile(os.path.join(savedir, f"{det_file}.{step}"), cis_det)
            else:
                _, _, _, occ_list, _, _ = tools.molden.load(os.path.join(workdir, "ORCA.molden.input"))
                froz = self.QMin.molecule["frozcore"]
                if isinstance(occ_list, tuple):
                    occ_list = convert_list(list(occ_list[0])[froz:] + list(occ_list[1] * 2)[froz:])
                else:
                    occ_list = [3 if x == 2.0 else 0 for x in occ_list[froz:]]
                # Convert to string and save file
                writefile(os.path.join(savedir, f"dets.{jobid}.{step}"), self.format_ci_vectors([{tuple(occ_list): 1.0}]))
                # TODO: also save dets_allelec files for this else

        shutil.copy(os.path.join(workdir, "ORCA.gbw"), os.path.join(savedir, f"ORCA.gbw.{jobid}.{step}"))

    def _get_mos(self, workdir: str, jobid: int, ignore_frozcore: bool = False) -> str:
        """
        Extract MO coefficients from ORCA gbw file

        workdir:   Directory of ORCA.gbw
        jobid:     ID number of job
        """
        restr = self.QMin.control["jobs"][jobid]["restr"]
        if ignore_frozcore:
            frozcore = 0
        else:
            frozcore = self.QMin.molecule["frozcore"]

        # run orca_fragovl
        string = f"{os.path.join(self.QMin.resources['orcadir'],'orca_fragovl')} ORCA.gbw ORCA.gbw"
        self.run_program(workdir, string, "fragovlp.out", "fragovlp.err")

        with open(os.path.join(workdir, "fragovlp.out"), "r", encoding="utf-8") as file:
            fragovlp = file.read()

            # Get number of atomic orbitals
            n_ao = re.findall(r"\s{3,}(\d+)\s{5}", fragovlp)
            n_ao = max(list(map(int, n_ao))) + 1

            # Parse matrix
            find_mat = re.search(r"FRAGMENT A MOs MATRIX\n-{32}\n([\s\d+\.-]*)\n\n", fragovlp)
            if not find_mat:
                raise ValueError
            ao_mat = list(map(float, re.findall(r"-?\d+\.\d{12}", find_mat.group(1))))
            if not restr:
                ao_mat_a = self._matrix_from_output(ao_mat[: len(ao_mat) // 2], n_ao)
                ao_mat_b = self._matrix_from_output(ao_mat[len(ao_mat) // 2 :], n_ao)
            else:
                ao_mat_a = self._matrix_from_output(ao_mat, n_ao)
                ao_mat_b = np.empty((0, 0))

            ao_mat_a = ao_mat_a[:, frozcore:]
            ao_mat_b = ao_mat_b[:, frozcore:]

        # make string
        n_mo = n_ao - frozcore
        if not restr:
            n_mo *= 2

        string = f"2mocoef\nheader\n1\nMO-coefficients from Orca\n1\n{n_ao}   {n_mo}\na\nmocoef\n(*)\n"

        for mat in [ao_mat_a, ao_mat_b]:
            for i in mat.T:
                for idx, j in enumerate(i):
                    if idx > 0 and idx % 3 == 0:
                        string += "\n"
                    string += f"{j: 6.12e} "
                if i.shape[0] - 1 % 3 != 0:
                    string += "\n"
        string += "orbocc\n(*)"

        for i in range(n_mo):
            if i % 3 == 0:
                string += "\n"
            string += f"{0.0: 6.12e} "
        return string

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
        if self.QMin.requests["theodore"]:
            nprop = len(self.QMin.resources["theodore_prop"]) + (nfrag := len(self.QMin.resources["theodore_fragment"])) ** 2
            labels = self.QMin.resources["theodore_prop"][:] + [f"Om_{i+1}_{j+1}" for i in range(nfrag) for j in range(nfrag)]
            theodore_arr = [[labels[j], np.zeros(self.QMin.molecule["nmstates"])] for j in range(nprop)]

        scratchdir = self.QMin.resources["scratchdir"]

        # Get contents of output file(s)
        for job in self.QMin.control["joblist"]:
            with open(os.path.join(scratchdir, f"master_{job}/ORCA.log"), "r", encoding="utf-8") as file:
                log_file = file.read()
                gs_mult, _ = self.QMin.control["jobs"][job].values()
                mults = self.QMin.control["jobs"][job]["mults"]
                nm_states = [0] + [x * m for m, x in enumerate(self.QMin.molecule["states"], 1)]
                states = [0] + self.QMin.molecule["states"]
                job_states = [x if i in mults else 0 for i, x in enumerate(nm_states)]

                # Populate SOC matrix
                if self.QMin.requests["soc"] and self.QMin.control["jobs"][job]["restr"]:
                    soc_mat = self._get_socs(log_file)
                    for mult in mults:
                        # Diagonal blocks
                        start1, stop1 = sum(job_states[:mult]), sum(job_states[: mult + 1])
                        start, stop = sum(nm_states[:mult]), sum(nm_states[: mult + 1])
                        self.QMout["h"][start:stop, start:stop] = soc_mat[start1:stop1, start1:stop1]

                    # Offdiagonals
                    self.QMout["h"][: nm_states[1], sum(nm_states[:3]) : sum(nm_states[:4])] = soc_mat[
                        : nm_states[1], nm_states[1] :
                    ]
                    self.QMout["h"][sum(nm_states[:3]) : sum(nm_states[:4]), : nm_states[1]] = soc_mat[
                        nm_states[1] :, : nm_states[1]
                    ]

                # Populate energies
                energies = self._get_energy(log_file, mults)
                for i in range(sum(nm_states)):
                    statemap = self.QMin.maps["statemap"][i + 1]
                    if statemap[0] in mults:
                        self.QMout["h"][i][i] = energies[(statemap[0], statemap[1])]

                # Populate dipole moments
                if self.QMin.requests["dm"]:
                    if self.QMin.resources["orcaversion"] >= (6, 0):
                        # Diagonal elements
                        dipole_dict = self._get_dipole_moment_orca6(log_file)
                        total_states = sum(job_states)
                        dipoles = np.zeros((total_states, 3))
                        start_idx = 0
                        for mult in mults:
                            for i in range(states[mult]):
                                key = (mult, i)  # e.g., (1, 0) is ground, (3, 1) is triplet first excited
                                if key in dipole_dict:
                                    dipoles[start_idx + i] = dipole_dict[key]
                            start_idx += states[mult] - 1  # only relevant for S+T
                        s_cnt = 0
                        for mult in mults:
                            for dim in range(3):
                                np.fill_diagonal(
                                    self.QMout["dm"][
                                        dim,
                                        sum(nm_states[:mult]) : sum(nm_states[: mult + 1]),
                                        sum(nm_states[:mult]) : sum(nm_states[: mult + 1]),
                                    ],
                                    np.tile(dipoles[s_cnt : s_cnt + states[mult], dim], mult),
                                )
                            s_cnt += states[mult]
                    else:
                        # Diagonal elements
                        dipoles_gs = self._get_dipole_moment(log_file, True)
                        dipoles_es = self._get_dipole_moment(log_file, False)
                        for mult in mults:
                            for dim in range(3):
                                np.fill_diagonal(
                                    self.QMout["dm"][
                                        dim,
                                        sum(nm_states[:mult]) : sum(nm_states[: mult + 1]),
                                        sum(nm_states[:mult]) : sum(nm_states[: mult + 1]),
                                    ],
                                    dipoles_es[dim],
                                )
                            if mult == gs_mult[0]:
                                for m in range(mult):
                                    self.QMout["dm"][
                                        :,
                                        sum(nm_states[:mult]) + m * states[mult],
                                        sum(nm_states[:mult]) + m * states[mult],
                                    ] = dipoles_gs

                    # Offdiagonals
                    if states[mults[0]] > 1:
                        if self.QMin.resources["orcaversion"] >= (6, 0):
                            dipoles_trans_dict = self._get_transition_dipoles_orca6(log_file)
                            for i in range(self.QMin.molecule["nmstates"]):
                                m1, s1, ms1 = self.QMin.maps["statemap"][i + 1]
                                for j in range(self.QMin.molecule["nmstates"]):
                                    m2, s2, ms2 = self.QMin.maps["statemap"][j + 1]
                                    if m1 != m2:
                                        continue
                                    if ms1 != ms2:
                                        continue
                                    if i == j:
                                        continue
                                    try:
                                        self.QMout["dm"][:, i, j] = dipoles_trans_dict[(m1, s1), (m2, s2)]
                                    except KeyError:
                                        pass
                        else:
                            dipoles_trans = self._get_transition_dipoles(log_file)
                            for idx, val in enumerate(dipoles_trans[: states[mults[0]] - 1], 1):  # States
                                for m in range(mults[0]):  # Make copies for multiplicities
                                    self.QMout["dm"][
                                        :,
                                        sum(nm_states[: mults[0]]) + m * states[mults[0]],
                                        sum(nm_states[: mults[0]]) + m * states[mults[0]] + idx,
                                    ] = val[:]
                                    self.QMout["dm"][
                                        :,
                                        sum(nm_states[: mults[0]]) + m * states[mults[0]] + idx,
                                        sum(nm_states[: mults[0]]) + m * states[mults[0]],
                                    ] = val[:]

                # TheoDORE
                if self.QMin.requests["theodore"]:
                    if self.QMin.control["jobs"][job]["restr"]:
                        ns = 0
                        for i in mults:
                            ns += states[i - 2] - (i == gs_mult)
                        if ns != 0:
                            props = self.get_theodore(
                                os.path.join(scratchdir, f"master_{job}", "tden_summ.txt"),
                                os.path.join(scratchdir, f"master_{job}", "OmFrag.txt"),
                            )
                            for i in range(self.QMin.molecule["nmstates"]):
                                m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                                if (m1, s1) in props:
                                    for j in range(
                                        len(self.QMin.resources["theodore_prop"])
                                        + len(self.QMin.resources["theodore_fragment"]) ** 2
                                    ):
                                        theodore_arr[j][1][i] = props[(m1, s1)][j]
            if self.QMin.requests["multipolar_fit"] or self.QMin.requests["density_matrices"]:
                if not self.QMout.mol:
                    mol2, _, _, _, _, _ = tools.molden.load(
                        os.path.join(self.QMin.resources["scratchdir"], f"master_{job}/ORCA.molden.input")
                    )
                    basis = {}
                    el_basis = {}
                    if self.QMin.template["basis_per_element"]:
                        for el, bas in batched(self.QMin.template["basis_per_element"]):
                            el_basis[el] = bas
                    for idx, atom in enumerate(self.QMin.molecule["elements"], 1):
                        if atom in el_basis:
                            basis[atom + str(idx)] = el_basis[atom]
                        else:
                            basis[atom + str(idx)] = self.QMin.template["basis"]
                    if self.QMin.template["basis_per_atom"]:
                        for idx, bas in batched(self.QMin.template["basis_per_atom"]):
                            basis[self.QMin.molecule["elements"][int(idx) - 1] + idx] = bas
                    mol = gto.Mole()
                    mol.atom = [[e[0], c] for e, c in zip(mol2.atom, self.QMin.coords["coords"].tolist())]
                    mol.unit = "Bohr"
                    mol.cart = False
                    mol.basis = basis
                    try:
                        mol.build()
                    except RuntimeError:
                        mol.spin = 1
                        mol.build()
                    self.QMout.mol = mol
                    self.QMin.molecule["SAO"] = mol.intor("int1e_ovlp")

                    self.ao_labels = [i.split()[::2] for i in mol2.ao_labels()]
                    self.order = sorted(list(range(len(self.ao_labels))), key=cmp_to_key(self._sort_ao))
                    self.ao_sign = []
                    for i, l in enumerate(self.ao_labels):
                        for j in ("f+3", "f-3", "g+3", "g-3", "g+4", "g-4", "h+3", "h-3", "h+4", "h-4"):
                            if j in l[1]:
                                self.ao_sign.append(i)
                with open(os.path.join(scratchdir, f"master_{job}/ORCA.json"), "r", encoding="utf-8") as f:
                    orca_json = json.load(f)
                restricted = job == 1 and not self.QMin.template["unrestricted_triplets"]
                for idx, s1 in enumerate(self.states):
                    if self.QMin.template["functional"] == "mp2":
                        dens = np.array(orca_json["Molecule"]["Densities"]["pmp2re"])
                        self.QMout.density_matrices[(self.states[0], self.states[0], "tot")] = dens[
                            np.ix_(self.order, self.order)
                        ]
                        break

                    if s1.S + 1 not in self.QMin.control["jobs"][job]["mults"]:
                        continue
                    for s2 in self.states[idx:]:
                        if s2.S + 1 not in self.QMin.control["jobs"][job]["mults"]:
                            continue
                        if s1 == s2:
                            if s1.N == 1:
                                if s1.S != 2 or (not restricted):
                                    dens = np.array(orca_json["Molecule"]["Densities"]["scfp"])
                                    self.log.debug(f"({s1}, {s2}) mult: {job} density: scfp")
                                    self.QMout.density_matrices[(s1, s2, "tot")] = dens[np.ix_(self.order, self.order)]
                                    continue
                            m = "triplet" if (s1.S == 2 and restricted) else "singlet"
                            try:
                                spin = 1 if m == "triplet" else 0
                                dens = np.array(orca_json["Molecule"]["Densities"][f"cispre.{m}.iroot{s1.N-1+spin}"])
                                self.QMout.density_matrices[(s1, s2, "tot")] = dens[np.ix_(self.order, self.order)]
                                self.log.debug(f"({s1}, {s2}) mult: {job} density: cispre.{m}.iroot{s1.N-1+spin}")
                            except KeyError:
                                self.log.warning(f"No relaxed density for ({s1}, {s2}, tot)")
                        elif s1.M == s2.M and s1.S == s2.S:
                            try:
                                spin = 1 if restricted and s1.S == 2 else 0
                                dens = np.array(orca_json["Molecule"]["Densities"][f"Tdens-CIS-0-{spin}-{s1.N-1}-{s2.N-1}"])
                                self.QMout.density_matrices[(s1, s2, "tot")] = dens[np.ix_(self.order, self.order)]
                                self.log.debug(f"({s1}, {s2}) mult: {job} density: Tdens-CIS-0-{spin}-{s1.N-1}-{s2.N-1}")
                            except KeyError:
                                self.log.warning(f"No density for ({s1}, {s2}, tot)")

        if self.QMin.requests["theodore"]:
            self.QMout["prop1d"].extend(theodore_arr)

        # Populate gradients
        if self.QMin.requests["grad"]:
            for grad in self.QMin.maps["gradmap"]:
                job_path, ground_state = self.QMin.control["jobgrad"][grad]
                grad_mult, _ = self.QMin.control["jobs"][int(job_path.split("_")[1])].values()
                grad_ext = f"{'singlet' if grad[0] == grad_mult[0] else IToMult[grad[0]].lower()}.root{grad[1] - (grad[0] == grad_mult[0])}"
                if ground_state:
                    if self.QMin.resources["orcaversion"] >= (6, 0):
                        gradients = self._get_grad(os.path.join(scratchdir, job_path, "ORCA.engrad.ground.grad.tmp"))
                    else:
                        gradients = self._get_grad(os.path.join(scratchdir, job_path, "ORCA.engrad"), True)
                else:
                    gradients = self._get_grad(os.path.join(scratchdir, job_path, f"ORCA.engrad.{grad_ext}.grad.tmp"))

                # Point charges
                point_charges = None
                if self.QMin.molecule["point_charges"]:
                    if ground_state:
                        point_charges = self._get_pc_grad(os.path.join(scratchdir, job_path, "ORCA.pcgrad"))
                    else:
                        point_charges = self._get_pc_grad(os.path.join(scratchdir, job_path, f"ORCA.pcgrad.{grad_ext}"))

                for key, val in self.QMin.maps["statemap"].items():
                    if (val[0], val[1]) == grad:
                        self.QMout["grad"][key - 1] = gradients

                        # Point charges
                        if self.QMin.molecule["point_charges"]:
                            self.QMout["grad_pc"][key - 1] = point_charges

            # Populate neglected gradients
            neglected_grads = [
                x for x in range(sum(nm_states)) if tuple(self.QMin.maps["statemap"][x + 1][0:2]) not in self.QMin.maps["gradmap"]
            ]
            match self.QMin.template["neglected_gradient"]:
                case "gs":
                    for state in neglected_grads:
                        self.QMout["grad"][state] = self.QMout["grad"][self.QMin.maps["gsmap"][state + 1] - 1]
                case "closest":
                    energies = np.array(np.diagonal(self.QMout["h"]))
                    energies_masked = deepcopy(energies)
                    energies_masked[neglected_grads] = 9999
                    for grad in neglected_grads:
                        idx = np.abs(energies_masked - energies[grad]).argmin()
                        self.QMout["grad"][grad] = self.QMout["grad"][idx]

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
        self.QMout["runtime"] = self.clock.measuretime(False)

        if self.QMin.requests["multipolar_fit"] or self.QMin.requests["density_matrices"]:
            for dens in self.QMout.density_matrices.values():
                dens[:, self.ao_sign] *= -1
                dens[self.ao_sign, :] *= -1
            self.get_densities()
            if self.QMin.requests["multipolar_fit"]:
                self.QMout.multipolar_fit = self._resp_fit_on_densities()
        return self.QMout

    def _sort_ao(self, ao1: int, ao2: int) -> int:
        """
        ORCA AO order to PySCF order
        """
        first = self.ao_labels[ao1]
        second = self.ao_labels[ao2]

        label_to_int = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5}
        angular_to_int = {
            "s": 1,
            "px": 3,
            "py": 1,
            "pz": 2,
            "dxy": 3,
            "dyz": 4,
            "dz^2": 2,
            "dxz": 5,
            "dx2-y2": 1,
            "f-3": 4,
            "f-2": 5,
            "f-1": 3,
            "f+0": 6,
            "f+1": 2,
            "f+2": 7,
            "f+3": 1,
            "g-4": 5,
            "g-3": 6,
            "g-2": 4,
            "g-1": 7,
            "g+0": 3,
            "g+1": 8,
            "g+2": 2,
            "g+3": 9,
            "g+4": 1,
            "h-5": 6,
            "h-4": 7,
            "h-3": 5,
            "h-2": 8,
            "h-1": 4,
            "h+0": 9,
            "h+1": 3,
            "h+2": 10,
            "h+3": 2,
            "h+4": 11,
            "h+5": 1,
        }
        return (
            (int(first[0]) - int(second[0])) * 10000  # Atom
            + (int(first[1][0]) - int(second[1][0])) * 100  # Shell
            + (label_to_int[first[1][1]] - label_to_int[second[1][1]]) * 1000  # Function
            + (angular_to_int[first[1][1:]] - angular_to_int[second[1][1:]]) * 1  # Angular
        )

    def _get_pc_grad(self, grad_path: str) -> np.ndarray:
        """
        Extract point charge gradients from ORCA.pcgrad
        """
        # read file
        out = readfile(grad_path)

        g = []
        for iatom in range(len(out) - 1):
            atom_grad = [0.0 for i in range(3)]
            s = out[iatom + 1].split()
            for ixyz in range(3):
                atom_grad[ixyz] = float(s[ixyz])
            g.append(atom_grad)
        return np.asarray(g)

    def _get_dipole_moment(self, output: str, ground_state: bool) -> np.ndarray:
        """
        Extract dipole moment from ORCA outfile
        output:     Content of outfile as string
        """
        find_dipole = re.findall(r"Total Dipole Moment[:\s]+(.*)", output)
        if not find_dipole:
            self.log.error("Cannot find dipole moment in ORCA outfile!")
            raise ValueError()
        find_dipole = [list(map(float, x.split())) for x in find_dipole]
        if self.QMin.resources["orcaversion"] >= (6, 0):
            return np.asarray(find_dipole)
        if len(find_dipole) == 1 and not ground_state:
            return np.zeros(3)
        return np.asarray(find_dipole[0] if ground_state else find_dipole[-1])

    def _get_dipole_moment_orca6(self, output: str) -> dict[tuple[int], list[float]]:
        """
        Extract dipole moment from ORCA6 outfile
        output:     Content of outfile as string
        """
        mults1 = re.findall(r"Multiplicity\s+:\s+(\d+)", output)  # gives ["1", "1", "3", ...]
        mults = [int(x) for x in mults1]

        states1 = re.findall(r"Method\s*:\s*(SCF)|State\s*:\s*(\d+)", output)  # gives "SCF", "1", "2", "1", ...]
        states = []
        for method, state in states1:
            if method:
                states.append(0)
            elif state:
                states.append(int(state))
        # now states is [0,1,2,3,...]

        dipole_vectors1 = re.findall(r"Total Dipole Moment\s+:\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", output)
        dipole_vectors = [list(map(float, match)) for match in dipole_vectors1]
        # is [ [1.,2.,3.], [...], ...]

        dipoles = {(m, s): d for m, s, d in zip(mults, states, dipole_vectors)}
        return dipoles  # dictionary {(mult, state): [x, y, z]}

    def _get_grad(self, grad_path: str, ground_state: bool = False) -> np.ndarray:
        """
        Extract gradients from ORCA outfile

        grad_path:  Path to gradient file
        """
        natom = self.QMin.molecule["natom"]

        with open(grad_path, "r" if ground_state else "rb") as grad_file:
            if ground_state:
                find_grads = re.search(r"bohr\n#\n(.*)#\n#", grad_file.read(), re.DOTALL)
                if not find_grads:
                    self.log.error(f"Gradients not found in {grad_path}!")
                    raise ValueError()
                gradients = find_grads.group(1).split()
            else:
                grad_file.read(8 + 28 * natom)  # Skip header
                gradients = struct.unpack(f"{natom*3}d", grad_file.read(8 * 3 * natom))
        return np.asarray(gradients).reshape(natom, 3)

    def _get_transition_dipoles(self, output: str) -> np.ndarray:
        """
        Extract transition dipole moments from ORCA outfile
        In TD-DFT with ORCA 5 only TDM between ground- and
        excited states of same multiplicity are calculated

        output:     Content of outfile as string
        """
        # Extract transition dipole table from output
        find_transition_dipoles = re.search(
            r"  ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS(.*?)ABS", output, re.DOTALL
        )
        if not find_transition_dipoles:
            self.log.error("Cannot find transition dipoles in ORCA output!")
            raise ValueError()
        # Filter dipole vectors, (states, (xyz))
        transition_dipoles = re.findall(r"([-\d.]+\s+[-\d.]+\s+[-\d.]+)\n", find_transition_dipoles.group(1))
        return np.asarray([list(map(float, x.split())) for x in transition_dipoles])

    def _get_transition_dipoles_orca6(self, output: str) -> dict:
        """ """
        # extract the table
        find_transition_dipoles = re.search(
            r"  ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS(.*?)ABS", output, re.DOTALL
        )
        if not find_transition_dipoles:
            self.log.error("Cannot find transition dipoles in ORCA output!")
            raise ValueError()

        def parse_label(string):
            "N-MA"
            s = string.replace("A", "").split("-")
            n = int(s[0]) + 1
            m = int(s[1])
            return (m, n)

        dictionary = {}
        for line in find_transition_dipoles.group(1).split("\n")[5:]:
            if not line:
                break
            s = line.split()
            s0 = s[0]
            si = s[2]
            d = np.asarray([float(i) for i in s[-3:]])
            m1, n1 = parse_label(s0)
            m2, n2 = parse_label(si)
            if m1 == m2:
                dictionary[(m1, n1), (m2, n2)] = d
                dictionary[(m2, n2), (m1, n1)] = d
        # self.log.info(dictionary)
        return dictionary

    def _get_socs(self, output: str) -> np.ndarray:
        """
        Extract SOC matrix from ORCA outfile

        output:     Content of outfile as string
        """
        # Get number of states
        n_roots = re.search(r"nroots\s+(\d+)", output)
        if not n_roots:
            self.log.error("Cannot find number of roots in ORCA outfile!")
            raise ValueError()

        n_trip = int(n_roots.group(1))
        n_sing = n_trip + 1

        # Extract raw matrix from outfile
        find_mat = re.search(r"Real part:([\s\d+-e]*)Image part:([\s\d+-e]*)\.{3}", output, re.DOTALL)
        if not find_mat:
            self.log.error("Cannot find SOC matrix in ORCA output!")
            raise ValueError()

        # Remove garbage and combine to complex
        real_part = list(map(float, re.sub(r"\s\d{1,5}\s", "", find_mat.group(1)).split()))
        imag_part = list(map(float, re.sub(r"\s\d{1,5}\s", "", find_mat.group(2)).split()))
        soc_matrix = []
        for real, imag in zip(real_part, imag_part):
            soc_matrix.append(complex(real, imag))

        # Convert raw matrix
        soc_matrix = self._matrix_from_output(soc_matrix, n_sing + 3 * n_trip)

        # Reorder matrix from T 0 -1 1 to -1 0 1
        for i in range(n_trip):
            soc_matrix[:, [n_sing + i, n_sing + i + n_trip]] = soc_matrix[:, [n_sing + i + n_trip, n_sing + i]]
            soc_matrix[[n_sing + i, n_sing + i + n_trip], :] = soc_matrix[[n_sing + i + n_trip, n_sing + i], :]

        # Create indices of rows/cols to skip
        counter = iter(count(n_sing))
        trip_slice = np.array(
            [[next(counter) for _ in range(n_trip)][self.QMin.molecule["states"][2] :] for _ in range(3)]
        ).flatten()
        sing_slice = [x for x in range(self.QMin.molecule["states"][0], n_sing)]

        # Remove extra triplets
        if trip_slice.size > 0:
            soc_matrix = np.delete(soc_matrix, trip_slice, axis=0)
            soc_matrix = np.delete(soc_matrix, trip_slice, axis=1)

        # Remove extra singlets
        if sing_slice:
            soc_matrix = np.delete(soc_matrix, sing_slice, axis=0)
            soc_matrix = np.delete(soc_matrix, sing_slice, axis=1)

        return soc_matrix

    def _get_energy(self, output: str, mults: list[int]) -> dict[tuple[int, int], float]:
        """
        Extract energies from ORCA outfile

        output:     Content of outfile as string
        mult:       Multiplicities
        """

        # Define variables
        gsmult = mults[0]
        states_extract = deepcopy(self.QMin.molecule["states"])
        states_extract[gsmult - 1] -= 1
        use_rpa = self.QMin.template["no_tda"]

        states_extract = [0 if idx + 1 not in mults else val for idx, val in enumerate(states_extract)]
        states_extract = [max(states_extract) if idx + 1 in mults else val for idx, val in enumerate(states_extract)]

        # Find ground state energy and apply dispersion correction
        find_energy = re.search(r"Total Energy[\s:]+([-\d\.]+)", output)
        if not find_energy:
            self.log.error("No energy in ORCA outfile found!")
            raise ValueError()

        gs_energy = float(find_energy.group(1))
        dispersion = re.search(r"Dispersion correction\s+(-?\d+\.\d+)", output)
        if dispersion:
            gs_energy += float(dispersion.group(1))

        energies = {(gsmult, int(1)): gs_energy}

        # Find excited states e.g. 2 sing + 2 trip: [(1, en1), (2, en2), (1,en_trip1), (2,en_trip2)
        exc_states = re.findall(r"STATE\s+(\d+):[A-Z\s=]+([-\d\.]+)\s+au", output)

        iter_states = iter(exc_states)
        sub_states = 0
        if self.QMin.resources["orcaversion"] >= (6, 0):
            sub_states = states_extract[gsmult - 1]
        if use_rpa:
            sub_states = 0
        for imult in mults:
            nstates = states_extract[imult - 1]
            for state, energy in iter_states:
                if (int(state) - (sub_states * (gsmult != imult))) <= self.QMin.molecule["states"][
                    imult - 1
                ]:  # Skip extra states
                    energies[(imult, int(state) - (sub_states * (gsmult != imult)) + (gsmult == imult))] = gs_energy + float(
                        energy
                    )
                if (int(state) - (sub_states * (gsmult != imult))) == nstates:
                    break
        return energies

    def printQMout(self) -> None:
        super().writeQMout()

    def _set_request(self, *args, **kwargs) -> None:
        super()._set_request(*args, **kwargs)
        self.QMin.requests["h"] = True

        if self.QMin.requests["soc"]:
            if (
                len(self.QMin.molecule["states"]) < 3
                or (self.QMin.molecule["states"][0] == 0 and self.QMin.molecule["states"][2] <= 1)
                or (self.QMin.molecule["states"][0] > 0 and self.QMin.molecule["states"][2] == 0)
            ):
                self.log.warning("SOCs requested but only 1 multiplicity given! Disable SOCs")
                self.QMin.requests["soc"] = False
            else:
                if self.QMin.template["unrestricted_triplets"]:
                    self.log.error("SOCs are not possible with unrestricted_triplets")
                    raise RuntimeError
                
        if self.QMin.requests["multipolar_fit"] or self.QMin.requests["density_matrices"]:
            if self.QMin.resources["orcaversion"] < (6, 0):
                self.log.error("multipolar_fit and density_matrices are only possible with ORCA 6")
                raise RuntimeError

    def read_resources(self, resources_file: str = "ORCA.resources", kw_whitelist: list[str] | None = None) -> None:
        if kw_whitelist is None:
            kw_whitelist = []
        super().read_resources(resources_file, kw_whitelist)

        # Check if frozcore specified
        if self.QMin.resources["numfrozcore"] >= 0:
            self.log.debug("Found numfrozcore in resources, overwriting frozcore")
            self.QMin.molecule["frozcore"] = self.QMin.resources["numfrozcore"]

        # LD PATH???
        if not self.QMin.resources["orcadir"]:
            raise ValueError("orcadir has to be set in resource file!")

        self.QMin.resources["orcadir"] = expand_path(self.QMin.resources["orcadir"])
        self.log.debug(f'orcadir set to {self.QMin.resources["orcadir"]}')

        self.QMin.resources["orcaversion"] = SHARC_ORCA.get_orca_version(self.QMin.resources["orcadir"])
        self.log.info(f'Detected ORCA version {".".join(str(i) for i in self.QMin.resources["orcaversion"])}')

        self.QMin.resources["scratchdir"] = expand_path(self.QMin.resources["scratchdir"])

        if self.QMin.resources["orcaversion"] < (5, 0):
            raise ValueError("This version of the SHARC-ORCA interface is only compatible to Orca 5.0 or higher!")

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)
        self.QMin.requests["h"] = True
        if self.QMin.requests["phases"]:
            self.QMin.requests["overlaps"] = True

    def read_template(self, template_file: str = "ORCA.template", kw_whitelist: list[str] | None = None) -> None:
        kw_whitelist = ["basis_per_element", "basis_per_atom", "ecp_per_element"]
        super().read_template(template_file, kw_whitelist)

        for key in kw_whitelist:
            if self.QMin.template[key] and isinstance(self.QMin.template[key][0], list):
                self.QMin.template[key] = list(chain.from_iterable(self.QMin.template[key]))

        # Convert keys to string if list
        if isinstance(self.QMin.template["keys"], list):
            self.QMin.template["keys"] = " ".join(self.QMin.template["keys"])
        elif self.QMin.template["keys"]:
            self.QMin.template["keys"] = self.QMin.template["keys"].lower()

        # Check for deprecated keys
        for depr in self._deprecated:
            if depr.casefold() in self.QMin.template:
                self.log.warning(f"Template key {depr} is deprecated and will be ignored!")

        # Check if unrestricted triplets needed
        if not self.QMin.template["unrestricted_triplets"]:
            if len(self.QMin.molecule["charge"]) >= 3 and self.QMin.molecule["charge"][0] != self.QMin.molecule["charge"][2]:
                self.log.error("Charges of singlets and triplets differ. Please enable the unrestricted_triplets option!")
                raise ValueError()

        if self.QMin.requests["soc"] and len(self.QMin.molecule["states"]) >= 3 and self.QMin.molecule["states"][2] > 0:
            self.log.error("Request SOC is not compatible with unrestricted_triplets!")
            raise ValueError()

        # Check if valid paste_input_file path is given
        if self.QMin.template["paste_input_file"]:
            self.QMin.template["paste_input_file"] = expand_path(self.QMin.template["paste_input_file"])
            if not os.path.isfile(self.QMin.template["paste_input_file"]):
                self.log.error(f"paste_input_file {self.QMin.template['paste_input_file']} does not exist!")
                raise FileNotFoundError()

        # Warn from Pople basis sets
        match = re.fullmatch(r"(?:STO|[\d+]+(?:-[\d+]+)?[Gg](?:\*{0,2}|\([^)]+\))?)", self.QMin.template["basis"])
        if match:
            self.log.warning("Detected a Pople basis set! Overlap calculations will not work properly")

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
        if not self.QMin.resources["dry_run"]:
            self.runjobs(self.QMin.scheduling["schedule"])

        # Run theodore
        if self.QMin.requests["theodore"]:
            self._run_theodore()

        # Run wfoverlap
        self._run_wfoverlap()

        self.log.debug("All jobs finished successful")

    def _create_aoovl(self) -> None:
        """
        Create AO_overl.mixed for overlap calculations
        """
        gbw_curr = f"ORCA.gbw.{self.QMin.control['joblist'][0]}.{self.QMin.save['step']}"
        gbw_prev = f"ORCA.gbw.{self.QMin.control['joblist'][0]}.{self.QMin.save['step']-1}"

        writefile(
            os.path.join(self.QMin.save["savedir"], "AO_overl.mixed"),
            self._get_ao_matrix(self.QMin.save["savedir"], gbw_prev, gbw_curr, 15, True),
        )

    def setup_interface(self) -> None:
        """
        Setup remaining maps (ionmap, gsmap) and build jobs dict
        """
        super().setup_interface()

        states_to_do = deepcopy(self.QMin.molecule["states"])
        for mult, state in enumerate(self.QMin.molecule["states"]):
            if state > 0:
                states_to_do[mult] += int(self.QMin.template["paddingstates"][mult])
        if (
            not self.QMin.template["unrestricted_triplets"]
            and len(self.QMin.molecule["states"]) >= 3
            and self.QMin.molecule["states"][2] > 0
        ):
            self.log.debug("Setup states_to_do")
            states_to_do[0] = max(self.QMin.molecule["states"][0], 1)
            req = max(self.QMin.molecule["states"][0] - 1, self.QMin.molecule["states"][2])
            states_to_do[0] = req + 1
            states_to_do[2] = req
        self.QMin.control["states_to_do"] = states_to_do

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
                    self.QMin.maps["gsmap"][i + 1] = j + 1
                    break
        # Populate initial orbitals dict
        self.QMin.control["initorbs"] = self._get_initorbs()

        for s in self.states:
            s.C["is_gs"] = False
            s.C["its_gs"] = None
            if s.N == 1:
                s.C["is_gs"] = True
                if s.S == 2:
                    s.C["is_gs"] = False

    def _build_jobs(self) -> None:
        """
        Build job dictionary from states_to_do
        """
        self.log.debug("Building job map.")
        jobs = {}
        for idx, state in enumerate(self.QMin.control["states_to_do"]):
            if state > 0 and idx != 2:
                jobs[idx + 1] = {"mults": [idx + 1], "restr": bool(idx == 0)}
            if state > 0 and idx == 2 and not self.QMin.template["unrestricted_triplets"]:
                jobs[1]["mults"].append(3)
            elif state > 0 and idx == 2 and self.QMin.template["unrestricted_triplets"]:
                jobs[3] = {"mults": [3], "restr": False}

        self.QMin.control["jobs"] = jobs
        self.QMin.control["joblist"] = sorted(set(jobs))

    def get_dets_from_cis(
        self, cis_path: str, jobid: int, ignore_frozencore: bool = False, filelabel: str = ""
    ) -> dict[str, str]:
        """
        Parse ORCA.cis file from WORKDIR
        """
        # Set variables
        cis_path = cis_path if os.path.isfile(cis_path) else os.path.join(cis_path, "ORCA.cis")
        restricted = self.QMin.control["jobs"][jobid]["restr"]
        mults = self.QMin.control["jobs"][jobid]["mults"]
        gsmult = self.QMin.maps["multmap"][-int(jobid)][0]
        if ignore_frozencore:
            frozcore = 0
        else:
            frozcore = self.QMin.molecule["frozcore"]
        states_extract = deepcopy(self.QMin.molecule["states"])
        states_skip = [self.QMin.control["states_to_do"][i] - states_extract[i] for i in range(len(states_extract))]

        for i, _ in enumerate(states_extract):
            if not i + 1 in mults:
                states_extract[i] = 0
                states_skip[i] = 0
            elif i + 1 == gsmult:
                states_extract[i] = max(0, states_extract[i] - 1)

        if states_extract[gsmult - 1] == 0:
            states_skip[gsmult - 1] -= 1

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

            # ground state configuration
            # 0: empty, 1: alpha, 2: beta, 3: double oppupied
            if restricted:
                occ_a = tuple([3] * (nfc + noa) + [0] * nva)
                occ_b = tuple()
            else:
                buffsize += (header[5] + 1 - header[4]) * (header[7] + 1 - header[6])
                occ_a = tuple([1] * (nfc + noa) + [0] * nva)
                occ_b = tuple([2] * (nfc + nob) + [0] * nvb)

            # Iterate over multiplicities and parse determinants
            eigenvectors = {}
            for mult in mults:
                eigenvectors[mult] = []

                if mult == gsmult:
                    key = tuple(occ_a[frozcore:] + occ_b[frozcore:])
                    eigenvectors[mult].append({key: 1.0})

                for _ in range(states_extract[mult - 1]):
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
                    self.trim_civecs(dets)

                    dets_exp = {}
                    for occ, virt, dummy in dets:
                        if restricted:
                            key = list(occ_a)
                            match mult:
                                case 1:
                                    # TODO: Check if relative sign of ab and ba is correct here!
                                    key[occ], key[virt] = 2, 1
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)] * math.sqrt(0.5)
                                    key[occ], key[virt] = 1, 2
                                    dets_exp[tuple(key)] = -dets[(occ, virt, dummy)] * math.sqrt(0.5)
                                case 3:
                                    key[occ], key[virt] = 1, 1
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)]
                        else:
                            key = list(occ_a + occ_b)
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
                            key2 = key[frozcore:]
                            dets_nofroz[key2] = val
                            continue
                        if any(map(lambda x: x != 1, key[:frozcore])) or any(
                            map(lambda x: x != 2, key[frozcore + noa + nva : noa + nva + 2 * frozcore])
                        ):
                            self.log.warning("Non-occupied orbital inside frozen core! Skipping ...")
                            continue
                        key2 = key[frozcore : frozcore + noa + nva] + key[noa + nva + 2 * frozcore :]
                        dets_nofroz[key2] = val
                    eigenvectors[mult].append(dets_nofroz)

                # Skip extra roots
                skip = 40 + buffsize * 8
                if self.QMin.template["no_tda"]:
                    skip *= 2
                skip *= states_skip[mult - 1]
                cis_file.read(skip)

            # Convert determinant lists to strings
            strings = {}
            for mult in mults:
                filename = f"dets{filelabel}.{mult}"
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
        self.QMin.control["jobgrad"] = jobgrad

        schedule = [{}]

        # add the master calculations
        ntasks = len([1 for g in gradjob if "master" in g])
        _, nslots, cpu_per_run = self.divide_slots(self.QMin.resources["ncpu"], ntasks, self.QMin.resources["schedule_scaling"])
        self.QMin.control["nslots_pool"].append(nslots)

        for idx, job in enumerate(sorted(gradjob)):
            if not "master" in job:
                continue
            qmin = deepcopy(self.QMin)
            qmin.control["master"] = True
            qmin.control["jobid"] = int(job.split("_")[1])
            qmin.resources["ncpu"] = cpu_per_run[idx]
            qmin.maps["gradmap"] = set(gradjob[job])
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
        states_to_do = deepcopy(qmin.control["states_to_do"])
        for imult, _ in enumerate(states_to_do):
            if not imult + 1 in qmin.maps["multmap"][-job]:
                states_to_do[imult] = 0
        states_to_do[gsmult - 1] -= 1

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

        # CPU cores
        if qmin.resources["ncpu"] > 1:
            string += f"%pal\n\tnprocs {qmin.resources['ncpu']}\nend\n\n"
        string += f"%maxcore {qmin.resources['memory']}\n\n"

        # Basis sets + ECP basis set
        if qmin.template["basis_per_element"]:
            string += "%basis\n"
            # basis_per_element key is list, need to iterate pairwise
            for batch in batched(qmin.template["basis_per_element"]):
                string += f'\tnewgto {batch[0]} "{batch[1]}" end\n'
            if qmin.template["ecp_per_element"]:
                for batch in batched(qmin.template["ecp_per_element"]):
                    string += f'\tnewECP {batch[0]} "{batch[1]}" end\n'
            string += "end\n\n"

        # Frozen core
        string += f"%method\n\tfrozencore {-2*qmin.molecule['frozcore'] if qmin.molecule['frozcore'] >0 else 'FC_NONE'}\nend\n\n"

        # HF exchange
        if qmin.template["hfexchange"] > 0:
            string += f"%method\n\tScalHFX = {qmin.template['hfexchange']}\nend\n\n"

        # Intacc
        if qmin.template["intacc"] > 0:
            string += f"%method\n\tintacc {qmin.template['intacc']:3.1f}\nend\n\n"

        # Gaussian point charges
        if qmin.template["keys"] and "cpcm" in qmin.template["keys"]:
            string += "%cpcm\n\tsurfacetype vdw_gaussian\nend\n\n"

        # Keep single points with and without gradients consistent
        if qmin.template["keys"] and "zora" in qmin.template["keys"]:
            string += "%rel\n\tonecenter true\nend\n\n"

        # Excited states
        if max(states_to_do) > 0:
            string += f"%tddft\n\ttda {'false' if qmin.template['no_tda'] else 'true'}\n"
            if qmin.requests["theodore"]:
                string += "\ttprint 0.0001\n"
            if restr and trip:
                string += "\ttriplets true\n"
            string += f"\tnroots {max(states_to_do)}\n"
            if restr and qmin.requests["soc"]:
                string += "\tdosoc true\n\tprintlevel 3\n"
            if do_grad:
                if singgrad:
                    string += "\tsgradlist " + ",".join([str(i) for i in sorted(singgrad)]) + "\n"
                if tripgrad:
                    string += "\ttgradlist " + ",".join([str(i) for i in sorted(tripgrad)]) + "\n"

            elif egrad:
                string += f"\tiroot {egrad[1] - (gsmult == egrad[0])}\n"
            if not qmin.requests["soc"] and qmin.template["alltrans"]:
                string += "\tDOTRANS ALL\n"
            string += "end\n\n"

        # Output
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
        string += f"\tmult {gsmult}\n"
        string += "\tcoords\n"
        for iatom, (label, coords) in enumerate(zip(qmin.molecule["elements"], qmin.coords["coords"]), 1):
            string += f"\t{label:4s} {coords[0]:16.9f} {coords[1]:16.9f} {coords[2]:16.9f}"
            if qmin.template["basis_per_atom"] and str(iatom) in qmin.template["basis_per_atom"]:
                idx = qmin.template["basis_per_atom"].index(str(iatom))
                string += f"\tnewgto \"{qmin.template['basis_per_atom'][idx+1]}\" end"
            string += "\n"
        string += "\tend\nend\n\n"

        # Point charges
        if qmin.molecule["point_charges"]:
            string += '%pointcharges "ORCA.pc"\n\n'

        # Paste input file
        if qmin.template["paste_input_file"]:
            with open(qmin.template["paste_input_file"], "r", encoding="utf-8") as paste:
                string += f"{paste.read()}\n"
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

    def get_readable_densities(self):
        return {}

    def read_and_append_densities(self):
        pass

    def read_dets_and_mos(self, directory, s, step):
        file = f"{directory}/dets_allelec.{s + 1}.{step}"
        if not os.path.isfile(file):
            file = f"{directory}/dets.{s + 1}.{step}"
        nst = np.loadtxt(file, usecols=(0,), max_rows=1, dtype=int)
        nst = int(nst)
        dets = np.loadtxt(file, usecols=(0,), skiprows=1, dtype=str, ndmin=1).tolist()
        ci = np.loadtxt(file, skiprows=1, usecols=list(range(1, nst + 1)), ndmin=2, dtype=float)
        mo_mult = s
        if not self.QMin.template["unrestricted_triplets"] and s == 2:
            mo_mult = 0
        file = f"{directory}/mos_allelec.{mo_mult + 1}.{step}"
        if not os.path.isfile(file):
            file = f"{directory}/mos.{mo_mult + 1}.{step}"
        nao = np.loadtxt(file, skiprows=5, max_rows=1, usecols=(0,), dtype=int)
        nmo = np.loadtxt(file, skiprows=5, max_rows=1, usecols=(1,), dtype=int)
        mos = np.zeros((nao, nmo))
        nr = nao // 3
        if nao % 3 != 0:
            nr += 1
        for i in range(nmo):
            mos[:, i] = np.concatenate(
                (
                    np.loadtxt(file, skiprows=9 + i * nr, max_rows=nr - 1).flatten(),
                    np.loadtxt(file, skiprows=9 + i * nr + nr - 1, max_rows=1, ndmin=1),
                )
            )
        mos = np.ascontiguousarray(mos)
        dets = np.char.replace(dets, old="d", new="7,")
        dets = np.char.replace(dets, old="a", new="5,")
        dets = np.char.replace(dets, old="b", new="1,")
        dets = np.char.replace(dets, old="e", new="-1,")
        dets = np.array([np.fromstring(i, dtype=int, sep=",") for i in dets])
        mos = mos[self.order]
        mos[self.ao_sign, :] *= -1
        return nst, dets, ci, mos


if __name__ == "__main__":
    SHARC_ORCA(loglevel=10).main()
