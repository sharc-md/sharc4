#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
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


# IMPORTS
# external
import datetime
import itertools
import os
import shutil
from copy import deepcopy
from io import TextIOWrapper
from typing import Optional

import numpy as np

# internal
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import InDir, expand_path, mkdir, question, writefile

version = "4.0"
versiondate = datetime.datetime(2025, 4, 1)

changelogstring = """
"""


__all__ = ["SHARC_NUMDIFF"]


def phase_correction(log, matrix: np.ndarray) -> tuple[np.ndarray]:
    """
    Do a phase correction of a matrix.
    Follows algorithm from J. Chem. Theory Comput. 2020, 16, 2, 835-846 (https://doi.org/10.1021/acs.jctc.9b00952)
    """
    phases = np.ones(matrix.shape[-1])
    U = matrix.real.copy()
    det_U = np.linalg.det(U)
    if det_U < 0:
        U[:, 0] *= -1.0  # this row/column convention is correct
        phases[0] *= -1.0
    U_sq = U * U

    # sweeps
    length = len(U)
    sweeps = 0
    done = False
    while not done:
        done = True
        for j in range(length):
            for k in range(j + 1, length):
                delta = 3.0 * (U_sq[j, j] + U_sq[k, k])
                delta += 6.0 * U[j, k] * U[k, j]
                delta += 8.0 * (U[k, k] + U[j, j])
                delta -= 3.0 * (U[j, :] @ U[:, j] + U[k, :] @ U[:, k])

                # Test if delta < 0
                num_zero_thres = -1e-15  # needs proper threshold towards 0
                if delta < num_zero_thres:
                    U[:, j] *= -1.0  # this row/column convention is correct
                    U[:, k] *= -1.0  # this row/column convention is correct
                    phases[j] *= -1.0
                    phases[k] *= -1.0
                    done = False
        sweeps += 1

    log.debug(f"Finished phase correction after {sweeps} sweeps.")

    return U, phases


def loewdin_orthonormalization(A):
    """
    Do Loewdin orthonormalization of a matrix.
    """
    S = A.T @ A
    eigenvals, eigenvecs = np.linalg.eigh(S)
    idx = eigenvals > 1e-15
    S_sqrt = np.dot(
        eigenvecs[:, idx] / np.sqrt(eigenvals[idx]), eigenvecs[:, idx].conj().T
    )
    A_ortho = A @ S_sqrt

    # Normalize the matrix
    A_lo = A_ortho.T
    length = len(A_lo)
    A_lon = np.zeros((length, length))

    for i in range(length):
        norm_of_col = np.linalg.norm(A_lo[i])
        A_lon[i] = [e / (norm_of_col**0.5) for e in A_lo[i]]

    return A_lon.T


def post_process_overlap_matrix(log, overlap_matrix):
    """
    Process an overlap matrix to ensure that it has correct phases
    and in orthonormal.
    """
    # First fix phases
    phase_corrected_overlap, phases = phase_correction(log, overlap_matrix)

    # Do a Loewdin orthonormalization
    orthogonal_overlap = loewdin_orthonormalization(phase_corrected_overlap)

    # Extra phase correction (probably not needed)
    final, phases2 = phase_correction(log, orthogonal_overlap)
    return final, phases * phases2


class SHARC_NUMDIFF(SHARC_HYBRID):

    _version = version
    _versiondate = versiondate
    _changelogstring = changelogstring

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Update resource keys
        self.QMin.resources.update(
            {
                "use_all_cores_for_ref": True,
            }
        )
        self.QMin.resources.types.update(
            {
                "use_all_cores_for_ref": bool,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "qm-program": None,
                "qm-dir": None,  # that's where the QM template/resource are
                "numdiff_method": "central-diff",  # or "central-quad"
                "numdiff_representation": "adiabatic",  # or "diabatic"
                "numdiff_stepsize": 0.01,  # TODO: should be a list of displacements per-DOF
                "coord_type": "cartesian",  # or 'displacement' -> 'normal_modes'
                "normal_modes_file": None,
                "whitelist": [],
            }
        )
        self.QMin.template.types.update(
            {
                "qm-program": str,
                "qm-dir": str,
                "numdiff_method": str,
                "numdiff_representation": str,
                "numdiff_stepsize": float,
                "coord_type": str,
                "normal_modes_file": None,
                "whitelist": list,
            }
        )

        self.do_numerically = None
        self.do_numerically_now = None
        self.ref_interface: SHARC_INTERFACE = None

    @staticmethod
    def authors() -> str:
        return "Nicolai Machholdt Høyer and Sebastian Mai"

    @staticmethod
    def version():
        return SHARC_NUMDIFF._version

    @staticmethod
    def versiondate():
        return SHARC_NUMDIFF._versiondate

    @staticmethod
    def name() -> str:
        return "NUMDIFF"

    @staticmethod
    def description():
        return (
            "   HYBRID interface for numerical derivatives (grad, NACdr, SOCdr, DMdr)"
        )

    @staticmethod
    def changelogstring():
        return SHARC_NUMDIFF._changelogstring

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None):
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        # TODO: if not self._read_template then ask for path and read the template
        # TODO: instantiate reference child and get features
        # check what differentiation is possible and add to features
        # remember which requests require differentiation and which are done by reference child

        # if in setup mode, ask for the template file, read it, and instantiate a child to get features
        # if not in setup mode, reference child is already there
        if not self._read_template:
            self.template_file = question(
                "Please specify the path to your NUMDIFF.template file",
                str,
                KEYSTROKES=KEYSTROKES,
                default="NUMDIFF.template",
            )
            self.read_template(self.template_file)
        if not hasattr(self, "ref_interface"):
            qm_program = self.QMin.template["qm-program"]
            self.ref_interface = self._load_interface(qm_program)()
            # self.ref_interface.QMin.molecule['states'] = self.QMin.molecule['states']
            if isinstance(self.ref_interface, SHARC_HYBRID):
                self.log.error(
                    "Currently, Hybrid interfaces cannot be used as children of SHARC_NUMDIFF.py"
                )
                raise NotImplementedError

        ref_features = self.ref_interface.get_features(KEYSTROKES=KEYSTROKES)
        needed = {
            "grad": set(["h"]),
            "socdr": set(["soc"]),
            "dmdr": set(["dm"]),
            "nacdr": set(["h", "overlap"]),
        }
        if self.QMin.template["numdiff_representation"] == "diabatic":
            for i in needed:
                needed[i].add("overlap")
        possible = set()
        for i in needed:
            if all([j in ref_features for j in needed[i]]):
                possible.add(i)
        qm_features = ref_features.union(possible)

        # NUMDIFF cannot displace point charges
        not_supported = {"point_charges"}
        qm_features -= not_supported

        # Make QM features into a set a return these
        self.log.debug(qm_features)  # log features
        return set(qm_features)

    def get_infos(self, INFOS, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """communicate requests from setup and asks for additional paths or info

        The `INFOS` dict holds all global informations like paths to programs
        and requests in `INFOS['needed_requests']`

        all interface specific information like additional files etc should be stored
        in the interface intance itself.

        use the `question()` function from the `utils` module and write the answers
        into `KEYSTROKES`

        Parameters:
        ---
        INFOS
            dict[str]: dictionary with all previously collected infos during setup
        KEYSTROKES
            str: object as returned by open() to be used with question()
        """
        # Setup some output to log
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'NUMDIFF interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        # Interactive setting of file
        # TODO: add options
        if question(
            "Do you have an NUMDIFF.resources file?",
            bool,
            KEYSTROKES=KEYSTROKES,
            autocomplete=False,
            default=False,
        ):
            self.resources_file = question(
                "Specify path to NUMDIFF.resources",
                str,
                KEYSTROKES=KEYSTROKES,
                autocomplete=True,
            )
        else:
            self.log.info(f"{'NUMDIFF Ressource usage':-^60}\n")
            self.log.info(
                """Please specify the number of CPUs to be used by EACH trajectory.
        """
            )
            self.setupINFOS["ncpu_numdiff"] = abs(
                question("Number of CPUs:", int, KEYSTROKES=KEYSTROKES)[0]
            )
            self.setupINFOS["scratchdir_numdiff"] = question(
                "Path to scratch directory:", str, KEYSTROKES=KEYSTROKES
            )
            # self.setupINFOS["scratchdir_numdiff"] += '/$$/'

            # TODO: could use schedule scaling and Amdahl, but SHARC_HYBRID does not have it

        # if we need overlaps, we need to modify the INFOS['needed_requests'] to tell children to prepare for that
        needed_copy = deepcopy(INFOS["needed_requests"])
        if (
            self.QMin.template["numdiff_representation"] == "diabatic"
            or "nacdr" in INFOS["needed_requests"]
        ):
            INFOS["needed_requests"].add("overlap")

        # Get the infos from the child
        self.log.info(f"{' Setting up QM-interface ':=^80s}\n")
        self.ref_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        # reset the needed requests
        INFOS["needed_requests"] = needed_copy

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        """
        prepares the folder for an interface calculation

        Parameters
        ----------
        INFOS
            dict[str]: dictionary with all infos from the setup script
        dir_path
            str: *relative* path to the directory to setup (can be appended to `scratchdir`)
        """
        # Copy files to the nummdiff dir
        shutil.copy(
            self.template_file, os.path.join(dir_path, self.name() + ".template")
        )
        # shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".resources"))

        # write resource file
        string = "ncpu %i\nscratchdir %s/%s\nuse_all_cores_for_ref True\n" % (
            self.setupINFOS["ncpu_numdiff"],
            self.setupINFOS["scratchdir_numdiff"],
            dir_path,
        )
        writefile(os.path.join(dir_path, self.name() + ".resources"), string)

        # Setup sub-dir for the QM calcs
        if self.QMin.template["qm-dir"] is None:
            raise ValueError("Keyword 'qm-dir' not found in template file!")
        qmdir = dir_path + f"/{self.QMin.template['qm-dir']}"
        mkdir(qmdir)

        # Make savedir and scratchdir for the reference interface
        if not self.QMin.save["savedir"]:
            self.log.warning(
                "savedir not specified in QM.in, setting savedir to current directory!"
            )
            self.QMin.save["savedir"] = os.getcwd()

        ref_savedir = os.path.join(
            dir_path,
            self.QMin.save["savedir"],
            "QM_" + self.QMin.template["qm-program"].upper(),
        )
        self.log.debug(f"ref_savedir {ref_savedir}")
        if not os.path.isdir(ref_savedir):
            mkdir(ref_savedir)

        ref_scratchdir = os.path.join(
            self.QMin.resources["scratchdir"],
            "QM_" + self.QMin.template["qm-program"].upper(),
        )
        self.log.debug(f"ref_scratchdir {ref_scratchdir}")
        if not os.path.isdir(ref_scratchdir):
            mkdir(ref_scratchdir)

        self.ref_interface.QMin.save["savedir"] = ref_savedir
        self.ref_interface.QMin.resources["scratchdir"] = ref_scratchdir

        # if we need overlaps, we need to modify the INFOS['needed_requests'] to tell children to prepare for that
        needed_copy = deepcopy(INFOS["needed_requests"])
        if (
            self.QMin.template["numdiff_representation"] == "diabatic"
            or "nacdr" in INFOS["needed_requests"]
        ):
            INFOS["needed_requests"].add("overlap")

        # Call prepare for the reference interface
        self.ref_interface.prepare(INFOS, qmdir)

        # reset the needed requests
        INFOS["needed_requests"] = needed_copy

        return

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def read_template(self, template_file="NUMDIFF.template") -> None:
        super().read_template(template_file)

        # If we use normal mode coordinates we read them in here
        if self.QMin.template["coord_type"] == "normal_modes":
            self.read_displacement_coordinates(
                os.path.abspath(self.QMin.template["normal_modes_file"])
            )
            # TODO: change to V0.txt format!

        self.QMin.template["numdiff_stepsize"] = float(
            self.QMin.template["numdiff_stepsize"]
        )

    def read_resources(self, resources_file="NUMDIFF.resources") -> None:
        super().read_resources(resources_file)

    def read_displacement_coordinates(self, disp_coord_filename):
        # TODO: replace with V0.txt format
        # Need to decide what to do if the current geometry does not match with the reference geometry
        # I guess we should displace in Q and then transform...

        # Read the coord file
        disp_coords = []
        with open(disp_coord_filename, "r") as f:
            for line in f:
                if "units" in line:
                    line = f.readline()
                if "normal modes" in line:
                    line = f.readline()
                    n_coords = int(line)
                    for i_coord in range(n_coords):
                        line = f.readline()
                        line = line.split()
                        disp_coords.append(
                            np.resize(
                                np.asarray(line, dtype=float),
                                (self.QMin.molecule["natom"], 3),
                            )
                        )

        self.QMin.disp_coords = disp_coords
        return

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def setup_interface(self) -> None:
        """
        Prepare the interface for calculations
        """
        # --- reference child ---

        # paths
        self.QMin.resources["scratchdir"] = expand_path(
            self.QMin.resources["scratchdir"]
        )

        if self.QMin.template["qm-dir"] is None:
            raise ValueError("Keyword 'qm-dir' not found in template file!")
        qmdir = expand_path(self.QMin.template["qm-dir"])
        qm_program = self.QMin.template["qm-program"]

        # Create reference child
        ref_logname = f"Reference:{qm_program}"
        pwd = os.path.join(self.QMin.resources["scratchdir"], "PWD", "reference")

        self.ref_interface = self._load_interface(qm_program)(
            logname=ref_logname,
            loglevel=self.log.level,
            persistent=False,
            fast_queue=True,
        )

        # do setup molecule
        local_QMin = deepcopy(self.QMin)
        local_QMin.molecule["factor"] = 1.0
        local_QMin.molecule["unit"] = "bohr"
        self.ref_interface.setup_mol(local_QMin)

        ## then do setup_mol/template/resources
        self.log.info(f"Setting up reference child ...")
        with InDir(qmdir):
            self.ref_interface.read_resources()
            self.ref_interface.read_template()
            ## reassign scratchdir and savedir
            scratchdir = os.path.join(
                self.QMin.resources["scratchdir"], "SCRA", "reference"
            )
            savedir = os.path.join(self.QMin.save["savedir"], "children", "reference")
            self.ref_interface.QMin.resources["scratchdir"] = scratchdir
            self.ref_interface.QMin.save["savedir"] = savedir
            self.ref_interface.QMin.resources["pwd"] = pwd
            self.ref_interface.QMin.resources["cwd"] = pwd
            if self.QMin.resources["use_all_cores_for_ref"]:
                self.ref_interface.QMin.resources["ncpu"] = self.QMin.resources["ncpu"]
            self.ref_interface.setup_interface()

        # --- kindergarden ---

        # figure out how many children we have in the kindergarden and make labels
        labels = []
        match self.QMin.template["coord_type"]:
            case "cartesian":
                labels.append(["cartesian"])
                labels.append(list(range(self.QMin.molecule["natom"])))
                labels.append(["x", "y", "z"])
            case "normal_modes":
                labels.append(["normal_modes"])
                labels.append(list(range(len(self.QMin.disp_coords))))
                raise NotImplementedError
            case _:
                raise RuntimeError(
                    f"Input 'coord_type': {self.QMin.template['coord_type']} is not valid."
                )
        match self.QMin.template["numdiff_method"]:
            case "central-diff":
                labels.append(["p", "n"])
            case "central-quad":
                labels.append(["pp", "p", "n", "nn"])
            case _:
                raise RuntimeError(
                    f"Input 'numdiff_method': {self.QMin.template['numdiff_method']} is not valid."
                )

        # make full labels as direct product of the labels:
        full_labels = list(itertools.product(*labels))

        # make child_dict: define logfiles
        child_dict = {}
        for label in full_labels:
            name = "_".join(str(i) for i in label)
            pwd = os.path.join(self.QMin.resources["scratchdir"], "PWD", name)
            logname = f"Displacement:{qm_program}:{name}"
            child_dict[label] = (
                qm_program,
                [],
                {
                    "logname": logname,
                    "loglevel": self.log.level,
                    "persistent": False,
                },
            )
        self.instantiate_children(child_dict)

        # do full setup for all children
        for label, child in self._kindergarden.items():
            name = "_".join(str(i) for i in label)
            self.log.info(f"Setting up displaced child {label} ...")
            child.setup_mol(local_QMin)
            with InDir(qmdir):
                child.read_resources()
                child.read_template()
                # scratch
                scratchdir = os.path.join(
                    self.QMin.resources["scratchdir"], "SCRA", name
                )
                child.QMin.resources["scratchdir"] = scratchdir
                # save
                savedir = os.path.join(self.QMin.save["savedir"], "children", name)
                child.QMin.save["savedir"] = savedir
                # pwd
                pwd = os.path.join(self.QMin.resources["scratchdir"], "PWD", name)
                child.QMin.resources["pwd"] = pwd
                child.QMin.resources["cwd"] = pwd
                child.setup_interface()
            if not os.path.isfile(os.path.join(self.QMin.save["savedir"], "STEP")):
                child.read_requests({"step": 0})
                child.write_step_file()

        # --- feature setup ---

        # get possible features
        ref_features = self.ref_interface.get_features()
        own_features = self.get_features()
        self.log.info(ref_features)
        self.log.info(own_features)
        self.do_numerically = set()
        check_these = ["grad", "socdr", "dmdr", "nacdr"]
        for i in check_these:
            if i in self.QMin.template["whitelist"]:
                if i in ref_features:
                    self.log.info(
                        f"Request {i} white-listed and available from child, will be passed to reference"
                    )
                else:
                    self.do_numerically.add(i)
                    self.log.info(
                        f"Request {i} white-listed but not available from child, will be done numerically"
                    )
            elif i in own_features:
                self.do_numerically.add(i)
                self.log.info(f"Request {i} will be done numerically")
            else:
                self.log.info(f"Request {i} not available")

    def read_requests(self, requests_file="QM.in"):
        super().read_requests(requests_file)

        all_requests = {k: v for (k, v) in self.QMin.requests.items() if v is not None}
        self.log.debug(f"{all_requests}")
        ref_requests = {"step": self.QMin.save["step"], "h": True}
        num_requests = {
            "step": self.QMin.save["step"] + 1,
            "h": True,
            "overlap": self.QMin.template["numdiff_representation"] == "diabatic",
        }
        self.log.debug(f"num_requests before loop: {num_requests}")

        self.do_numerically_now = set()
        for k, v in all_requests.items():
            if k in self.do_numerically:
                match k:
                    case "grad":
                        self.do_numerically_now.add(k)
                        # h and overlaps are already included
                    case "dmdr":
                        if v:
                            self.do_numerically_now.add(k)
                            num_requests["dm"] = True
                            self.log.debug(f"Adding 'dm' due to {k} and {self.do_numerically}")
                    case "socdr":
                        if v:
                            self.do_numerically_now.add(k)
                            num_requests["soc"] = True
                            self.log.debug(f"Adding 'soc' due to {k} and {self.do_numerically}")
                    case "nacdr":
                        if v:
                            self.do_numerically_now.add(k)
                            num_requests["overlap"] = True
                            self.log.debug(f"Adding 'overlap' due to {k} and {self.do_numerically}")
            else:
                ref_requests[k] = v

        self.log.debug(f"Passing the following requests to the reference child: {ref_requests}")
        self.ref_interface.read_requests(ref_requests)
        for child in self._kindergarden.values():
            self.log.debug(f"Passing the following requests to displaced child {child}: {num_requests}")
            child.read_requests(num_requests)

    def set_coords(self, xyz, pc=False):
        if pc:
            raise NotImplementedError("Numdiff interface cannot deal with point charges!")
        super().set_coords(xyz, pc)
        
        coords = self.QMin.coords["coords"].copy()
        self.ref_interface.set_coords(coords, pc)

        if self.do_numerically:
            cart_directions = {"x": 0, "y": 1, "z": 2}
            displacements = {
                "pp": +2.0,
                "p": +1.0,
                "n": -1.0,
                "nn": -2.0,
            }  # for other differentiation than central, new labels are needed, e.g., "pp" or "nn"
            for label, child in self._kindergarden.items():
                coords = self.QMin.coords["coords"].copy()
                match label[0]:
                    case "cartesian":
                        iatom, idir, idisp = label[1:4]
                        idir = cart_directions[idir]
                        idisp = displacements[idisp]
                        coords[iatom, idir] += (
                            idisp * self.QMin.template["numdiff_stepsize"]
                        )
                    case "normal_modes":
                        raise NotImplementedError
                child.set_coords(coords)

    def write_step_file(self):
        super().write_step_file()
        self.ref_interface.write_step_file()

    def run(self) -> None:
        # --- reference child calculation ---

        # run the child
        with InDir(self.ref_interface.QMin.resources["pwd"]):
            self.log.info("Running reference child ...")
            self.ref_interface.run()
            self.ref_interface.getQMout()

        if self.do_numerically_now:
            # take savedir of reference child and copy to all displaced children and set step
            for child in self._kindergarden.values():
                if child.persistent:
                    child.savedict = deepcopy(self.ref_interface.savedict)
                else:
                    # copy all files from reference child to displaced child
                    shutil.copytree(
                        self.ref_interface.QMin.save["savedir"],
                        child.QMin.save["savedir"],
                        dirs_exist_ok=True,
                    )

            # run the children
            t1 = datetime.datetime.now()
            self.log.info(f"Running displaced children with {self.QMin.resources['ncpu']} CPU cores ...")
            self.log.info("\nSTART:\t%s" % (t1))
            self.run_children(self.log, self._kindergarden, self.QMin.resources["ncpu"])
            t2 = datetime.datetime.now()
            self.log.info("FINISH:\t%s\tRuntime: %s\n" % (t2, t2 - t1))

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def getQMout(self) -> dict[str, np.ndarray]:
        """
        Return QMout object
        """

        # create empty QMout
        self.QMout.allocate(
            self.QMin.molecule["states"],
            self.QMin.molecule["natom"],
            self.QMin.molecule["npc"],
            self.QMin.requests,
        )

        # Set QMout with stuff from the reference calculation
        for key, val in self.ref_interface.QMout.items():
            self.QMout[key] = deepcopy(val)
        self.QMout.charges = [
            0 for i in self.QMin.molecule["states"]
        ]  # TODO: remove later

        # do all the numerical requests
        if self.do_numerically_now:

            # make phase corrections if overlaps are present
            any_child = self._kindergarden[next(iter(self._kindergarden))]
            if any_child.QMin.requests["overlap"]:
                self.log.info("Doing phase correction ...")
                for child in self._kindergarden.values():
                    child.QMout["overlap"], phases = post_process_overlap_matrix(
                        self.log, child.QMout["overlap"]
                    )
                    phases2 = phases[:, None] * phases[None, :]
                    if child.QMout["h"] is not None:
                        child.QMout["h"] *= phases2
                    if child.QMout["dm"] is not None:
                        child.QMout["dm"] *= phases2[None, :, :]

            # preparation
            cart_directions = {"x": 0, "y": 1, "z": 2}
            displacements = {"pp": +2.0, "p": +1.0, "n": -1.0, "nn": -2.0}
            stepsize = self.QMin.template["numdiff_stepsize"]
            nstates = self.QMin.molecule["nmstates"]

            # compute derivatives
            match self.QMin.template["coord_type"]:
                case "cartesian":
                    for iatom in range(self.QMin.molecule["natom"]):
                        for idir in ["x", "y", "z"]:

                            # pick the involved children for this direction
                            # the run() function would decide how many children per direction, depending on numdiff_method
                            # and here we only pick those that correspond to the current direction
                            children = {}
                            for label, child in self._kindergarden.items():
                                if ("cartesian", iatom, idir) == label[:-1]:
                                    children[label[-1]] = child

                            # the trafo matrices for this direction
                            match self.QMin.template["numdiff_representation"]:
                                case "adiabatic":
                                    S = {}
                                    for label in children:
                                        S[label] = np.identity(nstates)
                                case "diabatic":
                                    S = {}
                                    for label, child in children.items():
                                        S[label] = child.QMout["overlap"]

                            # go through the requests for this direction
                            for request in self.do_numerically_now:
                                # self.log.info(f"{request}")

                                # pick quantity for this request
                                match request:
                                    case (
                                        "grad"
                                    ):  # differentiate the energies, i.e., the diagonal elements of the Hamiltonian
                                        A = {}
                                        for label, child in children.items():
                                            A[label] = np.diag(np.diag(child.QMout["h"]))
                                    case (
                                        "socdr"
                                    ):  # differentiate the off-diagonal elements of the Hamiltonian
                                        A = {}
                                        for label, child in children.items():
                                            A[label] = child.QMout["h"] - np.diag(np.diag(child.QMout["h"]))
                                    case (
                                        "dmdr"
                                    ):  # differentiate the dipole moment matrix
                                        A = {}
                                        for label, child in children.items():
                                            A[label] = child.QMout["dm"]
                                    case "nacdr":  # NACs are a bit more complicated
                                        match self.QMin.template["numdiff_representation"]:
                                            case (
                                                "adiabatic"
                                            ):  # differentiate the overlap matrix elements
                                                A = {}
                                                for label, child in children.items():
                                                    A[label] = child.QMout["overlap"]
                                            case (
                                                "diabatic"
                                            ):  # differentiate the diabatized diagonal of the Hamiltonian
                                                A = {}
                                                for label, child in children.items():
                                                    A[label] = np.diag(np.diag(child.QMout["h"]))

                                # make the transformation and differentiation for this request and direction
                                # if other differentiation schemes will be implemented, here one can simply
                                # use them based on their labels. In this way, this is the only block that depends on numdiff_method
                                match self.QMin.template["numdiff_method"]:
                                    case "central-diff":
                                        numerator = (
                                            S["p"].T @ A["p"] @ S["p"]
                                            - S["n"].T @ A["n"] @ S["n"]
                                        )
                                        denomimator = stepsize * (
                                            displacements["p"] - displacements["n"]
                                        )
                                        result = numerator / denomimator
                                    case "central-quad":
                                        numerator = (
                                            -S["pp"].T @ A["pp"] @ S["pp"]
                                            + 8.0 * S["p"].T @ A["p"] @ S["p"]
                                            - 8.0 * S["n"].T @ A["n"] @ S["n"]
                                            + S["nn"].T @ A["nn"] @ S["nn"]
                                        )
                                        denomimator = (
                                            6.0
                                            * stepsize
                                            * (displacements["p"] - displacements["n"])
                                        )
                                        result = numerator / denomimator
                                    case _:
                                        raise NotImplementedError(
                                            "Only central differences implemented"
                                        )

                                # assign correctly the resulting request elements
                                match request:
                                    case (
                                        "grad"
                                    ):  # QMout['grad'] has shape (nstates, natom, 3)
                                        self.QMout["grad"][
                                            :, iatom, cart_directions[idir]
                                        ] = np.diag(result)
                                    case (
                                        "socdr"
                                    ):  # QMout['socdr'] has shape (nstates, nstates, natom, 3)
                                        self.QMout["socdr"][
                                            :, :, iatom, cart_directions[idir]
                                        ] = result
                                    case (
                                        "dmdr"
                                    ):  # QMout['socdr'] has shape (3, nstates, nstates, natom, 3)
                                        self.QMout["dmdr"][
                                            :, :, :, iatom, cart_directions[idir]
                                        ] = result
                                    case (
                                        "nacdr"
                                    ):  # QMout['socdr'] has shape (nstates, nstates, natom, 3)
                                        match self.QMin.template[
                                            "numdiff_representation"
                                        ]:
                                            case "adiabatic":
                                                # we only make properly anti-Hermitian
                                                self.QMout["nacdr"][
                                                    :, :, iatom, cart_directions[idir]
                                                ] = (result - result.T) / 2.0
                                            case "diabatic":
                                                # result contains dH/dR. To get the NAC, we need to scale by the energy gaps
                                                result = (
                                                    result + result.T
                                                ) / 2.0  # maybe unnecessary
                                                E = np.diag(self.QMout["h"])
                                                denominator = E[:, None] - E[None, :]
                                                denominator[
                                                    np.diag_indices_from(denominator)
                                                ] = np.inf
                                                denominator[denominator == 0.0] = np.inf
                                                self.QMout["nacdr"][
                                                    :, :, iatom, cart_directions[idir]
                                                ] = (result / denominator)
                case "normal_modes":
                    raise NotImplementedError("Normal mode displacements not allowed")
                    # TODO: probably the same as for Cartesian, but afterwards we have to do a coordinate transformation of all derivatives
                    # because the interface should always return Cartesian derivatives
                case _:
                    raise NotImplementedError("Only Cartesian displacements allowed")

        self.QMout.runtime = self.clock.measuretime(False)
        return self.QMout


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


if __name__ == "__main__":
    from logger import loglevel

    try:
        num_diff = SHARC_NUMDIFF(loglevel=loglevel)
        num_diff.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
