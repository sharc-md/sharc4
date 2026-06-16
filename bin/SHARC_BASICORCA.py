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

import datetime
import os
from io import TextIOWrapper
from copy import deepcopy
import re
import struct
import shutil
import sys
import subprocess as sp

import numpy as np
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import batched, expand_path, itmult, mkdir, question, readfile, writefile, link


# ---------------------------------| Infos |---------------------------------------------------------------------------

#TODO: Change BASICORCA to your desired name

__all__ = ["SHARC_BASICORCA"]  # Only export interface class


#TODO: This will be shown in the header when running a single point or sharc.x
AUTHORS = "Hans Georg Gallmetzer"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2025, 11, 6)
#TODO: This will be shown in the setup scripts
NAME = "BASICORCA"
DESCRIPTION = "AB INITIO a very simple ORCA interface as an example of an ab initio interface"

CHANGELOGSTRING = """17.06.2024:     Initial version 0.1 by Sascha and Georg
- Only energies, TDMs and gradients
- Only singlets

25.07.25:     Refined and adapted to SHARC4.0 by Georg

06.11.25:     Release-ready Version 1.0 by Georg
- Bugfixes"""

all_features = set(
    [
        #TODO: requests that your interface can fullfill. Delete the ones that cannot be used. 
        "h",
        "dm",
        "grad",
        # Rest of the possible requests:
        # "molden",
        # "nacdr",
        # "overlap",
        # "phases",
        # "soc",
        # "ion",
        # "theodore",
    ]
)


class SHARC_BASICORCA(SHARC_ABINITIO):
    """
    Doc string of your interface
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


# ---------------------------------| Template/Resources Definition |----------------------------------------------------

        self._need_this_later = None

        self.QMin.resources.update(
            {
                "orcadir": None, # Path to the executable of the QC-program
            }
        )
        self.QMin.resources.types.update(
            {
                "orcadir": str,
            }
        )

        self.QMin.template.update(
            {
                "basis": "6-31G",
                "functional": "PBE",
            }
        )
        self.QMin.template.types.update(
            {
                "basis": str,
                "functional": str,
            }
        )


# ---------------------------------| Standard Methods |------------------------------------------------------------

    @staticmethod
    def version() -> str:
        return SHARC_BASICORCA._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_BASICORCA._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_BASICORCA._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_BASICORCA._authors

    @staticmethod
    def name() -> str:
        return SHARC_BASICORCA._name

    @staticmethod
    def description() -> str:
        return SHARC_BASICORCA._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_BASICORCA._name}\n{SHARC_BASICORCA._description}"


# ---------------------------------| Initialization |------------------------------------------------------------------

    def read_template(self, template_file: str = "BASICORCA.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        #TODO: Validate and/or process custom template keys here

    def read_resources(self, resources_file: str = "BASICORCA.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        #TODO: Validate and/or process custom resources keys here

    def setup_interface(self) -> None:
        super().setup_interface()
        self.QMin.control["states_to_do"] = deepcopy(self.QMin.molecule["states"])
        self.QMin.resources["orca_version"] = self.get_orca_version(self.QMin.resources["orcadir"])
        if not (4,9999) < self.QMin.resources["orca_version"] < (5,999):
            raise RuntimeError("The BASICORCA interface supports only ORCA 5. ORCA 6 can be used with SHARC_ORCA.py")
        if any(x != 0 for x in self.QMin.molecule["states"][1:]):
            raise RuntimeError("The BASICORCA interface can only compute singlet states. Use SHARC_ORCA.py for other multiplicities.")

        #TODO: Setup stuff that needs to be done after read_template and read_resources


# ---------------------------------| Run |---------------------------------------------------------------------
# and if needed WFoverlap/Theodore + Care for Restart Information

    def run(self) -> None:
        starttime = datetime.datetime.now()
        qmin = self.QMin
        input_str = self.generate_inputstr(self.QMin)

        # Setup workdir
        workdir = self.QMin.resources["scratchdir"]
        mkdir(workdir)

        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "BASICORCA.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        # Setup ORCA
        
        exec_str = f"{os.path.join(qmin.resources['orcadir'],'orca')} BASICORCA.inp"
        exit_code = self.run_program(workdir, exec_str, os.path.join(workdir, "BASICORCA.log"), os.path.join(workdir, "BASICORCA.err"))

        #TODO: Errorhandling in case exit_code != 0
        if (exit_code != 0):
            self.log.error(f"ORCA execution with {exec_str} failed!")
            sys.exit(1)
        
        #TODO: Post processing, molden file, wfoverlap det/mo files, ...

        #TODO: Copy restart files to savedir
        self._save_files(workdir)

        #TODO: If you need more calculation runs in order to get all of the necessary data 
        # you can use scheduling
        # #HINT: If no schduling is needed then do this:
        # # schedule = [{"calc" : self.QMin}] #Generate fake schedule
        # # self.QMin.control["nslots_pool"].append(1)
        # # self.runjobs(schedule)

        # #TODO: Build schedule executed by runjobs here
        

        # # Execute schedule, execute_from_qmin will be run inside runjobs
        # self.runjobs(self.QMin.scheduling["schedule"])

        # #TODO: Save files that you need to keep after program execution.

        # Run overlap calc here if needed
        if self.QMin.requests["overlap"]:
            self._run_wfoverlap()
        
        #TODO: ion/dyson calc and everything that has to be done after the actual QM calc

        self.log.debug("All jobs finished successful")

    def create_restart_files(self) -> None:
            pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

        #TODO: Do some request related checks here. Only important for hybrid interfaces.

    def set_coords(self, coords_file: str = "QM.in") -> None:
        super().set_coords(coords_file)

        #TODO: Nothing to do here, this method just update the coordinates.


# ---------------------------------| Scheduling |---------------------------------------------------------------------
# Generate schedule if needed

    def _gen_schedule(self) -> None:
        """
        Generates scheduling from joblist
        """
        pass

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Do QM calculation
        will be called in SHARC_ABINITIO.runjobs()
        """

        # Setup workdir
        mkdir(workdir)

        input_str = ""
        
        input_str += self.generate_inputstr(qmin)
        
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "BASICORCA.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        #TODO: Copy restart from savedir and input files needed for calculation here


        self._copy_gbw(qmin, workdir)

        # Setup ORCA
        starttime = datetime.datetime.now()
        exec_str = f"{os.path.join(qmin.resources['orcadir'],'orca')} BASICORCA.inp"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "BASICORCA.log"), os.path.join(workdir, "BASICORCA.err")
        )
        endtime = datetime.datetime.now()

        #TODO: Maybe some errorhandling in case exit_code != 0
        if exit_code != 0:
            self.log.error(f"ORCA execution with {exec_str} failed!")
            return exit_code, endtime - starttime
        
        #TODO: Post processing, molden file, wfoverlap det/mo files, ...

        #TODO: Copy restart files to savedir
        if exit_code == 0:
            # Save files
            if not qmin.save["samestep"]:
                self._save_files(workdir, qmin.control["jobid"])

            # Delete files not needed
            work_files = os.listdir(workdir)
            for file in work_files:
                if not re.search(r"\.log$|\.cis$|\.engrad|A\.err$|\.molden\.input$|\.gbw$|\.pc$|\.pcgrad.*$", file):
                    os.remove(os.path.join(workdir, file))

        
        return exit_code, endtime - starttime


# ---------------------------------| Get Data |-----------------------------------------------------------------------

    def getQMout(self) -> None:
        #TODO: Parse requested properties from outputs and populate QMout object. You can make as many parsing and helper functions as you want.
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
            requests=requests,
        )

        scratchdir = self.QMin.resources["scratchdir"]
        with open(os.path.join(scratchdir, f"BASICORCA.log"), "r", encoding="utf-8") as file:
            log_file = file.read()

            energies = self._get_energy(log_file)
            for i in range(sum(self.QMin.molecule["states"])):
                self.QMout["h"][i][i] = energies[(1, i + 1)]

            if self.QMin.requests["dm"]:
                states_to_do = deepcopy(self.QMin.control["states_to_do"])
                self.log.info("States to do: " + str(states_to_do))

                # Diagonal elements
                if states_to_do[0] > 1:
                    dipoles_trans = self._get_transition_dipoles(log_file)

                ex_state = list(self.QMin.maps["gradmap"])[0][1]
                states_to_do_max = max(states_to_do)-1
              
                for i in range(states_to_do_max):
                    self.QMout["dm"][:, 0, i+1] = dipoles_trans[i]
                    self.QMout["dm"][:, i+1, 0] = - dipoles_trans[i]

        if self.QMin.requests["grad"]:
            ex_state = list(self.QMin.maps["gradmap"])[0][1] - 1
            if (ex_state ==0):
                gradients = self._get_grad(os.path.join(scratchdir,f"BASICORCA.engrad.ground.grad.tmp"))
            else:
                gradients = self._get_grad(os.path.join(scratchdir,f"BASICORCA.engrad.singlet.root{ex_state}.grad.tmp"))

            self.QMout["grad"][ex_state] = gradients
        
        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout
    

# ---------------------------------| Setup Related |------------------------------------------------------------------

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
        self.log.info(f"||{'BASICORCA interface setup': ^76}||\n{'||':<78}||")
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

        if os.path.isfile("BASICORCA.template"):
            usethisone = question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                self.template_file = "BASICORCA.template"
        else:
            while True:
                self.template_file = question("Template filename:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(self.template_file):
                    self.log.info(f"File {self.template_file} does not exist!")
                    continue
                break
        self.log.info("")

        # Resources
        if question("Do you have a 'BASICORCA.resources' file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            while not os.path.isfile(
                (resources_file := question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="BASICORCA.resources"))
            ):
                self.log.info(f"file at {resources_file} does not exist!")
            self.resources_file = resources_file
        else:
            #Do nothing
            pass
             
        

        return INFOS


    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self.resources_file:
            with open(os.path.join(dir_path, "BASICORCA.resources"), "w", encoding="utf-8") as file:
                for key in (
                    "orcadir",
                    # "scratchdir",
                ):
                    if key in self.setupINFOS:
                        file.write(f"{key} {self.setupINFOS[key]}\n")
                if "scratchdir" in self.setupINFOS:
                    file.write(f"scratchdir {os.path.join(self.setupINFOS['scratchdir'], dir_path)}\n")
        else:
            create_file(expand_path(self.resources_file), os.path.join(dir_path, "BASICORCA.resources"))
        create_file(expand_path(self.template_file), os.path.join(dir_path, "BASICORCA.template"))


# ---------------------------------| Additional Methods |------------------------------------------------------------

#TODO: Put all of your extra methods in here. They all should start with and underscore "_". For example a method that parses the gradients from the output-file should be called _get_grad().

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

    @staticmethod
    def generate_inputstr(qmin: QMin) -> str:
        """
        Generate ORCA input file string from QMin object
        """
        job = qmin.control["jobid"]
        charge = qmin["molecule"]["charge"][0]  #Charge is provided by SHARC, no need to define it in the template file.

        # excited states to calculate
        states_to_do = deepcopy(qmin.control["states_to_do"])
        
        # gradients
        do_grad = False
        if qmin.requests["grad"] and qmin.maps["gradmap"]:
            do_grad = True


        string = "! "
        keys = ["basis", "functional"]
        string += " ".join(qmin.template[x] for x in keys if qmin.template[x] is not None)
        
        string += " engrad\n" if do_grad else "\n"
        #Excited states
        if max(states_to_do) > 1:
            string += f"%tddft\n\ttda false\n"
            string += f"\tnroots {max(states_to_do)-1}\n"
            if do_grad:
                string += "\tsgradlist " +  ",".join([str(i[1]-1) for i in qmin.maps["gradmap"]]) + "\n"
            string += "end\n\n"
        
        string += "%output\n"
        string += "\tPrint[ P_MOs ] 1\n"
        string += "end\n\n"

        string += "%coords\n\tCtyp xyz\n\tunits bohrs\n"
        string += f"\tcharge {charge}\n"
        string += f"\tmult 1\n"
        string += "\tcoords\n"
        for iatom, (label, coords) in enumerate(zip(qmin.molecule["elements"], qmin.coords["coords"])):
            string += f"\t{label:4s} {coords[0]:16.9f} {coords[1]:16.9f} {coords[2]:16.9f}\n"
        string += "\tend\nend\n\n"
    
        return string
    
    def _get_energy(self, output: str) -> dict[tuple[int, int], float]:
        """
        Extract energies from ORCA outfile

        output:     Content of outfile as string
        mult:       Multiplicities
        """

        find_energy = re.search(r"Total Energy[\s:]+([-\d\.]+)", output)
        if not find_energy:
            self.log.error("No energy in ORCA outfile found!")
            raise ValueError()

        gs_energy = float(find_energy.group(1))


        energies = {(1, int(1)): gs_energy}

        exc_states = re.findall(r"STATE\s+(\d+):[A-Z\s=]+([-\d\.]+)\s+au", output)

        iter_states = iter(exc_states)
 
        for state, energy in iter_states:
            energies[(1, int(state) + 1)] = gs_energy + float(energy)

        return energies


    def _get_dipole_moment(self, output: str) -> np.ndarray:
        """
        Extract dipole moment from ORCA outfile
        output:     Content of outfile as string
        """
        find_dipole = re.findall(r"Total Dipole Moment[:\s]+(.*)", output)
        if not find_dipole:
            self.log.error("Cannot find dipole moment in ORCA outfile!")
            raise ValueError()
        find_dipole = [list(map(float, x.split())) for x in find_dipole]
        return np.asarray(find_dipole)
    
    def _get_transition_dipoles(self, output: str) -> np.ndarray:
        """
        Extract transition dipole moments from ORCA outfile
        In TD-DFT with ORCA 5 only TDM between ground- and
        excited states of same multiplicity are calculated

        output:     Content of outfile as string
        """
        # Extract transition dipole table from output
        find_transition_dipoles = re.search(
            r"ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS([^ABCDFGH]*)", output, re.DOTALL
        )
        if not find_transition_dipoles:
            self.log.error("Cannot find transition dipoles in ORCA output!")
            raise ValueError()
        # Filter dipole vectors, (states, (xyz))
        transition_dipoles = re.findall(r"([-\d.]+\s+[-\d.]+\s+[-\d.]+)\n", find_transition_dipoles.group(1))
        return np.asarray([list(map(float, x.split())) for x in transition_dipoles])
    
    def _get_grad(self, grad_path: str) -> np.ndarray:
        """
        Extract gradients from ORCA outfile

        grad_path:  Path to gradient file
        """
        natom = self.QMin.molecule["natom"]

        with open(grad_path, "rb") as grad_file:
            grad_file.read(8 + 28 * natom)  # Skip header
            gradients = struct.unpack(f"{natom*3}d", grad_file.read(8 * 3 * natom))

        return np.asarray(gradients).reshape(natom, 3)

        
    def _save_files(self, workdir: str) -> None:
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]
        self.log.debug("Copying files to savedir")

        shutil.copy(os.path.join(workdir, "BASICORCA.gbw"), os.path.join(savedir, f"BASICORCA.gbw.{step}"))

    def _copy_gbw(self, qmin: QMin, workdir: str) -> None:
        """
        Copy gbw file from last/current time step

        jobid:      Job ID
        qmin:       QMin object
        workdir:    Current working directory
        """
        self.log.debug("Copy ORCA.gbw to work directory")
        gbw_file = os.path.join(qmin.save["savedir"], f"ORCA.gbw.{qmin.control['jobid']}.{qmin.save['step']-1}")
        if os.path.isfile(gbw_file):
            shutil.copy(gbw_file, os.path.join(workdir, "ORCA.gbw"))

    def _create_aoovl(self) -> None:
        """
        Create AO_overl.mixed for overlap calculations
        """
        pass
    

        
# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_BASICORCA().main()
