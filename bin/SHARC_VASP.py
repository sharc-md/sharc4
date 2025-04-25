#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper
import itertools

from numpy import ndarray
import numpy as np
from constants import *
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import containsstring, expand_path, question, itmult, link, makecmatrix, mkdir, readfile, writefile


__all__ = ["SHARC_VASP"]  # Only export interface class

AUTHORS = "Marco Romanelli"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2025, 4, 1)
NAME = "VASP"
DESCRIPTION ="AB INITIO interface for the Vienna Ab Initio Simulation Package (VASP)"

CHANGELOGSTRING = """
01.04.2025:    Very basic VASP interface relying on CPA approximation
(i.e. ground state gradients only) + Kohn-Sham excitation energies from 
ground-state periodic DFT calculations with VASP.
Refinements will follow.
"""

all_features = set(
    [
        "h",
        "grad",
        "overlap",
        # Rest of the possible requests to implement:
        # "phases",  (orbital phase tracking?)
    ]
)

class SHARC_VASP(SHARC_ABINITIO):
    """
    SHARC interface for VASP. Currently relying on CPA approximation and KS excitation energies.
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

        self.QMin.resources.update(
            {
                "vaspdir": None, # Path to the executable of VASP
                "potcardir": None, # Path to the POTCAR VASP file with PAW pseudopotentials
                "ncpu" : 2 #Default number of cpus for mpi run with VASP 
            }
        )
        self.QMin.resources.types.update(
            {
                "vaspdir": str,
                "potcardir": str, 
            }
        )

        self.QMin.template.update(
            {
                "system": "unspecified", #String for "SYSTEM" label of VASP INCAR
                "gga": "PE", #PBE functional by default (PE flag in VASP)
                "ismear": -2, #Smearing parameter for VASP, default set to "no smearing" (-2)
                "sigma": 0, #Smearing width
                "encut": 200, #Energy cutoff for plan waves in eV
                "nbands": 0, #If left to 0 VASP will automatically set this, it specifies n. of KS orbital
                "scale_param": 1, #scaling parameter for VASP unit cell
                # Note that current implementation only consider gamma-point only calculations i.e. k=0
                "a1": None, #1st unit cell lattice vector
                "a2": None, #2nd unit cell lattice vector
                "a3": None, #3rd unit cell lattice vector
            }
        )
        self.QMin.template.types.update(
            {
                "system": str, 
                "gga": str, 
                "ismear": int, 
                "sigma": int,
                "encut" : int,
                "nbands" : int,
                "scale_param": int, #scaling parameter for VASP unit cell
                "a1": list, #1st unit cell lattice vector
                "a2": list, #2nd unit cell lattice vector
                "a3": list, #3rd unit cell lattice vector
            }
        )


# ---------------------------------| Standard Methods |------------------------------------------------------------

    @staticmethod
    def version() -> str:
        return SHARC_VASP._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_VASP._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_VASP._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_VASP._authors

    @staticmethod
    def name() -> str:
        return SHARC_VASP._name

    @staticmethod
    def description() -> str:
        return SHARC_VASP._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_VASP._name}\n{SHARC_VASP._description}"


# ---------------------------------| Initialization |-----------------------------------------------------------------

    def read_template(self, template_file: str = "VASP.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

    #Control check to be added

    def read_resources(self, resources_file: str = "VASP.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        if not self.QMin.resources["vaspdir"]:
            self.log.error("vaspdir has to be set in resource file!")
            raise ValueError()

        if not self.QMin.resources["potcardir"]:
            self.log.error("Please specify pathway to POTCAR file in resource file!")
            raise ValueError()
    
        if not self.QMin.resources["ncpu"]:
            self.log.warning(" No ncpu keyword found in the resource file. Default value of 2 is applied.")
    
    def setup_interface(self) -> None:
        super().setup_interface()


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
        self.log.info(f"||{'VASP interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        self.log.info("\nSpecify path to VASP binary.")
        self.setupINFOS["vasp"] = question("Path to VASP:", str, KEYSTROKES=KEYSTROKES)

        self.log.info("\n\nSpecify a scratch directory. The scratch directory will be used to run the calculations.")
        self.setupINFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)

        if os.path.isfile("VASP.template"):
            self.log.info("Found VASP.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "VASP.template"
        else:
            self.log.info("Specify a path to a VASP template file.")
            while True:
                template_file = question("Template path:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(template_file):
                    self.log.info(f"File {template_file} does not exist!")
                    continue
                self._template_file = template_file
                break

        if question("Do you have a VASP.resources file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            self._resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES)
            while not os.path.isfile(self._resource_file):
                self.log.info(f"{self._resource_file} does not exist!")
                self._resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES)
        else:
            self.log.info("Specify the number of CPUs to be used.")
            self.setupINFOS["ncpu"] = question("Number of CPUs (at least 2):", int, default=[2], KEYSTROKES=KEYSTROKES)[0]

            self.log.info("Specify the amount of RAM to be used.")
            self.setupINFOS["memory"] = question("Memory (MB):", int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resource_file:
            with open(os.path.join(dir_path, "VASP.resources"), "w", encoding="utf-8") as file:
                for key in ("vaspdir", "potcardir", "scratchdir", "ncpu", "memory"):
                    if key in self.setupINFOS:
                        file.write(f"{key} {self.setupINFOS[key]}\n")
        else:
            create_file(expand_path(self._resource_file), os.path.join(dir_path, "VASP.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "VASP.template"))


# ---------------------------------| Run functions |----------------------------------------------------------------------------

    def run(self) -> None:

        starttime = datetime.datetime.now()
        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "vasp_calc")

        schedule = [{"vasp_calc" : self.QMin}] #Generate fake schedule
        self.QMin.control["nslots_pool"].append(1)
        self.runjobs(schedule)

        #self._save_files(self.QMin.control["workdir"])
        #self.clean_savedir()

        self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def create_restart_files(self) -> None:
            pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")


    def set_coords(self, coords_file: str = "QM.in") -> None:
        super().set_coords(coords_file)



# ---------------------------------| Scheduling & QMin execution |----------------------------------------------------

    def _gen_schedule(self) -> None:
        """
        Generates scheduling from joblist
        """
        pass

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Do VASP QM calculation
        """
        jobid = qmin.control["jobid"]
        step = qmin.save["step"]
        savedir = qmin.save["savedir"]
        ncpu=qmin.resources["ncpu"]
        potcar=os.path.join(qmin.resources["potcardir"],"POTCAR")
        
        # Setup workdir
        mkdir(workdir)

        # Write VASP input files
        #INCAR 
        input_str = self._generate_inputstr_INCAR()
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "INCAR")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)
        #POSCAR
        input_str = self._generate_inputstr_POSCAR()
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "POSCAR")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)
        #KPOINTS
        input_str = self._generate_inputstr_KPOINTS()
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "KPOINTS")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)
        #POTCAR
        self.log.debug(f"Coping POTCAR file from {potcar} to {workdir}")
        os.system(f"ln {potcar} {workdir}")
        
        # VASP running commands
        starttime = datetime.datetime.now()
        
        exec_str = f"mpirun -np {ncpu}{os.path.join(qmin.resources['vaspdir'],'vasp_std')} > {os.path.join(workdir, 'VASP.out')}"
        #exit_code = self.run_program(
        #    workdir, exec_str, os.path.join(workdir, "VASP.out"), os.path.join(workdir, "VASP.err"))
        exit_code=0 

        endtime = datetime.datetime.now()

        return exit_code, endtime - starttime


# ---------------------------------| Parsing output data from VASP calculations |-------------------------------------

    def getQMout(self) -> dict[str, ndarray]:
        #TODO: Parse requested properties from outputs and populate QMout object. You can make as many parsing and helper functions as you want.
        
        return self.QMout
    

#-------------------| Functions for generating inputstrings for writing VASP input files |---------------------------- 

    def _generate_inputstr_INCAR(self) -> str:
        """
        Generate INCAR input file string for VASP from QMin object
        """
        
        qmin=self.QMin 
        
        system = qmin.template["system"]
        ismear = qmin.template["ismear"]
        sigma = qmin.template["sigma"]
        nbands = qmin.template["nbands"]
        gga = qmin.template["gga"]
        encut = qmin.template["encut"]

        inputstring = f"SISTEM = {system}\n"
        inputstring += f"ISMEAR = {ismear}\n"
        inputstring += f"SIGMA = {sigma}\n"
        inputstring += f"ISPIN = 1\n" #Only singlets currently, hard coded!
        inputstring += f"GGA = {gga}\n"
        if nbands != 0:
            inputstring += f"NBANDS = {nbands}\n" 
        inputstring += f"ENCUT = {encut}" 
        
        return inputstring

    def _generate_inputstr_KPOINTS(self) -> str:
        """
        Generate KPOINTS input file string for VASP from QMin object.
        This is basically hard-coded at the moment as only gamma point sampling (K=0) is considered.
        """
        
        inputstring = f"Gamma-point only\n"
        inputstring += f"1    !one k-point only\n"
        inputstring += f"rec    !in units of reciprocal lattice vectors\n"
        inputstring += f"0 0 0 1    !k-point coords and weight\n"
        
        return inputstring


    def _generate_inputstr_POSCAR(self) -> str:
        """
        Generate POSCAR input file string for VASP from QMin object.
        """
        
        qmin=self.QMin 
        
        coords = qmin.coords["coords"]
        elements = qmin.molecule["elements"]
        scale_param = qmin.template["scale_param"]
        a1 = qmin.template["a1"]
        a2 = qmin.template["a2"]
        a3 = qmin.template["a3"]
        system = qmin.template["system"]

        inputstring = f"{system}\n"
        inputstring += f"{scale_param}\n"
        inputstring += f"{a1[0]} {a1[1]} {a1[2]}\n"
        inputstring += f"{a2[0]} {a2[1]} {a2[2]}\n"
        inputstring += f"{a3[0]} {a3[1]} {a3[2]}\n"

        elements_nr=list(set(elements)) #Non-redundant list of element types for proper VASP input file format
        indx=list() #List of list of indexes for each element, redundant elements lead to inner lists with more than one element
        for i in elements_nr:
            tmp=list()
            for n,j in enumerate(elements):
                tmp.append(n) if j==i else None
            indx.append(tmp)

        for i in elements_nr:  
            inputstring += f" {i}"
        inputstring += f"\n"
        for i in indx:  
            inputstring += f" {len(i)}"
        inputstring += f"\n"
        
        inputstring += f"cart\n" #Hard-coded cartesian coordinates (Ang.) for input. Other options may be available in VASP
        for i in indx:
            for j in i:
                inputstring += f" {coords[j][0]*au2a:>16.9f}  {coords[j][1]*au2a:>16.9f}  {coords[j][2]*au2a:>16.9f}\n" 

        return inputstring


# ---------------------------------| Empty trivial functions otherwise external checks complain |---------------------
    def _create_aoovl(self) -> None:
        #empty function
        pass


    def get_mole(self) -> None:
        #empty function
        pass


    def get_readable_densities(self) -> None:
        #empty function
        pass


    def read_and_append_densities(self) -> None:
        #empty function
        pass


# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_VASP().main()
