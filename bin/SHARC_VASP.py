#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper
import itertools
import shutil
import numpy as np
import re
from constants import *
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import  expand_path, question, link,  mkdir,  writefile, is_exec, phase_correction_cmplx, det_slog
from copy import deepcopy
import importlib.util
import sys
from scipy.linalg import lu_solve, lu_factor 
from joblib import Parallel, delayed

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
        "dm",
        "grad",
        "overlap",
        "phases", #only if overlap matrix is available 
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
                "hdf5vaspdir": None, # Path to the HDF5 libraries used for VASP compilation
                "potcardir": None, # Path to the POTCAR VASP file with PAW pseudopotentials
                "ncore" : 1, #Default number of compute cores to work on a single orbital in VASP 
                "ncpu" : 2, #Default number of cpus for mpi run with VASP 
                "memory" : 2000,
                "wfoverlap" : "", #easy workaround to prevent bin_executable check, pawpyseed used here! 
            }                      
        )
        self.QMin.resources.types.update(
            {
                "vaspdir": str,
                "hdf5vaspdir": str,
                "potcardir": str, 
                "ncore":int, 
            }
        )

        self.QMin.template.update(
            {
                "system": "unspecified", #String for "SYSTEM" label of VASP INCAR
                "gga": "PE", #PBE functional by default (PE flag in VASP)
                "ismear": 0, #Smearing parameter for VASP, default set to Gaussian smearing (0) 
                "sigma": 0.001, #Smearing width
                "encut": 200, #Energy cutoff for plan waves in eV
                "ispin": 1, #keyword for selecting spin calculation, only singlet ISPIN=1 is implemented below
                "nbands": None, #If unspecified it will not appear in INCAR, so VASP will determine it automatically
                "nelm": 60, #setting maximun number of SCF electronic steps
                "algo": "Normal", #selects the algorithm to optimize orbitals, ALGO
                "ialgo": None, #If unspecified it will not appear in INCAR, so VASP will use its default
                "time_vasp" : None, #If unspecified it will not appear in INCAR, so VASP will use its default
                "ediff" : 1e-4, # eV energy change for SCF break condition 
                "lreal" : None, #If unspecified it will not appear in INCAR, so VASP will determine it automatically 
                "scale_param": 1, #scaling parameter for VASP unit cell
                "a1": None, #1st unit cell lattice vector
                "a2": None, #2nd unit cell lattice vector
                "a3": None, #3rd unit cell lattice vector
                "overlap_method": "full", #method to compute overlaps via pawpyseed
                "phases_method": "simple" #method to compute phase correction based on overlap matrix
            }
        )
        self.QMin.template.types.update(
            {
                "system": str, 
                "gga": str, 
                "ismear": int, 
                "sigma": float,
                "encut" : float,
                "ispin": int, 
                "nbands" : int,
                "nelm" : int,
                "algo":str, 
                "ialgo": int, 
                "time_vasp": float,
                "ediff" : float,
                "lreal" : str, 
                "scale_param": int, #scaling parameter for VASP unit cell
                "a1": list, #1st unit cell lattice vector
                "a2": list, #2nd unit cell lattice vector
                "a3": list, #3rd unit cell lattice vector
                "overlap_method": str, #method to compute overlaps via pawpyseed
                "phases_method": str, #method to compute phase correction based on overlap matrix
            }
        )
        self._coords_vasp =  None #to store VASP input geometry, different from QMin format
        self._el_vasp =  None #to store VASP indices for sorting input geometry, different from QMin format
        self._indices_vasp =  None #to store indices which sort VASP forces/geometry according to QMin geometry format


        # Setup stuff
        self._template_file = None
        self._resource_file = None

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
        
        #Most likely some of these variable type checks are redudant, they might be performed already somewhere else, to be checked!
        
        if not isinstance(self.QMin.template["system"],str):
            self.log.error("system keyword in the template file must be a string")
            raise ValueError() 
        if not isinstance(self.QMin.template["gga"],str):
            self.log.error("set the gga keyword in the template to a string corresponding to available functionals in VASP, see VASP wiki")
            raise ValueError() 
        else:
            self.QMin.template["gga"]=self.QMin.template["gga"].upper() #Probably it is not case sensitive anyway

        if not isinstance(self.QMin.template["sigma"],float):
            self.log.error("sigma in template has to be a real number, check vasp wiki")
            raise ValueError()

        if not isinstance(self.QMin.template["ismear"],int):
            self.log.error("ismear in template has to be an integer, check vasp wiki")
            raise ValueError()

        if not isinstance(self.QMin.template["encut"],float):
            self.log.error("encut in template has to be a real number, check vasp wiki")
            raise ValueError()

        if not isinstance(self.QMin.template["ispin"],int):
            self.log.error("ispin in template has to be an integer, check vasp wiki")
            raise ValueError()
        else:
            if self.QMin.template["ispin"] != 1:
                self.log.error("ispin has to be set to 1, singlet calculation. Higher multiplicities are not implemented yet")
                raise ValueError()

        if self.QMin.template["nbands"] is not None and not isinstance(self.QMin.template["nbands"],int):
            self.log.error("nbands in template has to be an integer, check vasp wiki")
            raise ValueError()
        
        if not isinstance(self.QMin.template["nelm"],int):
            self.log.error("nelm in template has to be an integer, check vasp wiki")
            raise ValueError()
        
        if self.QMin.template["ialgo"] is not None and not isinstance(self.QMin.template["ialgo"],int):
            self.log.error("ialgo in template has to be an integer, check vasp wiki")
            raise ValueError()
        
        if not isinstance(self.QMin.template["ediff"],float):
            self.log.error("ediff in template has to be an integer, check vasp wiki")
            raise ValueError()
        
        if self.QMin.template["lreal"] is not None and not isinstance(self.QMin.template["lreal"],str):
            self.log.error("lreal in template has to be a string, check vasp wiki")
            raise ValueError()
        
        if self.QMin.template["time_vasp"] is not None and not isinstance(self.QMin.template["time_vasp"],float):
            self.log.error("time_vasp in template has to be a real number, check vasp wiki")
            raise ValueError()
        
        if not isinstance(self.QMin.template["scale_param"],int):
            self.log.error("scale_param in template has to be an integer, check vasp wiki")
            raise ValueError()
        
        if not isinstance(self.QMin.template["overlap_method"],str):
            self.log.error("overlap_method has to be a string. Only 'full' or 'pseudo' are supported. It selects pawpyseed method for performing overlap calculation")
            raise ValueError()
        else:
            if self.QMin.template["overlap_method"] == "full" or self.QMin.template["overlap_method"] == "pseudo":
                pass
            else:
                self.log.error("overlap_method can only be either 'full' or 'pseudo'")
                raise ValueError()

        if not isinstance(self.QMin.template["phases_method"],str):
            self.log.error("phases_method has to be a string. Only 'none', 'simple' or 'robust' are supported. It selects phase correction method using overlap matrix")
            raise ValueError()
        else:
            if self.QMin.template["phases_method"] == "simple" or self.QMin.template["phases_method"] == "robust" or self.QMin.template["phases_method"] == "none":
                pass
            else:
                self.log.error("phases_method can only be either 'none', 'simple' or 'robust'")
                raise ValueError()
        
        #Control check for lattice vectors
        if not isinstance(self.QMin.template["a1"],list):
            self.log.error("a1 has to be a list of 3 numbers, the lattice vector coordinates. Please provide 3 real numbers separated by whitespaces in the template")
            raise ValueError()
        else: 
            if not (all([bool(re.fullmatch(r'[+-]?\d+\.?\d+',i.strip())) for i in self.QMin.template["a1"]]) and len(self.QMin.template["a1"])==3):
                self.log.error("a1 has to be a list of 3 real numbers only! Please provide each real number as e.g. -300.45")
                raise ValueError()

        if not isinstance(self.QMin.template["a2"],list):
            self.log.error("a2 has to be a list of 3 numbers, the lattice vector coordinates. Please provide 3 real numbers separated by whitespaces in the template")
            raise ValueError()
        else: 
            if not (all([bool(re.fullmatch(r'[+-]?\d+\.?\d+',i.strip())) for i in self.QMin.template["a2"]]) and len(self.QMin.template["a2"])==3):
                self.log.error("a2 has to be a list of 3 real numbers only! Please provide each real number as e.g. -300.45")
                raise ValueError()

        if not isinstance(self.QMin.template["a3"],list):
            self.log.error("a3 has to be a list of 3 numbers, the lattice vector coordinates. Please provide 3 real numbers separated by whitespaces in the template")
            raise ValueError()
        else: 
            if not (all([bool(re.fullmatch(r'[+-]?\d+\.?\d+',i.strip())) for i in self.QMin.template["a3"]]) and len(self.QMin.template["a3"])==3):
                self.log.error("a3 has to be a list of 3 real numbers only! Please provide each real number as e.g. -300.45")
                raise ValueError()


    def read_resources(self, resources_file: str = "VASP.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)
        
        self.log.debug("Debugging resources in VASP")
        self.log.debug(self.QMin.resources)
        self.log.debug("Debugging savedir in VASP")
        self.log.debug(self.QMin.save["savedir"])

        if self.QMin.resources["scratchdir"] == os.path.join(self.QMin.resources["pwd"],"SCRATCH"):
            self.log.warning("You have not setup scratchdir in the VASP resource file, this may cause issues! Please do it")
        
        if self.QMin.save["savedir"] == os.path.join(self.QMin.resources["pwd"],"SAVE"):
            self.log.warning("You have not setup savedir in the VASP resource file, this may cause issues! Please do it")
        
        if not self.QMin.resources["vaspdir"]:
            self.log.error("vaspdir has to be set in resource file!")
            raise ValueError("vaspdir has to be set in resource file!")

        if not self.QMin.resources["hdf5vaspdir"]:
            hdf5vaspdir=os.path.join(self.QMin.resources["vaspdir"],"../libs/lib")
            self.log.debug(hdf5vaspdir)
            if os.path.isdir(hdf5vaspdir):
                self.QMin.resources["hdf5vaspdir"]=hdf5vaspdir
            else:
                self.log.error("No HDF5 libraries linked to your VASP installation can be found. Please set hdf5vaspdir explicitly in VASP.resources")
                self.log.error("If you have not compiled VASP with HDF5 support please do")
                raise ValueError("No HDF5 libraries linked to your VASP installation can be found. Please set hdf5vaspdir explicitly in VASP.resources")


        if not self.QMin.resources["potcardir"]:
            self.log.error("Please specify pathway to POTCAR file in resource file!")
            raise ValueError("Please specify pathway to POTCAR file in resource file!")
    
        if not self.QMin.resources["ncore"]:
            self.log.warning(" No ncore keyword found in the resource file. Default value of 1 is applied.")
    
        if not self.QMin.resources["ncpu"]:
            self.log.warning(" No ncpu keyword found in the resource file. Default value of 2 is applied.")
    
    def setup_interface(self) -> None:
        super().setup_interface()
        
        self.log.info(f"Scratchdir: {self.QMin.resources['scratchdir']}")
        self.log.info(f"Savedir: {self.QMin.save['savedir']}")
       
        if (any(num > 0 for num in self.QMin.molecule["states"][1:]) or self.QMin.molecule["states"][0] == 0):
            self.log.error("Current VASP implementation only deals with singlets!")
            raise ValueError()
        # Checking for MPI installation. Needed to run VASP properly
        if not is_exec("mpirun"):
            self.log.error("Cannot find mpirun executable, please check your MPI installation and load the proper environment or add the right " \
            "executable to $PATH")
            raise ValueError()

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

        if os.path.isfile("VASP.template"):
            self.log.info("Found VASP.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "VASP.template"
            else:
                self.log.info("Specify a path to a VASP template file.")
                template_file = question("Template path:", str, KEYSTROKES=KEYSTROKES,autocomplete=True)
                while not os.path.isfile(template_file):
                    self.log.info(f"File {template_file} does not exist!")
                    template_file = question("Template path:", str, KEYSTROKES=KEYSTROKES,autocomplete=True)
                self._template_file = template_file
        else:
            self.log.info("Specify a path to a VASP template file.")
            template_file = question("Template path:", str, KEYSTROKES=KEYSTROKES,autocomplete=True)
            while not os.path.isfile(template_file):
                self.log.info(f"File {template_file} does not exist!")
                template_file = question("Template path:", str, KEYSTROKES=KEYSTROKES,autocomplete=True)
            self._template_file = template_file

        if question("Do you have a VASP.resources file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            if os.path.isfile("VASP.resources"):
                self.log.info("Found VASP.resources in current directory")
                if question("Use this resources file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                    self._resource_file = "VASP.resources"
                else:
                    self.log.info("Specify path to VASP resource file.")
                    resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
                    while not os.path.isfile(resource_file):
                        self.log.info(f"{resource_file} does not exist!")
                        resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
                    self._resource_file=resource_file
            else:
                self.log.info("Specify path to VASP resource file.")
                resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
                while not os.path.isfile(resource_file):
                    self.log.info(f"{resource_file} does not exist!")
                    resource_file = question("Resource path:", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
                self._resource_file=resource_file
                #handling possible missing info in resource file
                with open(self._resource_file, "r", encoding="utf-8") as f:
                    resources_file = f.read()
                savedir_check=re.search(r"\s*savedir",resources_file) 
                scratchdir_check=re.search(r"\s*scratchdir",resources_file) 
                if savedir_check is None:
                    self.log.warning("You have not specified savedir in the resource file. Please do it, this may cause issues.")
                if scratchdir_check is None:
                    self.log.warning("You have not specified scratchdir in the resource file. Please do it, this may cause issues.")
        else:
            self.log.info("Specify the number of CPUs to be used.")
            self.setupINFOS["ncpu"] = question("Number of CPUs (at least 2):", int, default=[2], KEYSTROKES=KEYSTROKES)[0]

            self.log.info("Specify the amount of RAM to be used.")
            self.setupINFOS["memory"] = question("Memory (MB):", int, default=[2000], KEYSTROKES=KEYSTROKES)[0]

            self.log.info("Specify the path to the VASP binary files")
            self.setupINFOS["vaspdir"] = question("path to VASP binary files", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

            self.log.info("Specify the path to the VASP HDF5 libraries")
            self.setupINFOS["hdf5vaspdir"] = question("path to VASP HDF5 libraries", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

            self.log.info("Specify the path to the VASP potcar file")
            self.setupINFOS["potcardir"] = question("path to VASP POTCAR file", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

            self.log.info("Specify the number of cores you want to work on each single orbital")
            self.setupINFOS["ncore"] = question("NCORE: ", int, default=[1],KEYSTROKES=KEYSTROKES)[0]
            
            self.log.info("\n\nSpecify a scratch directory. The scratch directory will be used to run the VASP calculation.")
            self.setupINFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
            #self.setupINFOS["scratchdir"] += '/$$/'
            
            self.log.info("\n\nSpecify a save directory. The save directory will keep important files for each VASP run.")
            self.setupINFOS["savedir"] = question("Path to save directory:", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
        
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None: 
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resource_file:
            with open(os.path.join(dir_path, "VASP.resources"), "w", encoding="utf-8") as file:
                for key in ("vaspdir","hdf5vaspdir", "potcardir", "scratchdir", "ncpu", "memory", "savedir","ncore"):
                    if key in self.setupINFOS:
                        file.write(f"{key} {self.setupINFOS[key]}\n")
        else:
            create_file(expand_path(self._resource_file), os.path.join(dir_path, "VASP.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "VASP.template"))


# ---------------------------------| Run functions |----------------------------------------------------------------------------

    def run(self) -> None:

        starttime = datetime.datetime.now()
        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "")

        schedule = [{"" : self.QMin}] #Generate fake schedule
        self.QMin.control["nslots_pool"].append(1)
        self.runjobs(schedule)


        self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def create_restart_files(self) -> None:
            pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                self.log.error(f"Found unsupported request {req}.")
                raise ValueError(f"Found unsupported request {req}.")
            
        #Checking for pawpyseed (https://github.com/kylebystrom/pawpyseed) installation in the conda environment the user uses for running SHARC
        # This is compulsory for calculating overlaps and so run VASP-SHARC dynamics !!
        # SH with VASP can only run by using overlaps, no NACs available.
        if self.QMin.requests["overlap"]:
            if importlib.util.find_spec("pawpyseed") is None:
                self.log.error("You have requested overlap propagation but no pawpyseed was found in your python env. \
                                This is necessary for computing overlaps out of VASP.")
                self.log.error("install pawpyseed from: https://github.com/kylebystrom/pawpyseed  and check afterwards that it appears in 'pip list'")
                raise ValueError("pawpyseed is not installed in your python env! do it first https://github.com/kylebystrom/pawpyseed")
        
        if self.QMin.requests["phases"] and not self.QMin.requests["overlap"]:
            self.log.error("Phase correction is not supported without overlap calculation here!")
            raise ValueError("Phase correction is not supported without overlap calculation here!")
        
        self.log.debug("debugging grad requests")
        self.log.debug(self.QMin.requests["grad"])
        if isinstance(self.QMin.requests["grad"],list) and self.QMin.requests["grad"] != [1]: #SHARC_VASP is supposed to be called by SHARC_CPA now, only GS gradient from child interface.
            self.log.error("SHARC_VASP can only provide ground-state gradient only. You cannot request excited-state ones")
            raise ValueError("SHARC_VASP can only provide ground-state gradient only. You cannot request excited-state ones")


    def set_coords(self, coords_file: str = "QM.in", pc: bool = False) -> None:
        super().set_coords(coords_file, pc)

        # Checking whether QM.in coordinates comply with VASP POSCAR format.
        # Basically all atoms of the same type have to be grouped together back to back in the geometry input in QM.in
        # This is a strict constraint but it simplifies a lot following usage of VASP
        
        def check_far_duplicates(lst:list) -> bool: 
            last_seen = {}
            for i, val in enumerate(lst):
                if val in last_seen and i - last_seen[val] > 1:
                    return True
                last_seen[val] = i
            return False

        if check_far_duplicates(self.QMin.molecule["elements"]):
            self.log.error("Input geometry in QMin does not comply with POSCAR format.")
            self.log.error("Please format it so that all atoms of the same type appear in consecutive lines in the input geometry matrix")
            raise TypeError()
        
        #This is for proper formatting of VASP input geometry from QM.in format
        self._el_vasp=[] #Saving indexes for VASP input format, list of lists       
        for i in list(dict.fromkeys(self.QMin.molecule["elements"])):
            tmp=list()
            for n,j in enumerate(self.QMin.molecule["elements"]):
                tmp.append(n) if j==i else None
            self._el_vasp.append(tmp)
        self._coords_vasp=[self.QMin.coords["coords"][i] for i in sum(self._el_vasp,[])] #Input geometry for VASP format
        self._indices_vasp=[] #For sorting output forces from VASP according to QMin geometry format
        for i in self.QMin.coords["coords"]:
            for n,j in enumerate(self._coords_vasp):
                self._indices_vasp.append(n) if np.array_equal(j,i) else None

# ---------------------------------| Scheduling & QMin execution |----------------------------------------------------

    def _gen_schedule(self) -> None:
        """
        Generates scheduling from joblist
        """
        pass

    def _setup_env(self) -> dict[str, str]:
        """
        Generate an env dict to properly setup a local ENV for running VASP.
        This is required because the HDF5 library that SHARC uses for NETCDF and with which it is compiled through CONDA env is often different than 
        the that of VASP with HDF5 support.
        This only modify the local environment of the subprocess where VASP is executed.
        """
        vasp_env = deepcopy(os.environ)
        #Prepending path for VASP execution
        self.log.debug(self.QMin.resources["hdf5vaspdir"])
        vasp_env["LD_LIBRARY_PATH"]=f"{self.QMin.resources["hdf5vaspdir"]}:{vasp_env["LD_LIBRARY_PATH"]}"
        self.log.debug(vasp_env["LD_LIBRARY_PATH"])
        # Preventing VASP step to use OpenMP
        self.log.debug("Setting OMP_NUM_THREADS to 1. No OpenMP allowed in VASP here.")
        vasp_env["OMP_NUM_THREADS"]="1"
        return vasp_env

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
        
        # Reading wavefunction guess from step-1 for speeding up VASP calculation of step
        self._copy_files(workdir,savedir)

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
        shutil.copy(potcar,workdir) 
        
        # VASP running commands
        starttime = datetime.datetime.now()

        exec_str = f"mpirun -np {ncpu} {os.path.join(qmin.resources['vaspdir'],'vasp_std')} > {os.path.join(workdir, 'VASP.out')}"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "VASP.out"), os.path.join(workdir, "VASP.err"),self._setup_env())

        ### Checking correct execution of VASP calculation. ###
        #Checking correct POSCAR and POTCAR format, only at first step is enough to make sure remaining steps run smoothly
        if self.QMin.save["step"] == 0:
            with open(os.path.join(workdir,"VASP.out"), "r", encoding="utf-8") as f:
                output = f.read()
            vasp_warning=re.search(r"\s+WARNING:\s+type\s+information\s+on\s+POSCAR\s+and\s+POTCAR\s+are\s+incompatible",output) 
            if vasp_warning is not None:
                self.log.error("POTCAR AND POSCAR formats are incompatible, please check VASP.out. You probably got unphysical results.")
                raise ValueError("POTCAR AND POSCAR formats are incompatible, please check VASP.out. You probably got unphysical results.")  
        
        if exit_code != 0:
            with open(os.path.join(workdir, "VASP.err"), "r", encoding="utf-8") as f:
                self.log.error("Please check your VASP.out, something went wrong with the VASP calculation!")
                self.log.error(f.read())
        elif exit_code ==0 and not self.QMin.save["samestep"]: #Saving files for overlap calculation 
            self._save_files(workdir,savedir)

        endtime = datetime.datetime.now()
        
        return exit_code, endtime - starttime


# ---------------------------------| Parsing output data from VASP calculations |-------------------------------------

    def getQMout(self) -> None:
        """
        Parse VASP output files
        """

        self.log.debug(f"Setting OMP_NUM_THREADS to self.QMin.resources['ncpu']: {self.QMin.resources['ncpu']}")
        os.environ["OMP_NUM_THREADS"]=str(self.QMin.resources['ncpu'])
        self.log.debug(f"Checking OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")


        self.log.debug("Testing VASP geometry sorting indices")
        self.log.debug(self._indices_vasp)
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
        
        nmstates = self.QMin.molecule["nmstates"]

        #Opening OUTCAR output file for parsing
        with open(os.path.join(self.QMin.control["workdir"],"OUTCAR"), "r", encoding="utf-8") as f:
            OUTCAR = f.read()

        # Populate energies
        if self.QMin.requests["h"]:
            energies,det_t, ks_mo_index = self._get_energies(OUTCAR)
            for i in range(len(energies)):
                self.QMout["h"][i][i] = energies[i]
        #ks_mo_index is a dictionary with each orbital label and corresponding orbital index
        # det_t contains occupation strings for determinants of each active state of current timestep 
        # It is needed to compute overlaps
        
        # Populate dipole moments
        if self.QMin.requests["dm"]:
            self.QMout["dm"] = self._get_dipoles(OUTCAR)
        
        # Populate gradients
        if self.QMin.requests["grad"]:
            self.QMout.grad = self._get_gradients(OUTCAR) #This is gonna be used in the parent SHARC_CPA interface for each excited-state
            self.log.debug("Checking GS gradients assigned to QMout and shape")
            self.log.debug(self.QMout.grad)
            self.log.debug(self.QMout.grad.shape)

        # Populate overlaps
        if self.QMin.requests["overlap"]:
            self.QMout.overlap = self._get_overlap(det_t,ks_mo_index)
            self.log.debug("Checking population of self.QMout.overlap, overlap matrix")
            self.log.debug(self.QMout.overlap)


        # Populate phases
        if self.QMin.requests["overlap"] and self.QMin.requests["phases"]:
            self.QMout.phases=self._get_phases(self.QMin.template["phases_method"],self.QMout.overlap)
            self.log.debug("Checking population of self.QMout.phases")
            self.log.debug(self.QMout.phases)
        
        return self.QMout
    

    def _get_gradients(self, vasp_out: str) -> np.ndarray:
        """
        Get GS gradients from VASP output (OUTCAR) file.
        Each ES gradient does coincide with the GS one -> CPA approximation!!
        Gradients are output in Hartree/Bohr units and read in eV/Ang. from VASP OUTCAR

        vasp_out: VASP OUTCAR file
        """        
        
        start = datetime.datetime.now()
        
        nmstates = self.QMin.molecule["nmstates"]

        start_marker=r"\sPOSITION\s+TOTAL-FORCE \(eV\/Angst\)\n\s\-+\n"
        end_marker=r"\s\-+\n"
        pattern = rf'{start_marker}(.*?){end_marker}'
        match = re.search(pattern, vasp_out, re.DOTALL)
        #Forces from VASP in eV/Ang.
        forces=np.array([i.split() for i in match.group(1).splitlines()],dtype=np.float64)[:,3:]
        # Sorting output forces according to QMin geometry format 
        forces=forces[self._indices_vasp]
        # To be removed after testing
        self.log.debug("forces out of VASP upon sorting according to proper QMin ordering")
        self.log.debug(forces)
        
        forces=forces/au2eV*au2a #Changing to forces in atomic units
        
        self.log.debug("GS forces out of VASP with SHARC format")
        self.log.debug(forces)

        gradients= -forces.copy() #We get forces from VASP but we need to pass gradients to SHARC driver

        end = datetime.datetime.now()
        self.log.debug("==> Getting gradients out of VASP done." + check_timing(start,end))
        return gradients

    def _get_dipoles(self, vasp_out: str) -> np.ndarray:
        """
        Get dipole operator matrix. Currently this return an array of zeros
        because dipole matrix elements are meaningless in the KS orbital picture.
        This is trivially done to prevent other SHARC script from not working properly if no TDM are provided in QM.out.

        vasp_out: VASP OUTCAR file
        """
        
        nmstates = self.QMin.molecule["nmstates"]
        dip=np.zeros((3,nmstates,nmstates))
        
        return dip

    def _get_energies(self, vasp_out: str) -> tuple[np.ndarray,tuple[dict,dict]]:
        """
        Eigenstate energies from VASP. Excitation energies are computed as orbital energy difference between KS eigenvalues!
        GS energy is the correct DFT one, higher-lying state energies are obtained by summing excitation energy (from KS MOs difference) to GS energy.
        Currently, only single excitations are considered! No double, triple excitations etc. 
        Moreover, GS is assumed to be closed shell and only singlet excited states are accounted for.
        Refinements will follow.
        
        reference: J.Chem.TheoryComput.2013,9,4959−4972 (Akimov & Prezhdo)

        vasp_out: VASP OUTCAR file
        """
        natom = self.QMin.molecule["natom"]
        nmstates = self.QMin.molecule["nmstates"]

        start = datetime.datetime.now()
        
        #### Extracting KS MOs and corresponding energies first ####
        start_marker=r"\s+band\s+No\.\s+band\s+energies\s+occupation\s+\n"
        end_marker=r"\n\n\n\-+"
        pattern = rf'{start_marker}(.*?){end_marker}'
        match = re.search(pattern, vasp_out, re.DOTALL)
        data=np.array([i.split() for i in match.group(1).splitlines()],dtype=np.float64)
        pattern=rf'\s+Fermi energy:\s+(.*?)\n' #Fermi energy
        efermi=float(re.search(pattern,vasp_out).group(1))
        ks_en=np.copy(data[:,1])-efermi #ks eigenvalues upon subtracting fermi energy
        occ=np.copy(data[:,2]) #occ. n of each orbital
        self.log.debug("Orbitals occupancies from VASP output")
        self.log.debug(occ)
        for i in occ:
            if i != 2.0 and i != 0.0:
                self.log.error("Orbital occupancy from VASP calculation differ from 2 or 0. Open-shell or partial occupancies are not supported")
                raise ValueError("Orbital occupancy from VASP calculation differ from 2 or 0. Open-shell or partial occupancies are not supported") 
        #Reading and sorting KS orbital energies (Fermi energy set to 0!)
        n_o=int(np.sum(occ)/2) # N. of occupied MO, assuming closed shell
        n_u=len(ks_en)-n_o
        ks_o=dict([('H-'+ str(n_o-1-i),ks_en[i]) for i in range(0,n_o-1)]) #orbitals below H
        ks_o.update({'H':ks_en[n_o-1]}) # H orbital
        ks_u={'L':ks_en[n_o]} #L orbital
        ks_u.update(dict([('L+'+ str(i-n_o),ks_en[i]) for i in range(n_o+1,n_o+n_u)])) #obitals above L
        #### Computing excitation energies by orbital energy difference among KS MOs and selecting first "nmstates" only #### 
        ks_mo= ks_o|ks_u
        ks_mo_index=dict([(i,n) for n,i in enumerate(ks_mo.keys())]) #Dictionary {'orbital_label':index} 
        ks_es_all=dict()
        for i in ks_o:
            for j in ks_u:
                ks_es_all.update({ i+'->'+j : ks_u[j]-ks_o[i]})
        ks_es_all=dict(sorted(ks_es_all.items(), key=lambda x: x[1])) #sorting excitation energies
        ks_es= dict(itertools.islice(ks_es_all.items(), nmstates-1)) # Getting first nmstates-1 excitation energies
        if nmstates > 1: 
            self._write_transitions(ks_es,ks_mo_index) #Writing out states selected and their composition
        #### Create the output list with GS energy and nmstates-1 excited state energies for SHARC driver ####
        energies=np.zeros(nmstates,dtype=complex)
        pattern=rf'  energy  without entropy=.*energy\(sigma->0\)\s+=\s+(.*?)\n'
        gs_en=float(re.search(pattern,vasp_out).group(1))
        self.log.debug("debugging GS energy from VASP")
        self.log.debug(gs_en)
        energies[0]=(gs_en/au2eV)
        for n,i in enumerate(ks_es.values()):
            energies[n+1]=(energies[0]+i/au2eV)
        self.log.debug("TESTING ENERGIES PARSING")
        self.log.debug(energies)
        self.log.debug("KS orbitals and transitions out of VASP")
        self.log.debug(ks_es)
        #### Saving Slater determinants occupation indexes for overlap calculation #####
        #GS slater determinant first {'first_occupied_orbital_label' : first_occupied_orbital_index ...etc...}
        gs=dict(itertools.islice(ks_mo_index.items(), list(ks_mo_index.keys()).index("L")))
        self.log.debug("checking Slater determinant string of GS")
        self.log.debug(gs)
        #Adding excited state determinants now 
        es=list()
        for i in ks_es.keys():
            tmp={}
            from_orbital=(i.split('->')[0]) #Occupied orbital to excite from
            to_orbital=(i.split('->')[-1]) #Unoccupied orbital to excite into
            for k, v in gs.items():
                if k == from_orbital:
                # replace key in the same position
                    tmp[to_orbital] = ks_mo_index[to_orbital]
                else:
                    tmp[k] = v
            es.append(tmp)
        self.log.debug("checking Slater determinant strings for selected ES")
        self.log.debug(es)
        det_ind=np.zeros((nmstates,len(gs)),dtype=int)
        det_ind[0]=np.array(list(gs.values())) #array of orbital indexes for each slater determinant
        for n,i in enumerate(es):
            det_ind[n+1]=np.array(list(i.values()))
        self.log.debug("checking number of states selected from SHARC input")
        self.log.debug(self.QMin.molecule["nmstates"]) 
        self.log.debug("checking number of states for which overlap is computed")
        self.log.debug(det_ind.shape)
        #Saving Slater Determinant occupations for overlap
        filename=os.path.join(self.QMin.save["savedir"], f"det_index.{self.QMin.save['step']}") 
        np.savetxt(filename,det_ind,fmt='%d') 
        
        end = datetime.datetime.now()
        self.log.debug("==> Building excitations out of VASP orbitals done." + check_timing(start,end))

        return energies,det_ind,ks_mo_index
    
    def _get_overlap(self, det_t: np.ndarray,  ks_mo_index: dict) -> np.ndarray:
        ''' 
        Function to get S_{ij}(r,t+dt) overlap matrix by using pawpyseed to compute overlaps between SDs out of KS orbitals from VASP.
        
        This routine relies on matrix determinant lemma (rank-1 and rank-2 updates) for speeding up multiple SD overlaps computation.

        det_t: np.array with determinants occupation for current timestep with orbital occupations for selected states
        ks_mo_index: dictionary where full orbital labels and corresponding indexes are stored. First occupied orbital index is 0.
       
        self.QMin.template["overlap_method"] selects which method from pawpyseed to compute MO overlaps.
        if 'full',  default AE overlaps are calculated.
        if 'pseudo' only pseudowavefunction overlaps
        '''
       
        start = datetime.datetime.now()
       
        from pawpyseed.core.projector import Wavefunction,Projector #Check if this is installed in $CONDA_PREFIX is done above

        filename=os.path.join(self.QMin.save["savedir"], f"det_index.{self.QMin.save['step']-1}")
        det_t0=np.loadtxt(filename,dtype=int)
        self.log.debug("Occupation strings of Slater determinants at previous timestep")
        self.log.debug(det_t0)
        self.log.debug(f"Checking OMP_NUM_THREADS parallelization for pawpyseed: {os.environ['OMP_NUM_THREADS']}")

        #Initializing AE or PS wavefunctions (KS valence MO wavefunctions)
        self.log.debug("-----------------------------")
        self.log.debug("PAWPYSEED overlap calculation")
        self.log.debug("-----------------------------\n")
        with suppress_stdout_stderr():  
            wf_t = Wavefunction.from_files(struct=os.path.join(self.QMin.control["workdir"],"CONTCAR"),  #current timestep wf
                                            wavecar=os.path.join(self.QMin.control["workdir"],"WAVECAR"),
                                            cr=os.path.join(self.QMin.control["workdir"],"POTCAR"),
                                            vr=os.path.join(self.QMin.control["workdir"],"vasprun.xml"))
            wf_t0 = Wavefunction.from_files(struct=os.path.join(self.QMin.save["savedir"],f"CONTCAR.{self.QMin.save['step']-1}"), #previous timestep wf
                                            wavecar=os.path.join(self.QMin.save["savedir"],f"WAVECAR.{self.QMin.save['step']-1}"),
                                            cr=os.path.join(self.QMin.save["savedir"],"POTCAR"),
                                            vr=os.path.join(self.QMin.save["savedir"],f"vasprun.xml.{self.QMin.save['step']-1}"))
            if self.QMin.template["overlap_method"]=="pseudo":
                pr=Projector(wf_t, wf_t0,method=self.QMin.template["overlap_method"])
            else:
                pr=Projector(wf_t, wf_t0)
        end_setup= datetime.datetime.now()
        self.log.debug("==> Pawpyseed projectors setup"+ check_timing(start,end_setup))

        #Computing the whole S overlap matrix among MOs, including all VASP MOs from WAVECAR
        S=np.zeros((len(ks_mo_index),len(ks_mo_index)),dtype=complex)
        with suppress_stdout_stderr():  
            for idx,i in enumerate(ks_mo_index.values()):
                S[idx,:]=pr.single_band_projection(i) #Computing each ith row of the KS overlap matrix 
        S=S.T  #Because each single_band_projection() loop computes <\psi_1(t0)|\psi_i(t+dt)>....<\psi_n(t0)|\psi_i(t+dt)>
        #which is a column of S(t,t+dt) 
        #np.savetxt('overlap_VASP.dat',S) #Printing out full MOs overlap matrix for VASP check
        end_pawpyseed= datetime.datetime.now()
        self.log.debug("==> Pawpyseed overlap"+ check_timing(end_setup,end_pawpyseed))

        #Creating sub-determinants from whole S matrix in orbital space for each state-to-state overlap
        start_lu=datetime.datetime.now()
        S_GS=S[np.ix_(det_t0[0],det_t[0])] #<GS(t0)|GS(t)>
        #LU factorization for determinant evaluation and inverse
        lu_gs,piv_gs=lu_factor(S_GS)
        #Determinant from LU factorization
        diag = np.diag(lu_gs)
        logdet_gs = np.sum(np.log(np.abs(diag)))
        phase_gs  = np.prod(diag / np.abs(diag))
        piv_sign = (-1) ** np.sum(piv_gs != np.arange(len(piv_gs)))
        sign_gs = piv_sign * phase_gs
        det_gs  = sign_gs * np.exp(logdet_gs)
        #Setting up matrix determinant Lemma ingredients for speeding up multiple SD overlap computation.
        det_beta=det_gs #beta electrons always the same, alpha excitations only here.
        end_lu=datetime.datetime.now()
        self.log.debug("==> Determinant and LU factorization of GS SDs overlap done."+check_timing(start_lu,end_lu))

        #Computing the overlap matrix elements with joblib parallelization
        def compute_row(i):
            ''' Computes matrix elements of SDs overlap matrix by relying on 1st or 2nd order rank update 
                determinant matrix Lemma. 

                1-rank update:
                det(A+UV^T)=det(A)*(1+V^T @ A^-1 @ U)
                A -> nxn matrix
                U,V -> n-dimensional vectors
 
                2-rank update:
                same formula but U,V become nx2 matrices and so instead of 1 we have 2x2 identity matrix.

                In our case A is the precomputed SD overlap matrix <GS(t0)|GS(t)> and each update
                compute the overlap of other SDs (excited states) starting from the GS one.                          
            '''
            row = np.empty(nt, dtype=complex)
            for j in range(nt):
                # Update matrix on-the-fly depending on new SDs for determinant matrix lemma evaluation.
                if i==0 and j==0:
                    sgn_col=1
                    sgn_row=1
                    row[j]=sgn_col*sgn_row*det_beta*det_gs #Everything was computed already for this case.
                elif i==0 and j!=0: #1-rank update determinant lemma for columns
                    idx_c=np.argmax(det_t[j]-det_t[0])
                    sgn_col=(-1)**((det_length-1)-idx_c) #Permutation of columns to get SD string in correct energy order of orbitals.
                    sgn_row=1
                    diff_c=S[0:det_length,det_t[j][idx_c]]-S_GS[:,idx_c]
                    basis_c=np.zeros_like(diff_c)
                    basis_c[idx_c]=complex(1,0)
                    update=lu_solve((lu_gs,piv_gs),diff_c) #Inverse from LU factorization
                    det_alpha=det_gs*(1+update[idx_c]) #1-rank column update of new determinant
                    row[j]=sgn_col*sgn_row*det_alpha*det_beta 
                elif i!=0 and j==0: #1-rank update determinant lemma for rows
                    idx_r=np.argmax(det_t0[i]-det_t0[0]) 
                    sgn_col=1
                    sgn_row=(-1)**((det_length-1)-idx_r)
                    diff_r=S[det_t0[i][idx_r],0:det_length]-S_GS[idx_r,:]
                    basis_r=np.zeros_like(diff_r)
                    basis_r[idx_r]=complex(1,0)
                    update = lu_solve((lu_gs, piv_gs), diff_r, trans=1).T
                    det_alpha=det_gs*(1+update[idx_r]) #1-rank row update of new determinant
                    row[j]=sgn_col*sgn_row*det_alpha*det_beta 
                else: #2-rank update, both row and column change
                    #Column change
                    idx_c=np.argmax(det_t[j]-det_t[0]) 
                    sgn_col=(-1)**((det_length-1)-idx_c)
                    diff_c=S[0:det_length,det_t[j][idx_c]]-S_GS[:,idx_c]
                    basis_c=np.zeros_like(diff_c)
                    basis_c[idx_c]=complex(1,0)
                    #Row change
                    idx_r=np.argmax(det_t0[i]-det_t0[0]) 
                    sgn_row=(-1)**((det_length-1)-idx_r)
                    diff_r=S[det_t0[i][idx_r],0:det_length]-S_GS[idx_r,:] 
                    basis_r=np.zeros_like(diff_r)
                    basis_r[idx_r]=complex(1,0)
                    #cross terms have to be updated properly, column_takes precedence
                    diff_c[idx_r]=S[det_t0[i][idx_r],det_t[j][idx_c]]-S_GS[idx_r,idx_c]
                    diff_r[idx_c]=0
                    U=np.column_stack((diff_c,basis_r))
                    V=np.column_stack((basis_c,diff_r))
                    X=lu_solve((lu_gs,piv_gs),U)
                    M=np.eye(2)+V.T @ X
                    det_alpha=det_gs*det_slog(M)
                    row[j]=sgn_col*sgn_row*det_alpha*det_beta
            return row
       
        # Parallel computation over rows
        n0 = len(det_t0)
        nt = len(det_t)
        det_length=len(det_t0[0]) #Length of each SD string
        njobs=int(os.environ['OMP_NUM_THREADS']) 
        S_ij = np.array(Parallel(n_jobs=njobs)(delayed(compute_row)(i) for i in range(n0)))

        #Löwdin's orthogonalization -> we need to make S_{ij}(r,t+dt) unitary for local-diabatization, see Granucci JCP 2001
        #This may need to be commented, so it's the driver doing that, before checking for intruder states (to be tested!)
        #λ,V = LA.eigh(S_ij.T.conjugate() @ S_ij)
        #S_ij_lowdin=S_ij @ V @ np.diag(λ**(-1/2)) @ V.T.conjugate()
       
        end = datetime.datetime.now()
        self.log.debug("==> Slater determinants overlap done." + check_timing(end_lu,end))
        self.log.debug("==> Full overlap routine done." + check_timing(start,end))
       
        return S_ij
    
    @staticmethod
    def longest_common_prefix(A,B):
        """
        Compute the longest set of indexes k  for which A[:k,:k]==B[:k,:k] assuming that the same
        block, if presents, sits on the upper left corner of the matrices. Necessary for faster Schur complement determinant
        of many matrices that share the same A_11 block. 
        Stops scanning as soon as a mismatch is found.
        """
        n_rows, n_cols = A.shape
        common_seq = []
        for j in range(n_cols):
            # Check column j across all rows
            col_vals = np.concatenate((A[:, j], B[:, j]))
            if np.all(col_vals == col_vals[0]):  # All rows have same value at this position
                if len(common_seq) == 0:
                    common_seq.append(col_vals[0])
                else:
                    # Ensure it continues the increasing sequence by +1
                    if col_vals[0] == common_seq[-1] + 1:
                        common_seq.append(col_vals[0])
                    else:
                        break
            else:
                break  # mismatch in this column
        return common_seq
    
    #Routine above with determinant lemma should be faster!
    #Keeping this here just in case we need to go back to Schur complement
    def _get_overlap_schur(self, det_t: np.ndarray,  ks_mo_index: dict) -> np.ndarray:
        ''' 
        Function to get S_{ij}(r,t+dt) overlap matrix by using pawpyseed to compute overlaps between SDs out of KS orbitals from VASP.
        
        This routine relies on Schur complement to speed up multi SDs overlap evaluation.

        det_t: np.array with determinants occupation for current timestep with orbital occupations for selected states
        ks_mo_index: dictionary where full orbital labels and corresponding indexes are stored. First occupied orbital index is 0.
        
        self.QMin.template["overlap_method"] selects which method from pawpyseed to compute MO overlaps.
        if 'full',  default AE overlaps are calculated.
        if 'pseudo' only pseudowavefunction overlaps.
        '''
        
        start = datetime.datetime.now()
        
        from pawpyseed.core.projector import Wavefunction,Projector #Check if this is installed in $CONDA_PREFIX is done above

        filename=os.path.join(self.QMin.save["savedir"], f"det_index.{self.QMin.save['step']-1}")
        det_t0=np.loadtxt(filename,dtype=int)
        self.log.debug("Occupation strings of Slater determinants at previous timestep")
        self.log.debug(det_t0)
        self.log.debug(f"Checking OMP_NUM_THREADS parallelization for pawpyseed: {os.environ['OMP_NUM_THREADS']}")

        #Initializing AE or PS wavefunctions (KS valence MO wavefunctions)
        self.log.debug("-----------------------------")
        self.log.debug("PAWPYSEED overlap calculation")
        self.log.debug("-----------------------------\n")
        with suppress_stdout_stderr():  
            wf_t = Wavefunction.from_files(struct=os.path.join(self.QMin.control["workdir"],"CONTCAR"),  #current timestep wf
                                            wavecar=os.path.join(self.QMin.control["workdir"],"WAVECAR"),
                                            cr=os.path.join(self.QMin.control["workdir"],"POTCAR"),
                                            vr=os.path.join(self.QMin.control["workdir"],"vasprun.xml"))
            wf_t0 = Wavefunction.from_files(struct=os.path.join(self.QMin.save["savedir"],f"CONTCAR.{self.QMin.save['step']-1}"), #previous timestep wf
                                            wavecar=os.path.join(self.QMin.save["savedir"],f"WAVECAR.{self.QMin.save['step']-1}"),
                                            cr=os.path.join(self.QMin.save["savedir"],"POTCAR"),
                                            vr=os.path.join(self.QMin.save["savedir"],f"vasprun.xml.{self.QMin.save['step']-1}"))
            if self.QMin.template["overlap_method"]=="pseudo":
                pr=Projector(wf_t, wf_t0,method=self.QMin.template["overlap_method"])
            else:
                pr=Projector(wf_t, wf_t0)
        end_setup= datetime.datetime.now()
        self.log.debug("==> Pawpyseed projectors setup"+ check_timing(start,end_setup))

        #Computing the whole S overlap matrix among MOs, including all VASP MOs from WAVECAR
        S=np.zeros((len(ks_mo_index),len(ks_mo_index)),dtype=complex)
        with suppress_stdout_stderr():  
            for idx,i in enumerate(ks_mo_index.values()):
                S[idx,:]=pr.single_band_projection(i) #Computing each ith row of the KS overlap matrix 
        S=S.T  #Because each single_band_projection() loop computes <\psi_1(t0)|\psi_i(t+dt)>....<\psi_n(t0)|\psi_i(t+dt)>
        #which is a column of S(t,t+dt) 
        #np.savetxt('overlap_VASP.dat',S) #Printing out full MOs overlap matrix for VASP check
        end_pawpyseed= datetime.datetime.now()
        self.log.debug("==> Pawpyseed overlap"+ check_timing(end_setup,end_pawpyseed))

        #Creating sub-determinants from whole S matrix in orbital space for each state-to-state overlap
        det_beta=det_slog(S[np.ix_(det_t0[0],det_t[0])]) #beta electrons always the same, alpha excitations.
        #Schur complement for speeding up determinant evaluation for SD overlaps for alpha electrons.
        #All these determinants share a big block, which is always the same and can be pre-computed to make use of Schur complement for full determinant.
        start_lu=datetime.datetime.now()
        common_idx=SHARC_VASP.longest_common_prefix(det_t0,det_t)
        self.log.debug(f"common set of orbitals among all SDs: {common_idx}")
        sub_block=S[np.ix_(common_idx,common_idx)]
        lu_block, piv_block = lu_factor(sub_block)
        # determinant info from LU
        diag = np.diag(lu_block)
        logdet_block = np.sum(np.log(np.abs(diag)))
        phase_block = np.prod(diag / np.abs(diag))
        piv_sign = (-1) ** np.sum(piv_block != np.arange(len(piv_block)))
        sign_block = piv_sign * phase_block
        det_block = sign_block * np.exp(logdet_block)
        end_lu=datetime.datetime.now()
        self.log.debug("==> Determinant of the sub_block and LU factorization done."+check_timing(start_lu,end_lu))

        #Computing the overlap matrix elements with joblib parallelization
        def compute_row(i):
            row = np.empty(nt, dtype=complex)
            for j in range(nt):
                # Compute submatrix on-the-fly to save memory
                submatrix = S[np.ix_(det_t0[i], det_t[j])]
                #Change of determinant sign because of re-ordering of orbitals in SD string
                #Reordering assume the new orbital after excitation will be put at the end of the string
                #Energy-based order of orbitals in SD
                if i!=0:
                    sgn_row=(-1)**((det_length-1)-np.argmax(det_t0[i]-det_t0[0]))
                else:
                    sgn_row=1
                if j!=0:
                    sgn_col=(-1)**((det_length-1)-np.argmax(det_t[j]-det_t[0]))
                else:
                    sgn_col=1
                row[j] = sgn_col*sgn_row*schur_det(submatrix, len(common_idx), det_block, lu_block,piv_block) * det_beta
            return row
        # Parallel computation over rows
        n0 = len(det_t0)
        nt = len(det_t)
        det_length=len(det_t0[0]) #Length of each SD string
        njobs=int(os.environ['OMP_NUM_THREADS']) 
        S_ij = np.array(Parallel(n_jobs=njobs)(delayed(compute_row)(i) for i in range(n0)))

        end = datetime.datetime.now()
        self.log.debug("==> Slater determinants overlap done." + check_timing(end_lu,end))
        self.log.debug("==> Full overlap routine done." + check_timing(start,end))
        
        return S_ij

    #Routine for computing overlap matrix with no speedup. Only for debugging purposes.
    def _get_overlap_fulldet(self, det_t: np.ndarray,  ks_mo_index: dict) -> np.ndarray:
        ''' 
        Function to get S_{ij}(r,t+dt) overlap matrix by using pawpyseed to compute overlaps between SDs out of KS orbitals from VASP.

        This does not provide any speedup for multi SDs evaluation. ONLY FOR DEBUGGING!

        det_t: np.array with determinants occupation for current timestep with orbital occupations for selected states
        ks_mo_index: dictionary where full orbital labels and corresponding indexes are stored. First occupied orbital index is 0.
        
        self.QMin.template["overlap_method"] selects which method from pawpyseed to compute MO overlaps.
        if 'full',  default AE overlaps are calculated.
        if 'pseudo' only pseudowavefunction overlaps.
        '''
        
        start = datetime.datetime.now()
        
        from pawpyseed.core.projector import Wavefunction,Projector #Check if this is installed in $CONDA_PREFIX is done above

        filename=os.path.join(self.QMin.save["savedir"], f"det_index.{self.QMin.save['step']-1}")
        det_t0=np.loadtxt(filename,dtype=int)
        self.log.debug("Occupation strings of Slater determinants at previous timestep")
        self.log.debug(det_t0)
        self.log.debug(f"Checking OMP_NUM_THREADS parallelization for pawpyseed: {os.environ['OMP_NUM_THREADS']}")

        #Initializing AE or PS wavefunctions (KS valence MO wavefunctions)
        self.log.debug("-----------------------------")
        self.log.debug("PAWPYSEED overlap calculation")
        self.log.debug("-----------------------------\n")
        with suppress_stdout_stderr():  
            wf_t = Wavefunction.from_files(struct=os.path.join(self.QMin.control["workdir"],"CONTCAR"),  #current timestep wf
                                            wavecar=os.path.join(self.QMin.control["workdir"],"WAVECAR"),
                                            cr=os.path.join(self.QMin.control["workdir"],"POTCAR"),
                                            vr=os.path.join(self.QMin.control["workdir"],"vasprun.xml"))
            wf_t0 = Wavefunction.from_files(struct=os.path.join(self.QMin.save["savedir"],f"CONTCAR.{self.QMin.save['step']-1}"), #previous timestep wf
                                            wavecar=os.path.join(self.QMin.save["savedir"],f"WAVECAR.{self.QMin.save['step']-1}"),
                                            cr=os.path.join(self.QMin.save["savedir"],"POTCAR"),
                                            vr=os.path.join(self.QMin.save["savedir"],f"vasprun.xml.{self.QMin.save['step']-1}"))
            if self.QMin.template["overlap_method"]=="pseudo":
                pr=Projector(wf_t, wf_t0,method=self.QMin.template["overlap_method"])
            else:
                pr=Projector(wf_t, wf_t0)
        end_setup= datetime.datetime.now()
        self.log.debug("==> Pawpyseed projectors setup"+ check_timing(start,end_setup))

        #Computing the whole S overlap matrix among MOs, including all VASP MOs from WAVECAR
        S=np.zeros((len(ks_mo_index),len(ks_mo_index)),dtype=complex)
        with suppress_stdout_stderr():  
            for idx,i in enumerate(ks_mo_index.values()):
                S[idx,:]=pr.single_band_projection(i) #Computing each ith row of the KS overlap matrix 
        S=S.T  #Because each single_band_projection() loop computes <\psi_1(t0)|\psi_i(t+dt)>....<\psi_n(t0)|\psi_i(t+dt)>
        #which is a column of S(t,t+dt) 
        #np.savetxt('overlap_VASP.dat',S) #Printing out full MOs overlap matrix for VASP check
        end_pawpyseed= datetime.datetime.now()
        self.log.debug("==> Pawpyseed overlap"+ check_timing(end_setup,end_pawpyseed))

        #Creating sub-determinants from whole S matrix in orbital space for each state-to-state overlap
        det_beta=det_slog(S[np.ix_(det_t0[0],det_t[0])]) #beta electrons always the same, alpha excitations.
        end_gs=datetime.datetime.now()
        self.log.debug("==> Determinant of the GS overlap evaluated."+check_timing(end_pawpyseed,end_gs))

        #Computing the overlap matrix elements with joblib parallelization
        def compute_row(i):
            row = np.empty(nt, dtype=complex)
            for j in range(nt):
                # Compute submatrix on-the-fly to save memory
                submatrix = S[np.ix_(det_t0[i], det_t[j])]
                #Change of determinant sign because of re-ordering of orbitals in SD string
                #Reordering assume the new orbital after excitation will be put at the end of the string
                #Energy-based order of orbitals in SD
                if i!=0:
                    sgn_row=(-1)**((det_length-1)-np.argmax(det_t0[i]-det_t0[0]))
                else:
                    sgn_row=1
                if j!=0:
                    sgn_col=(-1)**((det_length-1)-np.argmax(det_t[j]-det_t[0]))
                else:
                    sgn_col=1
                row[j] = sgn_col*sgn_row*np.linalg.det(submatrix) * det_beta
            return row
        # Parallel computation over rows
        n0 = len(det_t0)
        nt = len(det_t)
        det_length=len(det_t0[0]) #Length of each SD string
        njobs=int(os.environ['OMP_NUM_THREADS']) 
        S_ij = np.array(Parallel(n_jobs=njobs)(delayed(compute_row)(i) for i in range(n0)))
        
        end = datetime.datetime.now()
        self.log.debug("==> Slater determinants overlap done." + check_timing(end_gs,end))
        self.log.debug("==> Full overlap routine done." + check_timing(start,end))
        
        return S_ij

    def _get_phases(self,flag: str, overlap: np.ndarray[complex,2] ) -> np.ndarray[complex,1]:
        ''' 
        Function for phase correction of adiabatic states at different time steps relying on calculation of overlap matrix.
        Either 'none' phase correction or 'simple' and 'robust' algorithms can be chosen.
        the actual function is in utils.py

        flag: 'none', 'simple' (Alekey et al. J. Phys. Chem. Lett. 2018, 9, 6096−6102) , 'robust' (Subotnik et al. JCTC 2020, 16, 835−846)
        overlap: overlap matrix 

        returns: 1D array of complex numbers for phase correction.
        '''
        
        start = datetime.datetime.now()

        if flag=="none":
            phases=np.ones(self.QMin.molecule["nmstates"],dtype=complex) 
        
        elif flag=="simple" or flag=="robust":
            phases=phase_correction_cmplx(overlap,flag)

        end = datetime.datetime.now()
        self.log.debug("==> Phase correction routine done." + check_timing(start,end))
        
        return phases
    
    def _write_transitions(self,ks_es: dict, mo_index : dict):
        ''' 
        Function for writing out to a text file the orbitals involved in the selected transitions.
        '''

        step=self.QMin.save['step']
        if step==0:
            input_path = os.path.join(self.QMin.save["savedir"], "TRANSITIONS_t0")
        else:
            input_path = os.path.join(self.QMin.save["savedir"], f"TRANSITIONS.{step}")
        input_str = f"VASP states and info for step n.{step}\n"
        input_str += f"Bangap: {ks_es['H->L']:<10.5f} eV\n"
        input_str += f"{'Excited state n.':<20}{'orbitals(vasp band indexes)':<30}{'Energy(eV)':<20}\n"
        for n,(i,j) in enumerate(ks_es.items()):
            tmp=i.split('->')
            label=i+" ("+str(mo_index[tmp[0]]+1)+'->'+str(mo_index[tmp[1]]+1)+")" # +1 is because VASP index start from 1 and not 0
            input_str += f"{n+1:<20}{label:<30}{j:<10.5f}\n"
        writefile(input_path, input_str)
        
        return 


#-------------------| Functions for generating inputstrings for writing VASP input files |---------------------------- 

    def _generate_inputstr_INCAR(self) -> str:
        """
        Generate INCAR input file string for VASP from QMin object
        """
        
        inputstring = f"SISTEM = {self.QMin.template['system']}\n"
        inputstring += f"MAXMEM = {self.QMin.resources['memory']}\n" #allocated memory in Mb for each MPI rank
        inputstring += f"NCORE = {self.QMin.resources['ncore']}\n" #n. of cores working on a single orbital.
        inputstring += f"ISMEAR = {self.QMin.template['ismear']}\n"
        inputstring += f"SIGMA = {self.QMin.template['sigma']}\n"
        inputstring += f"EFERMI = MIDGAP\n"
        inputstring += f"ISPIN = {self.QMin.template['ispin']}\n" #Only singlets currently available
        inputstring += f"GGA = {self.QMin.template['gga']}\n"
        if self.QMin.template['time_vasp'] is not None: 
            inputstring += f"TIME = {self.QMin.template['time_vasp']}\n"
        if self.QMin.template['ialgo'] is not None:
            inputstring += f"IALGO = {self.QMin.template['ialgo']}\n"
        inputstring += f"ALGO = {self.QMin.template['algo']}\n"
        inputstring += f"NELM = {self.QMin.template['nelm']}\n"
        inputstring += f"EDIFF = {self.QMin.template['ediff']}\n"
        if self.QMin.template["nbands"] is not None:
            inputstring += f"NBANDS = {self.QMin.template['nbands']}\n" 
        if self.QMin.template["lreal"] is not None:
            inputstring += f"LREAL = {self.QMin.template['lreal']}\n" 
        inputstring += f"ENCUT = {self.QMin.template['encut']}" 
        
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
        
        elements = self.QMin.molecule["elements"]
        scale_param = self.QMin.template["scale_param"]
        a1 = self.QMin.template["a1"]
        a2 = self.QMin.template["a2"]
        a3 = self.QMin.template["a3"]
        system = self.QMin.template["system"]

        inputstring = f"{system}\n"
        inputstring += f"{scale_param}\n"
        inputstring += f"{a1[0]} {a1[1]} {a1[2]}\n"
        inputstring += f"{a2[0]} {a2[1]} {a2[2]}\n"
        inputstring += f"{a3[0]} {a3[1]} {a3[2]}\n"

        for i in list(dict.fromkeys(elements)):  
            inputstring += f" {i}"
        inputstring += f"\n"

        self.log.debug("sorting VASP geometry indexes") 
        self.log.debug(self._el_vasp) 
        for i in self._el_vasp:  
            inputstring += f" {len(i)}"
        inputstring += f"\n"
        inputstring += f"Cartesian\n" #Hard-coded cartesian coordinates (Ang.) for input. Other options may be available in VASP
        
        self.log.debug("sorted VASP geometry") 
        self.log.debug(self._coords_vasp) 
        for i in self._coords_vasp:
            inputstring += f" {i[0]*au2a:>16.7f}  {i[1]*au2a:>16.7f}  {i[2]*au2a:>16.7f}\n"
        
        self.log.debug("Testing VASP geometry sorting indices")
        self.log.debug(self._indices_vasp)
        
        return inputstring


# ---------------------------------| Saving & copying files after each run |---------------------------------------------------
    def _save_files(self, workdir : str, savedir : str ) -> None:
        """
        Save files (WAVECAR, vasprun.xml , CONTCAR, POTCAR, OUTCAR) to savedir
        Necessary for pawpyseed computation of overlap matrix 
        Naming convention: file.job.step
        """
        step = self.QMin.save["step"]

        self.log.debug(f"Saving files from step {step}")
        
        #POTCAR only once, never changes, to suppress pawpyseed warning
        if self.QMin.save["step"]==0:
            fromfile = os.path.join(workdir, "POTCAR")
            tofile = os.path.join(savedir, f"POTCAR")
            shutil.copy(fromfile, tofile)

        #vasprun.xml
        fromfile = os.path.join(workdir, "vasprun.xml")
        tofile = os.path.join(savedir, f"vasprun.xml.{step}")
        shutil.copy(fromfile, tofile)
        
        #WAVECAR
        fromfile = os.path.join(workdir, "WAVECAR")
        tofile = os.path.join(savedir, f"WAVECAR.{step}")
        shutil.copy(fromfile, tofile)
        
        #CONTCAR
        fromfile = os.path.join(workdir, "CONTCAR")
        tofile = os.path.join(savedir, f"CONTCAR.{step}")
        shutil.copy(fromfile, tofile)
        
        #OUTCAR
        fromfile = os.path.join(workdir, "OUTCAR")
        tofile = os.path.join(savedir, f"OUTCAR.{step}")
        shutil.copy(fromfile, tofile)

        return
    
    def _copy_files(self, workdir: str, savedir : str) -> None:
        """
        Copy WAVECAR from previous time step for speeding up density convergence of VASP calculation

        workdir:    Working directory
        savedir:    Save directory
        """
        
        step = self.QMin.save["step"]

        if step > 0:
            if os.path.isfile(os.path.join(savedir,f"WAVECAR.{step-1}")):
                self.log.debug(f"Using WAVECAR from step {step-1} for density preconditioning.")
                shutil.copy(os.path.join(savedir, f"WAVECAR.{step-1}"), os.path.join(workdir, "WAVECAR"))
        
        return

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

# --------------------------------------------------------------------------------------------------------------------

# Some usefull functions
def check_timing(starttime : datetime.datetime ,endtime : datetime.datetime):
    """ Simple function for computing runtime between starttime and endtime.

        starttime: initial datetime.datetime.now() object
        endtime: final datetime.datetime.now() object

        return: Output string with Runtime in days, hours, minutes and seconds
    """

    runtime = endtime-starttime
    hours = runtime.seconds // 3600
    minutes = runtime.seconds // 60 - hours * 60
    seconds = runtime.seconds % 60
    seconds += 1.0e-6 * runtime.microseconds
    output=(" Timings:  %i d  %i h  %i m  %f s\n" % (runtime.days, hours, minutes, seconds))

    return output

def schur_det(matrix,size_block,det_block,lu_block,piv_block):
    '''
    Computes determinant of a big input 'matrix' by relying on Schur complement.
    Assuming  a partitioning -> matrix=A=(A_11 A_12; A_21 A_22) then the Schur complement is S=A_22-A_21*A_11^-1*A_12
    from which it follows: det(A)=det(A_11)*det(S).
    This assumes that A_11 and its inverse are precomputed to speed up multideterminant evauluation when A_11 is a fixed block.
    it actually relies on LU decomposition of A_11 instead of numerically storing A_11^-1 for more numerical stability
    
    input quantitites:
    matrix -> Full-matrix (A in the notation above)
    size_block -> size n of the upper A_11 nxn block 
    det_block -> precomputed determinant of A_11
    lu_block -> LU factorization output of A_11 
    piv_block -> LU factorization output of A_11 

    return:
    det(matrix)
    '''

    n = matrix.shape[0]
    assert matrix.shape[1] == n, "Matrix must be square. Something is wrong with Schur complement"
    assert 0 < size_block < n, "Block size k must be valid. Check Schur complement routine."
    # Partition blocks
    A12 = matrix[:size_block, size_block:]
    A21 = matrix[size_block:, :size_block]
    A22 = matrix[size_block:, size_block:]
    # Calculation of Schur determinant
    X = lu_solve((lu_block,piv_block),A12)
    S = A22 - A21 @ X
    det=det_block*det_slog(S)
    return det

class suppress_stdout_stderr:
    """
    A context manager that redirects stdout and stderr to /dev/null (on Unix)
    or NUL (on Windows), suppressing all output including C-level prints.
    Added to suppress output from pawpyseed call to stdout in get_overlap().
    So that no need to comment out printf statements in pawpyseed .c source files is necessary.
    """

    def __enter__(self):
        # Flush Python's sys.stdout and sys.stderr
        sys.stdout.flush()
        sys.stderr.flush()
        # Open null files for stdout and stderr
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the original file descriptors
        self.save_fds = [os.dup(1), os.dup(2)]
        # Redirect stdout and stderr to null
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush Python's sys.stdout and sys.stderr
        sys.stdout.flush()
        sys.stderr.flush()
        # Restore original stdout and stderr file descriptors
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all fds
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_VASP().main()
