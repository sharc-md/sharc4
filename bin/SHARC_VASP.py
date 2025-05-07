#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper
import itertools
import shutil
from numpy import ndarray
import numpy as np
import re
from constants import *
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import  expand_path, question, link,  mkdir,  writefile,is_exec


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
                "ncpu" : 2, #Default number of cpus for mpi run with VASP 
                "memory" : 2000 #resetting memory default value for each VASP MPI rank to 2Mb instead of 1Mb
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
                "ismear": -2, #Smearing parameter for VASP, default set to "no smearing" (-2) or read from previous WAVECAR
                "sigma": 0.0, #Smearing width
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
                "sigma": float,
                "encut" : int,
                "nbands" : int,
                "scale_param": int, #scaling parameter for VASP unit cell
                "a1": list, #1st unit cell lattice vector
                "a2": list, #2nd unit cell lattice vector
                "a3": list, #3rd unit cell lattice vector
            }
        )
        self._indices =  None #Needed for sorting VASP input/output properly according to QM.in geometry format


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
            self.log.error("sigma in template has to be a float, check vasp wiki")
            raise ValueError()

        if not isinstance(self.QMin.template["ismear"],int):
            self.log.error("ismear in template has to be an integer, check vasp wiki")
            raise ValueError()

        if not isinstance(self.QMin.template["encut"],int):
            self.log.error("encut in template has to be an integer, check vasp wiki")
            raise ValueError()

        if not isinstance(self.QMin.template["scale_param"],int):
            self.log.error("scale_param in template has to be an integer, check vasp wiki")
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

        self.log.info(f"Scratchdir: {self.QMin.resources['scratchdir']}")
        self.log.info(f"Savedir: {self.QMin.save['savedir']}")
        
        if (any(num > 0 for num in self.QMin.molecule["states"][1:]) or self.QMin.molecule["states"][0] == 0):
            self.log.error("Current VASP implementation only deals with singlets!")
            raise ValueError()
        # Checking for MPI installation. It is strongly recommended for proper usage of VASP
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

        #probably this has tio be refined -> later stage, not so fundamental currently.

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

            self.log.info("Specify the path to the VASP binary files")
            self.setupINFOS["vaspdir"] = question("path to VASP binary files", str, KEYSTROKES=KEYSTROKES)

            self.log.info("Specify the path to the VASP potcar file")
            self.setupINFOS["potcardir"] = question("path to VASP POTCAR file", str, KEYSTROKES=KEYSTROKES)
        
            self.log.info("\n\nSpecify a scratch directory. The scratch directory will be used to run the calculations.")
            self.setupINFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
        
        return INFOS

    #That's for setuop script as well, toghter with get_infos and get_features. Probably has to be further refined later.
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

        self._save_files(self.QMin.control["workdir"])
        #self.clean_savedir()

        self.log.debug("All jobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def create_restart_files(self) -> None:
            pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        #Check for pawpyseed (https://github.com/kylebystrom/pawpyseed) installation in the conda environment the user uses for running SHARC
        # This is compulsory for calculating overlaps and so run VASP-SHARC dynamics !!
        # SH with VASP can only run by using overlaps, no NACs available.
        if self.QMin.requests["overlap"]:
            try: 
                from pawpyseed.core.projector import Wavefunction,Projector
            except: 
                self.log.error("pawpyseed is not installed in your environment, please do it first and make sure it appears in your 'pip list'!")
                self.log.error("pawpyseed repo: https://github.com/kylebystrom/pawpyseed")
                raise ValueError("pawpyseed is not installed in your python env! do it first https://github.com/kylebystrom/pawpyseed")


        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                self.log.error(f"Found unsupported request {req}.")
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
        self.log.debug(self._indices)
        #"indices" is necessary for proper sorting of output forces, see _generate_inputstr_POSCAR
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
        
        exec_str = f"mpirun -np {ncpu} {os.path.join(qmin.resources['vaspdir'],'vasp_std')} > {os.path.join(workdir, 'VASP.out')}"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "VASP.out"), os.path.join(workdir, "VASP.err"))

        endtime = datetime.datetime.now()

        return exit_code, endtime - starttime


# ---------------------------------| Parsing output data from VASP calculations |-------------------------------------

    def getQMout(self) -> None:
        """
        Parse VASP output files
        """

        self.log.debug(self._indices)
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
            energies,ks_info = self._get_energies(OUTCAR)
            for i in range(len(energies)):
                self.QMout["h"][i][i] = energies[i]
        #ks_info contain MOs excitation energies in eV and corresponding info for the MOs involved in the excitation
        # It is needed to compute overlaps, see corresponding self._ functions.
        
        # Populate forces (gradients)
        if self.QMin.requests["grad"]:
            self.QMout.grad = self._get_forces(OUTCAR)

        # Populate overlaps
        # to be done 
        
        return self.QMout
    

    def _get_forces(self, vasp_out: str) -> np.ndarray:
        """
        Get GS forces from VASP output (OUTCAR) file.
        Each ES gradient does coincide with the GS one -> CPA approximation!!
        Forces are output in au units and read in eV/Ang. from VASP OUTCAR

        vasp_out: VASP OUTCAR file
        """
        
        natom = self.QMin.molecule["natom"]
        nmstates = self.QMin.molecule["nmstates"]
        coords=self.QMin.coords["coords"] #needed for sorting output forces from VASP properly, see below
        elements = self.QMin.molecule["elements"]

        start_marker=r"\sPOSITION\s+TOTAL-FORCE \(eV\/Angst\)\n\s\-+\n"
        end_marker=r"\s\-+\n"
        pattern = rf'{start_marker}(.*?){end_marker}'
        match = re.search(pattern, vasp_out, re.DOTALL)
        #Forces from VASP in eV/Ang.
        forces=np.array([i.split() for i in match.group(1).splitlines()],dtype=np.float64)[:,3:]
        # Sorting output forces according to QMin geometry format 
        self.log.debug(self._indices)
        #forces=forces[self.indices]
        # To be removed after testing
        print("TESTING FORCES PARSING")
        print(forces)
        
        forces=forces/au2eV*au2a #Changing to forces in atomic units
        gradients=np.array([forces for i in range(0,nmstates)]) #(nmstates,natom,3) Each ES gradient is equal to GS one, CPA approximation!!
        
        # To be removed after testing
        print("TESTING FORCES PARSING FOR SHARC")
        print(gradients)
        
        return gradients


    def _get_energies(self, vasp_out: str) -> tuple[list,tuple[dict,dict]]:
        """
        Eigenstate energies from VASP. Excitation energies are computed as orbital energy difference between KS eigenvalues!
        GS energy is the correct DFT one, higher-lying state energies are obtained by summing excitation energy (from KS MOs difference) to GS energy.
        Currently, only single excitations are considered! No double, triple excitations etc. 
        Moreover, GS is assumed to be closed shell and only singlet excited states are accounter for.
        Refinements will follow.
        
        reference: J.Chem.TheoryComput.2013,9,4959−4972 (Akimov & Prezhdo)

        vasp_out: VASP OUTCAR file
        """
        natom = self.QMin.molecule["natom"]
        nmstates = self.QMin.molecule["nmstates"]

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
        #Reading and sorting KS orbital energies (Fermi energy set to 0!)
        n_o=int(np.sum(occ)/2) # N. of occupied MO, assuming closed shell
        n_u=len(ks_en)-n_o
        ks_o=dict([('H-'+ str(n_o-1-i),ks_en[i]) for i in range(0,n_o-1)]) #orbitals below H
        ks_o.update({'H':ks_en[n_o-1]}) # H orbital
        ks_u={'L':ks_en[n_o]} #L orbital
        ks_u.update(dict([('L+'+ str(i-n_o),ks_en[i]) for i in range(n_o+1,n_o+n_u)])) #obitals above L
        #### Computing excitation energies by orbital energy difference among KS MOs and selecting first "nmstates" only #### 
        ks_mo= ks_o|ks_u
        ks_es_all=dict()
        for i in ks_o:
            for j in ks_u:
                ks_es_all.update({ i+'->'+j : ks_u[j]-ks_o[i]})
        ks_es_all=dict(sorted(ks_es_all.items(), key=lambda x: x[1])) #sorting excitation energies
        ks_es= dict(itertools.islice(ks_es_all.items(), nmstates-1)) # Getting first nmstates-1 excitation energies
        #### Create the output list with GS energy and nmstates-1 excited state energies for SHARC driver ####
        energies=list()
        pattern=rf'  energy  without entropy=\s+(.*?)  energy\('
        gs_en=float(re.search(pattern,vasp_out).group(1))/au2eV #GS energy in atomic units
        energies.append(gs_en)
        for i in ks_es.values():
            energies.append(energies[0]+i/au2eV)

        #To be removed after check
        print("TESTING ENERGIES PARSING")
        print(energies)
        print(ks_es)
        
        return energies,(ks_es,ks_mo)




#-------------------| Functions for generating inputstrings for writing VASP input files |---------------------------- 

    def _generate_inputstr_INCAR(self) -> str:
        """
        Generate INCAR input file string for VASP from QMin object
        """
        
        system = self.QMin.template["system"]
        ismear = self.QMin.template["ismear"]
        sigma = self.QMin.template["sigma"]
        nbands = self.QMin.template["nbands"]
        gga = self.QMin.template["gga"]
        encut = self.QMin.template["encut"]
        memory= self.QMin.resources["memory"]

        
        inputstring = f"SISTEM = {system}\n"
        inputstring = f"MAXMEM = {memory}\n" #allocated memory in Mb for each MPI rank
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
        
        coords = self.QMin.coords["coords"]
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

        elements_nr=list(dict.fromkeys(elements)) #Non-redundant list of element types for proper VASP input file format
        
        indx=list() #List of list of indexes for each element, redundant elements lead to inner lists with more than one element
        
        for i in elements_nr:
            tmp=list()
            for n,j in enumerate(elements):
                tmp.append(n) if j==i else None
            indx.append(tmp)
        
        coords_vasp=[coords[i] for i in sum(indx,[])]
        
        for i in elements_nr:  
            inputstring += f" {i}"
        inputstring += f"\n"
        for i in indx:  
            inputstring += f" {len(i)}"
        inputstring += f"\n"
        inputstring += f"cart\n" #Hard-coded cartesian coordinates (Ang.) for input. Other options may be available in VASP
        
        for i in coords_vasp:
            inputstring += f" {i[0]*au2a:>16.9f}  {i[1]*au2a:>16.9f}  {i[2]*au2a:>16.9f}\n"
        
        self._indices=[] #For sorting different geometry format of QMin and VASP POSCAR
        for i in coords:
            for n,j in enumerate(coords_vasp):
                self._indices.append(n) if np.array_equal(j,i) else None
        
        self.log.debug(self._indices)

        return inputstring


# ---------------------------------| Saving files after each run |---------------------------------------------------
    def _save_files(self, workdir: str) -> None:
        """
        Save files (WAVECAR, OUTCAR, POTCAR) to savedir
        Naming convention: file.job.step
        """
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]
        
        #OUTCAR not sure this is needed, let's just leave it now for testing purposes
        fromfile = os.path.join(workdir, "OUTCAR")
        tofile = os.path.join(savedir, f"OUTCAR.{step}")
        shutil.copy(fromfile, tofile)

        #WAVECAR
        fromfile = os.path.join(workdir, "WAVECAR")
        tofile = os.path.join(savedir, f"WAVECAR.{step}")
        shutil.copy(fromfile, tofile)
        
        #POTCAR we copy only once, they are all equal -> needed by pawpyseed
        if step==0:
            fromfile = os.path.join(workdir, "POTCAR")
            tofile = os.path.join(savedir, f"POTCAR")
            shutil.copy(fromfile, tofile)
        
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


# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_VASP().main()
