#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper
from copy import deepcopy
import re

import numpy as np
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import batched, expand_path, itmult, mkdir, question, readfile, writefile


# ---------------------------------| Infos |---------------------------------------------------------------------------

#TODO: Change BASICORCA to your desired name

__all__ = ["SHARC_BASICORCA"]  # Only export interface class


#TODO: This will be shown in the header when running a single point or sharc.x
AUTHORS = "Hans Georg Gallmetzer"
VERSION = "0.1"
VERSIONDATE = datetime.datetime(2024, 6, 17)
#TODO: This will be shown in the setup scripts
NAME = "BASICORCA"
DESCRIPTION = "a really basic orca interface, just to show you how an inteface works"

CHANGELOGSTRING = """17.06.2024:     Initial version 0.1 by Georg
- Only energies, TDMs and gradients
- Only singlets"""

all_features = set(
    [
        #TODO: requests that your interface can fullfill. Delete the ones that cannot be used. 
        "h",
        "dm",
        "grad",
        # Rest of the possible requests:
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
                "molcharge": 0,
            }
        )
        self.QMin.template.types.update(
            {
                "basis": str,
                "functional": str,
                "molcharge": int,
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
        input_path = os.path.join(workdir, "ORCA.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        # Setup ORCA
        
        exec_str = f"{os.path.join(qmin.resources['orcadir'],'orca')} ORCA.inp"
        exit_code = self.run_program(workdir, exec_str, os.path.join(workdir, "ORCA.log"), os.path.join(workdir, "ORCA.err"))

        #TODO: Errorhandling in case exit_code != 0

        #TODO: Post processing, molden file, wfoverlap det/mo files, ...

        #TODO: Copy restart files to savedir


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

        self.QMout["runtime"] = datetime.datetime.now() - starttime

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

        #TODO: Copy restart from savedir and input files needed for calculation here

        # Run QM
        starttime = datetime.datetime.now()
        exec_str = "<command to run QM program>"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "BASICORCA.log"), os.path.join(workdir, "BASICORCA.err")
        )
        endtime = datetime.datetime.now()

        #TODO: Maybe some errorhandling in case exit_code != 0

        #TODO: Post processing, molden file, wfoverlap det/mo files, ...

        #TODO: Copy restart files to savedir
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
        with open(os.path.join(scratchdir, f"ORCA.log"), "r", encoding="utf-8") as file:
                log_file = file.read()

                energies = self._get_energy(log_file)
                for i in range(sum(self.QMin.molecule["states"])):
                    self.QMout["h"][i][i] = energies[(1, i + 1)]

                if self.QMin.requests["dm"]:
                    # Diagonal elements
                    dipoles_gs = self._get_dipole_moment(log_file)
                    dipoles_trans = self._get_transition_dipoles(log_file)
                    print("dipoles: ", dipoles_gs)
                    print("tdms: ", dipoles_trans)
                    print("grad: ", self.QMin.maps["gradmap"])

                    states_to_do = deepcopy(self.QMin.control["states_to_do"])
                    print("states_to_do: ", max(states_to_do)-1)
                    ex_state = list(self.QMin.maps["gradmap"])[0][1]
                    
                    states_to_do_max = max(states_to_do)-1
                    self.QMout["dm"][:,0,0] = dipoles_gs[-1]
                    self.QMout["dm"][:,ex_state,ex_state] = dipoles_gs[0]
                    for i in range(states_to_do_max):
                        self.QMout["dm"][:, 0, i+1] = dipoles_trans[i]
                        self.QMout["dm"][:, i+1, 0] = - dipoles_trans[i]
        
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
        #TODO: Setup things that should be asked during setup here
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        #TODO: Copy files that are needed for interface in setup
        pass


# ---------------------------------| Additional Methods |------------------------------------------------------------

#TODO: Put all of your extra methods in here. They all should start with and underscore "_". For example a method that parses the gradients from the output-file should be called _get_grad().
    @staticmethod
    def generate_inputstr(qmin: QMin) -> str:
        """
        Generate ORCA input file string from QMin object
        """
        job = qmin.control["jobid"]
        charge = qmin.template["molcharge"]

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
        if max(states_to_do) > 0:
            string += f"%tddft\n\ttda false\n"
            string += f"\tnroots {max(states_to_do)-1}\n"
            if do_grad:
                string += "\tsgradlist " +  ",".join([str(i[1]) for i in qmin.maps["gradmap"]]) + "\n"

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

    def _create_aoovl(self) -> None:
        pass

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
    
# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_BASICORCA().main()
