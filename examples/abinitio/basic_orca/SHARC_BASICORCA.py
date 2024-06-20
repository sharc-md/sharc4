#!/usr/bin/env python3
import datetime
import os
from io import TextIOWrapper
from copy import deepcopy
import re
import struct
import shutil
import sys

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

                states_to_do = deepcopy(self.QMin.control["states_to_do"])
                ex_state = list(self.QMin.maps["gradmap"])[0][1]
                states_to_do_max = max(states_to_do)-1
                self.QMout["dm"][:,0,0] = dipoles_gs[0]
                if (ex_state !=1):
                    self.QMout["dm"][:,ex_state,ex_state] = dipoles_gs[-1]
                for i in range(states_to_do_max):
                    self.QMout["dm"][:, 0, i+1] = dipoles_trans[i]
                    self.QMout["dm"][:, i+1, 0] = - dipoles_trans[i]

        if self.QMin.requests["grad"]:
            ex_state = list(self.QMin.maps["gradmap"])[0][1] - 1
            if (ex_state ==0):
                gradients = self._get_grad(os.path.join(scratchdir,f"ORCA.engrad.ground.grad.tmp"))
            else:
                gradients = self._get_grad(os.path.join(scratchdir,f"ORCA.engrad.singlet.root{ex_state}.grad.tmp"))

            self.QMout["grad"][ex_state] = gradients
        
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
                string += "\tsgradlist " +  ",".join([str(i[1]-1) for i in qmin.maps["gradmap"]]) + "\n"

            string += "end\n\n"
        
        string += "%output\n"
        string += "\tPrint[ P_Overlap ] 1\n"
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

    def _create_aoovl(self) -> None:
        """
        Create AO_overl.mixed for overlap calculations
        """
        gbw_curr = f"ORCA.gbw.{self.QMin.save['step']}"
        gbw_prev = f"ORCA.gbw.{self.QMin.save['step']-1}"

        writefile(
            os.path.join(self.QMin.save["savedir"], "AO_overl.mixed"),
            self._get_ao_matrix(self.QMin.save["savedir"], gbw_prev, gbw_curr),
        )

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
    
    def _get_ao_matrix(
        self, workdir: str, gbw_first: str = "ORCA.gbw", gbw_second: str = "ORCA.gbw") -> str:
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

        # run orca_fragovl
        string = f"orca_fragovl {gbw_first} {gbw_second}"
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
            
            ovlp_mat = ovlp_mat.T

            # Convert matrix to string
            ao_mat = f"{n_ao} {n_ao}\n"
            for i in ovlp_mat:
                ao_mat += "".join(f"{j: .15e} " for j in i) + "\n"
            return ao_mat
        
    def _save_files(self, workdir: str) -> None:
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]
        self.log.debug("Copying files to savedir")

        # Save gbw and dets from cis
        self.log.debug("Write MO coefficients to savedir")
        writefile(os.path.join(savedir, f"mos.{step}"), self._get_mos(workdir))
        
        with open(os.path.join(workdir, "ORCA.log"), "r", encoding="utf-8") as orca_log:
            # Extract list of orbital energies and filter occupation numbers
            orbital_list = re.search(r"ORBITAL ENERGIES\n-{16}(.*)MOLECULAR ORBITALS", orca_log.read(), re.DOTALL)
            print(orbital_list)
            occ_list = re.findall(r"\d+\s+([0-2]\.0{4})", orbital_list.group(1))
            occ_list = list(map(lambda x: int(float(x)), occ_list))

            # Convert to string and save file
            writefile(os.path.join(savedir, f"dets.{step}"), self.format_ci_vectors([{tuple(occ_list): 1.0}]))

        shutil.copy(os.path.join(workdir, "ORCA.gbw"), os.path.join(savedir, f"ORCA.gbw.{step}"))

    def _get_mos(self, workdir: str) -> str:
        """
        Extract MO coefficients from ORCA gbw file

        workdir:   Directory of ORCA.gbw
        """

        # run orca_fragovl
        string = "orca_fragovl ORCA.gbw ORCA.gbw"
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
            ao_mat = np.asarray(ao_mat)
            ao_mat = np.hstack(ao_mat.reshape(-1, n_ao, n_ao))

        # make string
        n_mo = n_ao * 2

        string = f"2mocoef\nheader\n1\nMO-coefficients from Orca\n1\n{n_ao}   {n_mo}\na\nmocoef\n(*)\n"

        
        for i in ao_mat.T:
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
    
    
    def get_dets_from_cis(self, cis_path: str) -> dict[str, str]:
        """
        Parse ORCA.cis file from WORKDIR
        """
        # Set variables
        cis_path = cis_path if os.path.isfile(cis_path) else os.path.join(cis_path, "ORCA.cis")
        states_extract = deepcopy(self.QMin.molecule["states"])
        states_skip = [self.QMin.control["states_to_do"][i] - states_extract[i] for i in range(len(states_extract))]

        

        # Parse file
        with open(cis_path, "rb") as cis_file:
            cis_file.read(4)
            header = struct.unpack("8i", cis_file.read(32))

            # Extract information from header
            # Number occupied A/B and number virtual A/B
            nfc = header[0]
            noa = header[1] - header[0] + 1
            nva = header[3] - header[2] + 1
            nob = header[5] - header[4] + 1 
            nvb = header[7] - header[6] + 1 
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
                                    key[occ], key[virt] = 2, 1
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)] * math.sqrt(0.5)
                                    key[occ], key[virt] = 1, 2
                                    dets_exp[tuple(key)] = dets[(occ, virt, dummy)] * math.sqrt(0.5)
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
                filename = f"dets.{mult}"
                strings[filename] = self.format_ci_vectors(eigenvectors[mult])
            return strings
        
# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_BASICORCA().main()
