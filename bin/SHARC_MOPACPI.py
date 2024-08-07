#!/usr/bin/env python3
import datetime
import itertools
import math
import os
import re
import shutil
from copy import deepcopy
from io import TextIOWrapper
from textwrap import dedent, wrap
from typing import Optional

import numpy as np
from constants import *
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import containsstring, expand_path, itmult, link, makecmatrix, mkdir, readfile, writefile

__all__ = ["SHARC_MOPACPI"]

AUTHORS = "Eduarda Sangiogo Gil"
VERSION = "0.1"
VERSIONDATE = datetime.datetime(2024, 7, 30)
NAME = "MOPACPI"
DESCRIPTION = "SHARC interface for the MOPAC-PI program"

CHANGELOGSTRING = """For now, this implementation is the simplest possible...
"""

all_features = set(
    [
        "h",
        "dm",
        "grad",
        "nacdr",
        "overlap",
        "phases",
    ]
)

au2ang = 0.529176125

class SHARC_MOPACPI(SHARC_ABINITIO):
    """
    SHARC interface for MOPAC-PI
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loglevel = 10  

        # Add resource keys
        self.QMin.resources.update(
            {
                "mopacpidir": None,
            }
        )
        self.QMin.resources.types.update(
            {
                "mopacpidir": str,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "ham"           : "AM1",
                "external_par"  : None,
                "numb_elec"     : None,
                "numb_orb"      : None,
                "flocc"         : 0.10,
                "memory"        : 1100,
                "meci"          : 20,
                "mxroot"        : 20,
                "singlet"       : True,
                "micros"        : None,
                "add_pot"       : False,
                "qmmm"          : None,
                "link_atoms"    : None,
                "link_atom_pos" : None,
                "force_field"   : None,
            }
        )
        self.QMin.template.types.update(
            {
                "ham"           : str,
                "external_par"  : int,
                "numb_elec"     : int,
                "numb_orb"      : int,
                "flocc"         : float,
                "memory"        : int,
                "meci"          : int,
                "mxroot"        : int,
                "single"        : bool,
                "micros"        : int,
                "add_pot"       : bool,
                "qmmm"          : int,
                "link_atoms"    : list,
                "link_atom_pos" : list,
                "force_field"   : str,
            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_MOPACPI._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_MOPACPI._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_MOPACPI._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_MOPACPI._authors

    @staticmethod
    def name() -> str:
        return SHARC_MOPACPI._name

    @staticmethod
    def description() -> str:
        return SHARC_MOPACPI._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_MOPACPI._name}\n{SHARC_MOPACPI._description}"

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    def create_restart_files(self):
        pass

    @staticmethod
    def generate_inputstr(qmin: QMin) -> str:
        """
        Generate MOPACPI input file string from QMin object
        """

        natom        = qmin["molecule"]["natom"]
        coords       = qmin["coords"]["coords"]
        elements     = qmin["molecule"]["elements"]
        nstates      = qmin["molecule"]["nstates"]
        ham          = qmin["template"]["ham"]
        external_par = qmin["template"]["external_par"]
        numb_elec    = qmin["template"]["numb_elec"]
        numb_orb     = qmin["template"]["numb_orb"]
        flocc        = qmin["template"]["flocc"]
        memory       = qmin["template"]["memory"]
        meci         = qmin["template"]["meci"]
        mxroot       = qmin["template"]["mxroot"]
        singlet      = qmin["template"]["singlet"]
        micros       = qmin ["template"]["micros"]
        add_pot      = qmin ["template"]["add_pot"]
        qmmm         = qmin ["template"]["qmmm"]
        link_atoms   = qmin ["template"]["link_atoms"]
        link_atom_pos= qmin ["template"]["link_atom_pos"]


        inpstring = ""
        # input top:
        if micros != None:
            inpstring += f"{ham} OPEN({numb_elec}, {numb_orb}) MICROS={micros} FLOCC={flocc} +\n"
        else:
            inpstring += f"{ham} OPEN({numb_elec}, {numb_orb}) FLOCC={flocc} +\n"
        if external_par != None:
            inpstring += "VECTORS GEO-OK PULAY EXTERNAL=external_par +\n"
        else:
            inpstring += "VECTORS GEO-OK PULAY +\n"
        if add_pot:
            inpstring += f"ITRY=1500 MEMORY={memory} NSTAT={nstates} MECI={meci} MXROOT={mxroot} ADDPOT + \n"
        else:
            inpstring += f"ITRY=1500 MEMORY={memory} NSTAT={nstates} MECI={meci} MXROOT={mxroot} + \n"
        if qmmm != None:
            inpstring += f"QMMM={qmmm} TINKER + \n"
        if qmin.requests["nacdr"]:
            inpstring += f"NMULT={nstates} S2 COMMECI NWTNX DYNAM SINGLET NACM \n"
        else:
            inpstring += f"NMULT={nstates} S2 COMMECI NWTNX DYNAM SINGLET \n"
        
        inpstring +="input generated by sharc\n\n"
        
        #input coordinates
        if qmmm != None:
            elements_link = []
            elements_link = elements
            for i in range(len(link_atoms)):
                elements_link[int(link_atom_pos[i])-1] = link_atoms[i]
            for i in range(natom-qmmm):
                inpstring += f"{elements_link[i]}\t{coords[i][0]*au2ang:>10,.5f} 1\t{coords[i][1]*au2ang:>10,.5f} 1\t{coords[i][2]*au2ang:>10,.5f} 1\n"
        else:
            for i in range(natom):
                inpstring += f"{elements[i]}\t{coords[i][0]*au2ang:>10,.5f} 1\t{coords[i][1]*au2ang:>10,.5f} 1\t{coords[i][2]*au2ang:>10,.5f} 1\n"

        inpstring += "\n"
        
        #MICROS
        if micros != None:
            allmicros = []
            with open('MOPACPI.template', 'r') as file:
                for line in file:
                    if 'MICROS' in line:
                        for _ in range(micros-1):
                            next_line = next(file).strip()
                            allmicros.append(next_line)
            
            inpstring += "MICROS \n"
            for i in range(micros-1):
                inpstring += f"{allmicros[i]}\n"
        
        par_str = ""
        if external_par != None:
            allpar = []
            with open('MOPACPI.template', 'r') as file:
                for line in file:
                    if 'EXTERNAL PARAMETERS' in line:
                        for _ in range(external_par-1):
                            next_line = next(file).strip()
                            allpar.append(next_line)
            
            for i in range(external_par-1):
                par_str += f"{allpar[i]}\n"

        if add_pot:
            inpstring += "\n"
            inpstring += "ADDED POTENTIAL \n"
            with open('MOPACPI.template', 'r') as file:
                for line in file:
                    if 'ADDED POTENTIAL' in line:
                            for _ in range(10):
                                next_line = next(file).strip()
                                if 'END ADDED POTENTIAL' not in next_line:
                                    inpstring += f"{next_line} \n"
                                else:
                                    break
        
        return inpstring, par_str

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        # Setup workdir
        mkdir(workdir)
        
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]
        external_par = qmin["template"]["external_par"]
        force_field = qmin["template"]["force_field"]
        main_dir = "./"
        
        filecopy = os.path.join(self.QMin.control["workdir"], "QM.in")
        saved_file = os.path.join(main_dir, "QM.in")
        shutil.copy(saved_file,filecopy)

        filecopy = os.path.join(self.QMin.control["workdir"], "QM.in")
        saved_file = os.path.join(main_dir, "QM.in")
        shutil.copy(saved_file,filecopy)

        filecopy = os.path.join(self.QMin.control["workdir"], "MOPACPI_tnk.xyz")
        saved_file = os.path.join(main_dir, "MOPACPI_tnk.xyz")
        shutil.copy(saved_file,filecopy)

        filecopy = os.path.join(self.QMin.control["workdir"], "MOPACPI_tnk.key")
        saved_file = os.path.join(main_dir, "MOPACPI_tnk.key")
        shutil.copy(saved_file,filecopy)

        filecopy = os.path.join(self.QMin.control["workdir"], force_field)
        saved_file = os.path.join(main_dir, force_field)
        shutil.copy(saved_file,filecopy)

        if step > 0:
            fromfile = os.path.join(savedir, "MOPACPI_nx.mopac_oldvecs")
            tofile = os.path.join(workdir, "MOPACPI_nx.mopac_oldvecs")
            shutil.copy(fromfile, tofile)

        # Write input
        input_str, par_str = self.generate_inputstr(qmin)
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "MOPACPI.dat")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        if external_par !=  None:
            input_par = os.path.join(workdir, "external_par")
            self.log.debug(f"Write external files into file {input_par}")
            writefile(input_par, par_str)
                  
        #Min.molecule["point_charges"]:
        #    pc_str = ""
        #    for coords, charge in zip(self.QMin.coords["pccoords"], self.QMin.coords["pccharge"]):
        #        pc_str += f"{' '.join(map(str, coords))} {charge}\n"
        #    writefile(os.path.join(workdir, "fort.20"), pc_str)

        # Setup MNDO
        starttime = datetime.datetime.now()
        exec_str = f"{os.path.join(qmin.resources['mopacpidir'],'mopacpi.x')} MOPACPI"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "MOPACPI.out"), os.path.join(workdir, "MOPACPI.err")
        ) 
        endtime = datetime.datetime.now()

        return exit_code, endtime - starttime
    
    def run(self) -> None:

        starttime = datetime.datetime.now()

        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "mopacpi_calc")

        #schedule = [{"mopacpi_calc" : self.QMin}] #Generate fake schedule
        #self.QMin.control["nslots_pool"].append(1)
        #self.runjobs(schedule)
        self.execute_from_qmin(self.QMin.control["workdir"], self.QMin)

        self._save_files(self.QMin.control["workdir"])

        self.log.debug("All jobs finished successfully")

        self.QMout["runtime"] = datetime.datetime.now() - starttime


    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        return

    def printQMout(self) -> None:
        super().writeQMout()

    def print_qmin(self) -> None:
        pass

    def getQMout(self) -> None:
        """
        Parse MOPACPI output files
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
            requests=requests,
        )

        nmstates = self.QMin.molecule["nmstates"]

        log_energies = os.path.join(self.QMin.control["workdir"], "MOPACPI_nx.epot")
        log_grads = os.path.join(self.QMin.control["workdir"], "MOPACPI_nx.grad.all")
        log_nacs = os.path.join(self.QMin.control["workdir"], "MOPACPI_nx.nad_vectors")
        log_ovl = os.path.join(self.QMin.control["workdir"], "MOPACPI_nx.run_cioverlap.log")
        log_dip = os.path.join(self.QMin.control["workdir"], "MOPACPI_nx.dipoles")
        
        # Populate energies
        if self.QMin.requests["h"]:
            energies = self._get_energy(log_energies)
            for i in range(len(energies)):
                self.QMout["h"][i][i] = energies[i]

        if self.QMin.requests["dm"]:
            self.QMout.dm = self._get_dip(log_dip)

        # Populate gradients
        if self.QMin.requests["grad"]:
            self.QMout.grad = self._get_grad(log_grads)

        # Populate NACV
        if self.QMin.requests["nacdr"]:
            self.QMout.nacdr = self._get_nac(log_nacs)
            if self.QMin.requests["phases"]:
                for i in range(nmstates):
                    self.QMout["phases"][i] = -1 
        
        # Populate overlaps
        if self.QMin.requests["overlap"]:
            self.QMout.overlap = self._get_overlap(log_ovl)
            if self.QMin.requests["phases"]:
                for i in range(nmstates):
                    self.QMout["phases"][i] = -1 if self.QMout["overlap"][i, i] < 0 else 1

    def _get_energy(self, log_energies: str):

        f = readfile(log_energies)
        energies = []
        for iline, line in enumerate(f):
            line = f[iline]
            en = line.split()
            energies.append(float(en[0]))

        return energies
    

    def _get_dip(self, log_dip: str):
        
        nmstates = self.QMin.molecule["nmstates"]

        f = readfile(log_dip)

        dip = [[[0.0 for j in range(nmstates)] for k in range(nmstates)] for i in range(3)]

        lines = f
        iline = 0
        for xyz in range(3):
            for i in range(nmstates):
                dips = lines[iline].split()
                if len(dips) != nmstates:
                    raise ValueError(f"Line {iline + 1} does not contain {nmstates} elements: {dips}")
                for j in range(nmstates):
                    dip[xyz][i][j] = float(dips[j])
                iline += 1
            iline += 1

        return dip
    
    def _get_grad(self, log_grad: str):
        
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]

        f = readfile(log_grad)

        grads = np.zeros((nmstates,natom,3))

        lines = f
        iline = 0
        for i in range(nmstates):
            for j in range(natom):
                gr = lines[iline].split()
                for xyz in range(3):
                    grads[i][j][xyz] = float(gr[xyz])
                iline += 1

        return grads

    def _get_nac(self, log_nac: str):
        
        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]

        f = readfile(log_nac)

        nacs = np.zeros((nmstates,nmstates,natom,3))

        lines = f
        iline = 0
        for i in range(nmstates):
            for j in range(i+1,nmstates):
                for k in range(natom):
                    nac = lines[iline].split()
                    for xyz in range(3):
                        nacs[j][i][k][xyz] = float(nac[xyz])
                        nacs[i][j][k][xyz] = (float(nac[xyz]))*(-1.00)
                    iline += 1

        return nacs
    
    
    def _get_overlap(self, log_ovl: str):
        
        nmstates = self.QMin.molecule["nmstates"]

        f = readfile(log_ovl)

        overlap = np.zeros((nmstates,nmstates))

        lines = f
        iline = 1
        for i in range(nmstates):
            ovl = lines[iline].split()
            for j in range(nmstates):
                overlap[i][j] = float(ovl[j])
            iline += 1

        return overlap
    
    def read_resources(self, resources_file: str = "MOPACPI.resources") -> None:
        super().read_resources(resources_file)
        # LD PATH???
        if not self.QMin.resources["mopacpidir"]:
            raise ValueError("mndodir has to be set in resource file!")

        self.QMin.resources["mopacpidir"] = expand_path(self.QMin.resources["mopacpidir"])
        self.log.debug(f'mopacpidir set to {self.QMin.resources["mopacpidir"]}')

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")
    
    def _save_files(self, workdir: str) -> None:
        """
        Save files (molden, mos) to savedir
        Naming convention: file.job.step
        """
        savedir = self.QMin.save["savedir"]

        fromfile = os.path.join(workdir, "MOPACPI_nx.mopac_oldvecs")
        tofile = os.path.join(savedir, "MOPACPI_nx.mopac_oldvecs")
        shutil.copy(fromfile, tofile)

        return

    def read_template(self, template_file: str = "MOPACPI.template") -> None:
        super().read_template(template_file)

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


    def setup_interface(self) -> None:
        """
        Setup remaining maps (ionmap, gsmap) and build jobs dict
        """
        super().setup_interface()


if __name__ == "__main__":
    SHARC_MOPACPI(loglevel=10).main()
