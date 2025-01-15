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
from utils import containsstring, expand_path, question, itmult, link, makecmatrix, mkdir, readfile, writefile

__all__ = ["SHARC_MOPACPI"]

AUTHORS = "Eduarda Sangiogo Gil, Hans Georg Gallmetzer"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2024, 12, 2)
NAME = "MOPACPI"
DESCRIPTION = "AB INITIO interface for the MOPAC-PI program"

CHANGELOGSTRING = """
30.07.2021:     Initial version 0.1 by Eduarda
For now, this implementation is the simplest possible....
Only Singlets are supported. Using local diabatization.
Will be extended to triplets and NACs int the future.

02.12.2024:     Version 1.0 by Georg
Added all the needed functionality to run the SHARC pipeline with MOPAC-PI. 
Minor and cosmetic changes to the code.
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
                "wfoverlap": "",
            }
        )
        self.QMin.resources.types.update(
            {
                "mopacpidir": str,
                "wfoverlap": str,
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
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()

        """
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'MOPAC-PI interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        self.files = []


        self.template_file = None
        self.log.info(f"{'MOPAC-PI input template file':-^60}\n")

        if os.path.isfile("MOPACPI.template"):
            usethisone = question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                self.template_file = "MOPACPI.template"
        else:
            while True:
                self.template_file = question("Template filename:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(self.template_file):
                    self.log.info(f"File {self.template_file} does not exist!")
                    continue
                break
            
        self.log.info("")
        self.files.append(self.template_file)

        if question("Do you have an 'ext_param' file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            while True:
                ext_param = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="ext_param")
                self.files.append(ext_param)
                if os.path.isfile(ext_param):
                    break
                else:
                    self.log.info(f"file at {ext_param} does not exist!")
        



        self.make_resources = False
        # Resources
        if question("Do you have a 'MOPACPI.resources' file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            while True:
                resources_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="MOPACPI.resources")
                self.files.append(resources_file)
                self.make_resources = False
                if os.path.isfile(resources_file):
                    break
                else:
                    self.log.info(f"file at {resources_file} does not exist!")
        else:
            self.make_resources = True
            self.log.info(
            "\nPlease specify path to MOPACPI directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n"
            )
            self.setupINFOS["mopacdir"] = question("Path to MOPACPI:", str, KEYSTROKES=KEYSTROKES)
            self.log.info("")

            # scratch
            self.log.info(f"{'Scratch directory':-^60}\n")
            self.log.info(
                "Please specify an appropriate scratch directory. This will be used to run the MOPAC-PI calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script."
            )
            INFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
            self.log.info("")

        if question("Do you want to run a QM/MM calculation?", bool, KEYSTROKES=KEYSTROKES, default=True):
            self.log.info(
            "\nThree files are needed:\n -MOPACPI_tnk.key\n -MOPACPI_tnk.xyz\n -oplsaa.prm\n"
            )
            while True:
                tinkerkey_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="MOPACPI_tnk.key")
                self.files.append(tinkerkey_file)
                if os.path.isfile(tinkerkey_file):
                    break
                else:
                    self.log.info(f"file at {tinkerkey_file} does not exist!")
            while True:
                tinkercoords_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="MOPACPI_tnk.xyz")
                self.files.append(tinkercoords_file)
                if os.path.isfile(tinkercoords_file):
                    break
                else:
                    self.log.info(f"file at {tinkercoords_file} does not exist!")
            while True:
                forcefield_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="oplsaa.prm")
                self.files.append(forcefield_file)
                if os.path.isfile(forcefield_file):
                    break
                else:
                    self.log.info(f"file at {forcefield_file} does not exist!")
            
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
        charge       = qmin["molecule"]["charge"][0]


        inpstring = ""
        # input top:
        if micros != None:
            inpstring += f"{ham} OPEN({numb_elec}, {numb_orb}) MICROS={micros} FLOCC={flocc} CHARGE={charge} +\n"
        else:
            inpstring += f"{ham} OPEN({numb_elec}, {numb_orb}) FLOCC={flocc} CHARGE={charge} +\n"
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
            if link_atoms != None:
                for i in range(len(link_atoms)):
                    elements_link[int(link_atom_pos[i])-1] = link_atoms[i]
            for i in range(natom-qmmm):
                inpstring += f"{elements_link[i]}\t{coords[i][0]*au2a:>10,.5f} 1\t{coords[i][1]*au2a:>10,.5f} 1\t{coords[i][2]*au2a:>10,.5f} 1\n"
        else:
            for i in range(natom):
                inpstring += f"{elements[i]}\t{coords[i][0]*au2a:>10,.5f} 1\t{coords[i][1]*au2a:>10,.5f} 1\t{coords[i][2]*au2a:>10,.5f} 1\n"

        inpstring += "\n"
        

        # Handle external parameters
        #Microstates
        if micros != None:
            allmicros = []
            with open('ext_param', 'r') as file:
                for line in file:
                    if 'MICROS' in line:
                        for _ in range(micros-1):
                            next_line = next(file).strip()
                            allmicros.append(next_line)
            
            inpstring += "MICROS \n"
            for i in range(micros-1):
                inpstring += f"{allmicros[i]}\n"
        
        par_str = ""

        if external_par is not None and external_par > 0:
            allpar = []
            lines = readfile('ext_param')  # Assuming readfile() returns a list of lines
            
            # Locate the "EXTERNAL PARAMETERS" block
            found = False
            for iline, line in enumerate(lines):
                if 'EXTERNAL PARAMETERS' in line:
                    found = True
                    iline += 1  # Move to the next line after "EXTERNAL PARAMETERS"
                    break
            
            # Collect the required number of external parameters
            if found:
                for _ in range(external_par - 1):
                    if iline < len(lines):  # Ensure we don't go out of bounds
                        allpar.append(lines[iline].strip())
                        iline += 1
            
            # Build the parameter string
            for par in allpar:
                par_str += f"{par}\n"

        # Handle added potential
        if add_pot:
            inpstring += "\n"
            inpstring += "ADDED POTENTIAL \n"
            added_pot = readfile('ext_param')  # Assuming readfile() returns a list of lines

            # Locate the "ADDED POTENTIAL" block
            found = False
            for iline, line in enumerate(added_pot):
                if 'ADDED POTENTIAL' in line:
                    found = True
                    iline += 1  # Move to the next line after "ADDED POTENTIAL"
                    break
            
            # Collect lines in the "ADDED POTENTIAL" block
            if found:
                while iline < len(added_pot):  # Ensure we don't go out of bounds
                    next_line = added_pot[iline].strip()
                    if 'END ADDED POTENTIAL' not in next_line:
                        inpstring += f"{next_line} \n"
                        iline += 1
                    else:
                        break

        
        return inpstring, par_str
    
    
    def writeQMin(self, qmin : QMin) -> str:
        coords = qmin["coords"]["coords"]
        elements = qmin["molecule"]["elements"]
        natom = qmin["molecule"]["natom"]
        step = qmin["save"]["step"]
        savedir = qmin["save"]["savedir"]

        qminstr = f"{natom}\n\n"
        for i in range(natom):
            qminstr += f"{elements[i]}\t{coords[i][0]*BOHR_TO_ANG:>5,.7f}\t{coords[i][1]*BOHR_TO_ANG:>5,.7f}\t{coords[i][2]*BOHR_TO_ANG:>5,.7f}\t{0:>10,.5f}\t{0:>10,.5f}\t{0:>10,.5f}\n"
        qminstr += f"unit angstrom\n"
        
        if (qmin.molecule["states"]):
            qminstr += "states\t"
            for i in qmin.molecule["states"]:
                qminstr += f" {i}"
            qminstr += "\n"

        qminstr += f"dt\t20.670687\nstep\t{step}\nsavedir\t{savedir}\n"

        if (qmin.requests["h"]):
            qminstr += "H\n"
        if (qmin.requests["dm"]):
            qminstr += "DM\n"
        if (qmin.requests["grad"]):
            qminstr += "GRAD\t"
            for i in qmin.requests["grad"]:
                qminstr += f" {i}"
            qminstr += "\n"
        # if (qmin.requests["nacdr"]): ##Still have to think about this!! I don't think it is necessary, becuase MOPAC-PI only needs this for the coordinates.
        #     qminstr += "NACDR\t"
        #     for i in qmin.requests["nacdr"]:
        #         qminstr += f" {i}"
        #     qminstr += "\n"
        if(qmin.requests["overlap"]):
            qminstr += "OVERLAP\n"
        
        return qminstr

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        # Setup workdir
        mkdir(workdir)
        
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]
        external_par = qmin["template"]["external_par"]
        force_field = qmin["template"]["force_field"]
        main_dir = "./"
        

        qminstring = self.writeQMin(qmin)
        writefile(os.path.join(workdir, "QM.in"), qminstring)

        if qmin["template"]["qmmm"] != None:
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
            fromfile = os.path.join(savedir, f"MOPACPI_nx.mopac_oldvecs.{step-1}")
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
                

        # Setup MOPAC-PI
        starttime = datetime.datetime.now()
        exec_str = f"{os.path.join(qmin.resources['mopacpidir'],'mopacpi.x')} MOPACPI"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "MOPACPI.out"), os.path.join(workdir, "MOPACPI.err")
        ) 
        endtime = datetime.datetime.now()

        return exit_code, endtime - starttime
    
    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """
    
    def run(self) -> None:

        starttime = datetime.datetime.now()

        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "mopacpi_calc")

        self.execute_from_qmin(self.QMin.control["workdir"], self.QMin)

        self._save_files(self.QMin.control["workdir"])

        self.clean_savedir()

        self.log.debug("All jobs finished successfully")

        self.QMout["runtime"] = datetime.datetime.now() - starttime


    def prepare(self, INFOS: dict, workdir: str):
        """
        prepare the workdir according to dictionary

        ---
        Parameters:
        INFOS: dictionary with infos
        workdir: path to workdir
        """
        if self.make_resources:
            try:
                resources_file = open('%s/MOPACPI.resources' % (workdir), 'w')
            except IOError:
                self.log.error('IOError during prepare MOPACPI, iconddir=%s' % (workdir))
                quit(1)

            string = 'scratchdir %s/\n' % INFOS['scratchdir']
            string = 'mopacdir %s\n' % INFOS['mopacdir']

            resources_file.write(string)
            resources_file.close()
            
        create_file = link if INFOS["link_files"] else shutil.copy
        
        for file in self.files:
            create_file(expand_path(file), os.path.join(workdir, file.split("/")[-1]))

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
        
        return self.QMout

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

        return np.array(dip)
    
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
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]

        fromfile = os.path.join(workdir, "MOPACPI_nx.mopac_oldvecs")
        tofile = os.path.join(savedir, f"MOPACPI_nx.mopac_oldvecs.{step}")
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

        if (any(num > 0 for num in self.QMin.molecule["states"][1:]) or self.QMin.molecule["states"][0] == 0):
            self.log.error("MOPAC-PI interface can only calculate singlets!!")
            raise ValueError()


if __name__ == "__main__":
    SHARC_MOPACPI(loglevel=10).main()
