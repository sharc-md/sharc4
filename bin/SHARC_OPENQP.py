#!/usr/bin/env python3
import datetime
import time
import os
import glob
import shutil
from io import TextIOWrapper
from copy import deepcopy
from itertools import chain

import math
import re
import numpy as np
#from numpy import ndarray
from constants import BOHR_TO_ANG
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import batched, mkdir, writefile, question, expand_path, link
import json

# ---------------------------------| Infos |---------------------------------------------------------------------------

__all__ = ["SHARC_OPENQP"]  # Only export interface class

AUTHORS = "Martina Hartinger, Sebastian Mai"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2026, 3, 1)

NAME = "OPENQP"
DESCRIPTION = "AB INITIO interface for OpenQP for MRSF-TDDFT"

CHANGELOGSTRING = " "

all_features = set(
    [
        "h",
        "dm", 
        "grad",
        "molden",
        "overlap",
        "phases",
        # Rest of the possible requests:
        # "theodore", 
        #"nacdr", 
        # "soc", 
        # "ion",
    ]
)


class SHARC_OPENQP(SHARC_ABINITIO):
    """
    SHARC interface for OpenQP
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
                "openqpdir": "",  
                "openqpexe": "",  
                "ncpu": 1,          # with 0, OpenQP uses all available CPUs
                "scratchdir": "/scratch/",
                "wfoverlap": "", 
                "wfthresh": 0.998,
                "wfmemory": 2000,
            }
        )
        self.QMin.resources.types.update(
            {
                "openqpdir": str,
                "openqpexe": str,
                "ncpu": int,
                "scratchdir": str,
                "wfoverlap": str,
                "wfthresh": float,
                "wfmemory": float,
            }
        )

        self.QMin.template.update(
            {
                "basis": "6-31G*",
                "functional": "BHHLYP",
                "scfiter": 150,
                "forced_attempt": 1,
                "d4dispersion": False,
                "pyscf_guess": False,
                "grid_keys": None,
                "scf_keys": None,
                "basis_per_element": None,
                "basis_per_atom": None,
            }
        )
        self.QMin.template.types.update(
            {
                "basis": str,
                "functional": str,
                "scfiter": int,
                "forced_attempt": int,
                "d4dispersion": bool,
                "pyscf_guess": bool,
                "grid_keys": str,
                "scf_keys": str,
                "basis_per_element": list,
                "basis_per_atom": list,
            }
        )


# ---------------------------------| Standard Methods |------------------------------------------------------------

    @staticmethod
    def version() -> str:
        return SHARC_OPENQP._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_OPENQP._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_OPENQP._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_OPENQP._authors

    @staticmethod
    def name() -> str:
        return SHARC_OPENQP._name

    @staticmethod
    def description() -> str:
        return SHARC_OPENQP._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_OPENQP._name}\n{SHARC_OPENQP._description}"


# ---------------------------------| Initialization |------------------------------------------------------------------

    def read_template(self, template_file: str = "OPENQP.template", kw_whitelist: list[str] | None = None) -> None:
        kw_whitelist = ["basis_per_element", "basis_per_atom"]
        super().read_template(template_file, kw_whitelist)

        for key in kw_whitelist:
            if self.QMin.template[key] and isinstance(self.QMin.template[key][0], list):
                self.QMin.template[key] = list(chain.from_iterable(self.QMin.template[key]))
        
        if self.QMin.template["basis_per_element"] and self.QMin.template["basis_per_atom"]:
            raise ValueError(f"Use either basis_per_element OR basis_per_atom!")
        

    def read_resources(self, resources_file: str = "OPENQP.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)
        
        if not self.QMin.resources["openqpdir"]:
            raise ValueError("openqpdir has to be set in resource file!")
        if not self.QMin.resources["openqpexe"]:
            raise ValueError("openqpexe has to be set in resource file!")
        self.QMin.resources["openqpdir"] = expand_path(self.QMin.resources["openqpdir"])
        self.QMin.resources["openqpexe"] = expand_path(self.QMin.resources["openqpexe"])

    def setup_interface(self) -> None:
        super().setup_interface()
        if self.QMin.resources["ncpu"] > 0:
            os.environ["OMP_NUM_THREADS"] = str(self.QMin.resources["ncpu"])
        os.environ["OPENQP_ROOT"] = self.QMin.resources["openqpdir"]
        os.environ['LD_LIBRARY_PATH'] = self.QMin.resources["openqpdir"] + '/lib' + ':%s' %  (os.environ['LD_LIBRARY_PATH'])

        if (any(num > 0 for num in self.QMin.molecule["states"][1:]) or self.QMin.molecule["states"][0] == 0):
            self.log.error("OpenQP can only calculate singlets!!")
            raise ValueError()



# ---------------------------------| Run |---------------------------------------------------------------------

    def run(self) -> None:
        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "oqp_calc")
        workdir = self.QMin.control["workdir"]
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]

        current_json_file = os.path.join(workdir, "OPENQP.json")  
        log_file = os.path.join(workdir, "OPENQP.log")
        self.execute_from_qmin(workdir, self.QMin)

        if not self.QMin.save["samestep"]:
            self._save_files(workdir)
        

        self.clean_savedir() ## löscht allzu alte Dateien aus SAVE
       
        if self.QMin.requests["overlap"] or step == 0: # overlap request cannot be set for step = 0 
            ### print MO coefficients to file for wfoverlap
            writefile(os.path.join(workdir, "mo_coeff"), self._get_mos(current_json_file))
            writefile(os.path.join(workdir, "determinants"), self._generate_mrsf_determinants(log_file))
       # copy the mo_coeff to savedir; this must be here for now bc _save_dir is before QMout is filled
            fromfile = os.path.join(workdir, "mo_coeff")
            tofile = os.path.join(savedir, f"mo_coeff.{step}")
            shutil.copy(fromfile, tofile)
       # copy the determinants to savedir; this must be here for now bc _save_dir is before QMout is filled
            fromfile = os.path.join(workdir, "determinants")
            tofile = os.path.join(savedir, f"dets.{step}")
            shutil.copy(fromfile, tofile)

        if self.QMin.requests["overlap"]:
            nmstates = self.QMin.molecule["nmstates"]
            self._run_wfoverlap()
            if self.QMin.requests["phases"]:
                for i in range(nmstates):
                    self.QMout["phases"][i] = -1 if self.QMout["overlap"][i, i] < 0 else 1

        self.log.debug("All done")



    def create_restart_files(self) -> None:
            pass

    def _save_files(self, workdir: str) -> None:
        """
        Save files from scratchdir/workdir to savedir after the job
        """
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]
         
        #json
        fromfile = os.path.join(workdir, "OPENQP.json")
        tofile = os.path.join(savedir, f"OPENQP.json.{step}")
        shutil.copy(fromfile, tofile)
        #molden
        if self.QMin.requests["molden"]:
            fromfile = glob.glob(os.path.join(workdir, "*molden"))[0]
            tofile = os.path.join(savedir, f"OPENQP.molden.{step}")
            shutil.copy(fromfile, tofile)
        #ao overlaps
        if self.QMin.requests["overlap"]:
            fromfile = os.path.join(workdir, "ao_overlap.dat")
            tofile = os.path.join(savedir, f"ao_overlap.{step}")
            shutil.copy(fromfile, tofile)

        ### mo coefficients and determinants are saved in their own functions ##

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)
        
        if self.QMin.requests["phases"]:
            self.QMin.requests["overlaps"] = True

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

    def set_coords(self, coords_file: str = "QM.in") -> None:
        super().set_coords(coords_file)

    def _run_wfoverlap(self) -> None:
        """ 
        Prepare the run of wfoverlap
        """
        
        workdir = self.QMin.control["workdir"]
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]

        wf_input = "a_mo = mocoef_a\nb_mo = mocoef_b\na_det = det_a\nb_det = det_b\nao_read=0\nmix_aoovl = ao_overlap"

        wf_exec_str = f"{self.QMin.resources['wfoverlap']} -m {self.QMin.resources['wfmemory']} -f wf_input.inp"
        
        writefile(os.path.join(workdir, "wf_input.inp"), wf_input)


        # Link files
        link(
            os.path.join(savedir, f"ao_overlap.{step}"),
            os.path.join(workdir, "ao_overlap"),
        ) 
        link(
            os.path.join(savedir, f"dets.{step-1}"),
            os.path.join(workdir, "det_a"),
        )
        link(
            os.path.join(savedir, f"dets.{step}"),
            os.path.join(workdir, "det_b"),
        )
        link(
            os.path.join(savedir, f"mo_coeff.{step-1}"),
            os.path.join(workdir, "mocoef_a"),
        )
        link(
            os.path.join(savedir, f"mo_coeff.{step}"),
            os.path.join(workdir, "mocoef_b"),
        )


        exit_code = self.run_program(workdir, wf_exec_str, os.path.join(workdir, "wfovl.out"), os.path.join(workdir, "wfovl.err"))
        if exit_code != 0:
            self.log.error("Something went wrong in wfoverlap")

            # read the content of wfovl.out elsewhere

# ---------------------------------| Scheduling |---------------------------------------------------------------------

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Do QM calculation
        will be called in SHARC_ABINITIO.runjobs()
        """
        
        mkdir(workdir)
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]
        openqpexe = self.QMin.resources["openqpexe"]

        ### copy stuff: old json from savedir to workdir for built-in overlap, and also as (potential) initial guess ###

        if step > 0:
            fromfile = os.path.join(savedir, f"OPENQP.json.{step-1}")
            tofile = os.path.join(workdir, "OPENQP.old.json")
            shutil.copy(fromfile, tofile)
        
        input_str = ""
        input_str += self.generate_inputstr(self.QMin)

        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "OPENQP.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        # Run QC Program
        starttime = datetime.datetime.now()
        exec_str = f"{openqpexe} OPENQP.inp"
        print(f"Executing {exec_str}")
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "OPENQP.out"), os.path.join(workdir, "OPENQP.err")
        )
        endtime = datetime.datetime.now()
        return exit_code, endtime - starttime


# ---------------------------------| Get Data |-----------------------------------------------------------------------

    def getQMout(self) -> dict[str, np.ndarray]:
        
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
        savedir = self.QMin.save["savedir"]
        workdir = self.QMin.control["workdir"]
        #files
        log_file = os.path.join(workdir, "OPENQP.log")
        energies_file = os.path.join(workdir, "energies")
        current_json_file = os.path.join(workdir, "OPENQP.json") 

        #file lists
        grad_files = glob.glob(os.path.join(workdir, "grad_*"))
        nac_files = glob.glob(os.path.join(workdir, "nac_*_*"))
        
        step = self.QMin.save["step"]

        if self.QMin.requests["h"]: # fill diagonal elements of H with energies
            energies = self._get_energies(energies_file)
            for i in range(len(energies)):
                self.QMout["h"][i][i] = energies[i]

        if self.QMin.requests["grad"]: 
            self.QMout.grad = self._get_grads(grad_files)

        if self.QMin.requests["dm"]:
            self.QMout.dm = self._get_dip(log_file)

        # fill overlap from wfoverlap calculation
        if self.QMin.requests["overlap"]:
            self.QMout["overlap"] = np.zeros((nmstates, nmstates))
            wfovl_file = os.path.join(workdir, "wfovl.out")

            ovlp_mat = self.parse_wfoverlap(wfovl_file)
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                    if not m1 == m2 == 1: # only singlets
                        continue
                    if not ms1 == ms2:
                        continue
                    self.QMout["overlap"][i][j] = ovlp_mat[s1-1,s2-1]


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
        self.log.info(f"||{'OpenQP interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        self.files = []
        # find template file
        self.template_file = None
        self.log.info(f"{'OpenQP input template file':-^60}\n")

        if os.path.isfile("OPENQP.template"):
            usethisone = question("Found OPENQP.template here. Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                self.template_file = "OPENQP.template"
            else:
                while True:
                    self.template_file = question("Template path and filename:", str, KEYSTROKES=KEYSTROKES)
                    if not os.path.isfile(expand_path(self.template_file)):
                        self.log.info(f"File {self.template_file} does not exist!")
                        continue
                    break
        else:
            while True:
                self.template_file = question("Template path and filename:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(expand_path(self.template_file)):
                    self.log.info(f"File {self.template_file} does not exist!")
                    continue
                break
        self.log.info("")
        self.files.append(self.template_file)
        
        # find resource file
        self.make_resources = False
        
        self.resource_file = None
        self.log.info(f"{'OpenQP input resource file':-^60}\n")

        if question("Do you have an OPENQP.resources file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            
            if os.path.isfile("OPENQP.resources"):
                usethisone = question("Found OPENQP.resources here. Use this resource file?", bool, KEYSTROKES=KEYSTROKES, default=True)
                if usethisone:
                    self.resource_file = "OPENQP.resources"
                else:
                    while True:
                        self.resource_file = question("Resource path and filename:", str, KEYSTROKES=KEYSTROKES)
                        if not os.path.isfile(expand_path(self.resource_file)):
                            self.log.info(f"File {self.resource_file} does not exist!")
                            continue
                        break
            self.files.append(self.resource_file)
        else:
            self.make_resources = True
            self.log.info("Specify path to OpenQP directory containing folders like lib, pyoqp, etc. (shell variables and ~ can be used, will be expanded when interface is started).\n")
            self.setupINFOS["openqpdir"] = question("Path to OpenQP folder:", str, KEYSTROKES=KEYSTROKES)
            self.log.info("")
            self.log.info("Specify path to the OpenQP executable (shell variables and ~ can be used, will be expanded when interface is started).\n")
            self.setupINFOS["openqpexe"] = question("Path to OpenQP executable:", str, KEYSTROKES=KEYSTROKES)
            self.log.info("")
            self.log.info(f"{'Scratch directory':-^60}\n")
            self.log.info(
                "Please specify an appropriate scratch directory. This will be used to run the OpenQP calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine.")
            self.setupINFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
            self.log.info(f"{'OpenQP Ressource usage':-^60}\n")
            self.setupINFOS["ncpu"] = question("Number of CPUs to use:", int, default=[1], KEYSTROKES=KEYSTROKES)

            if "overlap" in INFOS["needed_requests"]:
                self.log.info(f"\n{'WFoverlap setup':-^60}\n")
                self.setupINFOS["wfoverlap"] = question("Path to wavefunction overlap executable:", str, default="$SHARC/wfoverlap.x", KEYSTROKES=KEYSTROKES)

                self.setupINFOS["wfthres"] = question("Threshold for including determinants in the wfoverlap calculation:", float, default=[0.998], KEYSTROKES=KEYSTROKES)
                self.setupINFOS["wfmemory"] = question("Maximum memory (in MB) for wfoverlap calculation:", int, default=[2000], KEYSTROKES=KEYSTROKES)

        self.log.info("")


        return INFOS

    def prepare(self, INFOS: dict, workdir: str):
        if self.make_resources:
            try:
                resources_file = open('%s/OPENQP.resources' % (workdir), 'w')
            except IOError:
                self.log.error('IOError during prepareOpenQP, directory=%s' % (workdir))
                quit(1)
            string = 'scratchdir %s/%s/\n' % (self.setupINFOS['scratchdir'], workdir)
            string += 'openqpdir %s\n' % self.setupINFOS['openqpdir']
            string += 'ncpu %i\n' % (self.setupINFOS['ncpu'][0])
            if 'overlap' in INFOS['needed_requests']:
                string += 'wfoverlap %s\n' % (self.setupINFOS['wfoverlap'])
                string += 'wfthres %s\n' % (self.setupINFOS['wfthres'][0])
                string += 'wfmemory %s\n' % (self.setupINFOS['wfmemory'][0])
            resources_file.write(string)
            resources_file.close()


        #TODO: Copy files that are needed for interface in setup
        create_file = link if INFOS["link_files"] else shutil.copy
        for file in self.files:
            create_file(expand_path(file), os.path.join(workdir, file.split("/")[-1]))


# ---------------------------------| Additional Methods |------------------------------------------------------------

    @staticmethod
    def generate_inputstr(qmin: QMin) -> str:
        """
        Generate OpenQP input file string from QMin object
        """
        charge = qmin["molecule"]["charge"][0]
        states_to_do = deepcopy(qmin.control["states_to_do"])
        step = qmin.save["step"]
        do_grad = False
        do_overlap = False
        do_molden = False

        if qmin.requests["grad"] and qmin.maps["gradmap"]:
            do_grad = True
        if qmin.requests["overlap"]:
            do_overlap = True
        if qmin.requests["molden"]:
            do_molden = True

        # input section
        input_string = "[input]\n"
        input_string += f"method=tdhf\n"
        input_string += f"functional={qmin.template['functional']}\n"
        if do_overlap and step > 0: 
            input_string += f"runtype=prop\n" # this runtype can deliver mulitple gradients and built-in overlaps. ALWAYS does overlap, even without asking for nacme
        elif do_grad:
            input_string += f"runtype=data\n" # this runtype can deliver multiple gradients and nacs, but not the built-in overlaps. It should be used when overlaps are not required
        else:
            input_string += f"runtype=energy\n"
        #### geometry and basis set 
        input_string += f"system=\n"
        basis_dict = None
        mode = None

        if qmin.template["basis_per_element"] is not None:
            mode = "element"
            basis_dict = {
                batch[0]: batch[1]
                for batch in batched(qmin.template["basis_per_element"])
            }
        elif qmin.template["basis_per_atom"] is not None:
            mode = "atom"
            basis_dict = {
                int(batch[0]): batch[1]
                for batch in batched(qmin.template["basis_per_atom"])
            }
        for i, (label, coords) in enumerate(zip(qmin.molecule["elements"], qmin.coords["coords"])):
            tag = None
            if mode == "element":
                tag = f"{label.lower()}1" if label in basis_dict else "x1"
            elif mode == "atom":
                tag = f"{label.lower()}{i+1}" if (i+1) in basis_dict else "x1"
            if tag:
                input_string += (f"\t{label:4s} {coords[0]*BOHR_TO_ANG:16.9f}{coords[1]*BOHR_TO_ANG:16.9f} {coords[2]*BOHR_TO_ANG:16.9f} {tag}\n")
            else: 
                input_string += (f"\t{label:4s} {coords[0]*BOHR_TO_ANG:16.9f}{coords[1]*BOHR_TO_ANG:16.9f} {coords[2]*BOHR_TO_ANG:16.9f} \n")
        if basis_dict:
            input_string += "basis=library\n"
            input_string += "library=\n"
            if mode == "element":
                for element, basis in basis_dict.items():
                    input_string += f" {element}1 {basis}\n"
            elif mode == "atom":
                for atom_index, basis in basis_dict.items():
                    label = qmin.molecule["elements"][atom_index -1]
                    input_string += f" {label.lower()}{atom_index} {basis}\n"
            input_string += f" x1 {qmin.template['basis']}\n"
        else:
            input_string += f"basis={qmin.template['basis']}\n"


        input_string += f"charge= {charge}\n"
        if qmin.template["d4dispersion"]:
            input_string += f"d4=True\n"
        input_string += f"\n"
        
        # scf section
        scf_string = f"[scf]\n"
        scf_string += f"type=rohf\n"
        scf_string += f"maxit={qmin.template['scfiter']}\n" 
        scf_string += f"multiplicity=3\n"
        scf_string += f"forced_attempt={qmin.template['forced_attempt']}\n" 
        if do_molden:
            scf_string += f"save_molden = True\n"
        else:
            scf_string += f"save_molden = False\n"
        if qmin.template["scf_keys"]:
            scf_lines = [l.strip() for l in qmin.template["scf_keys"].split()]
            for l in scf_lines:
                scf_string += f"{l}\n"

        scf_string += "\n"

        grid_string = ""
        if qmin.template["grid_keys"]:
            grid_string += "[dftgrid]\n"
            grid_lines = [l.strip() for l in qmin.template["grid_keys"].split()]
            for l in grid_lines:
                grid_string += f"{l}\n"
            grid_string += "\n"

        #tdhf section
        tdhf_string = f"[tdhf]\n"
        tdhf_string += f"type=mrsf\n"
        tdhf_string += f"nstate={max(states_to_do)}\n"
        tdhf_string += f"conf_threshold=0.000001\n"
        tdhf_string += f"multiplicity=1\n\n" # for now only singlets
        
        #guess section
        guess_string = f"[guess]\n"
        guess_string += "save_mol = true \n" # creates json, do always
        if qmin.template["pyscf_guess"]:
            guess_string += f"type = pyscf\n" 
        elif qmin.save["always_guess"]:
            guess_string += "type=huckel\n"
        else:
            guess_string += "type=auto\n" # uses the json file given by file= , if present, and defaults to hueckel guess if not
            guess_string += "file=OPENQP.old.json\n"
        if do_overlap:
            guess_string += f"file2=OPENQP.old.json\n" ## for built-in overlaps and to get the AO overlap, the old json file must be read in like this
        guess_string += f"\n"

        #properties section
        properties_string = f"[properties]\n"
        properties_string += f"export=True\n" # save gradients, energies, nac, to files
        if do_overlap:
            properties_string += f"nac=nacme\n"
        if do_grad:
            properties_string += f"grad="+",".join([str(i[1]) for i in qmin.maps["gradmap"]]) + "\n\n" 

        
        #input done
        string = input_string + scf_string + grid_string + tdhf_string + guess_string + properties_string 
        return string

    def _get_energies(self, energiesfile: str) -> dict[int,float]:
        """
        Extract energies from OpenQP energies file
        """
        energies = {}
        with open(energiesfile, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines[1:]): #erste Energie wird weggeworfen weil T-Referenz
            line = line.strip()
            if not line:
                continue
            
            energy = float(line)
            energies[i] = energy # i=0 is GS, i=1 1st ex state
        
        return energies

    def _get_grads(self, grad_files: list[str]) -> np.ndarray: 
        """
        Reads gradients from OpenQP grad_x files. 
        This requires a list of all available grad_x files. 
        Only singlets.
        """

        nmstates = self.QMin.molecule["nmstates"]
        natom = self.QMin.molecule["natom"]

        grads = np.zeros((nmstates,natom,3))

        for grad_file in grad_files:
            m = re.search(r"grad_(\d+)", grad_file)
            i = int(m.group(1))-1
            with open(grad_file, "r") as f:
                lines = f.readlines()
                
            for j, line in enumerate(lines):
                s = line.split()
                grads[i, j, 0] = float(s[0])
                grads[i, j, 1] = float(s[1])
                grads[i, j, 2] = float(s[2])

        return grads

    def _get_dip(self, log_file: str):

        nmstates = self.QMin.molecule["nmstates"]
        pattern = re.compile(
            r"""
            ^\s*                        
            (\d+)\s*->\s*(\d+)          # states i -> j
            \s+
            [-+]?\d+\.\d+               # excitation energy (not used)
            \s+
            ([-+]?\d+\.\d+)             # mu_x
            \s+
            ([-+]?\d+\.\d+)             # mu_y
            \s+
            ([-+]?\d+\.\d+)             # mu_z
            """,
            re.VERBOSE
        )
        
        dip_entries = []
        with open(log_file, "r") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    i = int(m.group(1)) -1
                    j = int(m.group(2)) -1
                    mux = float(m.group(3))
                    muy = float(m.group(4))
                    muz = float(m.group(5))
                    dip_entries.append((i, j, mux, muy, muz))
        
        dip = np.zeros((3, nmstates, nmstates)) # no state dipoles, diagonal elements remain 0 for now
        for i, j, mux, muy, muz in dip_entries:
           dip[0,i,j] = mux
           dip[1,i,j] = muy
           dip[2,i,j] = muz
           dip[0,j,i] = mux
           dip[1,j,i] = muy
           dip[2,j,i] = muz

        return(dip)

    def _get_overlap(self, log_file: str):
        """
        Extract the overlaps calculated by OpenQP.
        """
        nmstates = self.QMin.molecule["nmstates"]
        pattern = "PyOQP: phase corrected state overlap (s_ij)"
        
        with open(log_file, "r") as f:
            lines = f.readlines()
        start = None
        for i, line in enumerate(lines):
            if pattern in line:
                start = i
                break
        ovl_start = start + 2
        
        S = np.zeros((nmstates,nmstates))

        for i in range(nmstates):
            row = lines[ovl_start + i].split()
            S[i, :] = [float(x) for x in row]
        return(S)

    def _create_aoovl(self) -> None:
        """
        Create AO_overl.mixed for overlap calculations
        """
        pass

    def _get_mos(self, json_file: str) -> str:
        """
        Extract MO coefficients from OpenQP json file
        """
        with open(json_file, "r") as file:
            data = json.load(file)
            mo_list = data["OQP::VEC_MO_A"]
            test = data["OQP::VEC_MO_B"]

            n_mo = len(mo_list)
            n_ao = len(mo_list) ## n_mo would be unequal n_ao if unrestricted or if frozcore is on

        string = f"2mocoef\nheader\n1\nMO coefficients from OpenQP JSON\n1\n{n_mo} {n_ao}\na\nmocoef\n(*)\n"
        for mo in mo_list:
            line = "\t".join(f"{coeff:6.12}" for coeff in mo) 
            string += line + "\n"

        string += "orbocc\n(*)\n"

        for _ in range(n_mo//3): ### fill occupation section with zeroes (3 per line)
            string += f"{0.0:6.12e}\t"*3 + "\n"
        rest = n_mo % 3
        string += f"{0.0:6.12e}\t"* rest
        
        return string

    def _generate_mrsf_determinants(self, log_file: str, outfile="determinants.dat") -> str:
        """
        Parse OPENQP MRSF-TDDFT logfile and generate spin-flip determinants.
        """

            ### Regex
        param_patterns = {
            "n_states": re.compile(r"Number of states:\s+(\d+)"),
            "n_mos": re.compile(r"Number of atomic orbitals:\s+(\d+)"),
            "n_occ_alpha": re.compile(r"Number of occupied alpha orbitals:\s+(\d+)"),
            "n_occ_beta": re.compile(r"Number of occupied beta orbitals:\s+(\d+)")
        }

        state_header = re.compile(r"State\s+#\s+(\d+)")
        excitation_line = re.compile(
            r"\s+\d+\s+([-+]?\d+\.\d+)\s+(\d+)\s+->\s+(\d+)"
        )

            ### Read infos and excitations
        params = {}
        states = {}

        current_state = None

        with open(log_file) as f:
            for line in f:
                    # header parameters
                for key, pat in param_patterns.items():
                    if key not in params:
                        m = pat.search(line)
                        if m:
                            params[key] = int(m.group(1))

                    # state blocks
                m = state_header.search(line)
                if m:
                    current_state = int(m.group(1))
                    states[current_state] = []
                    continue

                    # excitation lines
                m = excitation_line.match(line)
                if m and current_state is not None:
                    coeff = float(m.group(1))
                    occ = int(m.group(2))
                    vir = int(m.group(3))
                    states[current_state].append((coeff, occ, vir))

        missing = [k for k in param_patterns if k not in params]
        if missing:
            raise ValueError(f"Some header parameters were not found: {missing}")

        n_states = params["n_states"]
        n_mos = params["n_mos"]
        n_occ_alpha = params["n_occ_alpha"]
        n_occ_beta = params["n_occ_beta"]
        
        threshold = self.QMin.resources["wfthres"]
        threshold2 = threshold**0.5 ## TODO: is this the correct way to do it?

        for state, excitations in states.items():
            excitations.sort(key=lambda x: x[0]**2, reverse=True)
            trimmed = []
            norm = 0.0

            for coeff, occ, vir in excitations:
                trimmed.append((coeff, occ, vir))
                norm += coeff**2
                if norm >= threshold2:
                    break

            states[state] = trimmed

            self.log.debug(f"State {state:3d}: kept {len(trimmed):4d} excitations, "
                           f"norm = {norm**0.5:.6f}"
                        )
            ### Build reference determinant (Ms=+1)
        ref = []
        for i in range(1, n_mos + 1):
            if i <= n_occ_beta:
                ref.append("d")
            elif i <= n_occ_alpha:
                ref.append("a")
            else:
                ref.append("e")
        ref = "".join(ref)

            ### Create spin-flip excitations (from Ms=+1, alpha -> beta flips)
        def excite_spin_flip(ref_det, occ, vir):
            
            det = list(ref_det)

                # remove alpha at occ
            if det[occ - 1] == "a":
                det[occ - 1] = "e"
            elif det[occ - 1] == "d":
                det[occ - 1] = "b"
            else:
                raise ValueError(f"No alpha electron at MO {occ}")

                # add beta at vir
            if det[vir - 1] == "e":
                det[vir - 1] = "b"
            elif det[vir - 1] == "a":
                det[vir - 1] = "d"
            else:
                raise ValueError(f"Cannot add beta electron to MO {vir}")

            return "".join(det)

             ### Build determinants
        determinants = {}

        for state, excitations in states.items():
            for coeff, occ, vir in excitations:
                det = excite_spin_flip(ref, occ, vir)

                    # open-shell -> add "spin-complement", multiply by sqrt2
                if "a" in det or "b" in det:
                    factor = math.sqrt(0.5)
                    
                    det_list = [(det, coeff * factor), (det.translate(str.maketrans("ab", "ba")), -coeff * factor )]
                else:
                    # closed-shell -> nothing to do
                    det_list = [(det, coeff)]

                for d, cval in det_list:
                        if d not in determinants:
                            determinants[d] = [0.0] * n_states
                        determinants[d][state -1] = cval 


         ### Save and return string with trimmed determinants
        lines = []
        lines.append(f"{n_states} {n_mos} {len(determinants)}")
        for det, coeffs in determinants.items():
            coeff_str = " ".join(f"{c: .12f}" for c in coeffs)
            lines.append(f"{det} {coeff_str}")
        det_string = "\n".join(lines) + "\n"

        
        return det_string


# ---------------------------------| Main Function |--------------------------------------------------------------------       

if __name__ == "__main__":
    SHARC_OPENQP().main()

