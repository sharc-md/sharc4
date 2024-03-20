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

__all__ = ["SHARC_<INTERFACE_NAME>"]                    #TODO: Put in your interface name everywere where you find <INTERFACE_NAME>

AUTHORS = "Jane Doe, Ashok Kumar, Max Mustermann"       #TODO: Names of contributors
VERSION = "x.y"                                         #TODO: Version of your interface.
VERSIONDATE = datetime.datetime(<year>, <month>, <day>) #TODO: Put in year, month, date of the last update to the interface.
NAME = "<INTERFACE_NAME>"                               #TODO
DESCRIPTION = "SHARC interface for the <name> program"  #TODO: Put in the name of your QC program

#TODO: Here you can write what changed from one version to the other.
CHANGELOGSTRING = """27.10.2021:     Initial version 0.1 by Max

01.03.2024:     New implementation version 0.2 by Jane and Ashok
"""

all_features = set(
    [
        #TODO: requests that your interface can fullfill. Delete the ones that cannot be used. 
        "h",
        "dm",
        "grad",
        "nacdr",
        "overlap",
        "molden",
        # Rest of the possible requests:
        # "phases",
        # "ion",
        # "socdr",
        # "dmdr",
        # "multipolar_fit",
        # "theodore",
        # "basis_set",
        # "density_matrices",
        # "cleanup",
        # "backup",
        # "retain",
        # "molden",
        # "savestuff",
        # "nooverlap",
    ]
)

#TODO: Constants that you need. Examples:
KCAL_TO_EH = 0.0015936010974213599
EV_TO_EH = 0.03674930495120813
BOHR_TO_ANG = 0.529176125
D2AU = 1 / 0.393456

#TODO: Definition of your Interface class. You have to define the resources and template variables.
class SHARC_<INTERFACE_NAME>(SHARC_ABINITIO):
    """
    SHARC interface for <INTERFACE_NAME>
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loglevel = 10  # Added by Sascha for check-up

        # Add resource keys
        self.QMin.resources.update(
            {
                #Resources that your QC-program and interface need go here.
                #TODO:
                "programdir": None, # Path to the executable of the QC-program
                "wfthres": 1.0,     # Norm of wave function for writing determinant files
                "numocc": None,     # Number of orbitals to not ionize from in Dyson calculations
            }
        )
        self.QMin.resources.types.update(
            {
                #TODO:
                "programdir": str,
                "wfthres": float,
                "numocc": int,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                #Program specific informations go here. This data is read from the <INTERFACE_NAME>.template file.
                #You can put whatever that you need here. As an example here we have some inputs for the MNDO program.
                #You also have to define the datatype for each template-variable below.
                #TODO: Examples:
                "nciref": 1,
                "kitscf": 5000,
                "ici1": 0,
                "ici2": 0,
                "act_orbs": [1],
                "movo": 0,
                "kharge": 0,
                "imomap": 0,
            }
        )
        self.QMin.template.types.update(
            {
                #TODO:
                "nciref": int,
                "kitscf": int,
                "ici1": int,
                "ici2": int,
                "act_orbs": list,
                "movo": int,
                "kharge": int,
                "imomap": int,
            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_<INTERFACE_NAME>._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_<INTERFACE_NAME>._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_<INTERFACE_NAME>._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_<INTERFACE_NAME>._authors

    @staticmethod
    def name() -> str:
        return SHARC_<INTERFACE_NAME>._name

    @staticmethod
    def description() -> str:
        return SHARC_<INTERFACE_NAME>._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_<INTERFACE_NAME>._name}\n{SHARC_<INTERFACE_NAME>._description}"


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

    #TODO execute_from_qmin executes your QM program with the needed input files.
    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        First step, setup_workdir (write inputfiles, copy orbital guesses (if possible), xyz, pc)
        Run program
        checkstatus(), check in the workdir if calculation was successfull
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b generate molden file, stripping)
        """
        # Setup workdir. Defined in utils. Makes folder if its doesn't exist or cleans it if it already exists.
        mkdir(workdir)
        
        step = self.QMin.save["step"]

        savedir = self.QMin.save["savedir"]

        # Write MNDO input
        input_str = self.generate_inputstr(qmin)                    #TODO: write your generate_inputstr method.
        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "<Inputfilename>")       #TODO: write in the filename of the inputfile
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)

        #TODO: If your QC program can work with point charges you have to treat it accordingly. In the example a file (<PCFILE>) is generated with all the information about the point charges.
        if self.QMin.molecule["point_charges"]:
            pc_str = ""
            for coords, charge in zip(self.QMin.coords["pccoords"], self.QMin.coords["pccharge"]):
                pc_str += f"{' '.join(map(str, coords))} {charge}\n"
            writefile(os.path.join(workdir, "<PCFILE>"), pc_str)

        #TODO: Setup your QM program. Make sure the execute string is done right, depends on your program.
        # Setup MNDO
        starttime = datetime.datetime.now()
        exec_str = f"{os.path.join(qmin.resources['<PROGRAMDIR>'],'<EXECUTABLE>')} < {os.path.join(workdir, '<INPUTFILE>')} > {os.path.join(workdir, '<OUTFILE>')}"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "<OUTFILE>"), os.path.join(workdir, "<ERRORFILE>")
        ) 

        #TODO: Make error-handling in case something with the calculation goes wrong.
        if (os.path.getsize(os.path.join(workdir, "<ERRORFILE>")) > 0 ):
            with open(os.path.join(workdir, "<ERRORFILE>"), "r", encoding="utf-8") as err_file:
                    self.log.error(err_file.read()) #Write the error message to the logger
            exit_code = -1
        # If datafile to small then it only contains the error message 
        elif (os.path.getsize(os.path.join(workdir, "<OUTPUTFILE>")) < 100):
            with open(os.path.join(workdir, "<OUTPUTFILE>"), "r", encoding="utf-8") as err_file:
                    self.log.error(err_file.read()) #Write the error message to the logger
            exit_code = -1

        endtime = datetime.datetime.now()

        #TODO: Delete files that are not longer needed. But workdir is always overwritten.
        work_files = os.listdir(workdir)
        for file in work_files: 
            if not re.search(r"\.inp$|\.out$|\.err$|\.dat$", file):
                os.remove(os.path.join(workdir, file))

        return exit_code, endtime - starttime

    #TODO: With this method you can save important files to the save directory. E.g. AO overlaps, MO coefficents, determinants and CI-coefficents are needed to calculate the overlaps with wfoverlap.x
    def _save_files(self, workdir: str) -> None:
        """
        Save files (molden, mos) to savedir
        Naming convention: file.job.step
        """
        
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]
        # save files

        #TODO: Save the files that you need in each step.
        # molden
        moldenfile = os.path.join(workdir, "molden.dat")
        tofile = os.path.join(savedir, f"molden.{step}")
        shutil.copy(moldenfile, tofile)

        # MOs
        mos, MO_occ, *_ = self._get_MO_from_molden(moldenfile)
        mo = os.path.join(savedir, f"mos.{step}")
        writefile(mo, mos)

        # imomap for orbital mapping
        if self.QMin["template"]["imomap"] == 3:
            fromfile = os.path.join(workdir, "imomap.dat")
            tofile = os.path.join(savedir, f"imomap.{step}")
            shutil.copy(fromfile, tofile)

        # dets
        log_file = os.path.join(workdir, "MNDO.out")
        nstates = self.QMin.molecule["nstates"]
        determinants = self._get_determinants(log_file, MO_occ, nstates)
        det = os.path.join(savedir, f"dets.{step}")
        writefile(det, determinants)
        return

    #TODO: this method returns the MO coefficents from a molden file. Change it to your needs and preferences.
    def _get_MO_from_molden(self, molden_file: str):
        """
        Extract MO coefficients from molden file

        out_file:   Path of out file
        jobid:      ID number of job
        """
        # get MOs from molden file and  1) transform to string for mol-file
        #                              2) write GS occupation to MO_occ
        f = readfile(molden_file)
        mo_coeff_matrix = []
        # get MOs and MO_occ in a dict from molden file
        
        MOs = {}
        NMO = 0
        MO_occ = {}
        for iline, line in enumerate(f):
            if "Sym= " in line:
                NMO += 1
                AO = {}
                o = f[iline + 3].split()
                MO_occ[NMO] = o[1]
                jline = iline + 4
                line = f[jline]
                while "Sym= " not in line:
                    s = line.split()
                    AO[int(s[0])] = float(s[1])
                    jline += 1
                    if jline == len(f):
                        break
                    line = f[jline]
                MOs[NMO] = AO

        # make string
        string = """2mocoef
header
1
MO-coefficients from OM2/MRCI
1
%i   %i
a
mocoef
(*)
""" % (
            NMO,
            len(AO),
        )
        x = 0
        for i in range(NMO):
            mo = []
            for j in range(len(AO)):
                c = MOs[i + 1][j + 1]
                if x >= 3:
                    string += "\n"
                    x = 0
                string += "% 6.12e " % c
                x += 1
                mo.append(c)
            if x > 0:
                string += "\n"
                x = 0

            mo_coeff_matrix.append(mo)
        string += "orbocc\n(*)\n"
        x = 0
        for i in range(NMO):
            if x >= 3:
                string += "\n"
                x = 0
            string += "% 6.12e " % (0.0)
            x += 1
        return string, MO_occ, len(AO), mo_coeff_matrix


    def _get_csfs(self, log_file: str, active_mos, nstates):
        # get CSF composition of states

        # open file
        f = readfile(log_file)
        # note the lines in which the states are
        state_lines = []
        for iline, line in enumerate(f):
            for i in range(1, nstates + 1):
                if "State  " + str(i) in line:
                    state_lines.append(iline)
            if "Using basis sets ECP-3G (first-row elements) and ECP-4G (second-row" in line:
                state_lines.append(iline)

        # read csfs from logfile
        csf_ref = {}
        for i in range(nstates):
            for iline, line in enumerate(f[state_lines[i] : state_lines[i + 1]]):
                if str(" " + str(active_mos[0]) + " ") in line:
                    x = line.split()
                    ref = f[state_lines[i] + iline + 1]
                    y = ref.split()
                    if x[2] not in csf_ref.keys():
                        csf_ref[x[2]] = {}
                        csf_ref[x[2]]["coeffs"] = [0.0 for k in range(nstates)]
                        csf_ref[x[2]]["CSF"] = {}
                        for imo, mo in enumerate(active_mos):
                            csf_ref[x[2]]["CSF"][mo] = y[imo]
                    csf_ref[x[2]]["coeffs"][i] = float(x[1])

        return csf_ref


    def _decompose_csf(self, ms2, step):
        # ms2 is M_S value
        # step is step vector for CSF (e.g. 3333012021000)

        def powmin1(x):
            a = [1, -1]
            return a[x % 2]

        # calculate key numbers
        nopen = sum([i == 1 or i == 2 for i in step])
        nalpha = int(nopen / 2.0 + ms2)
        norb = len(step)

        # make reference determinant
        refdet = deepcopy(step)
        for i in range(len(refdet)):
            if refdet[i] == 1:
                refdet[i] = 2

        # get the b vector and the set of open shell orbitals
        bval = []
        openorbs = []
        b = 0
        for i in range(norb):
            if step[i] == 1:
                b += 1
            elif step[i] == 2:
                b -= 1
            bval.append(b)
            if refdet[i] == 2:
                openorbs.append(i)

        # loop over the possible determinants
        dets = {}
        # get all possible combinations of nalpha orbitals from the openorbs set
        for localpha in itertools.combinations(openorbs, nalpha):
            # make determinant string
            det = deepcopy(refdet)
            for i in localpha:
                det[i] = 1

            # get coefficient
            coeff = 1.0
            sign = +1
            m2 = 0
            for k in range(norb):
                if step[k] == 1:
                    m2 += powmin1(det[k] + 1)
                    num = bval[k] + powmin1(det[k] + 1) * m2
                    denom = 2.0 * bval[k]
                    if num == 0.0:
                        break
                    coeff *= 1.0 * num / denom
                elif step[k] == 2:
                    m2 += powmin1(det[k] - 1)
                    num = bval[k] + 2 + powmin1(det[k]) * m2
                    denom = 2.0 * (bval[k] + 2)
                    sign *= powmin1(bval[k] + 2 - det[k])
                    if num == 0.0:
                        break
                    coeff *= 1.0 * num / denom
                elif step[k] == 3:
                    sign *= powmin1(bval[k])
                    num = 1.0

            # add determinant to dict if coefficient non-zero
            if num != 0.0:
                dets[tuple(det)] = 1.0 * sign * math.sqrt(coeff)

        return dets


    def _format_ci_vectors(self, ci_vectors, MO_occ, nstates):

        norb = len(MO_occ)
        ndets = len(ci_vectors) - 1

        # sort determinant strings
        dets = []
        for key in ci_vectors:
            if key != "active MOs":
                dets.append(key)
        dets.sort(reverse=True)

        # write first line of det-file
        string = "%i %i %i\n" % (nstates, norb, ndets)

        # dictionary to get "a/b/d/e"-nomenclature
        dict_int_to_de = {"2.0": "d", "0.0": "e", 3: "d", 2: "b", 1: "a", 0: "e"}

        # take basis MO occupany: 1) change it according to det;     2) add it to string ;
        ##                       3) add coefficients;               4)return string
        for det in dets:
            MO_occ_cp = deepcopy(MO_occ)
            for i, orb in enumerate(ci_vectors["active MOs"]):
                MO_occ_cp[orb] = det[i]
            # raise TypeError(MO_occ_cp)
            for MO in MO_occ_cp:
                string += dict_int_to_de[MO_occ_cp[MO]]
            for c in ci_vectors[det]:
                string += " %16.12f " % c
            string += "\n"

        return string


    def _get_determinants(self, log_file, MO_occ, nstates):

        # dictionary to convert to "0123"-nomenclature
        dict_ab_to_int = {"ab": "3", "a": "1", "b": "2", "-": "0"}

        ci_vectors = {}

        # add MO occupancy to ci_vector
        active_mos = [*self._get_active_space(log_file)]
        ci_vectors["active MOs"] = active_mos

        # get CSFs from log_file
        csf = self._get_csfs(log_file, active_mos, nstates)

        # convert to determinants
        ## 1) Convert nomenclature
        for i in csf:
            ref = []
            for k in csf[i]["CSF"].keys():
                ref.append(dict_ab_to_int[csf[i]["CSF"][k]])
            ref = tuple([int(n) for n in ref])
            csf[i]["CSF"] = ref
        ## 2) build ci vector from CSFs
        for x in csf:
            dets = self._decompose_csf(0, list(csf[x]["CSF"]))
            coeff = csf[x]["coeffs"]
            for det in dets:
                c = [dets[det] * i for i in coeff]
                if det in ci_vectors:
                    for istate in range(len(coeff)):
                        ci_vectors[det][istate] += c[istate]
                else:
                    ci_vectors[det] = c

        determinants = self._format_ci_vectors(ci_vectors, MO_occ, nstates)

        return determinants


    #TODO: Populate all of the things that where requested in the QMin requests.
    def getQMout(self) -> None:
        """
        Parse <prgram_name> output files and populate QMout
        """
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

        #TODO: parse output files and populate QMout. Here is a simple example:
        # Allocate matrices
        nmstates = self.QMin.molecule["nmstates"]

        log_file = os.path.join(self.QMin.control["workdir"], "<your_program>.out")
        grads_nacs_file = os.path.join(self.QMin.control["workdir"], "fort.15")
        
        # Get contents of output file(s)
        states, interstates = self._get_states_interstates(log_file)

        mults = self.QMin.maps["mults"]

        # Populate energies
        if self.QMin.requests["h"]:
            file = open(log_file, "r")
            output = file.read()
            energies = self._get_energy(output)
            for i in range(len(energies)):
                self.QMout["h"][i][i] = energies[(1, i + 1)]

        # Populate dipole moments
        if self.QMin.requests["dm"]:
            self.QMout.dm = self._get_transition_dipoles(log_file)

        # Populate gradients
        if self.QMin.requests["grad"]:
            self.QMout.grad = self._get_grad(grads_nacs_file)
            if self.QMin.molecule["point_charges"]:
                self.QMout.grad_pc = self._get_grad_pc(grads_nacs_file)

        # Populates NACs
        if self.QMin.requests["nacdr"]:
            self.QMout.nacdr = self._get_nacs(grads_nacs_file, interstates)
            if self.QMin.molecule["point_charges"]:
                self.QMout.nacdr_pc = self._get_nacs_pc(grads_nacs_file, interstates)

        # Populate overlaps
        if self.QMin.requests["overlap"]:
            if "overlap" not in self.QMout:
                self.QMout["overlap"] = makecmatrix(nmstates, nmstates)
            for mult in itmult(self.QMin.molecule["states"]):
                outfile = os.path.join(self.QMin.resources["scratchdir"], "wfovl.out")
                out = readfile(outfile)
                for i in range(nmstates):
                    for j in range(nmstates):
                        self.QMout["overlap"][i][j] = self.getsmate(out, i + 1, j + 1)


    # TODO: Parse the output file to get the gradients. Attention: SHARC works with atomic units so gradients need to be Hartress/a_0 and preferably a numpy-array should be returned.
    def _get_grad(self, log_path: str) -> np.ndarray:
        """
        Extract gradients from MNDO outfile

        log_path:  Path to fort.15 file
        """
        nmstates = self.QMin.molecule["nmstates"]

        grad = [y - 1 for x,y in self.QMin["maps"]["gradmap"]]
        natom = self.QMin.molecule["natom"]
        f = readfile(log_path)

        line_marker = []
        regexp = re.compile(r" CARTESIAN GRADIENT FOR STATE \s+([\d]*)\n")
        for iline, line in enumerate(f):
            if regexp.search(line):
                line_marker.append(iline + 1)

        grads = np.zeros((nmstates,natom,3))
        
        for l, st in enumerate(grad):
            iline = line_marker[l]
            for j in range(natom):
                line = f[iline]
                s = line.split()
                if self.QMin.molecule["point_charges"]:                         # In the fort.15 file, depending if thhe calculation includs point charges or not, there is a different amount of columns for the gradients and NACs
                    grads[st, j, 0] =  float(s[-4]) * KCAL_TO_EH * BOHR_TO_ANG  # kcal/Ang --> H/a0
                    grads[st, j, 1] =  float(s[-3]) * KCAL_TO_EH * BOHR_TO_ANG
                    grads[st, j, 2] =  float(s[-2]) * KCAL_TO_EH * BOHR_TO_ANG
                else:
                    grads[st, j, 0] =  float(s[-3]) * KCAL_TO_EH * BOHR_TO_ANG
                    grads[st, j, 1] =  float(s[-2]) * KCAL_TO_EH * BOHR_TO_ANG
                    grads[st, j, 2] =  float(s[-1]) * KCAL_TO_EH * BOHR_TO_ANG
                iline += 1

        return grads

    #TODO: Gradients for point charges
    def _get_grad_pc(self, log_path: str) -> np.ndarray:
        """
        Extract gradients from MNDO outfile

        log_path:  Path to gradient file
        """
        nmstates = self.QMin.molecule["nmstates"]
        grad = [y - 1 for x,y in self.QMin["maps"]["gradmap"]]
        ncharges = self.QMin.molecule["npc"]
        f = readfile(log_path)

        line_marker = []
        regexp = re.compile(r" CARTESIAN GRADIENT OF MM ATOMS FOR STATE \s+([\d]*)\n")
        for iline, line in enumerate(f):
            if regexp.search(line):
                line_marker.append(iline + 1)

                
        grads = np.zeros((nmstates,ncharges,3))
        
        for l, st in enumerate(grad):
            iline = line_marker[l]
            for j in range(ncharges):
                line = f[iline]
                s = line.split()
                grads[st, j, 0] =  float(s[-3]) * KCAL_TO_EH * BOHR_TO_ANG
                grads[st, j, 1] =  float(s[-2]) * KCAL_TO_EH * BOHR_TO_ANG
                grads[st, j, 2] =  float(s[-1]) * KCAL_TO_EH * BOHR_TO_ANG
                iline += 1

        return grads

    #TODO: Parse non-adiabatic couplings from the output file. Attention: Atomic units --> 1/a_0
    def _get_nacs(self, file_path: str, interstates) -> np.ndarray:
        """
        Extract NACS from MNDO outfile

        log_path:  Path to fort.15 file
        """
        natom = self.QMin.molecule["natom"]
        nmstates = self.QMin.molecule["nmstates"]

        f = readfile(file_path)

        line_marker = []
        
        regexp = re.compile(r"CARTESIAN INTERSTATE COUPLING GRADIENT FOR STATES \s+([\d]*) \s+([\d]*)\n")
        for iline, line in enumerate(f):
            if regexp.search(line):
                line_marker.append(iline + 1)

                
        nac = np.zeros((nmstates, nmstates, natom, 3))
        
        
        # make nac matrix
        # nac = np.fromiter(map(), count=).reshape()
        for i, (s1, s2) in enumerate(interstates):
            iline = line_marker[i]
            dE = self.QMout["h"][s2,s2].real - self.QMout["h"][s1,s1].real # In MNDO cannot calculate imaginary energies
            for j in range(natom):
                line = f[iline]
                s = line.split()
                if self.QMin.molecule["point_charges"]: #In the fort.15 file, depending if thhe calculation includs point charges or not, there is a different amount of columns for the gradients and NACs
                    nac[s1, s2, j, 0] =  float(s[-4])   
                    nac[s1, s2, j, 1] =  float(s[-3])
                    nac[s1, s2, j, 2] =  float(s[-2])
                    nac[s2, s1, j, 0] = -float(s[-4])
                    nac[s2, s1, j, 1] = -float(s[-3])
                    nac[s2, s1, j, 2] = -float(s[-2])
                else: 
                    nac[s1, s2, j, 0] =  float(s[-3])  
                    nac[s1, s2, j, 1] =  float(s[-2])
                    nac[s1, s2, j, 2] =  float(s[-1])
                    nac[s2, s1, j, 0] = -float(s[-3])
                    nac[s2, s1, j, 1] = -float(s[-2])
                    nac[s2, s1, j, 2] = -float(s[-1])

                iline += 1
            if (dE != 0.0):
                nac[s1,s2,...] = nac[s1,s2,...] * KCAL_TO_EH * BOHR_TO_ANG / dE # kcal/mol*Ang --> 1/a_0
                nac[s2,s1,...] = nac[s2,s1,...] * KCAL_TO_EH * BOHR_TO_ANG / dE
        
        return nac
    
    #TODO: NACs for point charges.
    def _get_nacs_pc(self, file_path: str, interstates) -> np.ndarray:
        """
        Extract NACS from MNDO outfile

        log_path:  Path to fort.15 file
        """
        ncharges = self.QMin.molecule["npc"]
        nmstates = self.QMin.molecule["nmstates"]

        f = readfile(file_path)

        line_marker = []
        
        regexp = re.compile(r"CARTESIAN INTERSTATE COUPLING GRADIENT OF MM ATOMS FOR STATES \s+([\d]*) \s+([\d]*)\n")
        for iline, line in enumerate(f):
            if regexp.search(line):
                line_marker.append(iline + 1)

                
        nac = np.zeros((nmstates, nmstates, ncharges, 3))
        
        # make nac matrix
        for i, (s1, s2) in enumerate(interstates):
            iline = line_marker[i] 
            dE = self.QMout["h"][s2, s2].real - self.QMout["h"][s1, s1].real # In MNDO cannot calculate imaginary energies
            for j in range(ncharges):
                line = f[iline]
                s = line.split() 
                nac[s1, s2, j, 0] =  float(s[-3]) 
                nac[s1, s2, j, 1] =  float(s[-2])
                nac[s1, s2, j, 2] =  float(s[-1])
                nac[s2, s1, j, 0] = -float(s[-3])
                nac[s2, s1, j, 1] = -float(s[-2])
                nac[s2, s1, j, 2] = -float(s[-1])
                iline += 1
            if (dE != 0.0):
                nac[s1,s2,...] = nac[s1,s2,...] * KCAL_TO_EH * BOHR_TO_ANG / dE  # kcal/mol*Ang --> 1/a_0 
                nac[s2,s1,...] = nac[s2,s1,...] * KCAL_TO_EH * BOHR_TO_ANG / dE

        return nac

    #TODO: Parse output file to get the transition dipole moments (TDMs). Attention: Atomic units
    def _get_transition_dipoles(self, log_path: str) -> np.ndarray:
        """
        Extract transition dipole moments from MNDO outfile

        log_path:   Path to logfile
        """

        nmstates = self.QMin.molecule["nmstates"]

        # Extract transition dipole table from output
        f = readfile(log_path)

        dm = [[[0.0 for j in range(nmstates)] for k in range(nmstates)] for i in range(3)]
        states = []
        # diagonal elements
        for iline, line in enumerate(f):
            if "State dipole moments:" in line:
                iline += 3
                for st in range(nmstates):
                    line = f[iline]
                    s = line.split()
                    dmx = float(s[5]) * D2AU
                    dmy = float(s[6]) * D2AU
                    dmz = float(s[7]) * D2AU
                    state = int(s[0])
                    states.append(state)
                    dm[0][state - 1][state - 1] = dmx
                    dm[1][state - 1][state - 1] = dmy
                    dm[2][state - 1][state - 1] = dmz
                    iline += 1

        # off-diagonal elements
        line_offdiag = []
        for iline, line in enumerate(f):
            if "Dipole-length electric dipole transition moments" in line:
                line_offdiag.append(iline + 3)
        noffdiag = int(nmstates - 1)
        st = 0
        for i in line_offdiag:
            for j in range(noffdiag):
                line = f[i]
                s = line.split()
                dmx = float(s[5]) * D2AU
                dmy = float(s[6]) * D2AU
                dmz = float(s[7]) * D2AU
                dm[0][states[st] - 1][int(s[0]) - 1] = dmx
                dm[1][states[st] - 1][int(s[0]) - 1] = dmy
                dm[2][states[st] - 1][int(s[0]) - 1] = dmz
                dm[0][int(s[0]) - 1][states[st] - 1] = dmx
                dm[1][int(s[0]) - 1][states[st] - 1] = dmy
                dm[2][int(s[0]) - 1][states[st] - 1] = dmz
                i += 1
            noffdiag -= 1
            st += 1

        if not dm:
            self.log.error("Cannot find transition dipoles in MNDO output!")
            raise ValueError()
        # Filter dipole vectors, (states, (xyz))
        return np.array(dm)

    #TODO: Parse output file to get the energy of each state. Attention --> Hartree
    def _get_energy(self, output: str) -> dict[tuple[int, int], float]:
        """
        Extract energies from ORCA outfile

        output:     Content of outfile as string
        mult:       Multiplicities, not needed for now (deleted)
        """
        #TODO:
        pattern = re.compile(r"eV,  E=[\s:]+([-+]\d*\.*\d+) eV")
        energies = {}
        for i, match in enumerate(pattern.finditer(output)):
            energies[(1, i + 1)] = float(match.group(1)) * EV_TO_EH  # only singlets for now!!

        return energies




    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        return


    def printQMout(self) -> None:
        super().writeQMout()


    def print_qmin(self) -> None:
        pass

    #TODO: Reads the <Name>.resources file. Make checks if necessary.
    def read_resources(self, resources_file: str = "<Name>.resources") -> None:
        super().read_resources(resources_file)
        # LD PATH???
        #TODO:
        if not self.QMin.resources["mndodir"]:
            raise ValueError("mndodir has to be set in resource file!")

        self.QMin.resources["mndodir"] = expand_path(self.QMin.resources["mndodir"])
        self.log.debug(f'mndodir set to {self.QMin.resources["mndodir"]}')

    #TODO: Reads the requests from the QM.in file. Make checks if necessary
    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)
        #TODO:
        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                raise ValueError(f"Found unsupported request {req}.")

    #TODO: Reads the <NAME>.template file. Make necessary checks ans cast strings to integers, if needed.
    def read_template(self, template_file: str = "MNDO.template") -> None:
        super().read_template(template_file)

        #TODO:
        self.QMin["template"]["kharge"] = int(self.QMin["template"]["kharge"]) #cast template inputs to int
        self.QMin["template"]["imomap"] = int(self.QMin["template"]["imomap"])
        
        if self.QMin["template"]["imomap"] < 0 or self.QMin["template"]["imomap"] > 1:  # Check if imomap is not out of range.
            raise ValueError(f"imomap can either be 0 (false) or 1 (true). Negative numbers not supported!")
        elif self.QMin["template"]["imomap"] > 0:
            self.QMin["template"]["imomap"] = 3   #Orbital tracking activated when imomap=3 in the MNDO.inp file.

        
        self.QMin["template"]["movo"] = int(self.QMin["template"]["movo"])
        if self.QMin["template"]["movo"] > 1 or self.QMin["template"]["movo"] < 0 :
            raise ValueError(f"movo can only be 0 (false) or 1 (true).")
        
        if self.QMin["template"]["movo"] == 1 :
            self.QMin["template"]["act_orbs"] = [int(i) for i in self.QMin["template"]["act_orbs"]]

    #TODO: How many steps back (retain) do you want to keep your restart files?
    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """

    #TODO: This method runs the interface and basically makes one time step.
    def run(self) -> None:
        """
        request & other logic
            define requestmaps -> DONE IN SETUP_INTERFACE
            path to the different orbital restart files -> DONE IN SETUP_INTERFACE
        make schedule
        runjobs()
        run_wfoverlap (needs input files)
        save directory handling
        """

        starttime = datetime.datetime.now()
        
        #TODO: Think about what your program needs to do. Does it need to run more than one calculation to get the needed data?

        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "<FOLDER_NAME>")

        #TODO: Generate a schedule if several calculations are needed. In this example we just need one calculation.
        schedule = [{"<NAME>" : self.QMin}] #Generate fake schedule
        self.QMin.control["nslots_pool"].append(1)
        self.runjobs(schedule)

        #Save the files
        self._save_files(self.QMin.control["workdir"])

        # Run wfoverlap
        if self.QMin.requests["overlap"]:
            self._run_wfoverlap()

        self.log.debug("All jobs finished successfully")


        self.QMout["runtime"] = datetime.datetime.now() - starttime

    #TODO: If you need overlaps, then here you can run the wfoverlap program.
    def _run_wfoverlap(self) -> None:
        """
        Prepare files and folders for wfoverlap and execute wfoverlap
        """
        #TODO:
        # Content of wfoverlap input file
        wf_input = dedent(
            """\
        a_mo=mo.a
        b_mo=mo.b
        a_det=det.a
        b_det=det.b
        ao_read=-1
        same_aos
        """
        )
        if self.QMin.resources["numocc"]:
            wf_input += f"ndocc={self.QMin.resources['numocc']}\n"

        if self.QMin.resources["ncpu"] >= 8:
            wf_input += "force_direct_dets"

        # cmdline string
        wf_cmd = f"{self.QMin.resources['wfoverlap']} -m {self.QMin.resources['memory']} -f wfovl.inp"

        # Overlap calculations
        if self.QMin.requests["overlap"]:
            workdir = self.QMin.resources["scratchdir"]

            # Write input file
            writefile(os.path.join(workdir, "wfovl.inp"), wf_input)

            # Link files
            #TODO: Get the needed files form the save directory
            link(
                os.path.join(self.QMin.save["savedir"], f"dets.{self.QMin.save['step']}"),
                os.path.join(workdir, "det.a"),
            )
            link(
                os.path.join(self.QMin.save["savedir"], f"dets.{self.QMin.save['step']-1}"),
                os.path.join(workdir, "det.b"),
            )
            link(
                os.path.join(self.QMin.save["savedir"], f"mos.{self.QMin.save['step']}"),
                os.path.join(workdir, "mo.a"),
            )
            link(
                os.path.join(self.QMin.save["savedir"], f"mos.{self.QMin.save['step']-1}"),
                os.path.join(workdir, "mo.b"),
            )

            starttime = datetime.datetime.now()

            #TODO: Execute wfoverlap and handle eventual errors.
            code = self.run_program(workdir, wf_cmd, os.path.join(workdir, "wfovl.out"), os.path.join(workdir, "wfovl.err"))
            self.log.info(f"Finished wfoverlap job!!\nruntime: {datetime.datetime.now()-starttime}")
            if code != 0:
                self.log.error("wfoverlap did not finish successfully!")
                with open(os.path.join(workdir, "wfovl.err"), "r", encoding="utf-8") as err_file:
                    self.log.error(err_file.read())
                raise OSError()

    #TODO: Generate the inputsring for the input-file of your program of choice. Here you can apply the things defined in the template file.
    @staticmethod
    def generate_inputstr(qmin: QMin) -> str:
        """
        Generate <INTERFACE_NAME> input file string from QMin object
        """
        #TODO: get needed information from flags or the QMin object
        natom = qmin["molecule"]["natom"]
        ncigrd = len(qmin["maps"]["gradmap"])
        coords = qmin["coords"]["coords"]
        elements = qmin["molecule"]["elements"]
        movo = qmin["template"]["movo"]
        ici1 = qmin["template"]["ici1"]
        ici2 = qmin["template"]["ici2"]
        nciref = qmin["template"]["nciref"]
        act_orbs = qmin["template"]["act_orbs"]
        iroot = qmin["molecule"]["states"][0]
        ncharges = qmin["molecule"]["npc"]
        grads = [y for x,y in qmin["maps"]["gradmap"]]
        charge = qmin["template"]["kharge"]
        imomap = qmin["template"]["imomap"]


        #Flags for calculation
        inputstring = f"iop=-6 jop=-2 imult=0 iroot={iroot} icross=7 ncigrd={ncigrd} inac=0 imomap={imomap}  ici1={ici1} ici2={ici2} movo={movo} nciref={nciref} kharge={charge} multci=1 cilead=1 ncisym=-1 numatm={ncharges} nsav15=9"

        #Header comments
        inputstring = " +\n".join(wrap(inputstring, width=70))
        inputstring += "\nheader\n"
        inputstring += "header\n"

        #Geometry
        for i in range(natom):
            inputstring += f"{NUMBERS[elements[i]]:>3d}\t{coords[i][0]*BOHR_TO_ANG:>10,.5f} 1\t{coords[i][1]*BOHR_TO_ANG:>10,.5f} 1\t{coords[i][2]*BOHR_TO_ANG:>10,.5f} 1\n"
        inputstring += f"{0:>3d}\t{0:>10,.5f} 0\t{0:>10,.5f} 0\t{0:>10,.5f} 0\n"

        #Additional information
        if movo == 1:
            for j in act_orbs:
                inputstring += str(j) + " "
            inputstring += "\n"

        for l in grads:
            inputstring += str(l) + " "

        return inputstring
    
    
    #Only needed for ECI calculations
    def _create_aoovl(self) -> None:
        #empty function
        pass

    #Only needed for ECI calculations
    def get_mole(self) -> None:
        #empty function
        pass

    #Only needed for ECI calculations   
    def get_readable_densities(self) -> None:
        #empty function
        pass

    #Only needed for ECI calculations
    def read_and_append_densities(self) -> None:
        #empty function
        pass

    #TODO: This method sets up the interface. Do you need some steps to define your maps (gsmap, gradmap, nacmap)? Do you have padding states that you need to account for? Unrestricted triplets?
    def setup_interface(self) -> None:
        """
        Setup remaining maps (ionmap, gsmap) and build jobs dict
        """
        super().setup_interface()
        



if __name__ == "__main__":
    SHARC_<INTERFACE_NAME>(loglevel=2).main()
