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
from utils import containsstring, expand_path, question, link, makecmatrix, mkdir, readfile, writefile

__all__ = ["SHARC_MNDO"]

AUTHORS = "Nadja K. Singer, Hans Georg Gallmetzer"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2024, 12, 2)
NAME = "MNDO"
DESCRIPTION = "AB INITIO interface for the MNDO program (OM2-MRCI)"

CHANGELOGSTRING = """27.10.2021:     Initial version 0.1 by Nadja
- Only OM2/MRCI
- Only singlets
- Not functioning

24.04.2024:     New implementation version 0.2 by Georg
- also ODM2/MRCI
- Point charges
- Overlaps/Phases
- NACDR
- Problems with the MO-Tracking through imomap file.

02.12.2024:     Minor fixes/changes, version 1.0 by Georg
- Fully working version.
- Problems with MO-Tracking through imomap file could not be resolved.
"""

all_features = set(
    [
        "h",
        "dm",
        "grad",
        "nacdr",
        "overlap",
        "phases",
        "molden",
        "point_charges",
    ]
)



class SHARC_MNDO(SHARC_ABINITIO):
    """
    SHARC interface for MNDO
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
                "mndodir": None,
            }
        )
        self.QMin.resources.types.update(
            {
                "mndodir": str,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "nciref": 1,
                "kitscf": 5000,
                "ici1": 0,
                "ici2": 0,
                "act_orbs": [],
                "imomap": 0,
                "hamiltonian": None,
                "fomo": 0,
                "rohf": 0,
                "levexc": 2,
                "mciref": 0,
                
            }
        )
        self.QMin.template.types.update(
            {
                "nciref": int,
                "kitscf": int,
                "ici1": int,
                "ici2": int,
                "act_orbs": list,
                "imomap": int,
                "hamiltonian": str,
                "fomo": int,
                "rohf": int,
                "levexc": int,
                "mciref": int,

            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_MNDO._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_MNDO._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_MNDO._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_MNDO._authors

    @staticmethod
    def name() -> str:
        return SHARC_MNDO._name

    @staticmethod
    def description() -> str:
        return SHARC_MNDO._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_MNDO._name}\n{SHARC_MNDO._description}"


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
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'MNDO interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        self.files = []

        self.template_file = None
        self.log.info(f"{'MNDO input template file':-^60}\n")

        if os.path.isfile("MNDO.template"):
            usethisone = question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                self.template_file = "MNDO.template"
        else:
            while True:
                self.template_file = question("Template filename:", str, KEYSTROKES=KEYSTROKES)
                if not os.path.isfile(self.template_file):
                    self.log.info(f"File {self.template_file} does not exist!")
                    continue
                break
            
        self.log.info("")
        self.files.append(self.template_file)

        self.make_resources = False
        
        # Resources
        # TODO: either ask for resource file at the top of this routine or not at all...
        if question("Do you have a 'MNDO.resources' file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            while True:
                resources_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="MNDO.resources")
                self.files.append(resources_file)
                self.make_resources = False
                if os.path.isfile(resources_file):
                    break
                else:
                    self.log.info(f"file at {resources_file} does not exist!")
        else:
            self.make_resources = True
            self.log.info(
                "\nPlease specify path to MNDO directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n"
            )
            INFOS["mndodir"] = question("Path to MNDO:", str, KEYSTROKES=KEYSTROKES)
            self.log.info("")

            # scratch
            self.log.info(f"{'Scratch directory':-^60}\n")
            self.log.info(
                "Please specify an appropriate scratch directory. This will be used to run the MNDO calculations. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script."
            )
            INFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)

            self.log.info(f"{'MNDO Ressource usage':-^60}\n")

            self.setupINFOS["memory"] = question("Memory (MB):", int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

            
            if "overlap" in INFOS["needed_requests"]:
                self.log.info(f"\n{'WFoverlap setup':-^60}\n")
                self.setupINFOS["wfoverlap"] = question(
                    "Path to wavefunction overlap executable:", str, default="$SHARC/wfoverlap.x", KEYSTROKES=KEYSTROKES
                )

        return INFOS




    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen, stripping)
        """

        # Setup workdir
        mkdir(workdir)
        
        step = self.QMin.save["step"]

        savedir = self.QMin.save["savedir"]

        if step > 0 and self.QMin["template"]["imomap"] == 3:
            orbital_tracking = os.path.join(workdir, "imomap.dat")
            saved_file = os.path.join(savedir, f"imomap.{step-1}")
            shutil.copy(saved_file, orbital_tracking)

        # Write MNDO input
        input_str = self._generate_inputstr()


        self.log.debug(f"Generating input string\n{input_str}")
        input_path = os.path.join(workdir, "MNDO.inp")
        self.log.debug(f"Write input into file {input_path}")
        writefile(input_path, input_str)
        # Write point charges
        if self.QMin.molecule["point_charges"]:
            pc_str = ""
            pccoords = np.array(self.QMin.coords["pccoords"])
            pccharges = np.array(self.QMin.coords["pccharge"])
            for coords, charge in zip(pccoords * BOHR_TO_ANG, pccharges):
                pc_str += f"{' '.join(map(str, coords))} {charge}\n"
            writefile(os.path.join(workdir, "fort.20"), pc_str)

        # Setup MNDO
        starttime = datetime.datetime.now()
        exec_str = f"{os.path.join(qmin.resources['mndodir'],'mndo2020')} < {os.path.join(workdir, 'MNDO.inp')} > {os.path.join(workdir, 'MNDO.out')}"
        exit_code = self.run_program(
            workdir, exec_str, os.path.join(workdir, "MNDO.out"), os.path.join(workdir, "MNDO.err")
        ) 
        if (os.path.getsize(os.path.join(workdir, "MNDO.err")) > 0 ):
            with open(os.path.join(workdir, "MNDO.err"), "r", encoding="utf-8") as err_file:
                    self.log.error(err_file.read())
            exit_code = -1

        elif (os.path.getsize(os.path.join(workdir, "fort.15")) < 100):
            with open(os.path.join(workdir, "fort.15"), "r", encoding="utf-8") as err_file:
                    self.log.error(err_file.read())
            exit_code = -1

        endtime = datetime.datetime.now()

        #TODO: Think about deletion of certain files
        # Delete files not needed
        # work_files = os.listdir(workdir)
        # for file in work_files: 
        #     if not re.search(r"\.inp$|\.out$|\.err$|\.dat$", file):
        #         os.remove(os.path.join(workdir, file))

        return exit_code, endtime - starttime


    def _save_files(self, workdir: str) -> None:
        """
        Save files (molden, mos) to savedir
        Naming convention: file.job.step
        """
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]

        # molden
        moldenfile = os.path.join(workdir, "molden.dat")
        tofile = os.path.join(savedir, f"molden.{step}")
        shutil.copy(moldenfile, tofile)

        # MOs
        mos, MO_occ, NAO, _, mo_energies = self._get_MO_from_molden(moldenfile)

        mo = os.path.join(savedir, f"mos.{step}")
        writefile(mo, mos)

        
        #AO_OVL
        aos = self.get_Double_AOovl(NAO)
        ao = os.path.join(savedir, f"ao.{step}")
        writefile(ao, aos)

        # imomap
        if self.QMin["template"]["imomap"] == 3:
            fromfile = os.path.join(workdir, "imomap.dat")
            tofile = os.path.join(savedir, f"imomap.{step}")
            shutil.copy(fromfile, tofile)

        # dets
        nstates = self.QMin.molecule["nstates"]
        log_file = os.path.join(workdir, "MNDO.out")
        determinants = self._get_determinants(log_file, MO_occ, nstates)
        det = os.path.join(savedir, f"dets.{step}")
        writefile(det, determinants)


        return

    @staticmethod
    def get_Double_AOovl(NAO):
        
        string = '{} {}\n'.format(NAO, NAO)
        for irow in range(0, NAO):
            for icol in range(0, NAO):
                # OMx methods have globally orthogonalized AOs (10.1063/1.5022466)
                string += '{: .15e} '.format(
                    0. if irow != icol else 1.
                )    # note the exchanged indices => transposition
            string += '\n'
        return string


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
        mo_energies = []
        # get MOs and MO_occ in a dict from molden file
        MOs = {}
        NMO = 0
        MO_occ = {}
        for iline, line in enumerate(f):
            if "Sym= " in line:
                NMO += 1
                AO = {}
                o = f[iline + 3].split()
                mo_energies.append(f[iline + 1].split()[1])
                MO_occ[NMO] = o[1]
                jline = iline + 4
                line = f[jline]
                while "Sym= " not in line:
                    s = line.split()
                    if "[GEOCONV]" in line:
                        break
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
        return string, MO_occ, len(AO), mo_coeff_matrix, mo_energies


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
        dict_int_to_int = {'2.0': '3', '0.0': '0'}
        

        ci_vectors = {}

        # add MO occupancy to ci_vector
        active_mos = [*self._get_active_space(log_file)]
        ci_vectors["active MOs"] = [*range(1, len(MO_occ)+1)]

        # get CSFs from log_file
        csf = self._get_csfs(log_file, active_mos, nstates)

        # convert to determinants
        ## 1) Convert nomenclature
        for i in csf:
            ref = []
            for k in MO_occ:
                if (k in active_mos) :
                    ref.append(dict_ab_to_int[csf[i]['CSF'][k]])
                else:
                    ref.append(dict_int_to_int[MO_occ[k]])
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


    def _get_active_space(self, log_file: str) -> dict:
        """get the active space from the log file"""
        # get active space
        f = readfile(log_file)

        active_mos = {}
        for iline, line in enumerate(f):
            if "OCC.    ACTIVE" in line:
                jline = iline + 2
                line = f[jline]
                while line != "\n":
                    s = line.split()
                    if s[6] != "-":
                        active_mos[int(s[0])] = int(s[6])
                    jline += 1
                    line = f[jline]
                break

        return active_mos


    def getQMout(self) -> None:
        """
        Parse MNDO output files
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
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )

        nmstates = self.QMin.molecule["nmstates"]
        scratchdir = self.QMin.resources["scratchdir"]

        #Energies and TDMs are taken from the MNDO.out file
        log_file = os.path.join(self.QMin.control["workdir"], "MNDO.out")

        #Gradients and NACs are taken from the fort.15 file, this file has many more significant digits 
        grads_nacs_file = os.path.join(self.QMin.control["workdir"], "fort.15")
        
        # Get contents of output file(s)
        states, interstates = self._get_states_interstates(log_file)


        # Populate energies no SOCs, so no diagonal elements
        if self.QMin.requests["h"]:
            file = open(log_file, "r")
            output = file.read()
            energies = self._get_energy(output)
            for i in range(len(energies)):
                self.QMout["h"][i][i] = energies[(1, i + 1)]

        # Populate dipole moments
        if self.QMin.requests["dm"]:
            self.QMout.dm = self._get_transition_dipoles(log_file)

        # Populate gradients, also for point charges
        if self.QMin.requests["grad"]:
            self.QMout.grad = self._get_grad(grads_nacs_file)
            if self.QMin.molecule["point_charges"]:
                self.QMout.grad_pc = self._get_grad_pc(grads_nacs_file)

        # Populate NACs, also for point charges
        if self.QMin.requests["nacdr"]:
            self.QMout.nacdr = self._get_nacs(grads_nacs_file, interstates)
            if self.QMin.molecule["point_charges"]:
                self.QMout.nacdr_pc = self._get_nacs_pc(grads_nacs_file, interstates)

        # Populate overlaps, only singlets so this function is simpler than normal
        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            if "overlap" not in self.QMout:
                self.QMout["overlap"] = np.zeros((nmstates, nmstates))
                
            outfile = os.path.join(self.QMin.resources["scratchdir"], "wfovl.out")
            ovlp_mat = self.parse_wfoverlap(outfile)
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                    if not m1 == m2 == 1: # only singlets
                        continue
                    if not ms1 == ms2:
                        continue
                    self.QMout["overlap"][i][j] = ovlp_mat[s1-1,s2-1]
        
        #Populate Phases if requested
        if self.QMin.requests["phases"]:
                for i in range(self.QMin.molecule["nmstates"]):
                    self.QMout["phases"][i] = -1 if self.QMout["overlap"][i][i] < 0 else 1
        
        return self.QMout


    def _get_states_interstates(self, log_path: str):
        f = readfile(log_path)
        states = []

        interstates = []

        for iline, line in enumerate(f):
            if "CI CALCULATION FOR STATE:" in line:  # find lines where calc of state begins and write out the number of the state
                line = f[iline]
                s = line.split()
                state = int(s[4]) - 1  # python counts from 0
                states.append(state)
            if (
                "CI CALCULATION FOR INTERSTATE COUPLING OF STATES:" in line
            ):  # find lines where interstate coupling of states begins and write out the numbers of the states
                line = f[iline]
                s = line.split()
                istate_a = int(s[7]) - 1
                istate_b = int(s[8]) - 1
                interstates.append((istate_a, istate_b))

        return states, interstates


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
                    grads[st, j, 0] =  float(s[-4])
                    grads[st, j, 1] =  float(s[-3])
                    grads[st, j, 2] =  float(s[-2])
                else:
                    grads[st, j, 0] =  float(s[-3])
                    grads[st, j, 1] =  float(s[-2])
                    grads[st, j, 2] =  float(s[-1])
                iline += 1

            grads[st, ...] = grads[st, ...] * kcal_to_Eh * BOHR_TO_ANG  # kcal/Ang --> H/a0

        return grads


    def _get_grad_pc(self, log_path: str) -> np.ndarray:
        """
        Extract gradients from MNDO outfile

        log_path:  Path to gradient file
        """
        nmstates = self.QMin.molecule["nmstates"]
        grad = [y - 1 for x,y in self.QMin["maps"]["gradmap"]] #Get the gradients that need to be calculated
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
                grads[st, j, 0] =  float(s[-3])  # Indexing from the back
                grads[st, j, 1] =  float(s[-2])
                grads[st, j, 2] =  float(s[-1])
                iline += 1
            grads[st, ...] = grads[st, ...] * kcal_to_Eh * BOHR_TO_ANG  # kcal/Ang --> H/a0

        return grads


    def _get_nacs(self, file_path: str, interstates):
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
            nac[s1,s2,...] = nac[s1,s2,...] * BOHR_TO_ANG # 1/Ang --> 1/a_0
            nac[s2,s1,...] = nac[s2,s1,...] * BOHR_TO_ANG
        
        return nac
    

    def _get_nacs_pc(self, file_path: str, interstates):
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

            nac[s1,s2,...] = nac[s1,s2,...] * BOHR_TO_ANG # 1/Ang --> 1/a_0
            nac[s2,s1,...] = nac[s2,s1,...] * BOHR_TO_ANG


        return nac


    def _get_transition_dipoles(self, log_path: str):
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
                    dmx = float(s[5]) * D2au
                    dmy = float(s[6]) * D2au
                    dmz = float(s[7]) * D2au
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
                dmx = float(s[5]) * D2au
                dmy = float(s[6]) * D2au
                dmz = float(s[7]) * D2au
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


    def _get_energy(self, output: str) -> dict[tuple[int, int], float]:
        """
        Extract energies from ORCA outfile

        output:     Content of outfile as string
        mult:       Multiplicities, not needed for now (deleted)
        """

        pattern = re.compile(r"eV,  E=[\s:]+([-+]\d*\.*\d+) eV")
        energies = {}
        for i, match in enumerate(pattern.finditer(output)):
            if i == self.QMin.molecule["nmstates"]:
                break
            energies[(1, i + 1)] = float(match.group(1)) * EV_TO_EH  # only singlets for now!!

        return energies


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
                resources_file = open('%s/MNDO.resources' % (workdir), 'w')
            except IOError:
                self.log.error('IOError during prepareMNDO, iconddir=%s' % (workdir))
                quit(1)
            string = 'scratchdir %s/\n' % self.setupINFOS['scratchdir']
            string += 'mndodir %s\n' % self.setupINFOS['mndodir']
            string += 'memory %i\n' % (self.setupINFOS['memory'])
            if 'overlap' in INFOS['needed_requests']:
                string += 'wfoverlap %s\n' % (self.setupINFOS['wfoverlap'])

            resources_file.write(string)
            resources_file.close()
            
        create_file = link if INFOS["link_files"] else shutil.copy
        # print(self.files)
        for file in self.files:
            create_file(expand_path(file), os.path.join(workdir, file.split("/")[-1]))



    def printQMout(self) -> None:
        super().writeQMout()


    def print_qmin(self) -> None:
        pass


    def read_resources(self, resources_file: str = "MNDO.resources") -> None:
        super().read_resources(resources_file)
        
        if not self.QMin.resources["mndodir"]:
            raise ValueError("mndodir has to be set in resource file!")

        self.QMin.resources["mndodir"] = expand_path(self.QMin.resources["mndodir"])
        self.log.debug(f'mndodir set to {self.QMin.resources["mndodir"]}')

    def read_template(self, template_file: str = "MNDO.template") -> None:
        super().read_template(template_file)

        self.QMin["template"]["nciref"] = int(self.QMin["template"]["nciref"])
        if self.QMin["template"]["nciref"] < 1 or self.QMin["template"]["nciref"] > 20:
            raise ValueError(f"number of references can only be between 1 and 20.")

        self.QMin["template"]["kharge"] = self.QMin.molecule['charge'][0] #int(self.QMin["template"]["kharge"]) #cast template inputs to int
        self.QMin["template"]["imomap"] = int(self.QMin["template"]["imomap"])
        # self.QMin["template"]["disp"] = int(self.QMin["template"]["disp"])
        
        if self.QMin["template"]["imomap"] < 0 or self.QMin["template"]["imomap"] > 1:  # Check if imomap is not out of range.
            raise ValueError(f"imomap can either be 0 (false) or 1 (true). Negative numbers not supported!")
        if self.QMin["template"]["imomap"] == 1:
            self.QMin["template"]["imomap"] = 3   #Orbital tracking activated when imomap=3 in the MNDO.inp file.
        
        if self.QMin["template"]["hamiltonian"] != None:
            if self.QMin["template"]["hamiltonian"].lower() == "om2":
                self.QMin["template"]["iop"] = -6
            elif self.QMin["template"]["hamiltonian"].lower() == "odm2":
                self.QMin["template"]["iop"] = -22
            else:
                raise ValueError(f"Hamiltonian can either be OM2 or ODM2 (with dispersion correction). Other hamiltonians are currently not supported!")
        else:
            raise ValueError(f"You have to set the hamiltonian keyword. Hamiltonian can either be OM2 or ODM2 (with dispersion correction). Other hamiltonians are currently not supported!")
                 

        
        # self.QMin["template"]["movo"] = int(self.QMin["template"]["movo"])
        # if self.QMin["template"]["movo"] > 1 or self.QMin["template"]["movo"] < 0 :
        #     raise ValueError(f"movo can only be 0 (false) or 1 (true).")
        
        if len(self.QMin["template"]["act_orbs"]) > 0 :
            self.QMin["template"]["act_orbs"] = [int(i) for i in self.QMin["template"]["act_orbs"]]
            self.QMin["template"]["movo"] = 1
        else:
            self.QMin["template"]["movo"] = 0

        
        self.QMin["template"]["fomo"] = int(self.QMin["template"]["fomo"])
        if self.QMin["template"]["fomo"] > 1 or self.QMin["template"]["fomo"] < 0 :
            raise ValueError(f"fomo can only be 0 (false) or 1 (true).")
        
        self.QMin["template"]["rohf"] = int(self.QMin["template"]["rohf"])
        if self.QMin["template"]["rohf"] > 1 or self.QMin["template"]["rohf"] < 0 :
            raise ValueError(f"rohf can only be 0 (false) or 1 (true).")
        
        self.QMin["template"]["levexc"] = int(self.QMin["template"]["levexc"])
        if self.QMin["template"]["levexc"] > 6 or self.QMin["template"]["levexc"] < 1 :
            raise ValueError(f"levexc can only be between 1 (singlets) and 6 (sextets).")
        
        self.QMin["template"]["mciref"] = int(self.QMin["template"]["mciref"])
        if self.QMin["template"]["mciref"] != 3 and self.QMin["template"]["mciref"] != 0:
            raise ValueError(f"mciref can only be between 0 (automatic definition) or 3 (mciref 0 plus 85% of something).")




    def remove_old_restart_files(self, retain: int = 5) -> None:
        """
        Garbage collection after runjobs()
        """


    def run(self) -> None:
        """
        request & other logic
            requestmaps anlegen -> DONE IN SETUP_INTERFACE
            pfade für verschiedene orbital restart files -> DONE IN SETUP_INTERFACE
        make schedule
        runjobs()
        run_wfoverlap (braucht input files)
        save directory handling
        """

        starttime = datetime.datetime.now()
        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "mndo_calc")

        schedule = [{"mndo_calc" : self.QMin}] #Generate fake schedule
        self.QMin.control["nslots_pool"].append(1)
        self.runjobs(schedule)
        

        self._save_files(self.QMin.control["workdir"])
        self.clean_savedir()
        # Run wfoverlap
        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            self._run_wfoverlap()

        self.log.debug("All jobs finished successfully")

        self.QMout["runtime"] = datetime.datetime.now() - starttime


    def _run_wfoverlap(self) -> None:
        """
        Prepare files and folders for wfoverlap and execute wfoverlap
        """

        # Content of wfoverlap input file
        # wf_input = dedent(
        #     """\
        # a_mo=mo.a
        # b_mo=mo.b
        # a_det=det.a
        # b_det=det.b
        # ao_read=-1
        # same_aos
        # """
        # )

        wf_input = dedent(
            """\
        a_mo=mo.a
        b_mo=mo.b
        a_det=det.a
        b_det=det.b
        ao_read=0
        """
        )
    
        if self.QMin.resources["ncpu"] >= 8:
            wf_input += "force_direct_dets"

        # cmdline string
        wf_cmd = f"{self.QMin.resources['wfoverlap']} -m {self.QMin.resources['memory']} -f wfovl.inp"

        # Overlap calculations
        workdir = self.QMin.resources["scratchdir"]

        # Write input
        writefile(os.path.join(workdir, "wfovl.inp"), wf_input)

        link(
            os.path.join(self.QMin.save["savedir"], f"ao.{self.QMin.save['step']}"),
            os.path.join(workdir, "S_mix"),
        )

        # Link files
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

        # Execute wfoverlap, maybe better using time.perf_counter() ??
        starttime = datetime.datetime.now()

        code = self.run_program(workdir, wf_cmd, os.path.join(workdir, "wfovl.out"), os.path.join(workdir, "wfovl.err"))
        self.log.info(f"Finished wfoverlap job!!\nruntime: {datetime.datetime.now()-starttime}")
        if code != 0:
            self.log.error("wfoverlap did not finish successfully!")
            with open(os.path.join(workdir, "wfovl.err"), "r", encoding="utf-8") as err_file:
                self.log.error(err_file.read())
            raise OSError()


    def _generate_inputstr(self) -> str:
        """
        Generate MNDO input file string from QMin object
        """
        qmin = self.QMin
        ncigrd = 1
        grads = [1]
        icross = 1                                              #calc gradients
        natom = qmin["molecule"]["natom"]
        if qmin.requests["grad"] or qmin.requests["nacdr"]:
            if qmin.requests["nacdr"]:
                icross = 7                                      #calc gradients and NACs
                grads = self._check_grads_request(qmin)
                ncigrd = len(grads)
            else:
                ncigrd = len(qmin["maps"]["gradmap"])
                grads = [y for _,y in qmin["maps"]["gradmap"]]


        coords = qmin["coords"]["coords"]
        elements = qmin["molecule"]["elements"]
        movo = qmin["template"]["movo"]
        ici1 = qmin["template"]["ici1"]
        ici2 = qmin["template"]["ici2"]
        nciref = qmin["template"]["nciref"]
        act_orbs = qmin["template"]["act_orbs"]
        iroot = qmin["molecule"]["states"][0]
        ncharges = qmin["molecule"]["npc"]
        kharge = qmin["template"]["kharge"]
        kitscf = qmin["template"]["kitscf"]
        imomap = qmin["template"]["imomap"]
        iop = qmin["template"]["iop"]
        rohf = qmin["template"]["rohf"]
        levexc = qmin["template"]["levexc"]
        mciref = qmin["template"]["mciref"]

        nfloat = ici1 + ici2


        

        if qmin["template"]["fomo"] == 1:
            inputstring = f"iop={iop} jop=-2 imult={rohf} iform=1 igeom=1 mprint=1 icuts=-1 icutg=-1 dstep=1e-05 kci=5 ioutci=1 iroot={iroot} icross={icross} ncigrd={ncigrd} inac=0 imomap={imomap} iscf=9 iplscf=9 kitscf={kitscf} nciref={nciref} mciref={mciref} levexc={levexc} mapthr=70 iuvcd=3 nsav13=2 kharge={kharge} multci=1 cilead=1 ncisym=-1 nsav15=9 iuhf=-6 nfloat={nfloat}"
        else:
            inputstring = f"iop={iop} jop=-2 imult={rohf} iform=1 igeom=1 mprint=1 icuts=-1 icutg=-1 dstep=1e-05 kci=5 ioutci=1 iroot={iroot} icross={icross} ncigrd={ncigrd} inac=0 imomap={imomap} iscf=9 iplscf=9 kitscf={kitscf} ici1={ici1} ici2={ici2} movo={movo} nciref={nciref} mciref={mciref} levexc={levexc} mapthr=70 iuvcd=3 nsav13=2 kharge={kharge} multci=1 cilead=1 ncisym=-1 nsav15=9"
        
        if rohf == 1:
            inputstring += " idiis=1"

        if qmin["molecule"]["point_charges"]:
            inputstring += f" numatm={ncharges} mmcoup=2 mmfile=1 mmskip=0 mminp=2"
            

        inputstring = " +\n".join(wrap(inputstring, width=70))
        inputstring += "\nheader\n"
        inputstring += "header\n"
        for i in range(natom):
            inputstring += f"{NUMBERS[elements[i]]:>3d}\t{coords[i][0]*BOHR_TO_ANG:>10,.5f} 1\t{coords[i][1]*BOHR_TO_ANG:>10,.5f} 1\t{coords[i][2]*BOHR_TO_ANG:>10,.5f} 1\n"
        inputstring += f"{0:>3d}\t{0:>10,.5f} 0\t{0:>10,.5f} 0\t{0:>10,.5f} 0\n"

        if movo == 1:
            for j in act_orbs:
                inputstring += str(j) + " "
            inputstring += "\n"

        for l in grads:
            inputstring += str(l) + " "

        return inputstring

    @staticmethod
    def _check_grads_request(qmin: QMin) -> list:
        
        grads = [y for _,y in qmin["maps"]["gradmap"]]
        nac_i = list(map(lambda x: x[1], qmin["maps"]["nacmap"]))
        nac_j = list(map(lambda x: x[3], qmin["maps"]["nacmap"]))
        naclist = nac_i + nac_j
        together = grads + naclist
        no_duplicates = list(dict.fromkeys(together))
        ordered = sorted(no_duplicates)
        
        return ordered


    
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
            self.log.error("MNDO can only calculate singlets!!")
            raise ValueError()

        




if __name__ == "__main__":
    SHARC_MNDO(loglevel=10).main()
