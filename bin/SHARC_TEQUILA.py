#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************
import numpy as np
import datetime
from typing import Optional
from io import TextIOWrapper
import os
import subprocess as sp
# import shutil
# from qmin import QMin
import re
import tequila as tq
from pyscf import gto
import openfermion
# from pathlib import Path
from textwrap import dedent
import time
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

try:
    import qulacs
except ImportError:
    raise RuntimeError(
        "The Qulacs backend is required for SHARC-Tequila calculations. "
        "Install it with: pip install qulacs"
    )

# internal
from SHARC_FAST import SHARC_FAST
from utils import question, mkdir, readfile, writefile, InDir, link
from constants import au2a, au2eV

date = datetime.date

# ======================================================================= #

__author__ = "Eduarda Sangiogo Gil"
__version__ = "4.0"
versiondate = datetime.datetime(2024, 10, 15)
changelogstring = """
"""

all_features = set(
    [
        "h",
        "grad",
        "dm",
        "overlap",
        "phases",
        "nacdr",
    ]
)

# ======================================================================= #

# Code that redirects Tequilas print() statements to our logger
class LoggerIO(StringIO):
    """Send prints to a specified logger method."""
    def __init__(self, log_method):
        super().__init__()
        self.log_method = log_method

    def write(self, message):
        message = message.strip()
        if message:
            self.log_method(message)

    def flush(self):
        pass  # no-op for logger




# ======================================================================= #

class SHARC_TEQUILA(SHARC_FAST):
    """
    Interface for the [TEQUILA program](https://github.com/tequilahub/tequila)

    ---
    Evaluates the electronic properties using a hybird quantum-classical approach.
    The ground state properties are calculated using the variational quantum eigensolver (VQE) algorithm,
    while the excited state properties are obtained according to the variational quantum deflation (VQD) algorithm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # intended for all of Tequilas print statements
        self.stdout_logger = LoggerIO(self.log.info)

        self.QMin.template.update(
            {
                "basis"          : "sto-3g",
                "transformation" : "jordan-wigner",
                "active_orbitals": None,
                "method_opt"     : "BFGS",
                # "ext_input"      : None,
                # "norm_gr_check"  : 1000,
                # "spin_op"        : None,
                "vqd_par"        : 1.000000,
                "hellman_feynman_grad"       : True,
                "shift_par"      : 0.001,
                "ansatz"         : "UpCCGSD",
                # "sort_states"    : False,
                "rms_par"        : 99999,
                # "read_ext_ang"   : True,
                # "edges"          : None,
            }
        )
        self.QMin.template.types.update(
            {
                "basis"          : str,
                "transformation" : str,
                "active_orbitals": list,
                "method_opt"     : str,
                # "ext_input"      : str,
                # "norm_gr_check"  : float,
                # "spin_op"        : list,
                "vqd_par"        : float,
                "hellman_feynman_grad"       : bool,
                "shift_par"      : float,
                "ansatz"         : str,
                # "sort_states"    : bool,
                "rms_par"        : float,
                # "read_ext_ang"   : bool,
                # "edges"          : list,
            }
        )

        self.QMin.resources.update(
            {
                "wfoverlap": "$SHARC/wfoverlap.x",
            }
        )

        self.QMin.resources.types.update(
            {
                "wfoverlap": str,
            }

        )

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    @staticmethod
    def version():
        return "4.0"

    @staticmethod
    def versiondate() -> date:
        return versiondate

    @staticmethod
    def changelogstring() -> str:
        return changelogstring

    @staticmethod
    def authors() -> str:
        return __author__

    @staticmethod
    def name() -> str:
        return "TEQUILA"

    @staticmethod
    def description() -> str:
        return "Interface for the TEQUILA program for VQE calculation"

    def create_restart_files(self):
        pass

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

# ======================================================================= #

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
        self.log.info(f"||{'Tequila interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        self.files = []

        # scratch
        self.log.info(f"{'Scratch directory':-^60}\n")
        self.log.info(
            "Please specify an appropriate scratch directory. This will be used to run the calculations. " \
            "The scratch directory will be deleted after the calculation. " \
            "Remember that this script cannot check whether the path is valid, " \
            "since you may run the calculations on a different machine. " \
            "The path will not be expanded by this script."
        )
        INFOS["scratchdir"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
        self.log.info("")

        self.template_file = None
        self.log.info(f"{'TEQUILA input template file':-^60}\n")

        if os.path.isfile("TEQUILA.template"):
            usethisone = question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True)
            if usethisone:
                self.template_file = "TEQUILA.template"
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
        if question("Do you have a 'TEQUILA.resources' file?", bool, KEYSTROKES=KEYSTROKES, default=False):
            while True:
                resources_file = question("Specify the path:", str, KEYSTROKES=KEYSTROKES, default="TEQUILA.resources")
                self.files.append(resources_file)
                self.make_resources = False
                if os.path.isfile(resources_file):
                    break
                else:
                    self.log.info(f"file at {resources_file} does not exist!")
        else:
            self.make_resources = True
            self.log.info(f"{'TEQUILA` Ressource usage':-^60}\n")

            INFOS["memory"] = question("Memory (MB):", int, default=[1000], KEYSTROKES=KEYSTROKES)[0]

            
            if "overlap" in INFOS["needed_requests"]:
                self.log.info(f"\n{'WFoverlap setup':-^60}\n")
                INFOS["wfoverlap"] = question(
                    "Path to wavefunction overlap executable:", str, default="$SHARC/wfoverlap.x", KEYSTROKES=KEYSTROKES
                )

        return INFOS
    
# ======================================================================= #

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
                resources_file = open('%s/TEQUILA.resources' % (workdir), 'w')
            except IOError:
                self.log.error('IOError during prepare TEQUILA, iconddir=%s' % (workdir))
                quit(1)
            string = 'scratchdir %s/\n' % INFOS['scratchdir']
            string += 'memory %i\n' % (INFOS['memory'])
            if 'overlap' in INFOS['needed_requests']:
                string += 'wfoverlap %s\n' % (INFOS['wfoverlap'])

            resources_file.write(string)
            resources_file.close()
            
        INFOS["link_files"] = False

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def reset_timer(self, label="Timer reset"):
        """Reset the internal timer."""
        self._start_time = time.time()
        if label:
            formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._start_time))
            self.log.info(f"{label}: Timer reset at {formatted}")

    def log_elapsed(self, label="Elapsed time"):
        """Log the elapsed time since last reset."""
        if self._start_time is None:
            self.log.warning("Timer has not been reset.")
            return
        elapsed = time.time() - self._start_time
        self.log.info(f"{label}: {elapsed:.2f} seconds")


# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def calc_energies(self):

        # Tequila runs some PySCF calculations, so we run them in the scratch dir
        with InDir(self.QMin.control["workdir"]):

            # fetch infos from self.QMin
            qmin = self.QMin
            nstate = qmin.molecule["nmstates"]
            step = qmin.save["step"]
            basis = qmin["template"]["basis"]
            qubit_space_trans = qmin["template"]["transformation"]
            optimizer = qmin["template"]["method_opt"]
            ansatz = qmin["template"]["ansatz"]
            vqd_par = qmin["template"]["vqd_par"]
            # edges = qmin["template"]["edges"]
            input_active_orbitals = qmin["template"]["active_orbitals"]
            savedir = qmin.save["savedir"]

            # active space handling
            if input_active_orbitals is None:
                self.log.error("No active orbitals provided! Use 'active_orbitals' in the template file!")
                raise ValueError()
            act_orb = [int(i) for i in input_active_orbitals]

            # setting up the molecule
            elements = self.QMin.molecule["elements"]  
            coords   = self.QMin.coords["coords"]  
            string = ""
            for atom, (x, y, z) in zip(elements, coords):
                # Tequila uses Angstrom for geometry
                string += f"\n{atom:2s} {x*au2a:15.8f} {y*au2a:15.8f} {z*au2a:15.8f}"
            mol = tq.chemistry.Molecule(geometry=string, basis_set=basis, transformation=qubit_space_trans, active_orbitals=act_orb, backend="pyscf")

            # prepare Hamiltonian and Pauli matrices
            with redirect_stdout(self.stdout_logger), redirect_stderr(self.stdout_logger):
                self.log.info("Make Hamiltonian ...")
                H = mol.make_hamiltonian()
            self.log.info("Make Qubits ...")
            P0 = tq.paulis.Qp(qubit=H.qubits)

            # prepare lists with results
            energies = []
            results = []
            wfns = []
            Us = []

            # loop over the states ---------------
            for istate in range(nstate):

                # handling the ansatz - quantum circuit U
                # if ansatz == "SPA":
                #     if edges == None:
                #         self.log.error("The edges must be provided in the template file.")
                #         raise ValueError("The edges must be provided in the template file.")
                #     else:
                #         U = mol.make_ansatz(name=ansatz, edges=edges, label=istate)
                # else: 
                U = mol.make_ansatz(name=ansatz, label=istate)
                if istate == 1:  # add an extra gate to enforce orthogonality to state i=0
                    U += mol.UR(1, 2, angle=(tq.Variable("a")+0.5)*np.pi)

                # get the variables to optimize
                E = tq.ExpectationValue(U, H)
                active_vars = E.extract_variables()

                # handle initial angles
                path = None
                if step == 0:
                    path = os.path.join(qmin.resources["pwd"], f"angles_json.{istate}.init")
                else:
                    path = os.path.join(savedir, f"angles_json.{istate}.{step-1}")
                if not os.path.isfile(path):
                    path = None
                if path:
                    with open(path) as f:
                        data = json.load(f)
                    angles = {}
                    varmap = {str(v): v for v in active_vars}
                    for k, v in data.items():
                        if k in varmap:
                            angles[varmap[k]] = v
                else:
                    angles = {angle: 0.0 for angle in active_vars}

                # incorporate the results from the previous states
                for data, U2 in results:
                    S2 = tq.ExpectationValue(H=P0, U=U + U2.dagger())
                    E -= ((data.energy*vqd_par) * S2) 
                    angles = {**angles, **data.angles}

                # run the computation for that state
                self.log.info(f"Minimize state {istate} ...")
                result = tq.minimize(E, silent=False, method=optimizer, variables=active_vars, initial_values=angles)
                self.log.info(f"Simulate state {istate} ...")
                wfn = tq.simulate(U, variables=result.variables)

                # add all relevant results to the lists
                results.append((result, U))
                energies.append(float(result.energy))
                Us.append(U)
                wfns.append(wfn)

                filename = os.path.join(savedir, f"angles_json.{istate}.{step}")
                angles_dict = {str(k): float(v) for k, v in result.variables.items()}
                with open(filename, 'w') as outangle:
                    json.dump(angles_dict, outangle, indent=2)
            
        return energies, wfns, Us, mol, result.variables

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def get_MO_coefficients(self, mol):
        C = mol.integral_manager.orbital_coefficients
        return C

# ======================================================================= #

    def get_AO_double_mol_overlap(self, basis):
        elements = self.QMin.molecule["elements"]  
        coords   = self.QMin.coords["coords"]  
        new_geom = ""
        for atom, (x, y, z) in zip(elements, coords):
            # PySCF uses Angstrom for input
            new_geom += f"\n{atom:2s} {x*au2a:15.8f} {y*au2a:15.8f} {z*au2a:15.8f}"

        step = step = self.QMin.save["step"] - 1
        filename = os.path.join(self.QMin.save["savedir"], f"geom.xyz.{step}")
        elements, coords = self.read_geom_xyz(filename)
        old_geom = ""
        for atom, (x, y, z) in zip(elements, coords):
            # PySCF uses Angstrom for input
            old_geom += f"\n{atom:2s} {x*au2a:15.8f} {y*au2a:15.8f} {z*au2a:15.8f}"

        mol2 = gto.M(atom=old_geom, basis = basis)
        mol1 = gto.M(atom=new_geom, basis = basis)
        S = gto.intor_cross('int1e_ovlp', mol2, mol1) # first index: old geometry
        return S

# ======================================================================= #

    def read_geom_xyz(self, filename: str) -> tuple[list[str], np.ndarray]:
        """Read an XYZ file into atoms (labels) and coords (Nx3 array)."""
        lines = readfile(filename)
        try:
            natoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"First line of {filename} must be atom count")
        if len(lines) < natoms + 2:
            raise ValueError(f"File {filename} does not contain {natoms} atoms")
        atoms = []
        coords = []
        for line in lines[2:2 + natoms]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ line: {line.strip()}")
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return atoms, np.array(coords, dtype=float)

    def write_geom_xyz(self, atoms: list[str], coords: list[list[float]] | np.ndarray, filename: str):
        """Write atoms and coords to an XYZ file."""
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        natoms = len(atoms)
        lines = [f"{natoms}\n", "\n"]
        for atom, (x, y, z) in zip(atoms, coords):
            lines.append(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
        writefile(filename, lines)

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def one_body_op(self, integrals, mol):
        op = 0.0
        for i in range(integrals.shape[0]):
            for j in range(integrals.shape[1]):
                op += integrals[i,j]*mol.make_creation_op(2*i)*mol.make_annihilation_op(2*j) # spin up
                op += integrals[i,j]*mol.make_creation_op(2*i+1)*mol.make_annihilation_op(2*j+1) # spin down 
        return op
    
# ======================================================================= #
    
    def calc_dipole_moment(self, C, Us, variables, mol):
        with InDir(self.QMin.control["workdir"]):
            # fetch infos
            nstat = self.QMin.molecule["nmstates"]
            basis = self.QMin["template"]["basis"]
            input_active_orbitals = self.QMin["template"]["active_orbitals"]
            act_orb = [int(i) for i in input_active_orbitals]
            
            # build molecule for PySCF
            elements = self.QMin.molecule["elements"]  
            coords   = self.QMin.coords["coords"]  
            geom = ""
            for atom, (x, y, z) in zip(elements, coords):
                # PySCF uses Angstrom for input
                geom += f"\n{atom:2s} {x*au2a:15.8f} {y*au2a:15.8f} {z*au2a:15.8f}"
            molecule = gto.M(atom=geom, basis=basis)
            molecule.build()

            # compute dipole integrals
            dipole_integrals_ao = molecule.intor('int1e_r', comp=3)

            # Transform dipole integrals from AO to MO basis
            dipole_integrals_mo = np.einsum('pi,kpq,qj->kij', C, dipole_integrals_ao, C)

            # extract active-orbital submatrices
            dipole_submatrices = [ dipole_integrals_mo[k][np.ix_(act_orb, act_orb)] for k in range(3) ]
            dip_ops = [ self.one_body_op(dipole_submatrices[k], mol) for k in range(3) ]

            # initiate full dipole
            dip = np.zeros((3, nstat, nstat))

            # compute diagonal dipoles
            for i in range(nstat):
                for k in range (3):
                    self.log.info(f"- Bra {i} Ket {i} direction {k} ")
                    dip_temp = tq.ExpectationValue(U=Us[i], H=dip_ops[k])
                    dip[k, i, i] = tq.simulate(dip_temp, variables=variables)

            # compute transition dipoles
            for i in range(nstat):
                for j in range(i+1, nstat):
                    for k in range(3):
                        self.log.info(f"- Bra {i} Ket {j} direction {k} ")
                        dip_temp, _ = tq.BraKet(bra=Us[i], ket=Us[j], H=dip_ops[k])
                        dip[k, i, j] = tq.simulate(dip_temp, variables=variables)
                        dip[k, j, i] = dip[k, i, j]  # enforce symmetry

        return dip

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def compute_grad(self, mol, variables, Us, gradients_indices, nac_indices, energies):
        with InDir(self.QMin.control["workdir"]):
            # fetch infos
            nstat = self.QMin.molecule["nmstates"]
            natom = self.QMin.molecule["natom"]
            basis = self.QMin["template"]["basis"]
            qubit_trans = self.QMin["template"]["transformation"]
            shift_par = self.QMin["template"]["shift_par"]
            grad_hf = self.QMin["template"]["hellman_feynman_grad"]
            # sort_states = self.QMin["template"]["sort_states"]
            input_active_orbitals = self.QMin["template"]["active_orbitals"]
            act_orb = [int(i) for i in input_active_orbitals]


            # prepare gradient array
            tot_grad = np.zeros((nstat,natom,3))
            tot_nac = np.zeros((nstat,nstat,natom,3))

            # big "if using Hellman-Feynman forces"
            if grad_hf:
                # using Hellman-Feynman forces

                # fetch original geometry
                elements = self.QMin.molecule["elements"]  
                coords   = self.QMin.coords["coords"]

                # loop over displacements
                for iatom in range(len(coords)):
                    for idir in range(3):
                        self.log.info(f"- Displacement of atom {iatom} and direction {idir}")

                        # positive displacement
                        coords_plus = coords.copy()
                        coords_plus[iatom,idir] += shift_par
                        geom = ""
                        for atom, (x, y, z) in zip(elements, coords_plus):
                            geom += f"\n{atom:2s} {x*au2a:15.8f} {y*au2a:15.8f} {z*au2a:15.8f}"
                        mol_plus = tq.chemistry.Molecule(geometry=geom, basis_set=basis, transformation=qubit_trans, active_orbitals=act_orb)
                        with redirect_stdout(self.stdout_logger), redirect_stderr(self.stdout_logger):
                            H_mol_plus = mol_plus.make_molecule()
                            H_mol_plus2 = H_mol_plus.get_molecular_hamiltonian()

                        # negative displacement
                        coords_minus = coords.copy()
                        coords_minus[iatom,idir] -= shift_par
                        geom = ""
                        for atom, (x, y, z) in zip(elements, coords_minus):
                            geom += f"\n{atom:2s} {x*au2a:15.8f} {y*au2a:15.8f} {z*au2a:15.8f}"
                        mol_minus = tq.chemistry.Molecule(geometry=geom, basis_set=basis, transformation=qubit_trans, active_orbitals=act_orb)
                        with redirect_stdout(self.stdout_logger), redirect_stderr(self.stdout_logger):
                            H_mol_minus = mol_minus.make_molecule()
                            H_mol_minus2 = H_mol_minus.get_molecular_hamiltonian()

                        # define the difference Hamiltonian and compute expectation value
                        dH_dR = (H_mol_plus2 - H_mol_minus2)/(2.*shift_par/au2a)
                        dH_dR = openfermion.transforms.get_fermion_operator(dH_dR)
                        dH_dR_q = mol.transformation(dH_dR)

                        # compute gradients
                        for k in gradients_indices:
                            dHF_dR = tq.ExpectationValue(H=dH_dR_q, U=Us[k])
                            dHF_dR = tq.simulate(dHF_dR, variables)
                            tot_grad[k,iatom,idir] = dHF_dR

                        # compute NACs
                        for (k,l) in nac_indices:
                            dHF_dR, _ = tq.BraKet(H=dH_dR_q, bra=Us[k], ket=Us[l])
                            dHF_dR = tq.simulate(dHF_dR, variables)
                            # TODO: divide by dE!
                            tot_nac[k,l,iatom,idir] = -dHF_dR.real/(energies[k]-energies[l])
                            tot_nac[l,k,iatom,idir] = -dHF_dR.real/(energies[l]-energies[k])
            else:
                raise NotImplementedError("No numerical gradients, use SHARC_NUMDIFF.py")
            # else:
            #     # do not use Hellman-Feynman forces
            #     geometry = []
            #     label = []
            #     self.read_geom(f'{workdir}/geom.xyz', geometry, label)
            #     geometry = np.array(geometry)
            #     geom_plus = []
            #     geom_minus = []
            #     for xyz in range(3):
            #         for A in range(natom):
            #             geom_plus = geometry.copy()
            #             geom_plus[A,xyz] += shift_par
            #             geom_str_plus = self.geom_string(label, geom_plus)
            #             en_plus, wfn_plus, Us_plus, mol_plus, variables_plus = self.calc_energies(geom_str_plus)
            #             if sort_states:
            #                 en_plus.sort()
    
            #             geom_minus = geometry.copy()
            #             geom_minus[A,xyz] -= shift_par
            #             geom_str_minus = self.geom_string(label, geom_minus)
            #             en_minus, wfn_minus, Us_minus, mol_minus, variables_minus = self.calc_energies(geom_str_minus)
            #             if sort_states:
            #                 en_minus.sort()

            #             for i in range(nstat):
            #                 tot_grad[i,A,xyz] = (en_plus[i] - en_minus[i])/(2*shift_par_au)

            #             en_plus.clear()
            #             en_minus.clear()

            return tot_grad, tot_nac
        
# ======================================================================= #

    def map_tequila_to_sharc(self, det_bin: list[str], first_act: int, last_act: int, n_mo: int) -> dict[str, str]:
        """
        Build a dictionary that maps Tequila determinant bitstrings to SHARC '2ab0' style strings.

        Args:
            det_bin: list of Tequila bitstrings (e.g., '1100', '1010', ...), one per determinant
            first_act: index of first active orbital
            last_act: index of last active orbital
            n_mo: total number of orbitals (space orbitals)

        Returns:
            mapping: dict with keys = Tequila bitstrings, values = '2ab0...' strings
        """
        mapping = {}
        n_spin = len(det_bin[0])
        for bitstring in det_bin:
            det_list = []

            # core orbitals = doubly occupied
            det_list.extend(['d'] * first_act)

            # active orbitals
            for i in range(n_spin//2):
                alpha = bitstring[2*i]   # even index = α-spin
                beta = bitstring[2*i+1]  # odd index  = β-spin

                if alpha == '1' and beta == '1':
                    det_list.append('d')
                elif alpha == '1' and beta == '0':
                    det_list.append('a')
                elif alpha == '0' and beta == '1':
                    det_list.append('b')
                else:
                    det_list.append('e')

            # virtual orbitals = empty
            det_list.extend(['e'] * (n_mo - last_act - 1))
            mapping[bitstring] = ''.join(det_list)
        return mapping

# ======================================================================= #

    def wfn_to_dict(self,w) -> dict[str, complex]:
        """
        Convert a QubitWaveFunction to {binary_string: amplitude} dictionary.
        Works for both dense and sparse wavefunctions.
        """
        n_qubits = w.n_qubits
        # use raw_items to get integer index and amplitude
        return {
            format(idx, f'0{n_qubits}b'): amp.real
            for idx, amp in w.raw_items()
            if abs(amp) > 1e-14
        }

# ======================================================================= #

    def save_wfn_infos(self, C, Us, wfn, variables):

        # fetch infos
        nstat = self.QMin.molecule["nstates"]
        input_active_orbitals = self.QMin["template"]["active_orbitals"]
        step = self.QMin.save["step"]
        savedir = self.QMin.save["savedir"]
        act_orb = [int(i) for i in input_active_orbitals]


        # Write the file with the MO coefficients
        CT = C.T
        n_mo, n_ao = C.shape
        mo_string = f"2mocoef\nheader\n1\nMO-coefficients from Tequila\n1\n{n_ao}   {n_mo}\na\nmocoef\n(*)\n"
        for mat in CT:
            for idx, i in enumerate(mat):
                if idx > 0 and idx % 3 == 0:
                    mo_string += "\n"
                mo_string += f"{i: 6.12e} "
            if n_ao - 1 % 3 != 0:
                mo_string += "\n"
        mo_string += "orbocc\n(*)"
        for i in range(n_ao):
            if i % 3 == 0:
                mo_string += "\n"
            mo_string += f"{0.0: 6.12e} "
        filename = os.path.join(savedir,f"mos.{step}")
        writefile(filename, mo_string)
        

        # get CI coefficients as dictionary
        ci_coefs_dict = {}
        for i, w in enumerate(wfn):
            ci_coefs_dict[i] = self.wfn_to_dict(w)

        # get list of determinants for the file
        all_det_set = set()
        for state_dict in ci_coefs_dict.values():
            all_det_set.update(state_dict.keys())
        all_det_list = sorted(all_det_set)
        n_det = len(all_det_list)

        # get SHARC-style determinant strings
        last_orbact = act_orb[-1]
        first_orbact = act_orb[0]
        mapping = self.map_tequila_to_sharc(all_det_list, first_orbact, last_orbact, n_mo)

        # make the determinant file string
        strings = [ f'{nstat} {n_mo} {n_det}' ]
        for tq_string in all_det_list:
            string = [ mapping[tq_string] ] + [ f"{ci_coefs_dict[i].get(tq_string, 0.0): .12f}" for i in range(nstat) ]
            strings.append(' '.join(string))
        final_string = '\n'.join(strings)

        # write to file
        filename = os.path.join(savedir, f"dets.{step}")
        writefile(filename, final_string)

# ======================================================================= #

    def run_program(self, workdir: str, cmd: str, out: str, err: str, env: dict | None = None) -> int:
        """
        Runs a ab-initio programm and returns the exit_code

        workdir:    Path of the working directory
        cmd:        Contains path and arguments for execution of ab-initio program
        out:        Name of the output file
        err:        Name of the error file (optional)
        env:        Pass environment variables
        """
        current_dir = os.getcwd()
        os.sched_setaffinity(0, list(range(os.cpu_count())))
        os.chdir(workdir)
        self.log.debug(f"ab-initio call:\t {cmd}")
        self.log.debug(f"Working directory:\t {workdir}")

        with open(out, "w", encoding="utf-8") as outfile, open(err, "w", encoding="utf-8") as errfile:
            try:
                exit_code = sp.call(cmd, shell=True, stdout=outfile, stderr=errfile, env=env)
            except OSError as error:
                self.log.error(f"Execution of {cmd} failed!")
                raise OSError from error

        os.chdir(current_dir)
        return exit_code

# ======================================================================= #

    def run(self):
        self.log.info("Entering run() ...")

        ###### preparation stuff ###### 

        # short cuts
        QMin = self.QMin
        QMout = self.QMout

        # workdir
        self.QMin.control["workdir"] = os.path.join(self.QMin.resources["scratchdir"], "tequila")
        mkdir(self.QMin.control["workdir"])

        savedir = QMin.save["savedir"]

        # fetch inputs
        nmstates = QMin.molecule["nmstates"]
        step = QMin.save["step"]
        basis = QMin["template"]["basis"]
        rms_par = QMin["template"]["rms_par"]

        # allocate the QMout object
        QMout.allocate(states=QMin.molecule["states"], natom=QMin.molecule["natom"], npc=0, requests=("h","grad","dm","overlap","phases"))
 
        # save geometry in savedir for overlaps
        filename = os.path.join(savedir, f"geom.xyz.{step}")
        self.write_geom_xyz(self.QMin.molecule["elements"], self.QMin.coords["coords"], filename)


        ###### calculate the energies ###### 

        # Start: calculate the energies
        self.reset_timer("Calculating the energies")
        energies, wfn, Us, mol, variables = self.calc_energies()
        self.log_elapsed("Calculating the energies DONE")

        # reorder states by energy to have adiabatic states
        sorted_data = sorted(zip(energies, wfn, Us), key=lambda x: x[0])
        energies, wfn, Us = map(list, zip(*sorted_data))

        # populate the QMout Hamiltonian
        for i,en in enumerate(energies):
            QMout["h"][i,i] = en
    
        # obtain the MO coefficients
        C = self.get_MO_coefficients(mol)


        ###### calculate the dipole moments ###### 

        if QMin.requests["dm"]:
            self.reset_timer("Calculating the dipole moments")
            dip = self.calc_dipole_moment(C, Us, variables, mol)
            QMout["dm"] = dip
            self.log_elapsed("Calculating the dipole moments DONE")


        ###### calculate the gradients ###### 
        
        if QMin.requests["grad"] or QMin.requests["nacdr"]:

            # get which gradients to compute
            gradmap = self.QMin.maps["gradmap"]
            gradients_indices = []
            if gradmap:
                for mult, state in gradmap:
                    if mult != 1:
                        self.log.error("Can only do singlets!")
                        raise ValueError("Can only do singlets!")
                    gradients_indices.append(state-1)

            # which NACs to compute
            nacmap = self.QMin.maps["nacmap"]
            nac_indices = []
            if nacmap:
                for m1, s1, m2, s2 in nacmap:
                    if m1 != 1 or m2 != 1:
                        self.log.error("Can only do singlets!")
                        raise ValueError("Can only do singlets in NAC!")
                    if s1 < s2:
                        nac_indices.append( (s1-1,s2-1) )
                    else:
                        nac_indices.append( (s2-1,s1-1) )

            # compute the gradient
            self.reset_timer("Calculating the gradients and NACs")
            tot_grad, tot_nac = self.compute_grad(mol, variables, Us, gradients_indices, nac_indices, energies)
            QMout["grad"] = tot_grad
            QMout["nacdr"] = tot_nac
            self.log_elapsed("Calculating the gradients and NACs DONE")
    
            # store the gradient for later checking
            filename = os.path.join(savedir, f"gradients_all.npy.{step}")
            with open(filename, "wb") as f:
                np.save(f, tot_grad)

            # retrieve the previous gradient for comparison
            if step > 0:
                filename = os.path.join(savedir, f"gradients_all.npy.{step-1}")
                prev_tot_grad = np.load(filename)

            # to stick with previous code, compute the RMS and compare
            # instead of using current_state, we use the states from the gradmap
            rms = np.sqrt(np.mean(tot_grad[gradients_indices, :, :]**2)) * au2eV / au2a
            if step > 0:
                rms_prev = np.sqrt(np.mean(prev_tot_grad[gradients_indices, :, :]**2)) * au2eV / au2a
                rms_diff = np.abs(rms - rms_prev)
                if rms_diff > rms_par:
                    self.log.error("The gradients differ too strongly from the previous ones! Aborting!")
                    raise ValueError("The gradients differ too strongly from the previous ones! Aborting!")


        ###### calculate the overlaps ###### 

        # store wave function information
        self.reset_timer("Writing wave function information")
        if not self.QMin.save["samestep"]:
            self.save_wfn_infos(C, Us, wfn, variables)
        self.log_elapsed("Writing wave function information DONE")

        # actual overlap computation
        if self.QMin.requests["overlap"] and step > 0:
            self.reset_timer("Calculating the wave function overlaps")

            # get and write AO overlap to savedir
            S = self.get_AO_double_mol_overlap(basis)
            filename = os.path.join(savedir, f"Sao.{step-1}.{step}")
            np.savetxt(filename, S, fmt="%0.12f", header=f"{S.shape[0]} {S.shape[1]}", comments="")

            # make workdir and link files
            workdir = os.path.join(self.QMin.resources["scratchdir"], f"WFOVL")
            mkdir(workdir)
            link(os.path.join(savedir, f"Sao.{step-1}.{step}"), os.path.join(workdir, "aoovl"))
            link(os.path.join(savedir, f"dets.{step-1}"), os.path.join(workdir, "det.a"))
            link(os.path.join(savedir, f"dets.{step}"), os.path.join(workdir, "det.b"))
            link(os.path.join(savedir, f"mos.{step-1}"), os.path.join(workdir, "mo.a"))
            link(os.path.join(savedir, f"mos.{step}"), os.path.join(workdir, "mo.b"))

            # write input file
            wf_input = dedent(f"""\
            mix_aoovl=aoovl
            a_mo=mo.a
            b_mo=mo.b
            a_det=det.a
            b_det=det.b
            a_mo_read=0
            b_mo_read=0
            ao_read=0 """)
            if self.QMin.resources["ncpu"] >= 8:
                wf_input += "\nforce_direct_dets"
            filename = os.path.join(workdir, "wfovl.inp")
            writefile(filename, wf_input)

            # run wfoverlap.x
            starttime = datetime.datetime.now()
            os.environ["OMP_NUM_THREADS"] = str(self.QMin.resources["ncpu"])
            wf_cmd = f"{self.QMin.resources['wfoverlap']} -m {self.QMin.resources['memory']} -f wfovl.inp"
            code = self.run_program(workdir, wf_cmd, "wfovl.out", "wfovl.err")
            self.log.info(f"Finished wfoverlap job: code {code:<4d} runtime: {datetime.datetime.now()-starttime}")
            if code != 0:
                self.log.error("wfoverlap did not finish successfully!")
                with open(os.path.join(workdir, "wfovl.err"), "r", encoding="utf-8") as err_file:
                    self.log.error(err_file.read())
                raise OSError()

            # read the wfoverlap output
            filename = os.path.join(workdir, "wfovl.out")
            overlap = self.parse_wfoverlap(filename)
            self.log_elapsed("Calculating the wave function overlaps DONE")

            QMout["overlap"] = overlap
            if QMin.requests["phases"]:
                for i in range(nmstates):
                    QMout["phases"][i] = -1 if QMout["overlap"][i, i] < 0 else 1

    
# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def getQMout(self):

       self.QMout.states = self.QMin.molecule["states"]
       self.QMout.nstates = self.QMin.molecule["nstates"]
       self.QMout.nmstates = self.QMin.molecule["nmstates"]
       self.QMout.natom = self.QMin.molecule["natom"]
       self.QMout.npc = self.QMin.molecule["npc"]
       self.QMout.point_charges = self.QMin.molecule["npc"] > 0
       self.QMout["runtime"] = self.clock.measuretime(False)
       return self.QMout

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def parse_wfoverlap(self, overlap_file: str) -> np.ndarray:
        """
        Parse overlap matrix from wfoverlap output

        overlap_file: path to wfovlp.out
        """
        with open(overlap_file, "r", encoding="utf-8") as wffile:
            wf_out = wffile.read()
            dim = re.search(r"Number of <bra\| states:\s+(\d+)", wf_out)
            if not dim:
                raise ValueError("No states found in overlap file.")
            ovlp_values = re.findall(r"Overlap matrix(.*?)Ren", wf_out, re.DOTALL)
            ovlp_values = re.findall(r"-?\d+\.\d{10}", ovlp_values[0])
        return np.asarray(ovlp_values, dtype=float).reshape(int(dim.group(1)), -1)
    
# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def read_template(self, template_file: str = "TEQUILA.template") -> None:
        super().read_template(template_file)

# ======================================================================= #

    def read_resources(self, resources_filename="TEQUILA.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning(f"{resources_filename} not found! Continueing without further settings.")
            self._read_resources = True
            return

        super().read_resources(resources_filename)

        self.log.debug(f"{self.QMin.resources}")
        self._read_resources = True

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def setup_interface(self):
        """
        Setup remaining maps (ionmap, gsmap) and build jobs dict
        """
        super().setup_interface()
    
# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

    def _request_logic(self):
        """
        Create maps from QMin object
        """
        self.log.debug("Setup interface -> building maps")
        super()._request_logic()
        # Setup gradmap
        if self.QMin.requests["grad"]:
            self.log.debug("Building gradmap")
            self.QMin.maps["gradmap"] = set({tuple(self.QMin.maps["statemap"][i][0:2]) for i in self.QMin.requests["grad"]})
        if self.QMin.requests["nacdr"]:
            if self.QMin.requests["nacdr"] == ["all"]:
                mat = [
                    (i + 1, j + 1) for i in range(self.QMin.molecule["nmstates"]) for j in range(self.QMin.molecule["nmstates"])
                ]
                # self.QMin.requests["nacdr"] = mat
            else:
                mat = self.QMin.requests["nacdr"]
            self.log.debug("Building nacmap")
            self.QMin.maps["nacmap"] = set()
            for i in mat:
                m1, s1, ms1 = self.QMin.maps["statemap"][int(i[0])]
                m2, s2, ms2 = self.QMin.maps["statemap"][int(i[1])]
                if m1 != m2 or i[0] == i[1] or ms1 != ms2 or s1 > s2:
                    continue
                self.QMin.maps["nacmap"].add(tuple([m1, s1, m2, s2]))

# ======================================================================= #
# ======================================================================= #
# ======================================================================= #

if __name__ == "__main__":
    SHARC_TEQUILA().main()

