#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
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

# This script calculates QC results for a model system
#
# Reads QM.in
# Calculates SOC matrix, dipole moments, gradients, nacs and overlaps
# Writes these back to QM.out

# IMPORTS
# external
import numpy as np
import os
import sys
import re
import datetime
import sympy

# internal
from SHARC_FAST import SHARC_FAST
from utils import readfile, question, phase_correction
from io import TextIOWrapper

authors = "Brigitta Bachmair"
version = "4.0"
versiondate = datetime.datetime(2024, 4, 11)

changelogstring = """
"""
np.set_printoptions(linewidth=400, formatter={"float": lambda x: f"{x: 9.7}"}, threshold=sys.maxsize)

class SHARC_ANALYTICAL(SHARC_FAST):
    _read_resources = True
    _diagonalize = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update({"diagonalize": True, "keep_U": False})
        self.QMin.resources.types.update(
            {
                "diagonalize": bool,
                "keep_U": bool,
            }
        )

    @staticmethod
    def name():
        return "ANALYTICAL"

    @staticmethod
    def version():
        return version

    @staticmethod
    def versiondate():
        return versiondate

    @staticmethod
    def changelogstring():
        return changelogstring

    @staticmethod
    def authors():
        return authors

    @staticmethod
    def about():
        return "Interface for calculations with analytical PESs"

    @staticmethod
    def description():
        return "     FAST interface for analytical model Hamiltonians with sympy"

    @staticmethod
    def find_lines(nlines, match, strings):
        smatch = match.lower().split()
        nlen = len(strings)
        iline = -1
        while True:
            iline += 1
            if iline == nlen:
                return []
            line = strings[iline].lower().split()
            if line == smatch:
                return [s.rstrip('\n \t\r;').split(';') for s in strings[iline + 1:iline +1 + nlines]]
    
    def check_dimensions(self, mat, nmstates):
        for i, elements in enumerate(mat):
            if (len(elements) < i+1) or (len(elements) > nmstates):
                self.log.error('Dimensions of defined matrix in template file not correct.')
                raise ValueError('Dimensions of defined matrix in template file not correct.')

    def read_template(self, template_filename="ANALYTICAL.template"):
        lines = readfile(os.path.abspath(template_filename))

        natom = self.QMin.molecule["natom"]
        self.template_natom = int(lines[0])
        if not natom == self.template_natom:
            self.log.error(
                f'Natom from QM.in and from {template_filename} are inconsistent! {natom} != {self.template_natom}'
            )
            raise ValueError(f'impossible to calculate {natom} atoms with template holding {self.template_natom} atoms')

        self.parsed_states = self.parseStates(lines[1])
        states = self.parsed_states["states"]
        if states != self.QMin.molecule["states"]:
            self.log.error(
                f'states from QM.in and nstates from {template_filename} are inconsistent! {self.QMin.molecule["states"]} != {states}'
            )
        nmstates = self.parsed_states["nmstates"]
        
        # read the coordinate <-> variable mapping
        gvar = []
        for i in range(natom):
            s = lines[i + 2].lower().split()
            if s[0] != self.QMin.molecule["elements"][i].lower():
                self.log.warning(f'Inconsistent atom label in {template_filename} ({s[0]}) compared to atom number in input corresponding to {self.QMin.molecule["elements"][i].lower()}.')
            gvar.append(s[1:4])
        gvar_d = {}
        for i in range(natom):
            for j in range(3):
                v = gvar[i][j]
                if v == '0':
                    continue
                if v[0:1] == '_':
                    self.log.error(f'Variable names must not start with an underscore! {v} is not a valid name.')
                    raise ValueError(f'Variable names must not start with an underscore! {v} is not a valid name.')
                if v in gvar_d:
                    self.log.error(f'Repeated variable in geom<->var mapping in {template_filename}: {v}')
                    raise ValueError(f'Repeated variable in geom<->var mapping in {template_filename}: {v}')
                gvar_d[v] = [i, j]

        # read additional variables
        var = {}
        iline = -1
        while True:
            iline += 1
            if iline == len(lines):
                break
            line = re.sub('#.*$', '', lines[iline])
            s = line.lower().split()
            if s == []:
                continue
            if 'variables' in s[0]:
                while True:
                    iline += 1
                    line = re.sub('#.*$', '', lines[iline])
                    s = line.split()
                    if s == []:
                        continue
                    if 'end' in s[0].lower():
                        break
                    if s[0][0:1] == '_':
                        self.log.error(f'Variable names must not start with an underscore! {v} is not a valid name.')
                        raise ValueError(f'Variable names must not start with an underscore {v} is not a valid name.!')
                    if (s[0] in gvar_d) or (s[0] in var):
                        self.log.error(f'Repeated variable in additional variables in {template_filename}: {s[0]}')
                        raise ValueError(f'Repeated variable in additional variables in {template_filename}: {s[0]}')
                    var[s[0]] = float(s[1])

        self._var = var
        self._gvar = np.array(gvar)
        self._gvar_d = gvar_d

        # obtain the Hamiltonian
        self._Hstring = self.find_lines(nmstates, 'Hamiltonian', lines)
        if self._Hstring == []:
            self.log.error(f'No Hamiltonian defined in {template_filename}!')
            raise ValueError(f'No Hamiltonian defined in {template_filename}!')
        self.check_dimensions(self._Hstring, nmstates)

        # obtain the derivatives
        self._dHstring = {}

        # obtain the dipole matrices
        self._Dstring = {}
        for idir in range(1, 4):
            self._Dstring[idir] = self.find_lines(nmstates, f'Dipole {idir}', lines)
            if self._Dstring[idir] != []:
                self.check_dimensions(self._Dstring[idir], nmstates)

        # obtain the dipole derivative matrices
        #if self.QMin.requests["dmdr"]:
        self._dDstring = {}
        for idir in range(1, 4):
            self._dDstring[idir] = {}

        # obtain the SO matrix
        self._soc_real = True
        #if self.QMin.requests["soc"]:
        self._socRstring = self.find_lines(nmstates, 'SOC Re', lines)
        if self._socRstring != []:
            self.check_dimensions(self._socRstring, nmstates)
            self._socIstring = self.find_lines(nmstates, 'SOC Im', lines)
            if self._socIstring != []:
                self.check_dimensions(self._socIstring, nmstates)
                self._soc_real = False

        self._read_template = True
        return

    def read_resources(self, resources_filename="ANALYTICAL.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning(f"Resources file {resources_filename} not found; continuing without further settings.")
            self._read_resources = True
            return
        super().read_resources(resources_filename)
        if "diagonalize" in self.QMin.resources:
            self._diagonalize =  self.QMin.resources["diagonalize"]

    def setup_interface(self):
        super().setup_interface()
        if self.persistent:
            for file in os.listdir(self.QMin.save["savedir"]):
                if re.match(r"^U\.npy\.\d+$", file):
                    step = int(file.split('.')[-1])
                    ufile = os.path.join(self.QMin.save["savedir"], file)
                    self.savedict[step] = {'U': np.load(ufile).reshape( (self.QMin.molecule['nmstates'], self.QMin.molecule['nmstates']) )}

        #do more preprocessing of expressions and set derivatives where necessary
        #set matrices/function _fH, _fdH, fD, fdD, _fsocR, _fsocI
        nmstates = self.parsed_states["nmstates"]
        self._gvar_symb = np.vectorize(sympy.symbols)(self._gvar[self._gvar != '0'])
        #process H matrix
        Hmat = sympy.zeros(nmstates,nmstates)
        self._fdH = {}
        for i in range(nmstates):
            for j in range(i+1):
                #replace (non-geometry) variables in H
                Hmat[i,j] = sympy.parse_expr(self._Hstring[i][j],self._var)
        if isinstance(Hmat, sympy.Matrix) and len(Hmat.free_symbols)>0:
            #create function for H
            self._fH = sympy.lambdify([self._gvar_symb], Hmat, "numpy")
            #if self._do_derivs:
            for sy in self._gvar_symb:
                #create function for H derivatives
                self._fdH[str(sy)] = sympy.lambdify([self._gvar_symb],sympy.diff(Hmat, sy), "numpy")
        else:
            self.log.error('Hamiltonian does not depend on any geometrical feature (coordinate value). Should not be like this!')
            raise ValueError('Hamiltonian does not depend on any geometrical feature (coordinate value). Should not be like this!')

        #process dipole matrix if it is set
        self._fD = {}
        self._fdD = {}
        for idir in range(1, 4):
            if self._Dstring[idir] != []:
                Dmat = sympy.zeros(nmstates,nmstates)
                for i in range(nmstates):
                    for j in range(i+1):
                        #replace (non-geometry) variables in D
                        Dmat[i,j] = sympy.parse_expr(self._Dstring[idir][i][j],self._var)
                if isinstance(Dmat, sympy.Matrix) and len(Dmat.free_symbols)>0:
                    #create function for D
                    self._fD[idir] = sympy.lambdify([self._gvar_symb], Dmat, "numpy")
                    for sy in self._gvar_symb:
                        #create function for D derivatives
                        self._fdD[idir][str(sy)] = sympy.lambdify([self._gvar_symb],sympy.diff(Dmat, sy), "numpy")
                else:
                    self._fD[idir] = np.array(Dmat)#np.zeros((nmstates,nmstates))
                    self._fdD[idir] = {}
            else:
                self._fD[idir] = np.zeros((nmstates,nmstates))

        #process soc matrix if it is set
        if self._socRstring != []:
            socRmat = sympy.zeros(nmstates,nmstates)
            for i in range(nmstates):
                for j in range(i+1):
                    #replace (non-geometry) variables in real part of SOCs
                    socRmat[i,j] = sympy.parse_expr(self._socRstring[i][j],self._var)
            if isinstance(socRmat, sympy.Matrix) and len(socRmat.free_symbols)>0:
                #create function for real part of SOCs
                self._fsocR = sympy.lambdify([self._gvar_symb], socRmat, "numpy")
            else:
                self._fsocR = np.array(socRmat)
            if not self._soc_real:
                socImat = sympy.zeros(nmstates,nmstates)
                for i in range(nmstates):
                    for j in range(i+1):
                        #replace (non-geometry) variables in imaginary part of SOCs
                        socImat[i,j] = sympy.parse_expr(self._socIstring[i][j],self._var)
                if isinstance(socImat, sympy.Matrix) and len(socImat.free_symbols)>0:
                    #create function for imaginary part of SOCs
                    self._fsocI = sympy.lambdify([self._gvar_symb], socImat, "numpy")
                else:
                    self._fsocI = np.array(socImat)


    def getQMout(self):
        return self.QMout

    def run(self):
        '''Calculates the MCH Hamiltonian, SOC matrix ,overlap matrix, gradients, DM'''
        nmstates = self.parsed_states["nmstates"]
        states = self.parsed_states["states"]
        natom = self.QMin.molecule["natom"]
        #r3N = 3 * natom
        coords: np.ndarray = self.QMin.coords["coords"].copy()
        #coords_needed = [coords.ravel()[i] for i in range(r3N) if self._gvar_list[i] != '0']
        coords_needed = coords[self._gvar != '0']

        if self.QMin.requests["grad"]: grad = np.zeros((nmstates, natom, 3), dtype=float)
        if self.QMin.requests["nacdr"]: 
            dH = np.zeros((nmstates, nmstates, natom, 3), dtype=float)
            #nacdr = np.zeros((nmstates, nmstates, natom, 3), dtype=float)
        if self.QMin.requests["dm"]: dipole = np.zeros((3, nmstates, nmstates), dtype=float)
        if self.QMin.requests["dmdr"]: dipoledr = np.zeros((3, nmstates, nmstates, natom, 3), dtype=float)
        if self.QMin.requests["soc"]:
            if self._soc_real:
                soc = np.zeros((nmstates, nmstates), dtype=float)
            else: soc = np.zeros((nmstates, nmstates), dtype=complex)

        self._do_derivs = self.QMin.requests["grad"] or self.QMin.requests["nacdr"]


        #with sympy: only evaluation at geometry values remains
        self._H = self._fH(coords_needed)

        if self._diagonalize:
            Hd, self._U = np.linalg.eigh(self._H, UPLO="L")
            Hd = np.diag(Hd)
        else:
            Hd = np.tril(self._H) + np.triu(self._H.T,1)#only lower triangle matrix was present
            self._U = np.identity(nmstates, dtype=float)

        if self._do_derivs:
            for sy in self._gvar_d.keys():
                dHmat = self._fdH[sy](coords_needed)
                dHmat =  np.tril(dHmat) + np.triu(dHmat.T,1)#only lower triangle matrix was present
                if self._diagonalize:
                    dHmat = np.einsum('ij,ik,jl->kl',dHmat,self._U,self._U,optimize=True)
                index = self._gvar_d[sy]
                if self.QMin.requests["grad"]: grad[:,index[0],index[1]] =  np.diagonal(dHmat)
                if self.QMin.requests["nacdr"]:
                    dH[:,:,index[0],index[1]] = dHmat

        if self.QMin.requests["nacdr"]:
            #scale derivatives with energy to get nacdr
            tmp = np.full((nmstates,nmstates), np.diagonal(Hd))
            tmp = tmp.T-tmp
            tmp[tmp!=0.] **= -1
            dH = np.einsum('ji,ijkl->ijkl', tmp, dH, optimize=True)

        if self.QMin.requests["dm"]:
            for idir in range(1, 4):
                if not isinstance(self._fD[idir], np.ndarray):
                    temp_dipole = self._fD[idir](coords_needed)
                else: temp_dipole = self._fD[idir]#if dipoles not found, they will just be set to zero (_fD is zero mat then)
                temp_dipole = np.tril(temp_dipole) + np.triu(temp_dipole.T,1)#only lower triangle matrix was present
                if self._diagonalize:
                    temp_dipole = np.einsum('ij,ik,jl->kl',temp_dipole,self._U,self._U,optimize=True)
                dipole[idir-1] = temp_dipole

        if self.QMin.requests["dmdr"]:
            if self._fdD == {}:
                self.log.error('dmdr requested, but no dipoles provided to be derived!')
                raise ValueError('dmdr requested, but no dipoles provided to be derived!')
            else:
                for idir in range(1, 4):
                    for sy in self._gvar_d.keys():
                        if type(self._self._fdD[idir][sy]) != np.ndarray:
                            temp_ddipole = self._fdD[idir][sy](coords_needed)
                        temp_ddipole = np.tril(temp_ddipole) + np.triu(temp_ddipole.T,1)#only lower triangle matrix was present
                        if self._diagonalize:
                            temp_ddipole = np.einsum('ij,ik,jl->kl',temp_ddipole,self._U,self._U,optimize=True)
                            index = self._gvar_d[sy]
                            dipoledr[idir-1,:,:,index[0],index[1]] = temp_ddipole

        if self.QMin.requests["soc"]:
            Hd = Hd.astype(soc.dtype)
        if self.QMin.requests["soc"] and np.sum(states[1:])>0:#if SOCs requested and there are not only singlet states present
            if self._socRstring == []:
                self.log.error('SOCs requested, but (real part of) SOCs not provided!')
                raise ValueError('SOCs requested, but (real part of) SOCs not provided!')
            if self._soc_real:
                if not type(self._fsocR) == np.ndarray:
                    soc = self._fsocR(coords_needed)
                else: soc = self._fsocR
                soc = np.tril(soc) + np.triu(soc.T,1)#only lower triangle matrix was present
            else:
                if not type(self._fsocR) == np.ndarray:
                    soc.real = self._fsocR(coords_needed)
                else: soc.real = self._fsocR
                if not self._soc_real:
                    if not type(self._fsocI) == np.ndarray:
                        soc.imag = self._fsocI(coords_needed)
                    else: soc.imag = self._fsocI
                soc = np.tril(soc) + np.triu(soc.conj().T,1)#only lower triangle matrix was present
            adia_soc = self._U.T @ soc @ self._U
            Hd += adia_soc


        # OVERLAP
        if self.QMin.requests["overlap"] or self.QMin.requests["phases"]:
            if self.QMin.save["step"] == 0:
                pass
            elif self.persistent:
                Uold = self.savedict[self.QMin.save['step']-1]["U"]
            else:
                Uold = np.load(os.path.join(self.QMin.save["savedir"], f"U.npy.{self.QMin.save['step']-1}")).reshape(self._U.shape)
            overlap = Uold.T @ self._U
            if self.QMin.requests["phases"]:
                _, phases = phase_correction(overlap)

        # store U matrix
        if not self.QMin.save["samestep"]:
            if self.persistent:
                self.savedict[self.QMin.save['step']] = {'U': np.copy(self._U)}
            else:
                with open(os.path.join(self.QMin.save["savedir"], f"U.npy.{self.QMin.save['step']}"), 'wb') as f:
                    np.save(f, self._U)  # writes a binary file (can be read with numpy.load())
            
            # keep all U matrices 
            # TODO: could be removed because is done by retain mechanism
            if self.QMin.resources["keep_U"]:
                if "all_U" not in self.__dict__:
                    self.all_U = []
                self.all_U.append(self._U)

        # ======================================== assign to QMout =========================================
        self.log.debug(f"requests: {self.QMin.requests}")
        self.QMout.states = states
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = nmstates#self.QMin.molecule["nmstates"]
        self.QMout.natom = natom#self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = self.QMin.molecule["point_charges"]
        self.QMout.h = Hd
        if self.QMin.requests["overlap"]:
            self.QMout.overlap = overlap
        if self.QMin.requests["phases"]:
            self.QMout.phases = phases
        if self.QMin.requests["grad"]:
            self.QMout.grad = grad
        if self.QMin.requests["nacdr"]:
            self.QMout.nacdr = dH
        if self.QMin.requests["dm"]:
            self.QMout.dm = dipole
        if self.QMin.requests["dmdr"]:
            self.QMout.dmdr = dipoledr

        return

    def create_restart_files(self):
        super().create_restart_files()
        if self.persistent:
            for istep in self.savedict:
                if not isinstance(istep,int):
                    continue
                with open( os.path.join(self.QMin.save["savedir"], f'U.npy.{istep}'), 'wb') as f:
                    np.save(f, self.savedict[istep]["U"])  # writes a binary file (can be read with numpy.load())

            if self.QMin.resources["keep_U"]:
                all_U = np.array(self.all_U)
                np.save(os.path.join(self.QMin.save["savedir"], f"U_0-{self.QMin.save['step']}.npy"), all_U)

    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        return {
            "h",
            "soc",
            "dm",
            "grad",
            "nacdr",
            "overlap",
            "phases",
            "dmdr",
        }

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'ANALYTICAL interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        if os.path.isfile('ANALYTICAL.template'):
            default = 'ANALYTICAL.template'
        else:
            default = None
        self.template_file = question("Specify path to ANALYTICAL.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True, default=default)
        while not os.path.isfile(self.template_file):
            self.template_file = question(
                f"'{self.template_file}' not found!\nSpecify path to ANALYTICAL.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True
            )

        # Check template for Soc and dipoles
        soc_found = False
        dm_found = False
        with open(self.template_file, "r") as f:
            for line in f:
                if "SOC" in line:
                    soc_found = True
                if "Dipole" in line:
                    dm_found = True
        if "soc" in INFOS["needed_requests"] and not soc_found:
            self.log.error(f"Requested SOC calculation but 'SOC' keyword not found in {self.template_file}")
            raise RuntimeError()

        if ("dm" in INFOS["needed_requests"] or "dmdr" in INFOS["needed_requests"]) and not dm_found:
            self.log.error(f"Calculation of dipole moment (or DMdr) requested but 'DM' keyword not found in {self.template_file}")
            raise RuntimeError()

        #if question("Do you have an ANALYTICAL.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
        #    self.resources_file = question("Specify path to ANALYTICAL.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

        return INFOS

if __name__ == "__main__":
    from logger import loglevel

    analytical = SHARC_ANALYTICAL(loglevel=loglevel)
    analytical.main()

