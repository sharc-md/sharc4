#!/usr/bin/env python3

# Short program to evaluate partial charges with the restircted electrostatic potential fit
# Source Phys.Chem.Chem.Phys.,2019,21, 4082--4095 cp/c8cp06567e and TheJournalofPhysicalChemistry,Vol.97,No.40,199 10.1021/j100142a004
#  gaussian can do RESP
from dataclasses import dataclass
from error import Error
from time import process_time_ns as t
from itertools import chain
from utils import swap_rows_and_cols, build_basis_dict
from constants import IAn2AName as CHARGEATOM, ATOMIC_RADII

import sys
import numpy as np
from scipy.spatial import distance_matrix
from asa_grid import mk_layers
from pyscf import gto, tools, df, scf
import h5py

au2a = 0.52917721092
np.set_printoptions(threshold=sys.maxsize, linewidth=10000, precision=5)


def get_resp_grid(atom_symbols: list[str], coords: np.ndarray):
    atom_radii = np.fromiter(map(lambda x: ATOMIC_RADII[x], atom_symbols), dtype=float)
    return mk_layers(coords, atom_radii)


@dataclass(init=True, eq=True)
class Cube:
    n1: int
    n2: int
    n3: int

    x0: float
    x1: float
    x2: float
    x3: float

    y0: float
    y1: float
    y2: float
    y3: float

    z0: float
    z1: float
    z2: float
    z3: float

    def get_points(self) -> np.ndarray:
        return np.asarray([p for p in self.points()], dtype=float)

    def points(self):
        point = np.array([self.x0, self.y0, self.z0])
        shift1 = np.array([self.x1, self.y1, self.z1])
        shift2 = np.array([self.x2, self.y2, self.z2])
        shift3 = np.array([self.x3, self.y3, self.z3])
        for p in range(self.n1):
            s1 = p * shift1
            for q in range(self.n2):
                s2 = q * shift2
                for r in range(self.n3):
                    yield point + r * shift3 + s2 + s1

    def volume(self):
        s = self
        d = s.x1 * s.y2 * s.z3 + s.x2 * s.y3 * s.z1 + s.x3 * s.y1 * s.z2 - s.x3 * s.y2 * s.z1 - s.x1 * s.y3 * s.z2 - s.x2 * s.y1 * s.z3
        return d

    def n_points(self):
        return self.n1 * self.n2 * self.n3


class Resp:
    def __init__(self, coords: np.ndarray, atom_symbols: list[str]):
        """
        creates an object with a fitting grid and precalculated properties for the molecule.

        Parameters:
        ----------
        coords: ndarray array with shape (natom,3) unit has to be Bohr

        atom_symbols: list[str] with atom symbols order corresponding to coords 
        """
        self.beta = 0.0005
        self.coords = coords
        self.atom_symbols = atom_symbols
        self.mk_grid = get_resp_grid(atom_symbols, coords * au2a) / au2a
        self.natom = coords.shape[0]
        self.ngp = self.mk_grid.shape[0]
        # Build 1/|R_A - r_i| m_A_i
        self.R_alpha: np.ndarray = np.full((self.natom, self.ngp, 3), self.mk_grid) - self.coords[:, None, :]  # rA-ri
        self.r_inv: np.ndarray = 1 / np.sqrt(np.sum((self.R_alpha)**2, axis=2))    # 1 / |ri-rA|

    @classmethod
    def from_molden(cls, moldenfile):
        is_ang = False
        coords = []
        atom_symbols = []
        with open(moldenfile, 'r') as f:
            while True:
                line = f.readline()
                if '[atoms]' in line.lower():
                    if 'ANG' in line.upper():
                        is_ang = True
                    break
            while True:
                llist = f.readline().split()
                if len(llist) != 6:
                    break
                atom_symbols.append(CHARGEATOM[int(llist[2])])
                coords.append([float(x) for x in llist[3:]])

        coords = np.array(coords) / au2a if is_ang else np.array(coords)
        return cls(coords, atom_symbols)

    @classmethod
    def from_cube(cls, cubefile):
        atom_symbols = []
        coords = []
        with open(cubefile, 'r') as f:
            f.readline()    # header
            f.readline()
            natom = int(f.readline().split()[0])
            f.readline()
            f.readline()
            f.readline()
            cast = [int, float, float, float, float]
            # coords block
            for _ in range(natom):
                IA, _, x, y, z = (c(x) for c, x in zip(cast, f.readline().split()))
                atom_symbols.append(CHARGEATOM[IA])
                coords.append([x, y, z])
        # coords already in atomic units
        coords = np.asarray(coords)
        return cls(coords, atom_symbols)

    @classmethod
    def from_chk(cls, chkfile):
        mol = scf.chkfile.load_mol(chkfile)
        coords = mol.atom_coords(unit="BOHR")
        atom_symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        return cls(atom_symbols, coords)

    @classmethod
    def from_fchk(cls, fchkfile):
        coords = []
        atom_symbols = None
        with open(fchkfile, 'r') as f:
            f.readline()
            f.readline()
            llist = f.readline().split()
            if ' '.join(llist[:3]) != 'Number of atoms':
                raise Error('Number of atoms not found')
            natom = int(llist[-1])
            while True:
                if 'Atomic numbers' in f.readline():
                    break
            atom_symbols = []
            for _ in range((natom - 1) // 6 + 1):
                atom_symbols.extend(map(lambda x: CHARGEATOM[int(x)], f.readline().split()))
            f.readline()
            for _ in range((natom - 1) // 5 + 1):
                f.readline()
            f.readline()
            for _ in range((natom * 3 - 1) // 5 + 1):
                coords.extend(map(float, f.readline().split()))
        coords = np.asarray(coords).reshape((-1, 3))
        return cls(coords, atom_symbols)

    def prepare(self, basis, cart_basis=False):
        natom = len(self.atom_symbols)
        atoms = [[f'{s.upper()}{j+1}', c.tolist()] for j, s, c in zip(range(natom), self.atom_symbols, self.coords)]
        mol = gto.Mole(atom=atoms, basis=basis, unit='BOHR', symmetry=False, cart=cart_basis)
        mol.build()
        Z = mol.atom_charges()
        self.Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        fakemol = gto.fakemol_for_charges(self.mk_grid)
        # NOTE This could be very big (fakemol could be broken up into multiple pieces)
        # NOTE the value of these integrals is not affected by the atom charge
        self.ints = df.incore.aux_e2(mol, fakemol)

    def multipoles_from_dens(self, dm: np.ndarray, include_core_charges: bool, order=2):
        if not (0 < order <= 2):
            raise Error("Specify order in the range of 0 - 2")
        n_fits = sum([1, 3, 6][:order + 1])
        Fesp_i = np.copy(self.Vnuc) if include_core_charges else np.zeros((self.ngp), dtype=float)
        Vele = np.einsum('ijp,ij->p', self.ints, dm)
        Fesp_i[...] -= Vele
        fits = np.empty((self.natom, n_fits))
        R_alpha = self.R_alpha
        r_inv = self.r_inv
        mp = self.fit_monopoles(Fesp_i)
        Fesp_i_res = Fesp_i - mp @ r_inv
        self.r_inv3 = self.rinv3 if 'rinv3' in self.__dict__ else r_inv**3
        dp = self.fit_dipoles(Fesp_i_res).reshape((3, -1))
        Fesp_i_res = Fesp_i_res - np.einsum('xi,inx,in->n', dp, R_alpha, self.r_inv3)
        qp = self.fit_quadrupoles(Fesp_i_res).reshape((6, -1))
        fits = np.vstack((mp, dp, qp)).T
        return fits

    # NOTE: This works but the density of core electron is not well represented!!!
    def ESP_from_cubes(self, cube_list: list[str]):
        s1 = t()
        densities = []
        atom_symbols = []
        atom_charges = []
        coords = []
        cube = None
        for icube, cube_file in enumerate(cube_list):
            with open(cube_file, 'r') as f:
                f.readline()    # header
                f.readline()
                cast = [int, float, float, float, float]
                natom, x0, y0, z0, _ = (c(x) for c, x in zip(cast, f.readline().split()))
                # notation from cubegen documentation https://gaussian.com/cubegen/?tabid=0#cubegen_utility
                n1, x1, y1, z1 = (c(x) for c, x in zip(cast, f.readline().split()))
                n2, x2, y2, z2 = (c(x) for c, x in zip(cast, f.readline().split()))
                n3, x3, y3, z3 = (c(x) for c, x in zip(cast, f.readline().split()))
                c = Cube(n1, n2, n3, x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3)
                # coords block
                for _ in range(natom):
                    IA, Chg, x, y, z = (c(x) for c, x in zip(cast, f.readline().split()))
                    atom_symbols.append(CHARGEATOM[IA])
                    atom_charges.append(IA)
                    coords.append([x, y, z])
                coords = np.array(coords)
                # sanity check
                if atom_symbols != self.atom_symbols:
                    raise Error(f'The files do not contain the same atom!!\n{atom_symbols}\n{self.atom_symbols}')
                assert np.array_equiv(coords, self.coords)
                if icube != 0:
                    assert c == cube
                cube = c
                densities.append([float(x) for x in f.read().split()])
        s2 = t()
        print('parsing done: {:8.8f}ms'.format((s2 - s1) * 1e-6))
        s1 = s2
        voxelV = cube.volume()
        densities = np.asarray(densities)
        Z = np.asarray(atom_charges)
        Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        s2 = t()
        print('Vnuc: {:8.8f}ms'.format((s2 - s1) * 1e-6))
        s1 = s2
        self.Fesp_s_i = np.full((len(cube_list), self.ngp), Vnuc[None, ...], dtype=float)
        Vele = np.zeros((len(cube_list), self.ngp), dtype=float)
        # self.Fesp_s_i = np.zeros((len(cube_list), self.ngp), dtype=float)
        cube_grid = cube.get_points()
        s2 = t()
        print('points: {:8.8f}ms'.format((s2 - s1) * 1e-6))
        s1 = s2
        s2 = t()
        print('pre loop: {:8.8f}ms'.format((s2 - s1) * 1e-6))
        s1 = s2
        for i, mk_point in enumerate(self.mk_grid):
            dist = 1 / distance_matrix(mk_point[None, ...], cube_grid, p=2, threshold=1e8)
            Vele[..., i] += voxelV * np.einsum('sx,x->s', densities, dist.flat, optimize=True)
            self.Fesp_s_i[..., i] -= Vele[..., i]
        s2 = t()
        print('done: {:8.8f}ms'.format((s2 - s1) * 1e-6))
        s1 = s2
        print("REF:\n[-5.14204 -4.85815 -4.62127 -4.85815 -4.41226 -4.36045 -4.41226 -4.85815 -4.62127 -4.85815]")
        print(Vele[0, :20])

    def ESP_from_molden(self, molden_list):
        mol: gto.Mole = None
        mo_coeffs = []
        mo_occs = []
        for imolden, molden_file in enumerate(molden_list):
            mol_t, _, mo_coeff_t, mo_occ_t, _, _ = tools.molden.load(molden_file)
            mol_t.build()
            atom_symbols = [mol_t.atom_pure_symbol(i) for i in range(mol_t.natm)]
            assert atom_symbols == self.atom_symbols
            coords = mol_t.atom_coords(unit='BOHR')
            assert np.allclose(coords, self.coords, rtol=1e-6)
            if imolden != 0:
                assert mol_t._basis == mol._basis
            mol = mol_t
            mo_coeffs.append(mo_coeff_t)
            mo_occs.append(mo_occ_t)
        Z = mol.atom_charges()
        Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        self.Fesp_s_i = np.full((len(molden_list), self.ngp), Vnuc[None, ...], dtype=float)
        for i, mo_coeff, mo_occ in zip(range(len(molden_list)), mo_coeffs, mo_occs):
            dm = np.einsum('pi,ij,qj->pq', mo_coeff, np.diag(mo_occ), mo_coeff)
            fakemol = gto.fakemol_for_charges(self.mk_grid)
            ints: np.ndarray = df.incore.aux_e2(mol, fakemol)
            Vele = np.einsum('ijp,ij->p', ints, dm)
            self.Fesp_s_i[i, ...] -= Vele

    def ESP_from_chk(self, chk_list, key='SCF'):
        densities = []
        mol: gto.Mole = None
        mo_coeffs = []
        # check chk files
        if any((not h5py.is_hdf5(file) for file in chk_list)):
            raise Error("not all files are h5py compliant!")

        for chkfile in chk_list:
            with h5py.File(chk_list, 'r') as fh5:
                pass

    def ESP_from_fchks(self, fchk_list):
        '''
        parses the densities specifed in dens_types from the corresponding fchk file (1-to-1 mapping)
        The first file will be used to parse the SCF, first ES and GS2ES densities, the other ES densities
        copme from the other files
        '''
        densities = []
        basis_name = ''
        mol = None
        nelectron = 0
        n_bf = 0
        for ifchk, fchkfile in enumerate(fchk_list):
            with open(fchkfile, 'r') as f:
                f.readline()
                basis_name_t = f.readline().split()[-1]
                if ifchk != 0:
                    assert basis_name == basis_name_t
                basis_name = basis_name_t
                llist = f.readline().split()
                if ' '.join(llist[:3]) != 'Number of atoms':
                    raise Error('Number of atoms not found')
                natom = int(llist[-1])
                while True:
                    line = f.readline()
                    if 'Number of electrons' in line:
                        nelectron = int(line.split()[-1])
                    if 'Atomic numbers' in line:
                        break
                    if 'Number of basis functions' in line:
                        n_bf = int(line.split()[-1])
                atom_symbols = []
                for _ in range((natom - 1) // 6 + 1):
                    atom_symbols.extend(map(lambda x: CHARGEATOM[int(x)], f.readline().split()))
                assert atom_symbols == self.atom_symbols
                f.readline()
                for _ in range((natom - 1) // 5 + 1):
                    f.readline()
                f.readline()
                coords = []
                for _ in range((natom * 3 - 1) // 5 + 1):
                    coords.extend(map(float, f.readline().split()))
                assert np.allclose(self.coords, np.array(coords).reshape((-1, 3)), rtol=1e-6)
                lines = f.readlines()
                if ifchk == 0:
                    i = 0
                    shell_types = []
                    n_prim = []
                    s_a_map = []
                    prim_exp = []
                    contr_coeff = []
                    ps_contr_coeff = None
                    while i != len(lines):
                        if 'Shell types' in lines[i]:
                            n = int(lines[i].split()[-1])
                            n_lines = (n - 1) // 6 + 1
                            i += 1
                            shell_types = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                            i += n_lines
                        if 'Number of primitives per shell' in lines[i]:
                            n = int(lines[i].split()[-1])
                            n_lines = (n - 1) // 6 + 1
                            i += 1
                            n_prim = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                            i += n_lines
                        if 'Shell to atom map' in lines[i]:
                            n = int(lines[i].split()[-1])
                            n_lines = (n - 1) // 6 + 1
                            i += 1
                            s_a_map = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                            i += n_lines
                        if 'Primitive exponents' in lines[i]:
                            n = int(lines[i].split()[-1])
                            n_lines = (n - 1) // 5 + 1
                            i += 1
                            prim_exp = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                            i += n_lines
                        if 'Contraction coefficients' in lines[i]:
                            n = int(lines[i].split()[-1])
                            n_lines = (n - 1) // 5 + 1
                            i += 1
                            contr_coeff = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                            i += n_lines
                            if 'P(S=P) Contraction coefficients' in lines[i]:
                                n = int(lines[i].split()[-1])
                                n_lines = (n - 1) // 5 + 1
                                i += 1
                                ps_contr_coeff = list(
                                    map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines])))
                                )
                            break
                        i += 1
                    basis = build_basis_dict(
                        self.atom_symbols, shell_types, n_prim, s_a_map, prim_exp, contr_coeff, ps_contr_coeff
                    )
                    atoms = [
                        [f'{s.upper()}{j+1}', c.tolist()]
                        for j, s, c in zip(range(natom), self.atom_symbols, self.coords)
                    ]
                    mol = gto.Mole(
                        atom=atoms, basis=basis, unit='BOHR', symmetry=False, nelectron=nelectron, cart=False
                    )

                i = 0
                while i < len(lines):
                    if ifchk == 0 and 'SCF Density' in lines[i]:
                        n = int(lines[i].split()[-1])
                        n_lines = (n - 1) // 5 + 1
                        i += 1
                        d = np.fromiter(
                            map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                        )
                        # d is triangular -> fold into full matrix (assuming lower triangular)
                        d_tril = np.zeros((n_bf, n_bf))
                        idx = np.tril_indices(n_bf)
                        d_tril[idx] = d
                        density = d_tril.T + d_tril
                        np.fill_diagonal(density, np.diag(d_tril))
                        swap_rows_and_cols(atom_symbols, basis, density)
                        densities.append(density)
                        i += n_lines
                    if 'CI Density' in lines[i]:
                        n = int(lines[i].split()[-1])
                        n_lines = (n - 1) // 5 + 1
                        i += 1
                        d = np.fromiter(
                            map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                        )
                        # d is triangular -> fold into full matrix (assuming lower triangular)
                        d_tril = np.zeros((n_bf, n_bf))
                        idx = np.tril_indices(n_bf)
                        d_tril[idx] = d
                        density = d_tril.T + d_tril
                        np.fill_diagonal(density, np.diag(d_tril))
                        swap_rows_and_cols(atom_symbols, basis, density)
                        densities.append(density)
                        i += n_lines
                    if 'Excited state NLR' in lines[i]:
                        i += 1
                        n_es = int(lines[i].split()[-1])
                        i += 1
                        n_g2e = int(lines[i].split()[-1])
                        i += 1
                        if n_es != 0:
                            i += 1
                    if ifchk == 0 and 'G to E trans densities' in lines[i]:
                        n = int(lines[i].split()[-1])
                        n_lines = (n - 1) // 5 + 1
                        i += 1
                        d = np.fromiter(
                            map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                        ).reshape((2 * n_g2e, n_bf, n_bf))
                        print('Hello')
                        # d = np.zeros((2*n_g2e, n_bf, n_bf), dtype=float)
                        for i_d in range(0, 2 * n_g2e, 2):
                            tmp = d[i_d, ...]
                            swap_rows_and_cols(atom_symbols, basis, tmp)
                            densities.append(tmp)
                        i += n_lines
                    i += 1

        mol.symmetry = None
        mol.build()
        # reorder densities
        test = [x for x in range(len(densities))]
        test2 = []
        test2.extend(test[:2])
        test2.extend(test[2 + n_g2e:])
        test2.extend(test[2:(2 + n_g2e)])
        print(test2)
        densities_sort = []
        densities_sort.extend(densities[:2])
        densities_sort.extend(densities[2 + n_g2e:])
        densities_sort.extend(densities[2:(2 + n_g2e)])
        # Test purpose calculate dipole moments
        for dens in densities_sort:
            dip = scf.hf.dip_moment(mol, dens)
            print(dip * 0.39343)
        print()
        Z = mol.atom_charges()
        Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        print(n_es)
        for_states = np.full((len(fchk_list) + 1, self.ngp), Vnuc[None, ...], dtype=float)
        for_tdm = np.zeros((n_g2e, self.ngp), dtype=float)
        self.Fesp_s_i = np.concatenate((for_states, for_tdm), axis=0)
        for i, dens in enumerate(densities_sort):
            fakemol = gto.fakemol_for_charges(self.mk_grid)
            ints = df.incore.aux_e2(mol, fakemol)
            Vele = np.einsum('ijp,ji->p', ints, dens)
            self.Fesp_s_i[i, ...] -= Vele

    def ESP_from_fchk(self, fchk_file, gs=True, es=True, gses=True):
        '''
        parses the densities specifed from the same fchk file
        '''
        densities = None
        mol = None
        nelectron = 0
        n_bf = 0
        with open(fchk_file, 'r') as f:
            f.readline()
            basis_name_t = f.readline().split()[-1]
            llist = f.readline().split()
            if ' '.join(llist[:3]) != 'Number of atoms':
                raise Error('Number of atoms not found')
            natom = int(llist[-1])
            while True:
                line = f.readline()
                if 'Number of electrons' in line:
                    nelectron = int(line.split()[-1])
                if 'Atomic numbers' in line:
                    break
                if 'Number of basis functions' in line:
                    n_bf = int(line.split()[-1])
            atom_symbols = []
            for _ in range((natom - 1) // 6 + 1):
                atom_symbols.extend(map(lambda x: CHARGEATOM[int(x)], f.readline().split()))
            assert atom_symbols == self.atom_symbols
            f.readline()
            for _ in range((natom - 1) // 5 + 1):
                f.readline()
            f.readline()
            coords = []
            for _ in range((natom * 3 - 1) // 5 + 1):
                coords.extend(map(float, f.readline().split()))
            assert np.allclose(self.coords, np.array(coords).reshape((-1, 3)), rtol=1e-6)
            lines = f.readlines()

            shell_types = []
            n_prim = []
            s_a_map = []
            prim_exp = []
            contr_coeff = []
            ps_contr_coeff = None
            i = 0
            while i != len(lines):
                if 'Shell types' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 6 + 1
                    i += 1
                    shell_types = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                    i += n_lines
                if 'Number of primitives per shell' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 6 + 1
                    i += 1
                    n_prim = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                    i += n_lines
                if 'Shell to atom map' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 6 + 1
                    i += 1
                    s_a_map = list(map(int, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                    i += n_lines
                if 'Primitive exponents' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    prim_exp = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                    i += n_lines
                if 'Contraction coefficients' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    contr_coeff = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                    i += n_lines
                    if 'P(S=P) Contraction coefficients' in lines[i]:
                        n = int(lines[i].split()[-1])
                        n_lines = (n - 1) // 5 + 1
                        i += 1
                        ps_contr_coeff = list(map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))))
                    break
                i += 1
            basis = build_basis_dict(
                self.atom_symbols, shell_types, n_prim, s_a_map, prim_exp, contr_coeff, ps_contr_coeff
            )
            atoms = [[f'{s.upper()}{j+1}', c.tolist()] for j, s, c in zip(range(natom), self.atom_symbols, self.coords)]
            mol = gto.Mole(atom=atoms, basis=basis, unit='BOHR', symmetry=False, nelectron=nelectron, cart=False)

            i = 0
            while i != len(lines):
                if lines[i][0] == ' ':
                    i += 1
                    continue
                # if 'Alpha MO ' in lines[i]:
                #     n = int(lines[i].split()[-1])
                #     n_lines = (n-1)//5 +1
                #     i += 1
                #     d = np.fromiter(map(float, chain(*map(lambda x: x.split(), lines[i: i+n_lines]))), dtype=float, count=n).reshape((n_bf,n_bf))
                if gs and 'SCF Density' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    d = np.fromiter(
                        map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                    )
                    # d is triangular -> fold into full matrix (assuming lower triangular)
                    d_tril = np.zeros((n_bf, n_bf))
                    idx = np.tril_indices(n_bf)
                    d_tril[idx] = d
                    density = d_tril.T + d_tril
                    np.fill_diagonal(density, np.diag(d_tril))
                    swap_rows_and_cols(atom_symbols, basis, density)
                    densities = density
                    i += n_lines
                if 'CI Density' in lines[i]:
                    n = int(lines[i].split()[-1])
                    n_lines = (n - 1) // 5 + 1
                    i += 1
                    d = np.fromiter(
                        map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                    )
                    # d is triangular -> fold into full matrix (assuming lower triangular)
                    d_tril = np.zeros((n_bf, n_bf))
                    idx = np.tril_indices(n_bf)
                    d_tril[idx] = d
                    density = d_tril.T + d_tril
                    np.fill_diagonal(density, np.diag(d_tril))
                    swap_rows_and_cols(atom_symbols, basis, density)
                    densities = np.concatenate((densities[None, ...], density[None, ...]))
                    i += n_lines
                if es and 'Excited state NLR' in lines[i]:
                    nlr = int(lines[i].split()[-1])
                    i += 1
                    n_es = int(lines[i].split()[-1])
                    i += 1
                    n_g2e = int(lines[i].split()[-1])
                    i += 1
                    n_e2e = int(lines[i].split()[-1])
                    i += 2
                    spins = list(map(int, lines[i].split()))
                    if n_es != 0:
                        i += 1
                        n = int(lines[i].split()[-1])
                        n_lines = (n - 1) // 5 + 1
                        i += 1
                        d = np.fromiter(
                            map(float, chain(*map(lambda x: x.split(), lines[i:i + n_lines]))), dtype=float, count=n
                        ).reshape((2 * n_es, -1))
                        i += n_lines
                        d_tril = np.zeros((n_es, n_bf, n_bf))
                        tmp = np.zeros((n_bf, n_bf))
                        tmp2: np.ndarray = np.zeros((n_bf, n_bf))
                        idx = np.tril_indices(n_bf)
                        for v_i in range(0, 2 * n_es, 2):
                            tmp[idx] = d[v_i] + d[v_i + 1]
                            tmp2 = tmp.T + tmp
                            # np.savetxt(f'/user/severin/workdir/RESP/test/es_{v_i}.txt', tmp2, fmt='% 8.8f')
                            np.fill_diagonal(tmp2, np.diag(tmp))
                            swap_rows_and_cols(atom_symbols, basis, tmp2)
                            d_tril[v_i // 2] = tmp2
                        if gs:
                            if len(densities.shape) == len(d_tril.shape):
                                densities = np.concatenate((densities, d_tril), axis=0)
                            else:
                                densities = np.concatenate((densities[None, ...], d_tril), axis=0)
                        else:
                            densities = d_tril
                i += 1
        mol.symmetry = None
        mol.build()
        print(mol.nao_2c())
        # Test purpose calculate dipole moments
        for dens in densities:
            dip = scf.hf.dip_moment(mol, dens)
            print(dip * 0.39343)
        print()
        Z = mol.atom_charges()
        Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        n_states = int(gs) + int(es)
        n_tdm = n_g2e
        if gs or es:
            for_states = np.full((n_states, self.ngp), Vnuc[None, ...], dtype=float)
            if gses:
                for_tdm = np.zeros((n_tdm, self.ngp), dtype=float)
                self.Fesp_s_i = np.concatenate((for_states, for_tdm), axis=0)
            else:
                self.Fesp_s_i = for_states
        elif gses:
            self.Fesp_s_i = np.zeros((n_tdm, self.ngp), dtype=float)


        fakemol = gto.fakemol_for_charges(self.mk_grid)
        # NOTE This could be very big (fakemol could be broken up into multiple pieces)
        ints = df.incore.aux_e2(mol, fakemol)
        Vele = np.einsum('ijp,nji->np', ints, densities)
        self.Fesp_s_i -= Vele

    def fit_multipoles(self, order=2):
        self.fits = np.empty((self.Fesp_s_i.shape[0], self.natom, 10))
        if not (0 < order <= 2):
            raise Error("Specify order in the range of 0 - 2")
        R_alpha = self.R_alpha
        r_inv = self.r_inv
        for s, Fesp_i in enumerate(self.Fesp_s_i):
            mp = self.fit_monopoles(Fesp_i)
            Fesp_i_res = Fesp_i - mp @ r_inv
            self.r_inv3 = self.rinv3 if 'rinv3' in self.__dict__ else r_inv**3
            dp = self.fit_dipoles(Fesp_i_res).reshape((3, -1))
            Fesp_i_res = Fesp_i_res - np.einsum('xi,inx,in->n', dp, R_alpha, self.r_inv3)
            qp = self.fit_quadrupoles(Fesp_i_res).reshape((6, -1))
            self.fits[s, ...] = np.vstack((mp, dp, qp)).T
            dip = np.sum(self.coords.T * mp, axis=1)
            dip += np.sum(dp, axis=1)
        return self.fits

    def fit_monopoles(self, Fesp_i):
        natom = self.natom
        # build B
        b = self.r_inv @ Fesp_i    # v_A
        # build A
        a = self.r_inv @ self.r_inv.T    # contract over ngp -> m_A_A
        # q_A
        A = np.zeros((natom + 1, natom + 1))
        A[:natom, :natom] += a
        A[:natom, natom] = 1
        A[natom, :natom] = 1

        B = np.zeros((natom + 1))
        B[:natom] += b
        B[natom] = 0    # TODO reintroduce charge!!

        Q1 = np.linalg.solve(A, B)
        Q2 = np.ones(natom + 1, float)

        def get_rest(Q):
            return self.beta / (np.sqrt(Q**2 + 0.1**2))    # b = 0.1

        vget_rest = np.vectorize(get_rest, cache=True)

        while np.linalg.norm(Q1 - Q2) >= 0.00001:
            Q1 = Q2.copy()
            rest = vget_rest(Q1)
            B_rest = B
            A_rest = A + np.diag(rest)
            Q2 = np.linalg.solve(A_rest, B_rest)

        return Q2[:natom]

    def fit_dipoles(self, Fesp_i: np.ndarray):
        natom = self.natom
        # Build 1/|R_A - r_i| m_A_i
        R_alpha = self.R_alpha

        r_inv3 = self.r_inv3

        tmp = np.vstack((R_alpha[:, :, 0] * r_inv3, R_alpha[:, :, 1] * r_inv3, R_alpha[:, :, 2] * r_inv3))    # m_A_i
        # build A
        A = tmp @ tmp.T    # contract over ngp -> m_A_A

        # build B'
        B = tmp @ Fesp_i    # v_A

        Q1: np.ndarray = np.linalg.solve(A, B)
        Q2 = np.ones(Q1.shape, float)

        def get_rest(Q):
            return self.beta / (np.sqrt(Q**2 + 0.1**2))    # b = 0.1

        vget_rest = np.vectorize(get_rest)

        while np.linalg.norm(Q1 - Q2) >= 0.00001:
            Q1 = Q2.copy()
            rest = vget_rest(Q1)
            rest[:natom] = 0.
            B_rest = B
            A_rest = A + np.diag(rest)
            Q2 = np.linalg.solve(A_rest, B_rest)

        return Q2

    def fit_quadrupoles(self, Fesp_i):
        natom = self.natom
        # Build 1/|R_A - r_i| m_A_i
        R_alpha = self.R_alpha

        r_inv = self.r_inv
        r_inv5 = self.rinv5 if 'rinv5' in self.__dict__ else r_inv**5
        r_inv5_2 = 0.5 * r_inv5
        # order xx, yy, zz, xy, xz, yz
        tmp = np.vstack(
            (
                R_alpha[:, :, 0] * R_alpha[:, :, 0] * r_inv5_2, R_alpha[:, :, 1] * R_alpha[:, :, 1] * r_inv5_2,
                R_alpha[:, :, 2] * R_alpha[:, :, 2] * r_inv5_2, R_alpha[:, :, 0] * R_alpha[:, :, 1] * r_inv5_2,
                R_alpha[:, :, 0] * R_alpha[:, :, 2] * r_inv5_2, R_alpha[:, :, 1] * R_alpha[:, :, 2] * r_inv5_2
            )
        )    # m_A_i
        # build A
        A = tmp @ tmp.T    # contract over ngp -> m_A_A

        # build B'
        B = tmp @ Fesp_i    # v_A

        Q1 = np.linalg.solve(A, B)
        Q2 = np.ones(Q1.shape, float)

        def get_rest(Q):
            return self.beta / (np.sqrt(Q**2 + 0.1**2))    # b = 0.1

        vget_rest = np.vectorize(get_rest)

        while np.linalg.norm(Q1 - Q2) >= 0.00001:
            Q1 = Q2.copy()
            rest = vget_rest(Q1)
            rest[:natom] = 0.
            B_rest = B
            A_rest = A + np.diag(rest)
            Q2 = np.linalg.solve(A_rest, B_rest)

        # make traceless (Source: Sebastian)
        quad = Q2.reshape((natom, -1))
        traces = np.sum(quad[:, :3], axis=1)
        quad[:, :3] -= 1 / 3 * traces[..., None]
        Q2 = quad.reshape(Q2.shape)
        return Q2
