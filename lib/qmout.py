import math
from itertools import chain

import numpy as np
from constants import IToMult, IToPol
from numpy import ndarray
from printing import formatcomplexmatrix, formatgrad
from utils import eformat, itnmstates, writefile
from logger import log
import pyscf
from utils import electronic_state
from functools import reduce
from itertools import islice
import ast
import re
import ast
import json


class QMout:
    """
    Storage container for all results of single point calculations.
    Call the `allocate()` function to initialize the value according to the system
    and requests
    """

    # dimensions
    nstates: int
    nmstates: int
    natom: int
    npc: int
    point_charges: bool
    # data
    runtime: int
    notes: dict[str, str]
    states: list[int]
    charges: list[int]
    h: ndarray[complex, 2]
    dm: ndarray[complex, 3]
    grad: ndarray[float, 3]
    grad_pc: ndarray[float, 3]
    nacdr: ndarray[float, 4]
    nacdr_pc: ndarray[float, 4]
    overlap: ndarray[float, 2]
    phases: ndarray[float]
    prop0d: list[tuple[str, float]]
    prop1d: list[tuple[str, ndarray[float]]]
    prop2d: list[tuple[str, ndarray[float, 2]]]
    socdr: ndarray[float, 4]
    socdr_pc: ndarray[float, 4]
    dmdr: ndarray[float, 5]
    dmdr_pc: ndarray[float, 5]
    multipolar_fit: dict
    density_matrices: dict
    mol: pyscf.gto.Mole
    #dyson_orbitals: dict[tuple(electronic_state,electronic_state,str), ndarray[float,1] ]

    def __init__(self, filepath=None, states: list[int] = None, natom: int = None, npc: int = None, charges: list[int] = None,
                 flags='all'):
        self.prop0d = []
        self.prop1d = []
        self.prop2d = []
        self.notes = {}
        self.runtime = 0
        if flags == 'all':
            flags = {k for k in range(30)}
        if states is not None:
            self.states = states
            self.nmstates = sum((i + 1) * n for i, n in enumerate(self.states))
            self.nstates = sum(self.states)
        if natom is not None:
            self.natom = natom
        if npc is not None:
            self.npc = npc
            self.point_charges = self.npc > 0
        if charges is None and "states" in self:
            self.charges = [i % 2 for i in range(len(self.states))]
        else:
            self.charges = charges
        if filepath is not None:
            # initialize the entire object from a QM.out file
            log.debug(f"Reading file {filepath}")
            try:
                f = open(filepath, "r", encoding="utf-8")
            except IOError:
                raise IOError("'Could not find %s!' % (filepath)")
            log.debug(f"Done raw reading {filepath}")
            # get basic information
            basic_info = {"states": list, "charges": list, "natom": int, "npc": int, "nmstates": int}
            # set from input
            line = "Start"
            while line:
                # skip to next flag
                if not line[0] == "!":
                    line = f.readline()
                    continue
                # get flag
                flag = int(line.split()[1])
                if flag == 0:
                    data = []
                    line = f.readline()
                    while line[0] != "!":
                        data.append(line)
                        line = f.readline()
                    shape = []
                    block_length = 0
                elif flag in {20, 23}:
                    data = [line]
                    line = f.readline()
                    while line != '\n':
                        data.append(line)
                        line = f.readline()
                    shape = []
                    block_length = len(data)
                    if flag not in flags:
                        continue
                else:
                    if flag in {8, 25}:
                        shape = [1]
                        block_length = 1
                    else:
                        shape = [int(n) for n in re.search(r"\(((\d+x)+\d+)", line).group(1).split('x')]
                        block_length = reduce(lambda agg, x: agg*x, shape[:-1])
                        if len(shape) > 2:
                            block_length += shape[0] - 1
                    # skip unwanted flags
                    if flags != "all" and flag not in flags:
                        # print(f"skipping flag {flag} with {block_length} lines")
                        next(islice(f, block_length, block_length), None)
                        # (f.readline() for _ in range(block_length))
                        line = f.readline()
                        continue

                    data = [line] + [f.readline() for _ in range(block_length+1)]
                iline = 0

                log.debug(f"Parsing flag: {flag}")
                # print(f"Parsing flag: {flag}, {shape} {block_length}")
                match flag:
                    case 0: # basis info
                        while iline < len(data):
                            # log.trace(data[iline])
                            if not data[iline].strip():
                                iline += 1
                                continue
                            k, v = data[iline].split(maxsplit=1)

                            if k not in basic_info:
                                log.warning(f"did not parse {k} from section 0!")
                                iline += 1
                                continue
                            match basic_info[k].__name__:
                                case "int":
                                    self.__dict__[k] = int(v)
                                case "list":
                                    self.__dict__[k] = [int(i) for i in v.split()]
                                case _:
                                    log.error(f"type {basic_info[k]} for {k} cannot be parsed")
                                    raise NotImplementedError()
                            iline += 1
                        for k in basic_info:
                            if k not in self:
                                log.warning(f"{k} not read from QMout!")
                                pass
                        self.nmstates = sum((i + 1) * n for i, n in enumerate(self.states))
                        self.nstates = sum(self.states)
                        self.point_charges = self.npc > 0
                        continue
                    case 1: # h
                        self.h, iline = QMout.get_quantity(data, iline, complex, (self.nmstates, self.nmstates))
                    case 2: # dm
                        self.dm, iline = QMout.get_quantity(data, iline, complex, (3, self.nmstates, self.nmstates))
                    case 3: # grad
                        self.grad, iline = QMout.get_quantity(data, iline, float, (self.nmstates, self.natom, 3))
                    case 30 if self.point_charges: # grad_pc
                        self.grad_pc, iline = QMout.get_quantity(data, iline, float, (self.nmstates, self.npc, 3))
                    case 5: # nacdr
                        self.nacdr, iline = QMout.get_quantity(data, iline, float, (self.nmstates, self.nmstates, self.natom, 3))
                    case 31 if self.point_charges: # nacdr_pc
                        self.nacdr_pc, iline = QMout.get_quantity(data, iline, float, (self.nmstates, self.nmstates, self.npc, 3))
                    case 6: # overlap
                        self.overlap, iline = QMout.get_quantity(data, iline, complex, (self.nmstates, self.nmstates))
                    case 7: # phases
                        self.phases, iline = QMout.get_quantity(data, iline, complex, (self.nmstates,))
                    case 13: # socdr
                        self.socdr, iline = QMout.get_quantity(data, iline, complex, (self.nmstates, self.nmstates, self.natom, 3))
                    case 33 if self.point_charges:
                        self.socdr_pc, iline = QMout.get_quantity(
                            data,
                            iline,
                            complex,
                            (self.nmstates, self.nmstates, self.npc, 3),
                        )
                    case 12: # dmdr
                        self.dmdr, iline = QMout.get_quantity(data, iline, float, (self.nmstates, self.nmstates, self.natom, 3))
                    case 32 if self.point_charges:
                        self.dmdr_pc, iline = QMout.get_quantity(
                            data,
                            iline,
                            float,
                            (3, self.nmstates, self.nmstates, self.npc, 3),
                        )
                    case 22: # multipolar_fit
                        self.multipolar_fit, iline = QMout.get_multipoles(data, iline, self.charges, shape)
                        if data[iline].find("settings") != -1:
                            self.notes["multipolar_fit"] = data[iline][data[iline].find("settings"):-1]
                    case 24: # Densities
                        self.density_matrices, iline = QMout.get_densities(data, iline, self.charges, shape)
                    case 23: # prop0d
                        self.prop0d, iline = QMout.get_property(data, iline, float, ())
                    case 25:
                        self.mol, iline  = QMout.get_mol(data, iline)
                    case 21: # prop1d
                        self.prop1d, iline = QMout.get_property(data, iline, float, (self.nmstates,))
                    case 20: # prop2d
                        self.prop2d, iline = QMout.get_property(data, iline, float, (self.nmstates, self.nmstates))
                    case 8: # runtime
                        self.runtime, iline = QMout.get_quantity(data, iline, float, ())
                    case 999: # notes
                        self.notes, iline = QMout.get_notes(data, iline)  
                        break  # as we do not know how many lines the notes are, we are not reading the QM.out file after the notes
                    case _:
                        iline += 1
                        log.warning(f"Warning!: property with flag {flag} not yet implemented in QMout class")
                line = f.readline()
            f.close()

    @staticmethod
    def find_line(data, flag):
        iline = 0
        for iline, line in enumerate(data):
            if line.startswith("! %i" % flag):
                return iline
        return None

    @staticmethod
    def get_quantity(data, iline, type, shape):
        log.debug(f"Parsing: {data[iline]}")
        if len(shape) == 0:
            iline += 1
            line = data[iline].split()
            if type == complex:
                result = complex(float(line[0]), float(line[1]))
            elif type == float:
                result = float(line[0])
            return result, iline
        else:
            result = np.zeros(shape=shape, dtype=type)
        if len(shape) == 1:
            iline += 2
            for irow in range(shape[0]):
                line = data[iline + irow].split()
                if type == complex:
                    result[irow] = complex(float(line[0]), float(line[1]))
                elif type == float:
                    result[irow] = float(line[0])
        elif len(shape) == 2:
            iline += 2
            for irow in range(shape[0]):
                line = data[iline + irow].split()
                if type == complex:
                    result[irow, :] = np.fromiter((complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(shape[1])),
                                                  dtype=complex, count=shape[1])
                elif type == float:
                    result[irow, :] = np.array([float(line[i]) for i in range(shape[1])])
        elif len(shape) == 3:
            iline += 2
            for iblock in range(shape[0]):
                for irow in range(shape[1]):
                    line = data[iline + irow].split()
                    if type == complex:
                        result[iblock, irow, :] = np.fromiter(
                            (complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(shape[2])),
                            dtype=complex,
                            count=shape[2]
                        )
                    elif type == float:
                        result[iblock, irow, :] = np.array([float(line[i]) for i in range(shape[2])])
                iline += 1 + shape[1]
        elif len(shape) == 4:
            iline += 2
            for iblock in range(shape[0]):
                for jblock in range(shape[1]):
                    for irow in range(shape[2]):
                        line = data[iline + irow].split()
                        if type == complex:
                            result[iblock, jblock, irow, :] = np.array(
                                [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(shape[3])]
                            )
                        elif type == float:
                            result[iblock, jblock, irow, :] = np.array([float(line[i]) for i in range(shape[3])])
                    iline += 1 + shape[2]
        # elif len(targets[t]["dim"]) == 4:
            # for iblocks in range(targets[t]["dim"][0]):
                # sblock = []
                # for jblocks in range(targets[t]["dim"][1]):
                    # iline += 1
                    # block = []
                    # for irow in range(targets[t]["dim"][2]):
                        # iline += 1
                        # line = lines[iline].split()
                        # if targets[t]["type"] == complex:
                            # row = [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(targets[t]["dim"][3])]
                        # elif targets[t]["type"] == float:
                            # row = [float(line[i]) for i in range(targets[t]["dim"][3])]
                        # else:
                            # row = line
                        # block.append(row)
                    # sblock.append(block)
                # values.append(sblock)
        elif len(shape) == 5:
            iline += 2
            for _ in range(shape[0]):
                for _ in range(shape[1]):
                    for iblock in range(shape[2]):
                        for irow in range(shape[3]):
                            line = data[iline + irow].split()
                            if type == complex:
                                result[iblock, irow, :] = np.array(
                                    [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(shape[4])]
                                )
                            elif type == float:
                                result[iblock, irow, :] = np.array([float(line[i]) for i in range(shape[4])])
                        iline += 1 + shape[3]
        if len(shape) in [3, 4, 5]:
            iline -= 1
        return result, iline

    @staticmethod                                   
    def get_notes(data, iline):                     
        num = int(data[iline+1].split()[0])         
        # currently only skipping                   
        toskip = 4 + 3*num                          
        return {'Notes': 'not read'}, iline + toskip
        # TODO: actually read in the notes as dict. Readig should stop at the first empty line

    @staticmethod
    def get_property(data, iline, type, shape):
        num = int(data[iline + 1].split()[0])
        keys = []
        for irow in range(num):
            keys.append(data[iline + 3 + irow].strip())
        iline += 4 + num
        res = []
        for irow in range(num):
            res.append(QMout.get_quantity(data, iline, type, shape)[0])
            iline += 2
        result = [(keys[i], res[i]) for i in range(num)]
        return result, iline - 1

    @staticmethod
    def get_multipoles(data, iline, charges, shape):
        res = {}
        # shape = [int(s) for s in data[iline].split()[-1][1:-1].split("x")]
        iline += 1
        for i in range(shape[0]):
            tmp = data[iline].split()
            y, z, m1, s1, ms1, m2, s2, ms2 = int(tmp[0]), int(tmp[1]), int(tmp[4]), int(tmp[6]), int(float(tmp[8])), int(tmp[10]), int(tmp[12]), int(float(tmp[14]))
            if y != shape[1] or z != shape[2]:
                log.error(f"shapes do not match {shape} vs {shape[0]}x{y}x{z}")
                raise ValueError()
            s1 = electronic_state(Z=charges[m1], S=m1, M=ms1, N=s1)
            s2 = electronic_state(Z=charges[m2], S=m2, M=ms2, N=s2)
            res[(s1, s2)] = np.fromiter(chain(*map(lambda x: map(float, x.split()), data[iline + 1: iline + 1 + y])), dtype=float,
                                        count=y*z).reshape((y,z))
            iline += y + 1
        return res, iline - 1

    @staticmethod
    def get_densities(data, iline, charges, shape):
        res = {}
        Nao, Nrho = int(shape[1]), int(shape[0])
        for d in range(Nrho):
            line = data[iline+d*(Nao+1)+1]
            if line[0] == '<':
                state1, spin, state2 = line.split('|')
                spin = spin.strip()
                s1, m1, n1 = (i.split()[-1] for i in state1.split(','))
                s2, m2, n2 = (i.split()[-1] for i in state2.strip()[:-1].split(','))
            else:
                state1, state2, spin = line.split('|')
                spin = spin.strip()
                s1, m1, n1 = state1.split(',')
                s2, m2, n2 = state2.split(',')

            s1 = int(2*float(s1))
            s2 = int(2*float(s2))
            m1 = int(2*float(m1))
            m2 = int(2*float(m2))
            n1 = int(n1)
            n2 = int(n2)
            state1 = electronic_state(Z=charges[s1], S=s1, M=m1, N=n1)
            state2 = electronic_state(Z=charges[s2], S=s2, M=m2, N=n2)
            rho = np.zeros((Nao,Nao))
            for i in range(Nao):
                row = data[iline+d*(Nao+1)+2+i].split()
                row = np.array([ float(r) for r in row ])
                rho[i,:] = row
            res[(state1,state2,spin)] = rho
        iline += Nao*Nrho
        return res, iline


    @staticmethod
    def get_mol(data, iline):
        mol = pyscf.gto.Mole.unpack(ast.literal_eval(data[iline+1].replace("'",'"'))) 
        mol.build()
        return mol, iline + 2

    def allocate(self, states=[], natom=0, npc=0, requests: set[str] = set()):
        self.nmstates = sum((i + 1) * n for i, n in enumerate(states))
        self.nstates = sum(states)
        self.states = states
        self.runtime = 0
        self.notes = {}
        self.natom = natom
        self.npc = npc
        self.point_charges = npc > 0
        if "h" in requests or "soc" in requests:
            self.h = np.zeros((self.nmstates, self.nmstates), dtype=complex)
        if "dm" in requests:
            self.dm = np.zeros((3, self.nmstates, self.nmstates), dtype=float)
        if "grad" in requests:
            self.grad = np.zeros((self.nmstates, natom, 3), dtype=float)
            if self.point_charges:
                self.grad_pc = np.zeros((self.nmstates, npc, 3), dtype=float)
        if "nacdr" in requests:
            self.nacdr = np.zeros((self.nmstates, self.nmstates, natom, 3), dtype=float)
            if self.point_charges:
                self.nacdr_pc = np.zeros((self.nmstates, self.nmstates, npc, 3), dtype=float)
        if "overlap" in requests:
            self.overlap = np.zeros((self.nmstates, self.nmstates), dtype=float)
        if "phases" in requests:
            self.phases = np.zeros((self.nmstates), dtype=float)
        self.prop0d = []
        self.prop1d = []
        self.prop2d = []

        if "socdr" in requests:
            self.socdr = np.zeros((self.nmstates, self.nmstates, natom, 3), dtype=complex)
            if self.point_charges:
                self.socdr_pc = np.zeros((self.nmstates, self.nmstates, npc, 3), dtype=complex)
        if "dmdr" in requests:
            self.dmdr = np.zeros((3, self.nmstates, self.nmstates, natom, 3), dtype=float)
            if self.point_charges:
                self.dmdr_pc = np.zeros((3, self.nmstates, self.nmstates, npc, 3), dtype=float)
        if "multipolar_fit" in requests:
            self.multipolar_fit = {}#np.zeros((self.nmstates, self.nmstates, natom, 10), dtype=float)
        if "density_matrices" in requests:
            self.density_matrices = {}
        self.mol = None 


    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key not in QMout.__dict__["__annotations__"].keys():
            raise KeyError
        else:
            return None

    def __setitem__(self, key, value):
        if key not in QMout.__dict__["__annotations__"].keys():
            raise KeyError(f"{key} not a valid entry in QMout! (Bug)")
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __str__(self):
        string = ""
        for k, v in self.__dict__.items():
            string += f"{k}:\n{v}\n\n"
        return string

    def items(self):
        return self.__dict__.items()





    # =============================================================================================== #
    # =============================================================================================== #
    # =========================================== QMout writing ===================================== #
    # =============================================================================================== #
    # =============================================================================================== #

    def write(self, filename, requests):
        """Writes the requested quantities to the file which SHARC reads in.
        The filename is QMinfilename with everything after the first dot replaced by "out".

        Arguments:
        1 dictionary: filename usually QM.out
        2 set: set of requests
        """

        string = ""
        # write basic info
        string += "! 0 Basic information\n"
        string += "states " + " ".join([str(i) for i in self.states]) + "\n"
        string += f"nmstates {self.nmstates}\n"
        string += f"natom {self.natom}\n"
        string += f"npc {self.npc}\n"
        string += "charges " + " ".join(map(str, self.charges))
        string += "\n"
        # write data
        if requests["soc"] or requests["h"]:
            string += self.writeQMoutsoc()
        if requests["dm"]:
            string += self.writeQMoutdm()
        if requests["grad"]:
            string += self.writeQMoutgrad()
            if self.point_charges:
                string += self.writeQMoutgrad_pc()
        if requests["overlap"]:
            string += self.writeQMoutnacsmat()
        if requests["nacdr"]:
            string += self.writeQMoutnacana()
            if self.point_charges:
                string += self.writeQMoutnacana_pc()
        if requests["socdr"]:
            string += self.writeQMoutsocdr()
            if self.point_charges:
                string += self.writeQMoutsocdr_pc()
        if requests["dmdr"]:
            string += self.writeQMoutdmdr()
            if self.point_charges:
                string += self.writeQMoutdmdr_pc()
        if self.prop0d:
            string += self.writeQMoutprop0d()
        if self.prop1d:
            string += self.writeQMoutprop1d()
        if self.prop2d:
            string += self.writeQMoutprop2d()
        if requests["phases"]:
            string += self.writeQmoutPhases()
        if requests["multipolar_fit"]:
            string += self.writeQMoutmultipolarfit()
        if requests["density_matrices"]:
            pass
            # string += self.writeQMoutDensityMatrices()
        if requests["dyson_orbitals"]:
            string += self.writeQMoutDysonOrbitals()
        if "mol" in requests and requests["mol"]:
            string += self.writeQMoutMole()

        if self.notes:
            string += self.writeQMoutnotes()
        string += self.writeQMouttime()
        writefile(filename, string)
        return

    # ======================================================================= #

    def writeQMoutsoc(self):
        """Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix"""
        nmstates = self.nmstates
        string = ""
        string += "! %i Hamiltonian Matrix (%ix%i, complex)\n" % (1, nmstates, nmstates)
        string += "%i %i\n" % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += "%s %s " % (
                    eformat(self.h[i][j].real, 12, 3),
                    eformat(self.h[i][j].imag, 12, 3),
                )
            string += "\n"
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutdm(self):
        """Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line. The string contains three such matrices.

        Returns:
        1 string: multiline string with the DM matrices"""
        nmstates = self.nmstates
        string = ""
        string += "! %i Dipole Moment Matrices (3x%ix%i, complex)\n" % (
            2,
            nmstates,
            nmstates,
        )
        for xyz in range(3):
            string += "%i %i\n" % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += "%s %s " % (
                        eformat(self.dm[xyz][i][j].real, 12, 3),
                        eformat(self.dm[xyz][i][j].imag, 12, 3),
                    )
                string += "\n"
            string += ""
        string += "\n"
        return string

    # ======================================================================= #
    def writeQMoutdmdr(self):
        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ""
        string += "! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n" % (
            12,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                for ipol in range(3):
                    string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n" % (
                        natom,
                        3,
                        imult,
                        istate,
                        ims,
                        jmult,
                        jstate,
                        jms,
                        ipol,
                    )
                    for atom in range(natom):
                        for xyz in range(3):
                            string += "%s " % (eformat(self.dmdr[ipol][i][j][atom][xyz], 12, 3))
                        string += "\n"
                    string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    # ======================================================================= #
    def writeQMoutdmdr_pc(self):
        states = self.states
        nmstates = self.nmstates
        natom = self.npc
        string = ""
        string += "! %i Dipole moment derivatives on point charges (%ix%ix3x%ix3, real)\n" % (
            32,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                for ipol in range(3):
                    string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n" % (
                        natom,
                        3,
                        imult,
                        istate,
                        ims,
                        jmult,
                        jstate,
                        jms,
                        ipol,
                    )
                    for atom in range(natom):
                        for xyz in range(3):
                            string += "%s " % (eformat(self.dmdr_pc[ipol][i][j][atom][xyz], 12, 3))
                        string += "\n"
                    string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutsocdr(self):
        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ""
        string += "! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n" % (
            13,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n" % (
                    natom,
                    3,
                    imult,
                    istate,
                    ims,
                    jmult,
                    jstate,
                    jms,
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += "%s %s " % (
                            eformat(self.socdr[i][j][atom][xyz].real, 12, 3),
                            eformat(self.socdr[i][j][atom][xyz].imag, 12, 3),
                        )
                string += "\n"
                string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutsocdr_pc(self):
        states = self.states
        nmstates = self.nmstates
        natom = self.npc
        string = ""
        string += "! %i Spin-Orbit coupling derivatives on point charges (%ix%ix3x%ix3, complex)\n" % (
            33,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n" % (
                    natom,
                    3,
                    imult,
                    istate,
                    ims,
                    jmult,
                    jstate,
                    jms,
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += "%s %s " % (
                            eformat(self.socdr_pc[i][j][atom][xyz].real, 12, 3),
                            eformat(self.socdr_pc[i][j][atom][xyz].imag, 12, 3),
                        )
                string += "\n"
                string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutgrad(self):
        """Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the Gradient vectors"""

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ""
        string += "! %i Gradient Vectors (%ix%ix3, real)\n" % (3, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += "%i %i ! m1 %i s1 %i ms1 %i\n" % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    string += "%s " % (eformat(self.grad[i][atom][xyz], 12, 3))
                string += "\n"
            string += ""
            i += 1
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutgrad_pc(self):
        """Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Returns:
        1 string: multiline string with the Gradient vectors"""

        states = self.states
        nmstates = self.nmstates
        npc = self.npc
        string = ""
        string += "! %i Point Charge Gradient Vectors (%ix%ix3, real)\n" % (
            30,
            nmstates,
            npc,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += "%i %i ! m1 %i s1 %i ms1 %i\n" % (npc, 3, imult, istate, ims)
            for atom in range(npc):
                for xyz in range(3):
                    string += "%s " % (eformat(self.grad_pc[i][atom][xyz], 12, 3))
                string += "\n"
            string += ""
            i += 1
        string += "\n"
        return string

    # ======================================================================= #

    # def writeQMoutnacnum(self):
    #     """Generates a string with the NAC matrix in SHARC format.

    #     The string starts with a ! followed by a flag specifying the type of data.
    #     In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
    #     Blocks are separated by a blank line.

    #     Returns:
    #     1 string: multiline string with the NAC matrix"""

    #     nmstates = self.nmstates
    #     string = ""
    #     string += "! %i Non-adiabatic couplings (ddt) (%ix%i, complex)\n" % (
    #         4,
    #         nmstates,
    #         nmstates,
    #     )
    #     string += "%i %i\n" % (nmstates, nmstates)
    #     for i in range(nmstates):
    #         for j in range(nmstates):
    #             string += "%s %s " % (
    #                 eformat(self.nacdt[i][j].real, 12, 3),
    #                 eformat(self.nacdt[i][j].imag, 12, 3),
    #             )
    #         string += "\n"
    #     string += ""
    #     # also write wavefunction phases
    #     string += "! %i Wavefunction phases (%i, complex)\n" % (7, nmstates)
    #     for i in range(nmstates):
    #         string += "%s %s\n" % (eformat(self.phases[i], 12, 3), eformat(0.0, 12, 3))
    #     string += "\n\n"
    #     return string

    # ======================================================================= #

    def writeQMoutnacana(self):
        """Generates a string with the NAC vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

        Returns:
        1 string: multiline string with the NAC vectors"""

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ""
        string += "! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n" % (
            5,
            nmstates,
            nmstates,
            natom,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n" % (
                    natom,
                    3,
                    imult,
                    istate,
                    ims,
                    jmult,
                    jstate,
                    jms,
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += "%s " % (eformat(self.nacdr[i][j][atom][xyz], 12, 3))
                    string += "\n"
                string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    def writeQMoutnacana_pc(self):
        """Generates a string with the NAC vectors of point charges in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

        Returns:
        1 string: multiline string with the NAC vectors"""

        states = self.states
        nmstates = self.nmstates
        npc = self.npc
        string = ""
        string += "! %i Non-adiabatic couplings on point charges (ddr) (%ix%ix%ix3, real)\n" % (
            31,
            nmstates,
            nmstates,
            npc,
        )
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += "%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n" % (
                    npc,
                    3,
                    imult,
                    istate,
                    ims,
                    jmult,
                    jstate,
                    jms,
                )
                for atom in range(npc):
                    for xyz in range(3):
                        string += "%s " % (eformat(self.nacdr_pc[i][j][atom][xyz], 12, 3))
                    string += "\n"
                string += ""
                j += 1
            i += 1
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutnacsmat(self):
        """Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the transformation matrix"""

        nmstates = self.nmstates
        string = ""
        string += "! %i Overlap matrix (%ix%i, complex)\n" % (6, nmstates, nmstates)
        string += "%i %i\n" % (nmstates, nmstates)
        for j in range(nmstates):
            for i in range(nmstates):
                string += "%s %s " % (
                    eformat(self.overlap[j][i].real, 12, 3),
                    eformat(self.overlap[j][i].imag, 12, 3),
                )
            string += "\n"
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMouttime(self):
        """Generates a string with the quantum mechanics total runtime in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the runtime is given.

        Returns:
        1 string: multiline string with the runtime"""

        string = "! 8 Runtime\n%s\n" % (eformat(self.runtime, 9, 3))
        return string

    # ======================================================================= #

    def writeQMoutprop0d(self):
        """Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix"""

        prop0d = self.prop0d
        string = "! %i Property Scalars\n" % (23)
        string += "%i    ! number of property scalars\n" % (len(prop0d))

        string += "! Property Scalar Labels (%i strings)\n" % (len(prop0d))
        for element in prop0d:
            string += element[0] + "\n"

        string += "! Property Scalars (%i, real)\n" % (len(prop0d))
        for ie, element in enumerate(prop0d):
            string += "! %i %s\n" % (ie, element[0])
            string += "%s\n" % (eformat(element[1], 12, 3),)
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutprop1d(self):
        """Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix"""

        prop1d = self.prop1d
        nmstates = self.nmstates
        string = "! %i Property Vectors\n" % (21)
        string += "%i    ! number of property vectors\n" % (len(prop1d))

        string += "! Property Vector Labels (%i strings)\n" % (len(prop1d))
        for element in prop1d:
            string += element[0] + "\n"

        string += "! Property Vectors (%ix%i, real)\n" % (len(prop1d), nmstates)
        for ie, element in enumerate(prop1d):
            string += "! %i %s\n" % (ie, element[0])
            for i in range(nmstates):
                string += "%s\n" % (eformat(element[1][i], 12, 3),)
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutprop2d(self):
        """Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix"""

        prop2d = self.prop2d
        nmstates = self.nmstates
        string = "! %i Property Matrices\n" % (20)
        string += "%i    ! number of property matrices\n" % (len(prop2d))

        string += "! Property Matrix Labels (%i strings)\n" % (len(prop2d))
        for element in prop2d:
            string += element[0] + "\n"

        string += "! Property Matrices (%ix%ix%i, complex)\n" % (
            len(prop2d),
            nmstates,
            nmstates,
        )
        for element in prop2d:
            string += "%i %i   ! %s\n" % (nmstates, nmstates, element[0])
            for i in range(nmstates):
                for j in range(nmstates):
                    string += "%s %s " % (
                        eformat(element[1][i][j].real, 12, 3),
                        eformat(element[1][i][j].imag, 12, 3),
                    )
                string += "\n"
            string += "\n"
        string += "\n"
        return string

    # ======================================================================= #

    def writeQMoutnotes(self):
        """Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix"""

        notes = self.notes
        string = "! %i Notes\n" % (24)
        string += "%i    ! number of notes\n" % (len(notes))

        string += "! Notes Labels (%i strings)\n" % (len(notes))
        for element in notes:
            string += element + "\n"

        string += "! Notes (%i, real)\n" % (len(notes))
        for ie, element in enumerate(notes):
            string += "! index %i %s\n" % (ie, element)
            string += "%s\n" % (notes[element])
        string += "\n"
        return string

    # ======================================================================= #

    def writeQmoutPhases(self):
        string = "! 7 Wave function phases (%ix1, complex)\n%i\n" % (self.nmstates, self.nmstates)
        for i in range(self.nmstates):
            string += "%s %s\n" % (
                eformat(self.phases[i].real, 9, 3),
                eformat(self.phases[i].imag, 9, 3),
            )
        return string

    # ======================================================================= #

    # Start TOMI
    def writeQMoutDensityMatrices(self) -> str:
        nao = self.mol.nao
        setting_str = "S1,M1,N1|S2,M2,N2|spin"
        if "density_matrices" in self.notes:
            setting_str = self.notes["density_matrices"]
        nrho = len(self.density_matrices.values())
        string = (
            f"! 24 Total/Spin/Partial 1-particle density matrices in AO-product basis ({nrho}x{nao}x{nao}) {setting_str}\n"
        )
        for key, rho in self.density_matrices.items():
            s1, s2, spin = key
            string += f" {s1.S/2: 3.1f},{s1.M/2: 3.1f},{s1.N}|{s2.S/2: 3.1f},{s2.M/2: 3.1f},{s2.N}|{spin} \n"
            for i in range(nao):
                string += ' '.join(map(lambda j: f"{float(rho[i,j]): 15.12f}", range(nao)))
                string += "\n"
        return string

    def writeQMoutMole(self) -> str:
        string = (
            "! 25 Mole PySCF object (dict, 1 line)\n"
        )
        string += str(pyscf.gto.Mole.pack(self.mol)) + '\n'
        return string

    def writeQMoutDysonOrbitals(self) -> str:
        setting_str = ""
        if "dyson_orbitals" in self.notes:
            setting_str = self.notes["dyson_orbitals"]
        nphi = len(self.dyson_orbitals.values())
        nao = self.mol.nbas
        string = (
            f"! 25 Dyson orbitals in AO basis ({nao}x{nao}x{nphi}) {setting_str}\n"
        )
        for key, rho in self.density_matrices.items():
            s1, s2, spin = key
            string += f"{nao}x{nao} ! S1 = {s1.S/2: 3.1f}, MS1 = {s1.M/2: 3.1f}, N1 = {s1.N}; S2 = {s2.S/2: 3.1f}, MS2 = {s2.M/2: 3.1f}, N2 = {s2.N}; {spin} \n"
            for i in range(nao):
                string += ' '.join( [ "{rho[i,j]: 15.12f}" for j in range(nao) ] )
                string += "\n"
        return string

    def writeQMoutmultipolarfit(self) -> str:
        """Generates a string with the fitted RESP charges for each pair of states specified.

        The string starts with a! followed by a flag specifying the type of data.
        Each line starts with the atom number (starting at 1), state i and state j.
        If i ==j: fit for single state, else fit for transition multipoles.
        One line per atom and a blank line at the end.

        Returns:
        1 string: multiline string with the Gradient vectors"""

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        setting_str = ""
        if "multipolar_fit" in self.notes:
            setting_str = self.notes["multipolar_fit"]
        sorted_states = sorted(self.multipolar_fit.keys(), key=lambda x: (x[0].S, x[0].N, x[0].M, x[1].S, x[1].N, x[1].M))
        string = (
            f"! 22 Atomwise multipolar density representation fits for states ({len(sorted_states)}x{natom}x10) {setting_str}\n"
        )

        for (s1, s2) in sorted_states:
            val = self.multipolar_fit[(s1, s2)]
            istate, imult, ims = s1.N, s1.S, s1.M
            jstate, jmult, jms = s2.N, s2.S, s2.M

            string += f"{natom} 10 ! m1 {imult} s1 {istate} ms1 {ims: 3.1f}   m2 {jmult} s2 {jstate} ms2 {jms: 3.1f}\n"
            string += (
                "\n".join(
                    map(
                        lambda x: " ".join(map(lambda y: "{: 10.8f}".format(y), x)),
                        val,
                    )
                )
                + "\n"
            )
        string += "\n"
        return string

    # ======================================================================= #

    def printQMout(self, QMin, DEBUG=False):
        print(self.formatQMout(QMin, DEBUG=DEBUG))

    def formatQMout(self, QMin, DEBUG=False):
        """If PRINT, prints a summary of all requested QM output values.
        Matrices are formatted using printcomplexmatrix, vectors using printgrad.
        """

        states = QMin.molecule["states"]
        nmstates = QMin.molecule["nmstates"]
        natom = QMin.molecule["natom"]

        string = ""
        string += "===> Results:\n\n"
        # Hamiltonian matrix, real or complex
        if QMin.requests["h"] or QMin.requests["soc"]:
            eshift = math.ceil(self["h"][0][0].real)
            string += "=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f\n" % (eshift)
            en = self.h.copy()
            np.einsum("ii->i", en)[:] -= eshift
            string += formatcomplexmatrix(en, states)
            string += "\n"
        # Dipole moment matrices
        if QMin.requests["dm"]:
            string += "=> Dipole Moment Matrices:\n\n"
            for xyz in range(3):
                string += "Polarisation %s:\n" % (IToPol[xyz])
                matrix = self["dm"][xyz]
                string += formatcomplexmatrix(matrix, states)
            string += "\n"
        # Gradients
        if QMin.requests["grad"]:
            string += "=> Gradient Vectors:\n\n"
            istate = 0
            for imult, i, ms in itnmstates(states):
                string += "%s\t%i\tMs= % .1f:\n" % (IToMult[imult], i, ms)
                string += formatgrad(
                    self["grad"][istate],
                    natom,
                    QMin.molecule["elements"],
                    DEBUG,
                )
                istate += 1
            string += "\n"
        # Nonadiabatic coupling vectors
        if QMin.requests["nacdr"]:
            string += "=> Nonadiabatic Coupling Vectors:\n\n"
            istate = 0
            for imult, i, ims in itnmstates(states):
                jstate = 0
                for jmult, j, jms in itnmstates(states):
                    if imult == jmult and ims == jms:
                        string += "%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:\n" % (IToMult[imult], i, ims, IToMult[jmult], j, jms)
                        string += formatgrad(self["nacdr"][istate][jstate], natom, QMin.molecule["elements"], DEBUG)
                    jstate += 1
                istate += 1
            string += "\n"
        # Overlaps
        if QMin.requests["overlap"]:
            string += "=> Overlap matrix:\n\n"
            matrix = self["overlap"]
            string += formatcomplexmatrix(matrix, states)
            if QMin.requests["phases"]:
                string += "=> Wavefunction Phases:\n\n"
                for i in range(nmstates):
                    string += "% 3.1f % 3.1f" % (
                        self["phases"][i].real,
                        self["phases"][i].imag,
                    )
                string += "\n"
            string += "\n"
        # Spin-orbit coupling derivatives
        if QMin.requests["socdr"]:
            string += "=> Spin-Orbit Gradient Vectors:\n\n"
            istate = 0
            for imult, i, ims in itnmstates(states):
                jstate = 0
                for jmult, j, jms in itnmstates(states):
                    string += "%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:\n" % (IToMult[imult], i, ims, IToMult[jmult], j, jms)
                    string += formatgrad(self["socdr"][istate][jstate], natom, QMin.molecule["elements"], DEBUG)
                    jstate += 1
                istate += 1
            string += "\n"
        # Dipole moment derivatives
        if QMin.requests["dmdr"]:
            string += "=> Dipole moment derivative vectors:\n\n"
            istate = 0
            for imult, i, msi in itnmstates(states):
                jstate = 0
                for jmult, j, msj in itnmstates(states):
                    if imult == jmult and msi == msj:
                        for ipol in range(3):
                            string += "%s\tStates %i - %i\tMs= % .1f\tPolarization %s:\n" % (
                                IToMult[imult],
                                i,
                                j,
                                msi,
                                IToPol[ipol],
                            )
                            string += formatgrad(
                                self["dmdr"][ipol][istate][jstate],
                                natom,
                                QMin.molecule["elements"],
                                DEBUG,
                            )
                    jstate += 1
                istate += 1
            string += "\n"
        # Property matrices
        if self["prop2d"]:
            string += "=> Property matrices:\n\n"
            for element in self["prop2d"]:
                string += f'Matrix with label "{element[0]}"\n'
                string += formatcomplexmatrix(element[1], states)
            string += "\n"
        # Property vectors
        if self["prop1d"]:
            string += "=> Property vectors:\n\n"
            string += "%5s " % ("State")
            for j in range(len(self["prop1d"])):
                string += "%12s " % self["prop1d"][j][0]
            string += "\n"
            for i in range(nmstates):
                string += "% 5i " % (i + 1)
                for j in range(len(self["prop1d"])):
                    string += "%12.9f " % self["prop1d"][j][1][i]
                string += "\n"
            string += "\n"
        # Property scalars
        if self["prop0d"]:
            string += "=> Property scalars:\n\n"
            for element in self["prop0d"]:
                string += f"{element[0]} {element[1]}\n"
            string += "\n"
        # Multipolar fit
        if QMin.requests["multipolar_fit"]:
            string += "=> Multipolar fit:\n\n"
            for (s1, s2), val in sorted(self["multipolar_fit"].items()):
                istate, imult, ims = s1.N, s1.S, s1.M
                jstate, jmult, jms = s2.N, s2.S, s2.M
                if imult == jmult and ims == jms:
                    string += "%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:\n" % (
                        IToMult[imult + 1],
                        istate,
                        ims,
                        IToMult[jmult + 1],
                        jstate,
                        jms,
                    )
                    string += formatgrad(val, natom, QMin.molecule["elements"], DEBUG)
            string += "\n"

        return string


if __name__ == "__main__":
    test = QMout()
    test.allocate([1], 1, 1, set(["h"]))
    print(test.formatQMout())
