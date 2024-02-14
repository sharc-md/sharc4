import math
from copy import deepcopy

import numpy as np
from constants import IToMult, IToPol
from numpy import ndarray
from printing import formatcomplexmatrix, formatgrad
from utils import eformat, itnmstates, writefile
from logger import log
import pyscf
#import SHARC_INTERFACE


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
    h: ndarray[complex, 2]
    dm: ndarray[float, 3]
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
    multipolar_fit: ndarray[float, 4]
    density_matrices: dict
    mol: pyscf.gto.Mole 
    #dyson_orbitals: dict[tuple(electronic_state,electronic_state,str), ndarray[float,1] ]

    def __init__(self, filepath=None, states: list[int] =None, natom: int =None, npc: int=None):
        self.prop0d = []
        self.prop1d = []
        self.prop2d = []
        self.notes = {}
        self.runtime = 0
        if states is not None and natom is not None and npc is not None:
            self.states = states
            self.natom = natom
            self.npc = npc
            self.nmstates = sum((i + 1) * n for i, n in enumerate(self.states))
            self.nstates = sum(self.states)
            self.point_charges = self.npc > 0
        if filepath is not None:
            # initialize the entire object from a QM.out file

            log.debug(f"Reading file {filepath}")
            try:
                f = open(filepath, "r", encoding="utf-8")
                data = f.readlines()
                f.close()
            except IOError:
                raise IOError("'Could not find %s!' % (filepath)")
            log.debug(f"Done raw reading {filepath}")
            # get basic information
            # set from input
            iline = 0
            while iline < len(data):
                # skip to next flag
                if not data[iline].startswith("! "):
                    iline += 1
                    continue
                # get flag
                flag = int(data[iline].split()[1])
                log.debug(f"Parsing flag: {flag}")
                match flag:
                    case 0: # basis info
                        iline += 1
                        if "states" in data[iline]:
                            s = data[iline].split()
                            self.states = [int(i) for i in s[1:]]
                            iline += 2
                        else:
                            raise KeyError(f"Could not find states in {filepath}")
                        if "natom" in data[iline]:
                            self.natom = int(data[iline].split()[-1])
                            iline += 1
                        else:
                            raise KeyError(f"Could not find natom in {filepath}")
                        if "npc" in data[iline]:
                            self.npc = int(data[iline].split()[-1])
                            iline += 1
                        else:
                            raise KeyError(f"Could not find npc in {filepath}")
                        iline += 1
                        self.nmstates = sum((i + 1) * n for i, n in enumerate(self.states))
                        self.nstates = sum(self.states)
                        self.point_charges = self.npc > 0
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
                        self.multipolar_fit, iline = QMout.get_quantity(data, iline, float, (self.nmstates, self.nmstates, self.natom, 10))
                        if data[iline].find("settings") != -1:
                            self.notes["multipolar_fit"] = data[iline][data[iline].find("settings"):-1]
                    case 23: # prop0d
                        self.prop0d, iline = QMout.get_property(data, iline, float, ())
                    case 21: # prop1d
                        self.prop1d, iline = QMout.get_property(data, iline, float, (self.nmstates,))
                    case 20: # prop2d
                        self.prop2d, iline = QMout.get_property(data, iline, float, (self.nmstates, self.nmstates))
                    case 8: # runtime
                        self.runtime, iline = QMout.get_quantity(data, iline, float, ())
                    case _:
                        log.warning(f"Warning!: property with flag {flag} not yet implemented in QMout class")

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
                    result[irow, :] = np.array([complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(shape[1])])
                elif type == float:
                    result[irow, :] = np.array([float(line[i]) for i in range(shape[1])])
        elif len(shape) == 3:
            iline += 2
            for iblock in range(shape[0]):
                for irow in range(shape[1]):
                    line = data[iline + irow].split()
                    if type == complex:
                        result[iblock, irow, :] = np.array(
                            [complex(float(line[2 * i]), float(line[2 * i + 1])) for i in range(shape[2])]
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
        if len(shape) in [3,4,5]:
            iline -= 1
        return result, iline

    @staticmethod
    def get_property(data, iline, type, shape):
        num = int(data[iline + 1].split()[0])
        keys = []
        for irow in range(num):
            keys.append(data[iline + 3 + irow].strip())
        iline += 3 + num
        res = []
        for irow in range(num):
            res.append(QMout.get_quantity(data, iline, type, shape))
            iline += 1 + shape[0]
        result = [(keys[i], res[i]) for i in range(num)]
        return result, iline - 1

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
            self.multipolar_fit = np.zeros((self.nmstates, self.nmstates, natom, 10), dtype=float)
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
            string += self.writeQMoutDensityMatrices()
        if requests["dyson_orbitals"]:
            string += self.writeQMoutDysonOrbitals()
        if requests["basis_set"]:
            string += self.writeQMoutBasisSet()

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
        string = "! 7 Phases\n%i ! for all nmstates\n" % (self.nmstates)
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
        setting_str = ""
        if "density_matrices" in self.notes:
            setting_str = self.notes["density_matrices"]
        nrho = len(self.density_matrices.values())
        string = (
            f"! 24 Total/Spin/Partial 1-particle density matrices in AO-product basis ({nao}x{nao}x{nrho}) {setting_str}\n"
        )
        for key, rho in self.density_matrices.items():
            s1, s2, spin = key
            string += f"<S1 = {s1.S/2: 3.1f}, MS1 = {s1.M/2: 3.1f}, N1 = {s1.N}| {spin} | S2 = {s2.S/2: 3.1f}, MS2 = {s2.M/2: 3.1f}, N2 = {s2.N}> \n"
            for i in range(nao):
                string += ' '.join( [ f"{float(rho[i,j]): 15.12f}" for j in range(nao) ] )
                string += "\n"
        return string

    def writeQMoutBasisSet(self) -> str:
        string = (
            f"! 25 Basis set in the PySCF format (dict, 1 line)\n"
        )
        string += str(self.mol.basis)+'\n'
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
        string = (
            f"! 22 Atomwise multipolar density representation fits for states ({nmstates}x{nmstates}x{natom}x10) {setting_str}\n"
        )

        sorted_states = sorted(self.multipolar_fit.keys(), key=lambda x: (x[0].S, x[0].N, x[0].M, x[1].S, x[1].N, x[1].M))
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
            matrix = deepcopy(self["h"])
            for i in range(nmstates):
                matrix[i][i] -= eshift
            string += formatcomplexmatrix(matrix, states)
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
            for (s1, s2), val in self["multipolar_fit"].items():
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
