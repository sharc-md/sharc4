
from numpy import ndarray
import numpy as np
from logger import log
from utils import writefile, eformat, itnmstates

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
    states: list[int]
    h: ndarray[complex, 2]
    dm: ndarray[float, 3]
    grad: ndarray[float, 3]
    grad_pc: ndarray[float, 3]
    nacdr: ndarray[float, 4]
    nacdr_pc: ndarray[float, 4]
    overlap: ndarray[float, 2]
    phases: ndarray[float]
    prop1d: list[tuple[str, ndarray[float]]]
    prop2d: list[tuple[str, ndarray[float, 2]]]
    socdr: ndarray[float, 4]
    socdr_pc: ndarray[float, 4]
    dmdr: ndarray[float, 5]
    dmdr_pc: ndarray[float, 5]
    multipolar_fit: ndarray[float, 4]

    def __init__(self):
        pass

    def allocate(self, states=[], natom=0, npc=0, requests: set[str] = set()):
        self.nmstates = sum((i + 1) * n for i, n in enumerate(states))
        self.nstates = sum(states)
        self.states = states
        self.runtime = 0
        self.natom = natom
        self.npc = npc
        self.point_charges = npc > 0
        if 'h' in requests or 'soc' in requests:
            self.h = np.zeros((self.nmstates, self.nmstates), dtype=complex)
        if 'dm' in requests:
            self.dm = np.zeros((3, self.nmstates, self.nmstates), dtype=float)
        if 'grad' in requests:
            self.grad = np.zeros((self.nmstates, natom, 3), dtype=float)
            if self.point_charges:
                self.grad_pc = np.zeros((self.nmstates, npc, 3), dtype=float)
        if 'nacdr' in requests:
            self.nacdr = np.zeros((self.nmstates, self.nmstates, natom, 3), dtype=float)
            if self.point_charges:
                self.nacdr_pc = np.zeros((self.nmstates, self.nmstates, npc, 3), dtype=float)
        if 'overlap' in requests:
            self.overlap = np.zeros((self.nmstates, self.nmstates), dtype=float)
        if 'phases' in requests:
            self.phases = np.zeros((self.nmstates), dtype=float)
        self.prop1d = []
        self.prop2d = []

        if 'socdr' in requests:
            self.socdr = np.zeros((self.nmstates, self.nmstates, natom, 3), dtype=complex)
            if self.point_charges:
                self.socdr_pc = np.zeros((self.nmstates, self.nmstates, npc, 3), dtype=complex)
        if 'dmdr' in requests:
            self.dmdr = np.zeros((3, self.nmstates, self.nmstates, natom, 3), dtype=float)
            if self.point_charges:
                self.dmdr_pc = np.zeros((3, self.nmstates, self.nmstates, npc, 3), dtype=float)
        if 'multipolar_fit' in requests:
            self.multipolar_fit = np.array((self.nmstates, self.nmstates, natom, 10), dtype=float)


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

    def __str__(self):
        string = ''
        for k, v in self.__dict__.items():
            string += f"{k}:\n{v}\n\n"
        return string


    #  def __str__(self):
# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #


    def write(self, filename, QMin):
        '''Writes the requested quantities to the file which SHARC reads in.
        The filename is QMinfilename with everything after the first dot replaced by "out".

        Arguments:
        1 dictionary: filename usually QM.out
        2 set: set of requests
        '''
        k = filename.rfind('.')
        if k == -1:
            outfilename = filename + '.out'
        else:
            outfilename = filename[:k] + '.out'
        log.print('===> Writing output to file %s in SHARC Format\n' % (outfilename))
        string = ''
        if QMin.requests["soc"] or QMin.requests["h"]:
            string += self.writeQMoutsoc()
        if QMin.requests["dm"]:
            string += self.writeQMoutdm()
        if QMin.requests["grad"]:
            string += self.writeQMoutgrad()
            if self.point_charges:
                string += self.writeQMoutgrad_pc()
        if QMin.requests["overlap"]:
            string += self.writeQMoutnacsmat()
        if QMin.requests["nacdr"]:
            string += self.writeQMoutnacana()
        if QMin.requests["socdr"]:
            string += self.writeQMoutsocdr()
        if QMin.requests["dmdr"]:
            string += self.writeQMoutdmdr()
        if QMin.requests["ion"]:
            string += self.writeQMoutprop()
        if QMin.requests["theodore"]:
            string += self.writeQMoutTHEODORE(QMin)
        if QMin.requests["phases"]:
            string += self.writeQmoutPhases()
        if QMin.requests["multipolar_fit"]:
            string += self.writeQMoutmultipolarfit(QMin)
        string += self.writeQMouttime()
        writefile(outfilename, string)
        return

    def writeQMoutsoc(self):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix'''
        nmstates = self.nmstates
        string = ''
        string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (eformat(self.h[i][j].real, 12, 3), eformat(self.h[i][j].imag, 12, 3))
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #

    def writeQMoutdm(self):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line. The string contains three such matrices.

        Returns:
        1 string: multiline string with the DM matrices'''
        nmstates = self.nmstates
        string = ''
        string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (
                        eformat(self.dm[xyz][i][j].real, 12, 3), eformat(self.dm[xyz][i][j].imag, 12, 3)
                    )
                string += '\n'
            string += ''
        return string

    # ======================================================================= #
    def writeQMoutdmdr(self):

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ''
        string += '! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n' % (12, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                for ipol in range(3):
                    string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n' % (
                        natom, 3, imult, istate, ims, jmult, jstate, jms, ipol
                    )
                    for atom in range(natom):
                        for xyz in range(3):
                            string += '%s ' % (eformat(self.dmdr[ipol][i][j][atom][xyz], 12, 3))
                        string += '\n'
                    string += ''
                j += 1
            i += 1
        string += '\n'
        return string

    # ======================================================================= #

    def writeQMoutsocdr(self):

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ''
        string += '! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n' % (13, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (
                    natom, 3, imult, istate, ims, jmult, jstate, jms
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s %s ' % (
                            eformat(self.socdr[i][j][atom][xyz].real, 12,
                                    3), eformat(self.socdr[i][j][atom][xyz].imag, 12, 3)
                        )
                string += '\n'
                string += ''
                j += 1
            i += 1
        string += '\n'
        return string

    def writeQMoutang(self):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line. The string contains three such matrices.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the DM matrices'''

        nmstates = self.nmstates
        string = ''
        string += '! %i Angular Momentum Matrices (3x%ix%i, complex)\n' % (9, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (
                        eformat(self.angular[xyz][i][j].real, 12,
                                3), eformat(self.angular[xyz][i][j].imag, 12, 3)
                    )
                string += '\n'
            string += ''
        return string

    # ======================================================================= #

    def writeQMoutgrad(self):
        '''Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the Gradient vectors'''

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ''
        string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += '%i %i ! m1 %i s1 %i ms1 %i\n' % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s ' % (eformat(self.grad[i][atom][xyz], 12, 3))
                string += '\n'
            string += ''
            i += 1
        return string

    def writeQMoutgrad_pc(self):
        '''Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
        a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Returns:
        1 string: multiline string with the Gradient vectors'''

        states = self.states
        nmstates = self.nmstates
        npc = self.npc
        string = ''
        string += '! %i Point Charge Gradient Vectors (%ix%ix3, real)\n' % (30, nmstates, npc)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += '%i %i ! m1 %i s1 %i ms1 %i\n' % (npc, 3, imult, istate, ims)
            for atom in range(npc):
                for xyz in range(3):
                    string += '%s ' % (eformat(self.grad_pc[i][atom][xyz], 12, 3))
                string += '\n'
            string += ''
            i += 1
        return string

    # ======================================================================= #

    def writeQMoutnacnum(self):
        '''Generates a string with the NAC matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.


        Returns:
        1 string: multiline string with the NAC matrix'''

        nmstates = self.nmstates
        string = ''
        string += '! %i Non-adiabatic couplings (ddt) (%ix%i, complex)\n' % (4, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(self.nacdt[i][j].real, 12, 3), eformat(self.nacdt[i][j].imag, 12, 3)
                )
            string += '\n'
        string += ''
        # also write wavefunction phases
        string += '! %i Wavefunction phases (%i, complex)\n' % (7, nmstates)
        for i in range(nmstates):
            string += '%s %s\n' % (eformat(self.phases[i], 12, 3), eformat(0., 12, 3))
        string += '\n\n'
        return string

    # ======================================================================= #

    def writeQMoutnacana(self):
        '''Generates a string with the NAC vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        On the next line, natom and 3 are written, followed by the gradient, with one line per atom and
         a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

        Returns:
        1 string: multiline string with the NAC vectors'''

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        string = ''
        string += '! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                # string+='%i %i ! %i %i %i %i %i %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms)
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (
                    natom, 3, imult, istate, ims, jmult, jstate, jms
                )
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s ' % (eformat(self.nacdr[i][j][atom][xyz], 12, 3))
                    string += '\n'
                string += ''
                j += 1
            i += 1
        return string

    # ======================================================================= #

    def writeQMoutnacsmat(self):
        '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the transformation matrix'''

        nmstates = self.nmstates
        string = ''
        string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for j in range(nmstates):
            for i in range(nmstates):
                string += '%s %s ' % (
                    eformat(self.overlap[j][i].real, 12, 3), eformat(self.overlap[j][i].imag, 12, 3)
                )
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #

    def writeQMouttime(self):
        '''Generates a string with the quantum mechanics total runtime in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the runtime is given.

        Returns:
        1 string: multiline string with the runtime'''

        string = '! 8 Runtime\n%s\n' % (eformat(self.runtime, 9, 3))
        return string

    # ======================================================================= #

    def writeQMoutprop(self):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data.
        In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements.
        Blocks are separated by a blank line.

        Returns:
        1 string: multiline string with the SOC matrix'''

        nmstates = self.nmstates
        string = ''
        string += '! %i Property Matrix (%ix%i, complex)\n' % (11, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(self.prop[i][j].real, 12, 3), eformat(self.prop[i][j].imag, 12, 3)
                )
            string += '\n'
        string += '\n'

        # print(property matrices (flag 20) in new format)
        string += '! %i Property Matrices\n' % (20)
        string += '%i    ! number of property matrices\n' % (1)

        string += '! Property Matrix Labels (%i strings)\n' % (1)
        string += 'Dyson norms\n'

        string += '! Property Matrices (%ix%ix%i, complex)\n' % (1, nmstates, nmstates)
        string += '%i %i   ! Dyson norms\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(self.prop[i][j].real, 12, 3), eformat(self.prop[i][j].imag, 12, 3)
                )
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #

    def writeQMoutTHEODORE(self, QMin):

        nmstates = self.nmstates
        nprop = len(self.prop1d) + len(self.prop2d)
        nprop += 1 if 'qmmm' in self and 'MMEnergy_terms' in self['qmmm'] else 0
        if nprop == 0:
            return '\n'

        string = ''

        string += '! %i Property Vectors\n' % (21)
        string += '%i    ! number of property vectors\n' % (nprop)

        string += '! Property Vector Labels (%i strings)\n' % (nprop)
        descriptors = []
        if 'theodore' in QMin:
            for i in QMin['resources']['theodore_prop']:
                descriptors.append('%s' % i)
                string += descriptors[-1] + '\n'
            for i in range(len(QMin['resources']['theodore_fragment'])):
                for j in range(len(QMin['resources']['theodore_fragment'])):
                    descriptors.append('Om_{%i,%i}' % (i + 1, j + 1))
                    string += descriptors[-1] + '\n'
        if QMin['template']['qmmm']:
            for label in sorted(QMout['qmmm']['MMEnergy_terms']):
                descriptors.append(label)
                string += label + '\n'

        string += '! Property Vectors (%ix%i, real)\n' % (nprop, nmstates)
        if 'theodore' in QMin:
            for i in range(QMin['resources']['theodore_n']):
                string += '! TheoDORE descriptor %i (%s)\n' % (i + 1, descriptors[i])
                for j in range(nmstates):
                    string += '%s\n' % (eformat(QMout['theodore'][j][i].real, 12, 3))
        if QMin['template']['qmmm']:
            for label in sorted(QMout['qmmm']['MMEnergy_terms']):
                string += '! QM/MM energy contribution (%s)\n' % (label)
                for j in range(nmstates):
                    string += '%s\n' % (eformat(QMout['qmmm']['MMEnergy_terms'][label], 12, 3))
        string += '\n'

        return string

    # ======================================================================= #

    def writeQmoutPhases(self):

        string = '! 7 Phases\n%i ! for all nmstates\n' % (self.nmstates)
        for i in range(self.nmstates):
            string += '%s %s\n' % (eformat(self.phases[i].real, 9, 3), eformat(self.phases[i].imag, 9, 3))
        return string

    def writeQMoutmultipolarfit(self, QMin):
        '''Generates a string with the fitted RESP charges for each pair of states specified.

        The string starts with a! followed by a flag specifying the type of data.
        Each line starts with the atom number (starting at 1), state i and state j.
        If i ==j: fit for single state, else fit for transition multipoles.
        One line per atom and a blank line at the end.

        Returns:
        1 string: multiline string with the Gradient vectors'''

        states = self.states
        nmstates = self.nmstates
        natom = self.natom
        resp_layers = self.resp_layers
        resp_density = QMin.resources['resp_density']
        resp_flayer = QMin.resources['resp_first_layer']
        resp_order = QMin.resources['resp_fit_order']
        resp_grid = QMin.resources['resp_grid']
        setting_str = f' settings [order grid firstlayer density layers] {resp_order} {resp_grid} {resp_flayer} {resp_density} {resp_layers}'
        string = f'! 22 Atomwise multipolar density representation fits for states ({nmstates}x{nmstates}x{natom}x10) {setting_str}\n'

        for i, (imult, istate, ims) in zip(range(nmstates), itnmstates(states)):
            for j, (jmult, jstate, jms) in zip(range(nmstates), itnmstates(states)):
                string += f'{natom} 10 ! m1 {imult} s1 {istate} ms1 {ims: 3.1f}   m2 {jmult} s2 {jstate} ms2 {jms: 3.1f}\n'
                string += "\n".join(map(lambda x: " ".join(map(lambda y: '{: 10.8f}'.format(y), x)), QMout.multipolar_fit[i][j])) + '\n'
                string += ''
        return string

    # ======================================================================= #



if __name__ == "__main__":
    test = QMout()
    print(test.__class__.__dict__)
    test['h'] = np.zeros((2, 2))
    print(test['dm'])
    print(test)
