import sys
import os
import struct
import math

from utils import readfile, writefile

class civfl_ana:

    def __init__(self, path, imult, maxsqnorm=1.0, debug=False, filestr='CCRE0'):
        self.det_dict = {}  # dictionary with determinant strings and cicoefficient information
        self.sqcinorms = {}  # CI-norms
        self.path = path
        if imult not in [1, 3]:
            print('CCR* file readout implemented only for singlets and triplets!')
            sys.exit(106)
        self.mult = imult
        self.maxsqnorm = maxsqnorm
        self.debug = debug
        self.nmos = -1  # number of MOs
        self.nfrz = 0  # number of frozen orbs
        self.nocc = -1  # number of occupied orbs (including frozen)
        self.nvir = -1  # number of virtuals
        self.filestr = filestr
        self.read_control()
# ================================================== #

    def read_control(self):
        '''
        Reads nmos, nfrz, nvir from control file
        '''
        controlfile = os.path.join(self.path, 'control')
        control = readfile(controlfile)
        for iline, line in enumerate(control):
            if 'nbf(AO)' in line:
                s = line.split('=')
                self.nmos = int(s[-1])
            if '$closed shells' in line:
                s = control[iline + 1].split()[1].split('-')
                self.nocc = int(s[-1])
            if 'implicit core' in line:
                s = line.split()
                self.nfrz = int(s[2])
        if self.nmos == -1:
            mosfile = os.path.join(self.path, 'mos')
            mos = readfile(mosfile)
            for line in mos:
                if "eigenvalue" in line:
                    self.nmos = int(line.split()[0])
        if any([self.nmos == -1, self.nfrz == -1, self.nocc == -1]):
            print('Number of orbitals not found: nmos=%i, nfrz=%i, nocc=%i' % (self.nmos, self.nfrz, self.nocc))
            sys.exit(107)
        self.nvir = self.nmos - self.nocc
# ================================================== #

    def get_state_dets(self, state):
        """
        Get the transition matrix from CCR* file and add to det_dict.
        """
        if (self.mult, state) == (1, 1):
            det = self.det_string(0, self.nocc, 'de')
            self.det_dict[det] = {1: 1.}
            return
        try:
            filename = ('%s% 2i% 3i% 4i' % (self.filestr, 1, self.mult, state - (self.mult == 1))).replace(' ', '-')
            filename = os.path.join(self.path, filename)
            CCfile = open(filename, 'rb')
        except IOError:
            # if the files are not there, use the right eigenvectors
            filename = ('%s% 2i% 3i% 4i' % ('CCRE0', 1, self.mult, state - (self.mult == 1))).replace(' ', '-')
            filename = os.path.join(self.path, filename)
            CCfile = open(filename, 'rb')
        # skip 8 byte
        CCfile.read(8)
        # read method from 8 byte
        method = str(struct.unpack('8s', CCfile.read(8))[0])
        # skip 8 byte
        CCfile.read(8)
        # read number of CSFs from 4 byte
        nentry = struct.unpack('i', CCfile.read(4))[0]
        # skip 4 byte
        CCfile.read(4)
        # read 8 byte as long int
        versioncheck = struct.unpack('l', CCfile.read(8))[0]
        if versioncheck == 0:
            # skip 16 byte in Turbomole >=7.1
            CCfile.read(16)
        else:
            # skip 8 byte in Turbomole <=7.0
            CCfile.read(8)
        # checks
        if 'CCS' in method:
            print('ERROR: preoptimization vector found in file: %s' % (filename))
            sys.exit(108)
        if not nentry == self.nvir * (self.nocc - self.nfrz):
            print('ERROR: wrong number of entries found in file: %s' % (filename))
        # get data
        state_dict = {}
        nact = self.nocc - self.nfrz
        for iocc in range(nact):
            for ivirt in range(self.nvir):
                coef = struct.unpack('d', CCfile.read(8))[0]
                if self.mult == 1:
                    det = self.det_string(iocc + self.nfrz, self.nocc + ivirt, 'ab')
                    state_dict[det] = coef
                elif self.mult == 3:
                    det = self.det_string(iocc + self.nfrz, self.nocc + ivirt, 'aa')
                    state_dict[det] = coef
        # renormalize
        vnorm = 0.
        for i in state_dict:
            vnorm += state_dict[i]**2
        vnorm = math.sqrt(vnorm)
        # truncate data
        state_dict2 = {}
        norm = 0.
        for i in sorted(state_dict, key=lambda x: state_dict[x]**2, reverse=True):
            state_dict2[i] = state_dict[i] / vnorm
            norm += state_dict2[i]**2
            if norm > self.maxsqnorm:
                break
        # put into general det_dict, also adding the b->a excitation for singlets
        if self.mult == 1:
            for i in state_dict2:
                coef = state_dict2[i] / math.sqrt(2.)
                j = i.replace('a', 't').replace('b', 'a').replace('t', 'b')
                if i in self.det_dict:
                    self.det_dict[i][state] = coef
                else:
                    self.det_dict[i] = {state: coef}
                if j in self.det_dict:
                    self.det_dict[j][state] = -coef
                else:
                    self.det_dict[j] = {state: -coef}
        elif self.mult == 3:
            for i in state_dict2:
                coef = state_dict2[i]
                if i in self.det_dict:
                    self.det_dict[i][state] = coef
                else:
                    self.det_dict[i] = {state: coef}
# ================================================== #

    def det_string(self, fromorb, toorb, spin):
        if fromorb >= self.nocc or toorb < self.nocc or fromorb >= self.nmos or toorb >= self.nmos:
            print('Error generating determinant string!')
            sys.exit(109)
        string = 'd' * self.nocc + 'e' * (self.nmos - self.nocc)
        string = string[:fromorb] + spin[0] + string[fromorb + 1:toorb] + spin[1] + string[toorb + 1:]
        return string
# ================================================== #

    def sort_key(self, key):
        """
        For specifying the sorting order of the determinants.
        """
        return key.replace('d', '0').replace('a', '1').replace('b', '1')
# ================================================== #

    def sort_key2(self, key):
        """
        For specifying the sorting order of the determinants.
        """
        return key.replace('d', '0').replace('a', '0').replace('b', '1').replace('e', '1')
# ================================================== #

    def write_det_file(self, nstate, wname='dets', wform=' % 14.10f'):
        string = '%i %i %i\n' % (nstate, self.nmos, len(self.det_dict))
        for det in sorted(sorted(self.det_dict, key=self.sort_key2), key=self.sort_key):
            string += det
            for istate in range(1, nstate + 1):
                try:
                    string += wform % (self.det_dict[det][istate])
                except KeyError:
                    string += wform % (0.)
            string += '\n'
        writefile(wname, string)

