#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
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

# IMPORTS
# external
import pprint
import sys
import math
import datetime
from multiprocessing import Pool
from copy import deepcopy
from textwrap import wrap
import itertools

# internal
from SHARC_INTERFACE import INTERFACE
from globals import DEBUG, PRINT
from utils import *
from constants import HARTREE_TO_EV, D2au, kcal_to_Eh, BOHR_TO_ANG, NUMBERS

authors = 'Nadja K. Singer'
version = '0.1'
versiondate = datetime.datetime(2021, 10, 27)

changelogstring = '''
27.10.2021:     Initial version 0.1
- Only OM2/MRCI
- Only singlets
'''


class MNDO(INTERFACE):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring

    @property
    def version(self):
        return self._version

    @property
    def versiondate(self):
        return self._versiondate

    @property
    def changelogstring(self):
        return self._changelogstring

    @property
    def authors(self):
        return self._authors

    def run(self):
        QMin = self._QMin
        # write MNDO input
        input = self.write_MNDO_input()
        # start MNDO
        errorcodes = self.run_job(input)
        if 'overlap' in QMin and QMin['step'] != 0:
            self.run_wfoverlap()
        # read MNDO output
        self._QMout = self.getQMout()

        # Remove Scratchfiles from SCRATCHDIR
        if not self._DEBUG:
            cleandir(QMin['scratchdir'])
            if 'cleanup' in QMin:
                cleandir(QMin['savedir'])
        return errorcodes

    # =============================================================================================== #
    # =============================================================================================== #
    # ====================================== MNDO setup ============================================= #
    # =============================================================================================== #
    # =============================================================================================== #

    def read_resources(self, resources_filename="MNDO.resources"):
        super().read_resources(resources_filename)
        self.QMin['resources']['ncpu'] = 1
        print('The MNDO interface is currently not parallelized.')
        self._read_resources = True
        return

    def read_template(self, template_filename="MNDO.template"):
        if not self._read_resources:
            raise Error('Interface is not set up correctly. Call read_resources with the .resources file first!', 23)
        QMin = self.QMin
        # define keywords and defaults
        strings = {
            'dstep': '2e-4',
        }
        integers = {'mult': 0, 'icuts': -1, 'icutg': -1}
        bools = {'unrestricted_triplets': False, 'qmmm': False, 'imomap': False}
        special = {
            'paddingstates': [0 for i in QMin['states']],
            'charge': [i % 2 for i in range(len(QMin['states']))],
            'mo_occ': [],
            'mo_unocc': []
        }
        lines = readfile(template_filename)
        QMin['template'] = {
            **strings,
            **integers,
            **bools,
            **special,
            **self.parse_keywords(lines, strings=strings, integers=integers, bools=bools, special=special)
        }
        # Sanity checks and preparations
        if len(QMin['states']) > 1:
            raise Error('Currently only singlets can be calculated using OM2/MRCI.', 135)

        QMin['template']['imomap'] = 3

        if len(QMin['template']['mo_occ']) == 0 or len(QMin['template']['mo_unocc']) == 0:
            QMin['movo'] = 0
            QMin['ici1'] = 1
            QMin['ici2'] = 1
        else:
            QMin['movo'] = 1
            QMin['ici1'] = len(QMin['template']['mo_occ'])
            QMin['ici2'] = len(QMin['template']['mo_unocc'])

        self._read_template = True
        return

    def write_MNDO_input(self):
        QMin = self._QMin
        natom = QMin["natom"]
        nmstates = QMin["nmstates"]
        coords = QMin["coords"]
        mo_occ = QMin['template']["mo_occ"]
        mo_unocc = QMin['template']["mo_unocc"]
        mult = QMin['template']["mult"]
        elements = QMin["elements"]
        grad = QMin["grad"] if "grad" in QMin else [i + 1 for i in range(QMin['nmstates'])]
        movo = QMin["movo"]
        ici1 = QMin['ici1']
        ici2 = QMin['ici2']
        icuts = QMin['template']["icuts"]
        icutg = QMin['template']["icutg"]
        dstep = QMin['template']["dstep"]
        imomap = QMin['template']['imomap']
        #make string
        #TODO You can either use charge or muliplicity! not both
        inputstring = f'iop=-6 jop=-2 imult={mult} kitscf=500 iform=1 igeom=1 mprint=1 icuts={icuts} icutg={icutg} dstep={dstep} kci=5 ioutci=1 iroot={nmstates} ncisym=-1 icross=7 ncigrd={len(grad)} imomap={imomap} mapthr=50 movo={movo} ici1={ici1} ici2={ici2} nciref=3 mciref=3 levexc=6 cilead=1 iuvcd=3 nsav13=2\n'
        inputstring = " +\n".join(wrap(inputstring, width=70))
        inputstring += '\nheader\n'
        inputstring += 'header\n'
        for i in range(natom):
            inputstring += f'{NUMBERS[elements[i]]:>3d}\t{coords[i][0]*BOHR_TO_ANG:>10,.5f} 1\t{coords[i][1]*BOHR_TO_ANG:>10,.5f} 1\t{coords[i][2]*BOHR_TO_ANG:>10,.5f} 1\n'
        inputstring += f'{0:>3d}\t{0:>10,.5f} 0\t{0:>10,.5f} 0\t{0:>10,.5f} 0\n'

        if movo != 0:
            for j in mo_occ:
                inputstring += str(j) + " "
            for k in mo_unocc:
                inputstring += str(k) + " "
            inputstring += "\n"
        for l in grad:
            inputstring += str(l) + " "

        return inputstring

    # =============================================================================================== #
    # =============================================================================================== #
    # ====================================== running MNDO =========================================== #
    # =============================================================================================== #
    # =============================================================================================== #

    def run_job(self, inputstring):
        QMin = self._QMin
        WORKDIR = QMin['scratchdir']
        self.setupWORKDIR(WORKDIR, QMin, inputstring)
        self.runMNDO(WORKDIR)
        self.save_files(WORKDIR, QMin)

        if 'backup' in QMin:
            self.backupdata(QMin['backup'])

    # =============================================================================================== #
    @staticmethod
    def setupWORKDIR(WORKDIR, QMin, inputstring):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
        # then put the MNDO.inp file

        # setup the directory
        mkdir(WORKDIR)

        #write inputfile
        if PRINT:
            print('>>>>>>>>>>>>> Writing MNDO input files')
        filename = os.path.join(WORKDIR, 'MNDO.inp')
        writefile(filename, inputstring)
        #??
        if DEBUG:
            print('================== DEBUG input file for WORKDIR {} ================='.format(shorten_DIR(WORKDIR)))
            print(inputstring)
            print('MNDO input written to: %s' % (filename))
            print('====================================================================')
        #copy imomap from previous step
        if QMin['template']['imomap'] == 3 and QMin['step'] != 0:
            fromfile = os.path.join(QMin['savedir'], f'imomap.{QMin["step"]-1}')
            tofile = os.path.join(WORKDIR, 'imomap.dat')
            shutil.copy(fromfile, tofile)

        return

    # =============================================================================================== #

    def runMNDO(self, WORKDIR):
        QMin = self._QMin
        if PRINT:
            print('>>>>>>>>>>>>> Running MNDO calculation')
        mndodir = QMin.resources.update['mndodir']
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = mndodir + ' < MNDO.inp'
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('START:\t{}\t{}\t"{}"\n'.format(shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
            sys.stdout.flush()
        stdoutfile = open(os.path.join(WORKDIR, 'MNDO.out'), 'w')
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile)
        except OSError:
            raise Error('MNDO call have had some serious problems:', OSError, 77)
        stdoutfile.close()
        with open(os.path.join(WORKDIR, 'MNDO.out')) as f:
            line = f.readlines()
            if 'COMPUTATION TIME' not in line[-3]:
                runerror = 1
            if 'UNABLE TO ACHIEVE SCF CONVERGENCE' in line[-50:-1]:
                raise Error('MNDO could not achieve SCF convergence:', OSError, 78)
                #runerror = 1
        stdoutfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write(
                'FINISH:\t{}\t{}\tRuntime: {}\tError Code: {}\n'.format(
                    shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror
                )
            )
            sys.stdout.flush()
        os.chdir(prevdir)
        #if not DEBUG and runerror == 0:
        #    keep = [
        #        'MNDO.inp$', 'MNDO.out$', 'molden.dat$', 'imomap.dat$'
        #    ]
        #    INTERFACE.stripWORKDIR(WORKDIR, keep)
        return runerror

    # =============================================================================================== #

    def run_wfoverlap(self):
        QMin = self._QMin
        print('>>>>>>>>>>>>> Starting the WFOVERLAP job execution')
        step = QMin['step']

        # do overlap calculations
        self.get_Double_AOovl()
        WORKDIR = os.path.join(QMin['scratchdir'], 'WFOVL')
        files = {
            'aoovl': 'AO_overl.mixed',
            'det.a': f'dets.{step - 1}',
            'det.b': f'dets.{step}',
            'mo.a': f'mos.{step - 1}',
            'mo.b': f'mos.{step}'
        }
        INTERFACE.setupWORKDIR_WF(WORKDIR, QMin, files, self._DEBUG)
        runerror = INTERFACE.runWFOVERLAP(WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])
        if runerror != 0:
            raise Error('WFOVERLAP calculation crashed! Error code=%i' % (runerror), 109)
        return

    # ======================================================================= #
    def get_Double_AOovl(self):
        QMin = self._QMin
        # get geometries
        NAO = QMin['nmo']

        string = '{} {}\n'.format(NAO, NAO)
        for irow in range(0, NAO):
            for icol in range(0, NAO):
                # OMx methods have globally orthogonalized AOs (10.1063/1.5022466)
                string += '{: .15e} '.format(
                    0. if irow != icol else 1.
                )    # note the exchanged indices => transposition
            string += '\n'
        filename = os.path.join(QMin['savedir'], 'AO_overl.mixed')
        writefile(filename, string)
        return

    # =============================================================================================== #
    @staticmethod
    def save_files(WORKDIR, QMin):
        # save files
        fromfile = os.path.join(WORKDIR, 'molden.dat')
        tofile = os.path.join(QMin['savedir'], f'molden.{QMin["step"]}')
        shutil.copy(fromfile, tofile)
        if QMin['template']['imomap'] == 3:
            fromfile = os.path.join(WORKDIR, 'imomap.dat')
            tofile = os.path.join(QMin['savedir'], f'imomap.{QMin["step"]}')
            shutil.copy(fromfile, tofile)

        #MOs
        moldenfile = os.path.join(QMin['scratchdir'], 'molden.dat')
        mos, MO_occ = MNDO.get_MO_from_molden(moldenfile)
        QMin['nmo'] = len(MO_occ.keys())
        mo = os.path.join(QMin['savedir'], 'mos.' + str(QMin['step']))
        writefile(mo, mos)

        #dets
        logfile = os.path.join(QMin['scratchdir'], 'MNDO.out')
        determinants = MNDO.get_determinants(QMin, logfile, MO_occ)
        det = os.path.join(QMin['savedir'], 'dets.' + str(QMin['step']))
        writefile(det, determinants)
        return

    # =============================================================================================== #
    # =============================================================================================== #
    # ====================================== MNDO output parsing ==================================== #
    # =============================================================================================== #
    # =============================================================================================== #

    def getQMout(self):
        QMin = self._QMin

        if PRINT:
            print('>>>>>>>>>>>>> Reading MNDO output files')
        starttime = datetime.datetime.now()

        QMout = {}
        nmstates = QMin['nmstates']

        # Hamiltonian
        if 'h' in QMin:
            # make Hamiltonian
            if 'h' not in QMout:
                QMout['h'] = makecmatrix(nmstates, nmstates)
                #QMout['h'] = [ [ complex(0.,0.) for i in range(nmstates) ] for j in range(nmstates) ]
                logfile = os.path.join(QMin['scratchdir'], 'MNDO.out')
                energies = self.get_energies(logfile)
                for i in range(nmstates):
                    QMout['h'][i][i] = energies[i]

        # Dipole Moments
        if 'dm' in QMin:
            # make matrix
            if 'dm' not in QMout:
                QMout['dm'] = [makecmatrix(nmstates, nmstates) for i in range(3)]
                #QMout['dm'] = [ [ [ complex(0.,0.) for i in range(nmstates) ] for j in range(nmstates) ] for i in range(3) ]
                logfile = os.path.join(QMin['scratchdir'], 'MNDO.out')
                dipoles = self.get_tdm(QMin, logfile)
                for xyz in [0, 1, 2]:
                    for i in range(nmstates):
                        for j in range(nmstates):
                            if i <= j:
                                QMout['dm'][xyz][i][j] = dipoles[(i + 1, j + 1)][xyz]
                            else:
                                QMout['dm'][xyz][i][j] = dipoles[(j + 1, i + 1)][xyz]

        # Gradients
        if 'grad' in QMin:
            if 'grad' not in QMout:
                logfile = os.path.join(QMin['scratchdir'], 'MNDO.out')
                grad, nac = self.get_grads_and_nacs(QMin, logfile, energies)
                QMout['grad'] = grad
                QMout['nac'] = nac

        # Regular Overlaps
        if 'overlap' in QMin and QMin['step'] != 0:
            if 'overlap' not in QMout:
                QMout['overlap'] = makecmatrix(nmstates, nmstates)
                outfile = os.path.join(QMin['scratchdir'], 'WFOVL/wfovl.out')
                out = readfile(outfile)
                if PRINT:
                    print('Overlaps: ' + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                        m2, s2, ms2 = tuple(QMin['statemap'][j + 1])

                        QMout['overlap'][i][j] = self.getsmate(out, s1, s2)

        endtime = datetime.datetime.now()
        if PRINT:
            print("Readout Runtime: %s" % (endtime - starttime))

        return QMout

    # ======================================================================= #
    @staticmethod
    def get_energies(logfile):    #getenergy(self,logfile):
        #QMin = self._QMin
        # open file
        f = readfile(logfile)
        # read energies in eV and convert to Eh
        energies = []
        for iline, line in enumerate(f):
            if 'E=' in line:
                energies.append(float(line.split()[8]) / HARTREE_TO_EV)
        return energies

    # ======================================================================= #
    @staticmethod
    def get_tdm(QMin, logfile):
        #get transition dipole moments
        #QMin = self._QMin
        nmstates = QMin["nmstates"]
        f = readfile(logfile)
        #if PRINT:
        #print('Dipoles:  ' + shorten_DIR(logfile))
        dm = {}

        #diagonal elements
        for iline, line in enumerate(f):
            if 'State dipole moments:' in line:
                iline += 3
                for st in range(nmstates):
                    line = f[iline]
                    s = line.split()
                    dmx = float(s[5]) * D2au
                    dmy = float(s[6]) * D2au
                    dmz = float(s[7]) * D2au
                    state = int(s[0])
                    dm[(state, state)] = [dmx, dmy, dmz]
                    iline += 1

        #off-diagonal elements
        line_offdiag = []
        for iline, line in enumerate(f):
            if 'Dipole-length electric dipole transition moments' in line:
                line_offdiag.append(iline + 3)
        noffdiag = int(nmstates - 1)
        for i in line_offdiag:
            for j in range(noffdiag):
                line = f[i]
                #print(line)
                s = line.split()
                dmx = float(s[5]) * D2au
                dmy = float(s[6]) * D2au
                dmz = float(s[7]) * D2au
                st = int(j + 1)
                dm[(st, int(s[0]))] = [dmx, dmy, dmz]
                i += 1
            noffdiag -= 1
        return dm

    # ======================================================================= #
    @staticmethod
    def get_grads_and_nacs(QMin, logfile, energies):    #getgrad(self,logfile):
        #get gradients in Eh/Bohr from kcal/(mol*angstrom)
        #  and nacs in 1/Bohr from kcal/(mol*angstrom)
        nmstates = QMin["nmstates"]
        natom = QMin["natom"]
        f = readfile(logfile)
        #if PRINT:
        #print('Dipoles:  ' + shorten_DIR(logfile))

        line_marker = []
        for iline, line in enumerate(f):
            if 'GRADIENTS (KCAL/(MOL*ANGSTROM))' in line:
                line_marker.append(iline + 4)

        #gradient
        grads = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
        for k in range(nmstates):
            iline = line_marker[k]
            for j in range(natom):
                line = f[iline]
                s = line.split()
                grads[k][j][0] = float(s[5]) * kcal_to_Eh * BOHR_TO_ANG
                grads[k][j][1] = float(s[6]) * kcal_to_Eh * BOHR_TO_ANG
                grads[k][j][2] = float(s[7]) * kcal_to_Eh * BOHR_TO_ANG
                iline += 1

        #nacs
        nac = [[[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)] for l in range(nmstates)]
        n = nmstates
        for l in range(nmstates):
            for k in range(1, nmstates):
                if l < k:
                    iline = line_marker[n]
                    for j in range(natom):
                        line = f[iline]
                        s = line.split()
                        nac[l][k][j][0] = float(s[5]) * kcal_to_Eh * BOHR_TO_ANG / (energies[l] - energies[k])
                        nac[l][k][j][1] = float(s[6]) * kcal_to_Eh * BOHR_TO_ANG / (energies[l] - energies[k])
                        nac[l][k][j][2] = float(s[7]) * kcal_to_Eh * BOHR_TO_ANG / (energies[l] - energies[k])
                        iline += 1
                    n += 1
                    nac[k][l] = nac[l][k]

        return grads, nac

    # ======================================================================= #

    @staticmethod
    def getsmate(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                raise Error('Overlap of states %i - %i not found!' % (s1, s2), 103)
            if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    # ======================================================================= #
    @staticmethod
    def get_active_space(logfile):
        #get active space
        f = readfile(logfile)

        active_mos = {}
        for iline, line in enumerate(f):
            if 'OCC.    ACTIVE' in line:
                jline = iline + 2
                line = f[jline]
                while line != '\n':
                    s = line.split()
                    if s[6] != '-':
                        active_mos[int(s[0])] = int(s[6])
                    jline += 1
                    line = f[jline]
                break

        return active_mos

    # ======================================================================= #
    @staticmethod
    def get_MO_from_molden(moldenfile):
        #get MOs from molden file and  1) transform to string for mol-file
        #                              2) write GS occupation to MO_occ
        f = readfile(moldenfile)

        # get MOs and MO_occ in a dict from molden file
        MOs = {}
        NMO = 0
        MO_occ = {}
        for iline, line in enumerate(f):
            if 'Sym= A' in line:
                NMO += 1
                AO = {}
                o = f[iline + 3].split()
                MO_occ[NMO] = o[1]
                jline = iline + 4
                line = f[jline]
                while 'Sym= A' not in line:
                    s = line.split()
                    AO[int(s[0])] = float(s[1])
                    jline += 1
                    if jline == len(f):
                        break
                    line = f[jline]
                MOs[NMO] = AO

        # make string
        string = '''2mocoef
header
1
MO-coefficients from OM2/MRCI
1
%i   %i
a
mocoef
(*)
''' % (NMO, NMO)
        x = 0
        for i in range(NMO):
            for j in range(NMO):
                c = MOs[i + 1][j + 1]
                if x >= 3:
                    string += '\n'
                    x = 0
                string += '% 6.12e ' % c
                x += 1
            if x > 0:
                string += '\n'
                x = 0
        string += 'orbocc\n(*)\n'
        x = 0
        for i in range(NMO):
            if x >= 3:
                string += '\n'
                x = 0
            string += '% 6.12e ' % (0.0)
            x += 1

        return string, MO_occ

    # ======================================================================= #
    @staticmethod
    def get_csfs(QMin, logfile, active_mos):
        #get CSF composition of states
        nmstates = QMin['nmstates']

        # open file
        f = readfile(logfile)
        # note the lines in which the states are
        state_lines = []
        for iline, line in enumerate(f):
            for i in range(1, nmstates + 1):
                if 'State  ' + str(i) in line:
                    state_lines.append(iline)
            if 'Using basis sets ECP-3G (first-row elements) and ECP-4G (second-row' in line:
                state_lines.append(iline)

        # read csfs from logfile
        csf_ref = {}
        for i in range(nmstates):
            for iline, line in enumerate(f[state_lines[i]:state_lines[i + 1]]):
                if str(" "+str(active_mos[0])+" ") in line:
                    x = line.split()
                    ref = f[state_lines[i] + iline + 1]
                    y = ref.split()
                    if x[2] not in csf_ref.keys():
                        csf_ref[x[2]] = {}
                        csf_ref[x[2]]["coeffs"] = [0. for k in range(nmstates)]
                        csf_ref[x[2]]['CSF'] = {}
                        for imo, mo in enumerate(active_mos):
                            csf_ref[x[2]]['CSF'][mo] = y[imo]
                    csf_ref[x[2]]["coeffs"][i] = float(x[1])

        return csf_ref

    # ======================================================================= #
    @staticmethod
    def decompose_csf(ms2, step):
        # ms2 is M_S value
        # step is step vector for CSF (e.g. 3333012021000)

        def powmin1(x):
            a = [1, -1]
            return a[x % 2]

        # calculate key numbers
        nopen = sum([i == 1 or i == 2 for i in step])
        nalpha = int(nopen / 2. + ms2)
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
            coeff = 1.
            sign = +1
            m2 = 0
            for k in range(norb):
                if step[k] == 1:
                    m2 += powmin1(det[k] + 1)
                    num = bval[k] + powmin1(det[k] + 1) * m2
                    denom = 2. * bval[k]
                    if num == 0.:
                        break
                    coeff *= 1. * num / denom
                elif step[k] == 2:
                    m2 += powmin1(det[k] - 1)
                    num = bval[k] + 2 + powmin1(det[k]) * m2
                    denom = 2. * (bval[k] + 2)
                    sign *= powmin1(bval[k] + 2 - det[k])
                    if num == 0.:
                        break
                    coeff *= 1. * num / denom
                elif step[k] == 3:
                    sign *= powmin1(bval[k])
                    num = 1.

            # add determinant to dict if coefficient non-zero
            if num != 0.:
                dets[tuple(det)] = 1. * sign * math.sqrt(coeff)

        #pprint.pprint( dets)
        return dets

    # ======================================================================= #
    @staticmethod
    def format_ci_vectors(QMin, ci_vectors, MO_occ):

        nstates = QMin['nmstates']
        norb = len(MO_occ)
        ndets = len(ci_vectors) - 1

        # sort determinant strings
        dets = []
        for key in ci_vectors:
            if key != "active MOs":
                dets.append(key)
        dets.sort(reverse=True)

        #write first line of det-file
        string = '%i %i %i\n' % (nstates, norb, ndets)

        #dictionary to get "a/b/d/e"-nomenclature
        dict_int_to_de = {"2.0": "d", "0.0": "e", 3: "d", 2: "b", 1: "a", 0: "e"}

        #take basis MO occupany: 1) change it according to det;     2) add it to string ;
        ##                       3) add coefficients;               4)return string
        for det in dets:
            MO_occ_cp = deepcopy(MO_occ)
            for i, orb in enumerate(ci_vectors["active MOs"]):
                MO_occ_cp[orb] = det[i]
            #raise TypeError(MO_occ_cp)
            for MO in MO_occ_cp:
                string += dict_int_to_de[MO_occ_cp[MO]]
            for c in ci_vectors[det]:
                string += ' %16.12f ' % c
            string += '\n'

        return string

    # ======================================================================= #
    @staticmethod
    def get_determinants(QMin, logfile, MO_occ):

        # dictionary to convert to "0123"-nomenclature
        dict_ab_to_int = {'ab': '3', 'a': '1', 'b': '2', '-': '0'}

        ci_vectors = {}

        # add MO occupancy to ci_vector
        active_mos = [*mndo.get_active_space(logfile)]
        ci_vectors["active MOs"] = active_mos

        # get CSFs from logfile
        csf = mndo.get_csfs(QMin, logfile, active_mos)

        # convert to determinants
        ## 1) Convert nomenclature
        for i in csf:
            ref = []
            for k in csf[i]['CSF'].keys():
                ref.append(dict_ab_to_int[csf[i]['CSF'][k]])
            ref = tuple([int(n) for n in ref])
            csf[i]['CSF'] = ref
        ## 2) build ci vector from CSFs
        for x in csf:
            dets = mndo.decompose_csf(0, list(csf[x]["CSF"]))
            coeff = csf[x]["coeffs"]
            for det in dets:
                c = [dets[det] * i for i in coeff]
                if det in ci_vectors:
                    for istate in range(len(coeff)):
                        ci_vectors[det][istate] += c[istate]
                else:
                    ci_vectors[det] = c

        # Write determinants
        determinants = mndo.format_ci_vectors(QMin, ci_vectors, MO_occ)

        return determinants


if __name__ == '__main__':
    import pprint
    mndo = MNDO(DEBUG, PRINT)
    mndo.printheader()
    mndo.setup_mol('QM.in')
    mndo.read_resources('MNDO.resources')
    mndo.read_template('MNDO.template')
    mndo.set_coords('QM.in')
    mndo.read_requests('QM.in')
    mndo.setup_run()
    mndo.run()
    mndo.write_step_file()

    #pprint.pprint(mndo._QMout)
    if PRINT or DEBUG:
        mndo.printQMout()
    mndo._QMout['runtime'] = mndo.clock.measuretime()
    mndo.writeQMout()

    #pprint.pprint(mndo._QMin)
    #print(mndo._QMin['elements'])
    #print(inputstr)
