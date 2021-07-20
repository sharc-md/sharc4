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
from copy import deepcopy
from datetime import date, datetime
import math
import sys
import os
import re
import shutil
import subprocess as sp
from abc import ABC, abstractmethod, abstractproperty

# internal
from error import Error
from printing import printcomplexmatrix, printgrad, printtheodore
from utils import itnmstates, eformat, readfile, writefile, containsstring, makecmatrix, safe_cast, link, mkdir, clock
from constants import *

# NOTE: Error handling especially import for processes in pools (error_callback)
# NOTE: gradient calculation necessitates multiple parallel calls (either inside interface) or one interface = one calculation (i.e. interface spawns multiple instances of itself)


class INTERFACE(ABC):
    _QMin = {}
    _QMout = {}
    _setup_mol = False

    # TODO: set Debug and Print flag
    # TODO: set persistant flag for file-io vs in-core
    def __init__(self):
        self.clock = clock()

    # ================== abstract methods and properties ===================

    @abstractproperty
    def authors(self) -> str:
        return 'Severin Polonius, Sebastian Mai'

    @abstractproperty
    def version(self) -> str:
        return '3.0'

    @abstractproperty
    def versiondate(self) -> date:
        return date(2021, 7, 15)

    @abstractproperty
    def changelogstring(self) -> str:
        return 'This is the changelog string'

    @abstractmethod
    def main(self):
        pass

    @abstractmethod
    def readQMin(self, QMinfilename):
        pass

    @abstractmethod
    def read_template(self, template_filename):
        pass

    @abstractmethod
    def read_resources(self, resource_filename):
        pass

    @abstractmethod
    def set_requests(self, QMinfilename):
        pass

    @abstractmethod
    def set_coords(self, xyz):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_QMout(self):
        pass
    # ============================ Implemented public methods ========================

    def setup_mol(self, QMinfilename: str):
        QMin = self._QMin
        QMinlines = readfile(QMinfilename)
        QMin['elements'] = INTERFACE.read_elements(QMinlines)
        QMin['Atomcharge'] = sum(map(lambda x: ATOMCHARGE[x], QMin['elements']))
        QMin['natom'] = len(QMin['elements'])

        # replaces all comments with white space. filters all empty lines
        filtered = filter(lambda x: not re.match(r'^\s*$', x),
                          map(lambda x: re.sub(r'#.*$', '', x), QMinlines[QMin['natom'] + 2:]))
        
        # naively parse all key argument pairs from QM.in
        for line in filtered:
            llist = line.split(None, 1)
            key = llist[0].lower()
            if key == 'states':
                try:
                    QMin['states'] = list(map(int, llist[1].split()))
                except (ValueError, IndexError):
                    # get traceback of currently handled exception
                    tb = sys.exc_info()[2]
                    raise Error('Keyword "states" has to be followed by integers!', 37).with_traceback(tb)
                reduc = 0
                for i in reversed(QMin['states']):
                    if i == 0:
                        reduc += 1
                    else:
                        break
                for i in range(reduc):
                    del QMin['states'][-1]
                nstates = 0
                nmstates = 0
                for i in range(len(QMin['states'])):
                    nstates += QMin['states'][i]
                    nmstates += QMin['states'][i] * (i + 1)
                QMin['nstates'] = nstates
                QMin['nmstates'] = nmstates
        # NOTE: Quantity requests (tasks) are dealt with later and potentially re-assigned
        return

    def set_coords(self, xyz):
        lines = readfile(xyz)
        self._QMin["geom"] = list(map(INTERFACE._parse_xyz, lines))

    @staticmethod
    def set_coords(xyz):
        lines = readfile(xyz)
        try:
            natom = int(lines[0])
        except ValueError:
            raise Error('first line must contain the number of atoms!', 2)
        return [[x[0], *x[1]] for x in map(INTERFACE._parse_xyz, lines[2:natom + 2])]

    @staticmethod
    def read_elements(QMinlines: list[str]) -> list[str]:

        try:
            natom = int(QMinlines[0])
        except ValueError:
            raise Error('first line must contain the number of atoms!', 2)
        if len(QMinlines) < natom + 4:
            raise Error('Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task', 3)
        atomlist = list(map(lambda x: INTERFACE._parse_xyz(x)[1], (QMinlines[2:natom + 2])))
        return atomlist

    # ======================================================================= #

    @staticmethod
    def _parse_xyz(line) -> tuple[str, list[float]]:
        match = re.match(r'([a-zA-Z]{1,2}\d?)((\s+-?\d+\.\d*){3,6})', line)
        if match:
            return match[1], list(map(float, match[2].split()))
        else:
            raise Error(f"line is not xyz\n\n{line}", 43)

    @staticmethod
    def _get_pairs(QMinlines, i):
        nacpairs = []
        while True:
            i += 1
            try:
                line = QMinlines[i].lower()
            except IndexError:
                raise Error('"keyword select" has to be completed with an "end" on another line!', 47)
            if 'end' in line:
                break
            fields = line.split()
            try:
                nacpairs.append([int(fields[0]), int(fields[1])])
            except ValueError:
                raise Error('"nacdr select" is followed by pairs of state indices, each pair on a new line!', 48)
        return nacpairs, i

    # ======================================================================= #


    @staticmethod
    def checkscratch(SCRATCHDIR):
        '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

        Arguments:
        1 string: path to SCRATCHDIR'''

        exist = os.path.exists(SCRATCHDIR)
        if exist:
            isfile = os.path.isfile(SCRATCHDIR)
            if isfile:
                raise Error('$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR), 42)
        else:
            try:
                os.makedirs(SCRATCHDIR)
            except OSError:
                raise Error('Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR), 43)

    @staticmethod
    def removequotes(string):
        if string.startswith("'") and string.endswith("'"):
            return string[1:-1]
        elif string.startswith('"') and string.endswith('"'):
            return string[1:-1]
        else:
            return string
# ======================================================================= #

    @staticmethod
    def get_smatel(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                raise Error('Overlap of states %i - %i not found!' % (s1, s2), 32)
            if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    @staticmethod
    def coords_same(coord1, coord2):
        thres = 1e-5
        s = 0.
        for i in range(3):
            s += (coord1[i] - coord2[i])**2
        s = math.sqrt(s)
        return s <= thres

    def runProgram(self, string, workdir, outfile, errfile=''):
        prevdir = os.getcwd()
        PRINT = self.PRINT
        DEBUG = self.DEBUG
        if DEBUG:
            print(workdir)
        os.chdir(workdir)
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('%s\n\t%s' % (string, starttime))
            sys.stdout.flush()
        stdoutfile = open(os.path.join(workdir, outfile), 'w')
        if errfile:
            stderrfile = open(os.path.join(workdir, errfile), 'w')
        else:
            stderrfile = sp.STDOUT
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            raise Error('Call have had some serious problems:', OSError, 96)
        stdoutfile.close()
        if errfile:
            stderrfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write('\t%s\t\tRuntime: %s\t\tError Code: %i\n\n' % (endtime, endtime - starttime, runerror))
        os.chdir(prevdir)
        return runerror

    def writegeom(self):
        QMin = self._QMin
        factor = au2a
        fname = QMin['scratchdir'] + '/JOB/geom.xyz'
        string = '%i\n\n' % (QMin['natom'])
        for atom in QMin['geo']:
            string += atom[0]
            for xyz in range(1, 4):
                string += '  %f' % (atom[xyz] * factor)
            string += '\n'
        writefile(fname, string)

        os.chdir(QMin['scratchdir'] + '/JOB')
        error = sp.call('x2t geom.xyz > coord', shell=True)
        if error != 0:
            raise Error('xyz2col call failed!', 95)
        os.chdir(QMin['pwd'])

        # QM/MM
        if QMin['qmmm']:
            string = '$point_charges nocheck\n'
            for atom in QMin['pointcharges']:
                string += '%16.12f %16.12f %16.12f %12.9f\n' % (atom[0] / au2a, atom[1] / au2a, atom[2] / au2a, atom[3])
            string += '$end\n'
            filename = QMin['scratchdir'] + '/JOB/pc'
            writefile(filename, string)

        # COBRAMM
        if QMin['cobramm']:
            # chargefiles='charge.dat'
            # tocharge=os.path.join(QMin['scratchdir']+'/JOB/point_charges')
            # shutil.copy(chargefiles,tocharge)
            cobcharges = open('charge.dat', 'r')
            charges = cobcharges.read()
            only_atom = charges.split()
            only_atom.pop(0)
            filename = QMin['scratchdir'] + '/JOB/point_charges'
            string = '$point_charges nocheck\n'
            string += charges
            # counter=0
            # for atom in only_atom:
            #   	string+=atom
            #    string+=' '
            #    counter+=1
            #    if counter == 4:
            #      string+='\n'
            #      counter=0
            #    #string+='\n'
            string += '$end'
            writefile(filename, string)

    def get_wfovlout(self, path, mult):

        QMin = self._QMin
        QMout = self._QMout
        outfile = os.path.join(path, 'wfovl.out')
        out = readfile(outfile)

        if 'overlap' in QMin:
            nmstates = QMin['nmstates']
            if 'overlap' not in QMout:
                QMout['overlap'] = makecmatrix(nmstates, nmstates)
            # read the overlap matrix
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if not m1 == m2 == mult:
                        continue
                    if not ms1 == ms2:
                        continue
                    QMout['overlap'][i][j] = self.get_smatel(out, s1, s2)

        return

    def wfoverlap(self, scradir, mult):
        QMin = self._QMin
        # link all input files for wfoverlap
        savedir = QMin['savedir']
        link(os.path.join(savedir, 'ao_ovl'), os.path.join(scradir, 'ao_ovl'), crucial=True, force=True)
        link(os.path.join(savedir, 'mos.old'), os.path.join(scradir, 'mos.a'), crucial=True, force=True)
        link(os.path.join(savedir, 'mos'), os.path.join(scradir, 'mos.b'), crucial=True, force=True)
        if QMin['template']['method'] == 'cc2':
            link(os.path.join(savedir, 'dets_left.%i.old' % (mult)), os.path.join(scradir, 'dets.a'), crucial=True, force=True)
        else:
            link(os.path.join(savedir, 'dets.%i.old' % (mult)), os.path.join(scradir, 'dets.a'), crucial=True, force=True)
        link(os.path.join(savedir, 'dets.%i' % (mult)), os.path.join(scradir, 'dets.b'), crucial=True, force=True)

        # write input file for wfoverlap
        string = '''mix_aoovl=ao_ovl
    a_mo=mos.a
    b_mo=mos.b
    a_det=dets.a
    b_det=dets.b
    a_mo_read=2
    b_mo_read=2
    '''
        if 'ncore' in QMin:
            icore = QMin['ncore']
        elif 'frozenmap' in QMin:
            icore = QMin['frozenmap'][mult]
        else:
            icore = 0
        string += 'ncore=%i' % (icore)
        writefile(os.path.join(scradir, 'wfovl.inp'), string)

        # run wfoverlap
        string = '%s -f wfovl.inp -m %i' % (QMin['wfoverlap'], QMin['memory'])
        self.runProgram(string, scradir, 'wfovl.out')

    def copymolden(self):
        QMin = self._QMin
        # run tm2molden in scratchdir
        string = 'molden.input\nY\n'
        filename = os.path.join(QMin['scratchdir'], 'JOB', 'tm2molden.input')
        writefile(filename, string)
        string = 'tm2molden < tm2molden.input'
        path = os.path.join(QMin['scratchdir'], 'JOB')
        self.runProgram(string, path, 'tm2molden.output')

        if 'molden' in QMin:
            # create directory
            moldendir = QMin['savedir'] + '/MOLDEN/'
            if not os.path.isdir(moldendir):
                mkdir(moldendir)

            # save the molden.input file
            f = QMin['scratchdir'] + '/JOB/molden.input'
            fdest = moldendir + '/step_%s.molden' % (QMin['step'][0])
            shutil.copy(f, fdest)

    # ======================================================================= #
    def run_theodore(self):
        QMin = self._QMin
        workdir = os.path.join(QMin['scratchdir'], 'JOB')
        string = 'python2 %s/bin/analyze_tden.py' % (QMin['theodir'])
        runerror = self.runProgram(string, workdir, 'theodore.out')
        if runerror != 0:
            raise Error('Theodore calculation crashed! Error code=%i' % (runerror), 105)
        return

    def setupWORKDIR_TH(self):
        QMin = self._QMin
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

        WORKDIR = os.path.join(QMin['scratchdir'], 'JOB')
        # write dens_ana.in
        inputstring = '''rtype='ricc2'
    rfile='ricc2.out'
    mo_file='molden.input'
    jmol_orbitals=False
    molden_orbitals=%s
    read_binary=True
    comp_ntos=True
    alphabeta=False
    Om_formula=2
    eh_pop=1
    print_OmFrag=True
    output_file='tden_summ.txt'
    prop_list=%s
    at_lists=%s
    ''' % (('molden' in QMin),
            str(QMin['template']['theodore_prop']),
            str(QMin['template']['theodore_fragment']))

        filename = os.path.join(WORKDIR, 'dens_ana.in')
        writefile(filename, inputstring)
        return

    def get_theodore(self):
        QMin = self._QMin
        QMout = self._QMout
        if 'theodore' not in QMout:
            QMout['theodore'] = makecmatrix(QMin['template']['theodore_n'], QMin['nmstates'])
            sumfile = os.path.join(QMin['scratchdir'], 'JOB/tden_summ.txt')
            omffile = os.path.join(QMin['scratchdir'], 'JOB/OmFrag.txt')
            props = self.get_props(sumfile, omffile, QMin)
            for i in range(QMin['nmstates']):
                m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                if (m1, s1) in props:
                    for j in range(QMin['template']['theodore_n']):
                        QMout['theodore'][i][j] = props[(m1, s1)][j]
        return QMout

    # ======================================================================= #

    @staticmethod
    def get_props(sumfile, omffile):
        out = readfile(sumfile)
        props = {}

        def theo_float(x):
            return safe_cast(x, float, 0.)
        for line in out[2:]:
            s = line.replace('(', ' ').replace(')', ' ').split()
            if len(s) == 0:
                continue
            n = int(s[0])
            m = int(s[1])
            props[(m, n + (m == 1))] = [theo_float(i) for i in s[5:]]

        out = readfile(omffile)
        for line in out[1:]:
            s = line.replace('(', ' ').replace(')', ' ').split()
            if len(s) == 0:
                continue
            n = int(s[0])
            m = int(s[1])
            props[(m, n + (m == 1))].extend([theo_float(i) for i in s[4:]])

        return props

    # ======================================================================= #


    def copy_ntos(self):
        QMin = self._QMin
        # create directory
        moldendir = QMin['savedir'] + '/MOLDEN/'
        if not os.path.isdir(moldendir):
            mkdir(moldendir)

        # save the nto_x-x.a.mld files
        for i in QMin['statemap']:
            m, s, ms = QMin['statemap'][i]
            if m == 1 and s == 1:
                continue
            if m > 1 and ms != float(m - 1) / 2:
                continue
            f = os.path.join(QMin['scratchdir'], 'JOB', 'nto_%i-%i-a.mld' % (s - (m == 1), m))
            fdest = moldendir + '/step_%s__nto_%i_%i.molden' % (QMin['step'][0], m, s)
            shutil.copy(f, fdest)


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QM/MM ============================================= #
# =============================================================================================== #
# =============================================================================================== #


    def prepare_QMMM(self, table_file):
        ''' creates dictionary with:
        MM coordinates (including connectivity and atom types)
        QM coordinates (including Link atom stuff)
        point charge data (including redistribution for Link atom neighbors)
        reorder arrays (for internal processing, all QM, then all LI, then all MM)

        is only allowed to read the following keys from QMin:
        geo
        natom
        QM/MM related infos from template
        '''
        QMin = self._QMin
        table = readfile(table_file)


        # read table file
        print('===== Running QM/MM preparation ====')
        print('Reading table file ...         ', datetime.now())
        QMMM = {}
        QMMM['qmmmtype'] = []
        QMMM['atomtype'] = []
        QMMM['connect'] = []
        allowed = ['qm', 'mm']
        # read table file
        for iline, line in enumerate(table):
            s = line.split()
            if len(s) == 0:
                continue
            if not s[0].lower() in allowed:
                raise Error('Not allowed QMMM-type "%s" on line %i!' % (s[0], iline + 1), 34)
            QMMM['qmmmtype'].append(s[0].lower())
            QMMM['atomtype'].append(s[1])
            QMMM['connect'].append(set())
            for i in s[2:]:
                QMMM['connect'][-1].add(int(i) - 1)           # internally, atom numbering starts at 0
        QMMM['natom_table'] = len(QMMM['qmmmtype'])


        # list of QM and MM atoms
        QMMM['QM_atoms'] = []
        QMMM['MM_atoms'] = []
        for iatom in range(QMMM['natom_table']):
            if QMMM['qmmmtype'][iatom] == 'qm':
                QMMM['QM_atoms'].append(iatom)
            elif QMMM['qmmmtype'][iatom] == 'mm':
                QMMM['MM_atoms'].append(iatom)

        # make connections redundant and fill bond array
        print('Checking connection table ...  ', datetime.now())
        QMMM['bonds'] = set()
        for iatom in range(QMMM['natom_table']):
            for jatom in QMMM['connect'][iatom]:
                QMMM['bonds'].add(tuple(sorted([iatom, jatom])))
                QMMM['connect'][jatom].add(iatom)
        QMMM['bonds'] = sorted(list(QMMM['bonds']))


        # find link bonds
        print('Finding link bonds ...         ', datetime.now())
        QMMM['linkbonds'] = []
        QMMM['LI_atoms'] = []
        for i, j in QMMM['bonds']:
            if QMMM['qmmmtype'][i] != QMMM['qmmmtype'][j]:
                link = {}
                if QMMM['qmmmtype'][i] == 'qm':
                    link['qm'] = i
                    link['mm'] = j
                elif QMMM['qmmmtype'][i] == 'mm':
                    link['qm'] = j
                    link['mm'] = i
                link['scaling'] = {'qm': 0.3, 'mm': 0.7}
                link['element'] = 'H'
                link['atom'] = [link['element'], 0., 0., 0.]
                for xyz in range(3):
                    link['atom'][xyz + 1] += link['scaling']['mm'] * QMin['geo'][link['mm']][xyz + 1]
                    link['atom'][xyz + 1] += link['scaling']['qm'] * QMin['geo'][link['qm']][xyz + 1]
                QMMM['linkbonds'].append(link)
                QMMM['LI_atoms'].append(QMMM['natom_table'] - 1 + len(QMMM['linkbonds']))
                QMMM['atomtype'].append('999')
                QMMM['connect'].append(set([link['qm'], link['mm']]))


        # check link bonds
        mm_in_links = []
        qm_in_links = []
        mm_in_link_neighbors = []
        for link in QMMM['linkbonds']:
            mm_in_links.append(link['mm'])
            qm_in_links.append(link['qm'])
            for j in QMMM['connect'][link['mm']]:
                if QMMM['qmmmtype'][j] == 'mm':
                    mm_in_link_neighbors.append(j)
        mm_in_link_neighbors.extend(mm_in_links)
        # no QM atom is allowed to be bonded to two MM atoms
        if not len(qm_in_links) == len(set(qm_in_links)):
            raise Error('Some QM atom is involved in more than one link bond!', 35)
        # no MM atom is allowed to be bonded to two QM atoms
        if not len(mm_in_links) == len(set(mm_in_links)):
            raise Error('Some MM atom is involved in more than one link bond!', 36)
        # no neighboring MM atoms are allowed to be involved in link bonds
        if not len(mm_in_link_neighbors) == len(set(mm_in_link_neighbors)):
            raise Error('An MM-link atom is bonded to another MM-link atom!', 37)


        # check geometry and connection table
        if not QMMM['natom_table'] == QMin['natom']:
            raise Error('Number of atoms in table file does not match number of atoms in QMin!', 38)


        # process MM geometry (and convert to angstrom!)
        QMMM['MM_coords'] = []
        for atom in QMin['geo']:
            QMMM['MM_coords'].append([atom[0]] + [i * au2a for i in atom[1:4]])
        for ilink, link in enumerate(QMMM['linkbonds']):
            QMMM['MM_coords'].append(['HLA'] + link['atom'][1:4])


        # create reordering dicts
        print('Creating reorder mappings ...  ', datetime.now())
        QMMM['reorder_input_MM'] = {}
        QMMM['reorder_MM_input'] = {}
        j = -1
        for i, t in enumerate(QMMM['qmmmtype']):
            if t == 'qm':
                j += 1
                QMMM['reorder_MM_input'][j] = i
        for ilink, link in enumerate(QMMM['linkbonds']):
            j += 1
            QMMM['reorder_MM_input'][j] = QMMM['natom_table'] + ilink
        for i, t in enumerate(QMMM['qmmmtype']):
            if t == 'mm':
                j += 1
                QMMM['reorder_MM_input'][j] = i
        for i in QMMM['reorder_MM_input']:
            QMMM['reorder_input_MM'][QMMM['reorder_MM_input'][i]] = i


        # process QM geometry (including link atoms), QM coords in bohr!
        QMMM['QM_coords'] = []
        QMMM['reorder_input_QM'] = {}
        QMMM['reorder_QM_input'] = {}
        j = -1
        for iatom in range(QMMM['natom_table']):
            if QMMM['qmmmtype'][iatom] == 'qm':
                QMMM['QM_coords'].append(deepcopy(QMin['geo'][iatom]))
                j += 1
                QMMM['reorder_input_QM'][iatom] = j
                QMMM['reorder_QM_input'][j] = iatom
        for ilink, link in enumerate(QMMM['linkbonds']):
            QMMM['QM_coords'].append(link['atom'])
            j += 1
            QMMM['reorder_input_QM'][-(ilink + 1)] = j
            QMMM['reorder_QM_input'][j] = -(ilink + 1)


        # process charge redistribution around link bonds
        # point charges are in input geometry ordering
        print('Charge redistribution ...      ', datetime.now())
        QMMM['charge_distr'] = []
        for iatom in range(QMMM['natom_table']):
            if QMMM['qmmmtype'][iatom] == 'qm':
                QMMM['charge_distr'].append([(0., 0)])
            elif QMMM['qmmmtype'][iatom] == 'mm':
                if iatom in mm_in_links:
                    QMMM['charge_distr'].append([(0., 0)])
                else:
                    QMMM['charge_distr'].append([(1., iatom)])
        for link in QMMM['linkbonds']:
            mm_neighbors = []
            for j in QMMM['connect'][link['mm']]:
                if QMMM['qmmmtype'][j] == 'mm':
                    mm_neighbors.append(j)
            if len(mm_neighbors) > 0:
                factor = 1. / len(mm_neighbors)
                for j in QMMM['connect'][link['mm']]:
                    if QMMM['qmmmtype'][j] == 'mm':
                        QMMM['charge_distr'][j].append((factor, link['mm']))

        # pprint.pprint(QMMM)
        return QMMM

    def transform_QM_QMMM(self):
        QMin = self._QMin
        QMout = self._QMout
        # Meta data
        QMin['natom'] = QMin['natom_orig']
        QMin['geo'] = QMin['geo_orig']

        # Hamiltonian
        if 'h' in QMout:
            for i in range(QMin['nmstates']):
                QMout['h'][i][i] += QMin['qmmm']['MMEnergy']

        # Gradients
        if 'grad' in QMout:
            nmstates = QMin['nmstates']
            natom = QMin['natom_orig']
            grad = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
            # QM gradient
            for iqm in QMin['qmmm']['reorder_QM_input']:
                iqmmm = QMin['qmmm']['reorder_QM_input'][iqm]
                if iqmmm < 0:
                    ilink = -iqmmm - 1
                    link = QMin['qmmm']['linkbonds'][ilink]
                    for istate in range(nmstates):
                        for ixyz in range(3):
                            grad[istate][link['qm']][ixyz] += QMout['grad'][istate][iqm][ixyz] * link['scaling']['qm']
                            grad[istate][link['mm']][ixyz] += QMout['grad'][istate][iqm][ixyz] * link['scaling']['mm']
                else:
                    for istate in range(nmstates):
                        for ixyz in range(3):
                            grad[istate][iqmmm][ixyz] += QMout['grad'][istate][iqm][ixyz]
            # PC gradient
            # for iqm,iqmmm in enumerate(QMin['qmmm']['MM_atoms']):
            for iqm in QMin['qmmm']['reorder_pc_input']:
                iqmmm = QMin['qmmm']['reorder_pc_input'][iqm]
                for istate in range(nmstates):
                    for ixyz in range(3):
                        grad[istate][iqmmm][ixyz] += QMout['pcgrad'][istate][iqm][ixyz]
            # MM gradient
            for iqmmm in range(QMin['qmmm']['natom_table']):
                for istate in range(nmstates):
                    for ixyz in range(3):
                        grad[istate][iqmmm][ixyz] += QMin['qmmm']['MMGradient'][iqmmm][ixyz]
            QMout['grad'] = grad

        # pprint.pprint(QMout)
        return
    # ============================PRINTING ROUTINES========================== #


    def printheader(self):
        '''Prints the formatted header of the log file. Prints version number and version date
        Takes nothing, returns nothing.'''

        rule = '=' * 80
        lines = [f'  {rule}',
                 '',
                 f'SHARC - {self.__class__.__name__} - Interface',
                 '',
                 f'Authors: {self.authors}',
                 '',
                 f'Version: {self.version}',
                 'Date: {:%d.%m.%Y}'.format(self.versiondate),
                 '',
                 f'  {rule}']
        lines[1:-1] = map(lambda s: '||{:^80}||'.format(s), lines[1:-1])
        print(*lines, sep='\n')

    def printQMout(self, QMin, QMout):
        '''If PRINT, prints a summary of all requested QM output values. Matrices are formatted using printcomplexmatrix, vectors using printgrad.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout'''

        # if DEBUG:
        # pprint.pprint(QMout)
        if not self.PRINT:
            return
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        print('===> Results:\n')
        # Hamiltonian matrix, real or complex
        if 'h' in QMin or 'soc' in QMin:
            eshift = math.ceil(QMout['h'][0][0].real)
            print('=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f' % (eshift))
            matrix = deepcopy(QMout['h'])
            for i in range(nmstates):
                matrix[i][i] -= eshift
            printcomplexmatrix(matrix, states)
        # Dipole moment matrices
        if 'dm' in QMin:
            print('=> Dipole Moment Matrices:\n')
            for xyz in range(3):
                print('Polarisation %s:' % (IToPol[xyz]))
                matrix = QMout['dm'][xyz]
                printcomplexmatrix(matrix, states)
        # Gradients
        if 'grad' in QMin:
            print('=> Gradient Vectors:\n')
            istate = 0
            for imult, i, ms in itnmstates(states):
                print('%s\t%i\tMs= % .1f:' % (IToMult[imult], i, ms))
                printgrad(QMout['grad'][istate], natom, QMin['geo'])
                istate += 1
        # Overlaps
        if 'overlap' in QMin:
            print('=> Overlap matrix:\n')
            matrix = QMout['overlap']
            printcomplexmatrix(matrix, states)
            if 'phases' in QMout:
                print('=> Wavefunction Phases:\n')
                for i in range(nmstates):
                    print('% 3.1f % 3.1f' % (QMout['phases'][i].real, QMout['phases'][i].imag))
                print('\n')
        # Spin-orbit coupling derivatives
        if 'socdr' in QMin:
            print('=> Spin-Orbit Gradient Vectors:\n')
            istate = 0
            for imult, i, ims in itnmstates(states):
                jstate = 0
                for jmult, j, jms in itnmstates(states):
                    print('%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:' % (IToMult[imult], i, ims, IToMult[jmult], j, jms))
                    printgrad(QMout['socdr'][istate][jstate], natom, QMin['geo'])
                    jstate += 1
                istate += 1
        # Dipole moment derivatives
        if 'dmdr' in QMin:
            print('=> Dipole moment derivative vectors:\n')
            istate = 0
            for imult, i, msi in itnmstates(states):
                jstate = 0
                for jmult, j, msj in itnmstates(states):
                    if imult == jmult and msi == msj:
                        for ipol in range(3):
                            print('%s\tStates %i - %i\tMs= % .1f\tPolarization %s:' % (IToMult[imult], i, j, msi, IToPol[ipol]))
                            printgrad(QMout['dmdr'][ipol][istate][jstate], natom, QMin['geo'])
                    jstate += 1
                istate += 1
        # Property matrix (dyson norms)
        if 'ion' in QMin and 'prop' in QMout:
            print('=> Property matrix:\n')
            matrix = QMout['prop']
            printcomplexmatrix(matrix, states)
        # TheoDORE
        if 'theodore' in QMin:
            print('=> TheoDORE results:\n')
            matrix = QMout['theodore']
            printtheodore(matrix, QMin)
        sys.stdout.flush()

    # ======================================================================= #
    def printgrad(self, grad, natom, geo):
        '''Prints a gradient or nac vector. Also prints the atom elements. If the gradient is identical zero, just prints one line.

        Arguments:
        1 list of list of float: gradient
        2 integer: natom
        3 list of list: geometry specs'''

        string = ''
        iszero = True
        for atom in range(natom):
            if not self.DEBUG:
                if atom == 5:
                    string += '...\t...\t     ...\t     ...\t     ...\n'
                if 5 <= atom < natom - 1:
                    continue
            string += '%i\t%s\t' % (atom + 1, geo[atom][0])
            for xyz in range(3):
                if grad[atom][xyz] != 0:
                    iszero = False
                string += '% .5f\t' % (grad[atom][xyz])
            string += '\n'
        if iszero:
            print('\t\t...is identical zero...\n')
        else:
            print(string)


    def printtheodore(matrix, QMin):
        string = '%6s ' % 'State'
        for i in QMin['template']['theodore_prop']:
            string += '%6s ' % i
        for i in range(len(QMin['template']['theodore_fragment'])):
            for j in range(len(QMin['template']['theodore_fragment'])):
                string += '  Om%1i%1i ' % (i + 1, j + 1)
        string += '\n' + '-------' * (1 + QMin['template']['theodore_n']) + '\n'
        istate = 0
        for imult, i, ms in itnmstates(QMin['states']):
            istate += 1
            string += '%6i ' % istate
            for i in matrix[istate - 1]:
                string += '%6.4f ' % i.real
            string += '\n'
        print(string)

    # ======================================================================= #
# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #
    def writeQMout(self, QMinfilename):
        '''Writes the requested quantities to the file which SHARC reads in. The filename is QMinfilename with everything after the first dot replaced by "out".

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout
        3 string: QMinfilename'''
        QMin = self._QMin
        QMout = self._QMout
        k = QMinfilename.find('.')
        if k == -1:
            outfilename = QMinfilename + '.out'
        else:
            outfilename = QMinfilename[:k] + '.out'
        if self.PRINT:
            print('===> Writing output to file %s in SHARC Format\n' % (outfilename))
        string = ''
        if 'h' in QMin or 'soc' in QMin:
            string += self.writeQMoutsoc()
        if 'dm' in QMin:
            string += self.writeQMoutdm()
        if 'grad' in QMin:
            string += self.writeQMoutgrad()
        if 'overlap' in QMin:
            string += self.writeQMoutnacsmat()
        if 'socdr' in QMin:
            string += self.writeQMoutsocdr()
        if 'dmdr' in QMin:
            string += self.writeQMoutdmdr()
        if 'ion' in QMin:
            string += self.writeQMoutprop()
        if 'theodore' in QMin or QMin['template']['qmmm']:
            string += self.writeQMoutTHEODORE()
        if 'phases' in QMin:
            string += self.writeQmoutPhases()
        if 'grad' in QMin:
            if QMin['template']['cobramm']:
                self.writeQMoutgradcobramm()
        string += self.writeQMouttime()
        outfile = os.path.join(QMin['pwd'], outfilename)
        writefile(outfile, string)
        return

    def writeQMoutsoc(self):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the SOC matrix'''
        QMin = self._QMin
        QMout = self._QMout
        nmstates = QMin['nmstates']
        string = ''
        string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (eformat(QMout['h'][i][j].real, 12, 3), eformat(QMout['h'][i][j].imag, 12, 3))
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #


    def writeQMoutdm(self):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The string contains three such matrices.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the DM matrices'''
        QMin = self._QMin
        QMout = self._QMout
        nmstates = QMin['nmstates']
        string = ''
        string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (eformat(QMout['dm'][xyz][i][j].real, 12, 3), eformat(QMout['dm'][xyz][i][j].imag, 12, 3))
                string += '\n'
            string += ''
        return string

    # ======================================================================= #
    def writeQMoutdmdr(self):

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n' % (12, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                for ipol in range(3):
                    string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms, ipol)
                    for atom in range(natom):
                        for xyz in range(3):
                            string += '%s ' % (eformat(QMout['dmdr'][ipol][i][j][atom][xyz], 12, 3))
                        string += '\n'
                    string += ''
                j += 1
            i += 1
        string += '\n'
        return string

    # ======================================================================= #


    def writeQMoutsocdr(self):

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n' % (13, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms)
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s %s ' % (eformat(QMout['socdr'][i][j][atom][xyz].real, 12, 3), eformat(QMout['socdr'][i][j][atom][xyz].imag, 12, 3))
                string += '\n'
                string += ''
                j += 1
            i += 1
        string += '\n'
        return string

    def writeQMoutang(self):
        '''Generates a string with the Dipole moment matrices in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The string contains three such matrices.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the DM matrices'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Angular Momentum Matrices (3x%ix%i, complex)\n' % (9, nmstates, nmstates)
        for xyz in range(3):
            string += '%i %i\n' % (nmstates, nmstates)
            for i in range(nmstates):
                for j in range(nmstates):
                    string += '%s %s ' % (eformat(QMout['angular'][xyz][i][j].real, 12, 3), eformat(QMout['angular'][xyz][i][j].imag, 12, 3))
                string += '\n'
            string += ''
        return string

    # ======================================================================= #


    def writeQMoutgrad(self):
        '''Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the Gradient vectors'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += '%i %i ! m1 %i s1 %i ms1 %i\n' % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s ' % (eformat(QMout['grad'][i][atom][xyz], 12, 3))
                string += '\n'
            string += ''
            i += 1
        return string

    # ======================================================================= #


    def writeQMoutnacnum(self):
        '''Generates a string with the NAC matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the NAC matrix'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Non-adiabatic couplings (ddt) (%ix%i, complex)\n' % (4, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (eformat(QMout['nacdt'][i][j].real, 12, 3), eformat(QMout['nacdt'][i][j].imag, 12, 3))
            string += '\n'
        string += ''
        # also write wavefunction phases
        string += '! %i Wavefunction phases (%i, complex)\n' % (7, nmstates)
        for i in range(nmstates):
            string += '%s %s\n' % (eformat(QMout['phases'][i], 12, 3), eformat(0., 12, 3))
        string += '\n\n'
        return string

    # ======================================================================= #


    def writeQMoutnacana(self):
        '''Generates a string with the NAC vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the NAC vectors'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5, nmstates, nmstates, natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            j = 0
            for jmult, jstate, jms in itnmstates(states):
                # string+='%i %i ! %i %i %i %i %i %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms)
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms)
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s ' % (eformat(QMout['nacdr'][i][j][atom][xyz], 12, 3))
                    string += '\n'
                string += ''
                j += 1
            i += 1
        return string

    # ======================================================================= #


    def writeQMoutnacsmat(self):
        '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the transformation matrix'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for j in range(nmstates):
            for i in range(nmstates):
                string += '%s %s ' % (eformat(QMout['overlap'][j][i].real, 12, 3), eformat(QMout['overlap'][j][i].imag, 12, 3))
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #


    def writeQMouttime(self):
        '''Generates a string with the quantum mechanics total runtime in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the runtime is given

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the runtime'''

        QMout = self._QMout
        string = '! 8 Runtime\n%s\n' % (eformat(QMout['runtime'], 9, 3))
        return string

    # ======================================================================= #


    def writeQMoutprop(self):
        '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the SOC matrix'''

        QMin = self._QMin
        QMout = self._QMout
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        string = ''
        string += '! %i Property Matrix (%ix%i, complex)\n' % (11, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
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
                string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
            string += '\n'
        string += '\n'
        return string

    # ======================================================================= #


    def writeQMoutTHEODORE(self):

        QMin = self._QMin
        QMout = self._QMout
        nmstates = QMin['nmstates']
        nprop = QMin['template']['theodore_n']
        if QMin['template']['qmmm']:
            nprop += len(QMin['qmmm']['MMEnergy_terms'])
        if nprop <= 0:
            return '\n'

        string = ''

        string += '! %i Property Vectors\n' % (21)
        string += '%i    ! number of property vectors\n' % (nprop)

        string += '! Property Vector Labels (%i strings)\n' % (nprop)
        descriptors = []
        if 'theodore' in QMin:
            for i in QMin['template']['theodore_prop']:
                descriptors.append('%s' % i)
                string += descriptors[-1] + '\n'
            for i in range(len(QMin['template']['theodore_fragment'])):
                for j in range(len(QMin['template']['theodore_fragment'])):
                    descriptors.append('Om_{%i,%i}' % (i + 1, j + 1))
                    string += descriptors[-1] + '\n'
        if QMin['template']['qmmm']:
            for label in sorted(QMin['qmmm']['MMEnergy_terms']):
                descriptors.append(label)
                string += label + '\n'

        string += '! Property Vectors (%ix%i, real)\n' % (nprop, nmstates)
        if 'theodore' in QMin:
            for i in range(QMin['template']['theodore_n']):
                string += '! TheoDORE descriptor %i (%s)\n' % (i + 1, descriptors[i])
                for j in range(nmstates):
                    string += '%s\n' % (eformat(QMout['theodore'][j][i].real, 12, 3))
        if QMin['template']['qmmm']:
            for label in sorted(QMin['qmmm']['MMEnergy_terms']):
                string += '! QM/MM energy contribution (%s)\n' % (label)
                for j in range(nmstates):
                    string += '%s\n' % (eformat(QMin['qmmm']['MMEnergy_terms'][label], 12, 3))
        string += '\n'

        return string

    # ======================================================================= #


    def writeQmoutPhases(self):

        QMin = self._QMin
        QMout = self._QMout
        string = '! 7 Phases\n%i ! for all nmstates\n' % (QMin['nmstates'])
        for i in range(QMin['nmstates']):
            string += '%s %s\n' % (eformat(QMout['phases'][i].real, 9, 3), eformat(QMout['phases'][i].imag, 9, 3))
        return string



    def writeQMoutgradcobramm(self):
        '''Generates a string with the Gradient vectors in SHARC format.

        The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by      the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

        Arguments:
        1 dictionary: QMin
        2 dictionary: QMout

        Returns:
        1 string: multiline string with the Gradient vectors'''
        QMin = self._QMin
        QMout = self._QMout
        ncharges = len(readfile(os.path.join(QMin['scratchdir'], 'JOB', 'pc_grad'))) - 2
        states = QMin['states']
        nstates = QMin['nstates']
        nmstates = QMin['nmstates']
        natom = len(QMout['pcgrad'][0])
        print(QMout['pcgrad'][1])
        string = ''
        print(natom)
        # string+='! %i Gradient Vectors (%ix%ix3, real)\n' % (3,nmstates,natom)
        i = 0
        for imult, istate, ims in itnmstates(states):
            string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
            for atom in range(natom):
                for xyz in range(3):
                    print((QMout['pcgrad'][i][atom], 9, 3), i, atom)
                    string += '%s ' % (eformat(QMout['pcgrad'][i][atom][xyz], 9, 3))
                string += '\n'
            # string+='\n'
            i += 1
        string += '\n'
        writefile("grad_charges", string)


    # ======================================================================= #


    def backupdata(self, backupdir):
        # save all files in savedir, except which have 'old' in their name
        QMin = self._QMin
        ls = os.listdir(self._QMin['savedir'])
        for f in ls:
            ff = self._QMin['savedir'] + '/' + f
            if os.path.isfile(ff) and 'old' not in ff:
                step = int(self._QMin['step'][0])
                fdest = backupdir + '/' + f + '.stp' + str(step)
                shutil.copy(ff, fdest)
        # save molden files
        if 'molden' in QMin:
            ff = os.path.join(QMin['savedir'], 'MOLDEN', 'step_%s.molden' % (QMin['step'][0]))
            fdest = os.path.join(backupdir, 'step_%s.molden' % (QMin['step'][0]))
            shutil.copy(ff, fdest)
