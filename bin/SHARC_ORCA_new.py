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
import re
import time
import datetime
import traceback
from functools import reduce
from multiprocessing import Pool
from copy import deepcopy
from socket import gethostname

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import au2a, kcal_to_Eh, NUMBERS, BASISSETS, IToMult, rcm_to_Eh

authors = 'Sebastian Mai, Severin Polonius'
version = '3.0'
versiondate = datetime.datetime(2021, 7, 15)

changelogstring = '''
12.02.2016:     Initial version 0.1
- CC2 and ADC(2) from Turbomole
- SOC from Orca call
- only singlets and triplets
  => Doublets and Dyson could be added later

15.03.2016:     0.1.1
- ridft can be used for the SCF calculation, but dscf has to be run afterwards anyways (for SOC and overlaps).
- Laplace-transformed SOS-CC2/ADC(2) can be used ("spin-scaling lt-sos"), but does not work for soc or trans-dm
- if ricc2 does not converge, it is rerun with different combinations of npre and nstart (until convergence or too many tries)

07.03.2017:
- wfthres is now interpreted as in other interfaces (default is 0.99 now)

25.04.2017:
- can use external basis set libraries

23.08.2017
- Resource file is now called "RICC2.resources" instead of "SH2CC2.inp" (old filename still works)

24.08.2017:
- numfrozcore in resources file can now be used to override number of frozen cores for overlaps
- added Theodore capabilities (compute descriptors, OmFrag, and NTOs (also activate MOLDEN key for that))

11.11.2020:
- COBRAMM can be used for QM/MM calculations
'''


class ORCA(INTERFACE):

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

    def read_template(self, template_filename='ORCA.template'):
        '''reads the template file
        has to be called after setup_mol!'''

        if not self._read_resources:
            raise Error('Interface is not set up correctly. Call read_resources with the .resources file first!', 23)
        QMin = self._QMin
        # define keywords and defaults
        bools = {'no_tda': False,
                 'unrestricted_triplets': False,
                 'qmmm': False,
                 'cobramm': False,
                 'picture_change': False
                 }
        strings = {'basis': '6-31G',
                   'auxbasis': '',
                   'functional': 'PBE',
                   'dispersion': '',
                   'grid': '2',
                   'gridx': '',
                   'gridxc': '',
                   'ri': '',
                   'scf': '',
                   'qmmm_table': 'ORCA.qmmm.table',
                   'qmmm_ff_file': 'ORCA.ff',
                   'keys': '',
                   'paste_input_file': ''
                   }
        integers = {
            'frozen': -1,
            'maxiter': 700
        }
        floats = {
            'hfexchange': -1.,
            'intacc': -1.
        }
        special = {'paddingstates': [0 for i in QMin['states']],
                   'charge': [i % 2 for i in range(len(QMin['states']))],

                   'basis_per_element': {},
                   'ecp_per_element': {},
                   'basis_per_atom': {},
                   'range_sep_settings': {'do': False, 'mu': 0.14, 'scal': 1.0, 'ACM1': 0.0, 'ACM2': 0.0, 'ACM3': 1.0},
                   'paste_input_file': ''
                   }
        lines = readfile(template_filename)
        QMin['template'] = {**bools, **strings, **integers, **floats, **special,
                            **self.parse_keywords(bools, strings, integers, floats, special, lines)}

        # do logic checks
        if not QMin['template']['unrestricted_triplets']:
            if QMin['OrcaVersion'] < (4, 1):
                if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                    raise Error('With Orca v<4.1, triplets can only be computed with the unrestricted_triplets option!', 62)
            if len(QMin['template']['charge']) >= 3 and QMin['template']['charge'][0] != QMin['template']['charge'][2]:
                raise Error('Charges of singlets and triplets differ. Please enable the "unrestricted_triplets" option!', 63)
        if QMin['template']['unrestricted_triplets'] and 'soc' in QMin:
            if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                raise Error('Request "SOC" is not compatible with "unrestricted_triplets"!', 64)
        if QMin['template']['ecp_per_element'] and 'soc' in QMin:
            if len(QMin['states']) >= 3 and QMin['states'][2] > 0:
                raise Error('Request "SOC" is not compatible with using ECPs!', 64)

        self._read_template = True
        return


    def read_resources(self, resources_filename="ORCA.resources"):

        if not self._setup_mol:
            raise Error('Interface is not set up for this template. Call setup_mol with the QM.in file first!', 23)
        QMin = self._QMin

        pwd = os.getcwd()
        QMin['pwd'] = pwd

        strings = {'orcadir': '',
                   'tinker': '',
                   'scratchdir': '',
                   'savedir': '',  # NOTE: savedir from QMin
                   'theodir': '',
                   'wfoverlap': os.path.join(os.path.expandvars(os.path.expanduser('$SHARC')), 'wfoverlap.x'),
                   'qmmm_table': '',
                   'qmmm_ff_file': ''
                   }
        bools = {'debug': False,
                 'save_stuff': False,
                 'no_print': False,
                 'no_overlap': False,
                 'always_orb_init': False,
                 'always_guess': False,
                 }
        integers = {'ncpu': 1,
                    'memory': 100,
                    'numfrozcore': 0,
                    'numocc': 0
                    }
        floats = {'delay': 0.0,
                  'schedule_scaling': 0.9,
                  'wfthres': 0.99
                  }
        special = {
            'neglect_gradient': 'zero',
            'theodore_prop': ['Om', 'PRNTO', 'S_HE', 'Z_HE', 'RMSeh'],
            'theodore_fragment': [],
        }
        lines = readfile(resources_filename)
        # assign defaults first, which get updated by the parsed entries, which are updated by the entries that were already in QMin
        QMin['resources'] = {**bools, **strings, **integers, **floats, **special,
                             **self.parse_keywords(bools, strings, integers, floats, special, lines)}
        paths = {k: QMin['resources'][k] for k in strings.keys()}
        for k, v in paths.items():
            vlist = map(lambda x: os.path.expanduser(os.path.expandvars(x)) if len(x) != 0 and x[0] == '$' else x, v.split('/'))
            QMin['resources'][k] = '/'.join(vlist)

        # reassign QMin after losing the reference
        QMin['OrcaVersion'] = self._getOrcaVersion(QMin['resources']['orcadir'])

        # NOTE: This is reall optional
        ncpu = QMin['resources']['ncpu']
        if os.environ.get('NSLOTS') is not None:
            ncpu = int(os.environ.get('NSLOTS'))
            print(f'Detected $NSLOTS variable. Will use ncpu={ncpu}')
        elif os.environ.get('SLURM_NTASKS_PER_NODE') is not None:
            ncpu = int(os.environ.get('SLURM_NTASKS_PER_NODE'))
            print('Detected $SLURM_NTASKS_PER_NODE variable. Will use ncpu={ncpu}')
        QMin['resources']['ncpu'] = max(1, ncpu)

        if 0 < QMin['resources']['schedule_scaling'] <= 1.:
            QMin['resources']['schedule_scaling'] = 0.9
        if 'always_orb_init' in QMin and 'always_guess' in QMin:
            raise Error('Keywords "always_orb_init" and "always_guess" cannot be used together!', 53)
        self._QMin = {**QMin['resources'], **QMin}
        self._read_resources = True
        return


    def set_requests(self, QMinfilename):
        raise NotImplementedError

    def moveOldFiles(self):

        def move(fromf, tof):
            if not os.path.isfile(fromf):
                raise Error(f'File {fromf} not found, cannot move to {tof}!', 78)
            if PRINT:
                print(shorten_DIR(fromf) + '   =>   ' + shorten_DIR(tof))
            shutil.copy(fromf, tof)

        QMin = self._QMin

        if self._PRINT:
            print('>' * 15, 'Moving old Files')
        basenames = ['ORCA.gbw', 'ORCA.molden']
        if 'nooverlap' not in QMin:
            basenames.append('mos')
        step = QMin['step']
        for job in QMin['joblist']:
            for base in basenames:
                fromfile = os.path.join(QMin['savedir'], f'{base}.{job}')
                tofile = os.path.join(QMin['savedir'], f'{base}_{step}.{job}')
                move(fromfile, tofile)
        if 'nooverlap' in QMin:
            for job in itmult(QMin['states']):
                fromfile = os.path.join(QMin['savedir'], f'dets.{job}')
                tofile = os.path.join(QMin['savedir'], f'dets_{step}.{job}')
                move(fromfile, tofile)

        for f in ['AO_overl', 'AO_overl.mixed']:
            rmfile = os.path.join(QMin['savedir'], f)
            if os.path.isfile(rmfile):
                os.remove(rmfile)
                if PRINT:
                    print('rm ' + rmfile)
        return

    def runjobs(self, schedule):

        def error_handler(e: BaseException, WORKDIR):
            print('*' * 50 + '\nException in run_calc(%s)!' % (WORKDIR))
            traceback.print_exc()
            print('*' * 50 + '\n')
            raise e
        QMin = self._QMin
        if 'newstep' in QMin:
            self.moveOldFiles()
        print('>' * 15, 'Starting the ORCA job execution')
        errorcodes = {}
        for ijobset, jobset in enumerate(schedule):
            if not jobset:
                continue
            pool = Pool(processes=QMin['nslots_pool'][ijobset])
            for job in jobset:
                QMin1 = jobset[job]
                WORKDIR = os.path.join(QMin['scratchdir'], job)
                errorcodes[job] = pool.apply_async(ORCA.runORCA, [WORKDIR, QMin1], error_callback=lambda e: error_handler(e, WORKDIR))
                time.sleep(QMin['delay'])
        string = 'Error Codes:\n'
        success = True
        for j, job in enumerate(errorcodes):
            code = errorcodes.get()
            if code != 0:
                success = False
            string += '\t{}\t{}'.format(job + ' ' * (10 - len(job)), code)
            if (j + 1) % 4 == 0:
                string += '\n'
        print(string)
        if not success:
            raise Error('Some subprocesses did not finish successfully!\n\
                See {}:{} for error messages in ORCA output.'.format(gethostname(), QMin['scratchdir']), 75)


    @staticmethod
    def runORCA(WORKDIR, QMin):
        ORCA.setupWORKDIR(WORKDIR, QMin)
        orcadir = QMin['orcadir']
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = os.path.join(orcadir, 'orca') + ' ORCA.inp'
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
            sys.stdout.flush()
        stdoutfile = open(os.path.join(WORKDIR, 'ORCA.log'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'ORCA.err'), 'w')
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            raise Error('Call have had some serious problems:', OSError, 76)
        stdoutfile.close()
        stderrfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
            sys.stdout.flush()
        os.chdir(prevdir)
        if not DEBUG and runerror == 0:
            keep = ['ORCA.inp$', 'ORCA.err$', 'ORCA.log$', 'ORCA.gbw', 'ORCA.cis', 'ORCA.engrad', 'ORCA.pcgrad', 'ORCA.molden.input', 'ORCA.pc']
            INTERFACE.stripWORKDIR(WORKDIR, keep)
        return runerror

    @staticmethod
    def setupWORKDIR(WORKDIR, QMin):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
        # then put the ORCA.inp file

        # setup the directory
        mkdir(WORKDIR)

        # write ORCA.inp
        inputstring = ORCA.writeORCAinput(QMin)
        filename = os.path.join(WORKDIR, 'ORCA.inp')
        writefile(filename, inputstring)
        if DEBUG:
            print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
            print(inputstring)
            print('ORCA input written to: %s' % (filename))
            print('====================================================================')
        # write point charges
        if QMin['qmmm']:
            inputstring = ORCA.write_pccoord_file(QMin)
            filename = os.path.join(WORKDIR, 'ORCA.pc')
            writefile(filename, inputstring)
            if DEBUG:
                print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
                print(inputstring)
                print('Point charges written to: %s' % (filename))
                print('====================================================================')
        if QMin['template']['cobramm']:
            currentDirectory = os.getcwd()
            fromfile = os.path.join(currentDirectory, 'charge.dat')
            tofile = tofile = os.path.join(WORKDIR, 'charge.dat')
            shutil.copy(fromfile, tofile)

        # wf file copying
        if 'master' in QMin:
            job = QMin['IJOB']
            if job in QMin['initorbs']:
                fromfile = QMin['initorbs'][job]
                tofile = os.path.join(WORKDIR, 'ORCA.gbw')
                shutil.copy(fromfile, tofile)
        elif 'grad' in QMin:
            job = QMin['IJOB']
            fromfile = os.path.join(QMin['scratchdir'], 'master_%i' % job, 'ORCA.gbw')
            tofile = os.path.join(WORKDIR, 'ORCA.gbw')
            shutil.copy(fromfile, tofile)

        return

    def writeORCAinput(QMin):
        # split gradmap into smaller chunks
        Nmax_gradlist = 255
        print('GRADMAP', QMin['gradmap'])
        gradmaps = [sorted(QMin['gradmap'])[i:i + Nmax_gradlist] for i in range(0, len(QMin['gradmap']), Nmax_gradlist)]

        # make multi-job input
        string = ''
        for ichunk, chunk in enumerate(gradmaps):
            if ichunk >= 1:
                string += '\n\n$new_job\n\n%base "ORCA"\n\n'
            QMin_copy = deepcopy(QMin)
            QMin_copy['gradmap'] = chunk
            string += ORCA.ORCAinput_string(QMin_copy)
        if not gradmaps:
            string += ORCA.ORCAinput_string(QMin)
        return string


    # ======================================================================= #


    @staticmethod
    def ORCAinput_string(QMin):
        # pprint.pprint(QMin)

        # general setup
        job = QMin['IJOB']
        gsmult = QMin['multmap'][-job][0]
        restr = QMin['jobs'][job]['restr']
        charge = QMin['chargemap'][gsmult]

        # excited states to calculate
        states_to_do = QMin['states_to_do']
        for imult in range(len(states_to_do)):
            if not imult + 1 in QMin['multmap'][-job]:
                states_to_do[imult] = 0
        states_to_do[gsmult - 1] -= 1

        # do minimum number of states for gradient jobs
        if 'gradonly' in QMin:
            gradmult = QMin['gradmap'][0][0]
            gradstat = QMin['gradmap'][0][1]
            for imult in range(len(states_to_do)):
                if imult + 1 == gradmult:
                    states_to_do[imult] = gradstat - (gradmult == gsmult)
                else:
                    states_to_do[imult] = 0

        # number of states to calculate
        if restr:
            ncalc = max(states_to_do)
            trip = (len(states_to_do) >= 3 and states_to_do[2] > 0)
        else:
            ncalc = max(states_to_do)

        # gradients
        multigrad = False
        if 'grad' in QMin and QMin['gradmap']:
            dograd = True
            egrad = ()
            for grad in QMin['gradmap']:
                if not (gsmult, 1) == grad:
                    egrad = grad
            if QMin['OrcaVersion'] >= (4, 1):
                multigrad = True
                singgrad = []
                tripgrad = []
                for grad in QMin['gradmap']:
                    if grad[0] == gsmult:
                        singgrad.append(grad[1] - 1)
                    if grad[0] == 3 and restr:
                        tripgrad.append(grad[1])
        else:
            dograd = False

        # construct the input string
        string = ''

        # main line
        string += '! '

        keys = ['basis',
                'auxbasis',
                'functional',
                'dispersion',
                'ri',
                'keys']
        for i in keys:
            string += '%s ' % (QMin['template'][i])
        string += 'nousesym '

        string += 'grid%s ' % QMin['template']['grid']
        if QMin['template']['gridx']:
            string += 'gridx%s ' % QMin['template']['gridx']
    # In this way, one can change grid on individual atoms:
    # %method
    # SpecialGridAtoms 26,15,-1,-4         # for element 26 and, for atom index 1 and 4 (cannot change on atom 0!)
    # SpecialGridIntAcc 7,6,5,5            # size of grid
    # end

        if dograd:
            string += 'engrad'
        string += '\n'

        # cpu cores
        if QMin['ncpu'] > 1 and 'AOoverlap' not in QMin:
            string += '%pal\n  nprocs %i\nend\n\n' % (QMin['ncpu'])
        string += '%maxcore %i\n\n' % (QMin['memory'])

        # basis sets
        if QMin['template']['basis_per_element']:
            string += '%basis\n'
            for i in QMin['template']['basis_per_element']:
                string += 'newgto %s "%s" end\n' % (i, QMin['template']['basis_per_element'][i])
            if not QMin['template']['ecp_per_element']:
                string += 'end\n\n'

        # ECP basis sets
        if QMin['template']['ecp_per_element']:
            if QMin['template']['basis_per_element']:
                for i in QMin['template']['ecp_per_element']:
                    string += 'newECP %s "%s" end\n' % (i, QMin['template']['ecp_per_element'][i])
                string += 'end\n\n'
            else:
                print("ECP defined without additional basis. Not implemented.")

        # frozen core
        if QMin['frozcore'] > 0:
            string += '%%method\nfrozencore -%i\nend\n\n' % (2 * QMin['frozcore'])
        else:
            string += '%method\nfrozencore FC_NONE\nend\n\n'

        # hf exchange
        if QMin['template']['hfexchange'] >= 0.:
            # string+='%%method\nScalHFX = %f\nScalDFX = %f\nend\n\n' % (QMin['template']['hfexchange'],1.-QMin['template']['hfexchange'])
            string += '%%method\nScalHFX = %f\nend\n\n' % (QMin['template']['hfexchange'])

        # Range separation
        if QMin['template']['range_sep_settings']['do']:
            string += '''%%method
    RangeSepEXX True
    RangeSepMu %f
    RangeSepScal %f
    ACM %f, %f, %f\nend\n\n
    ''' % (QMin['template']['range_sep_settings']['mu'],
                QMin['template']['range_sep_settings']['scal'],
                QMin['template']['range_sep_settings']['ACM1'],
                QMin['template']['range_sep_settings']['ACM2'],
                QMin['template']['range_sep_settings']['ACM3']
           )

        # Intacc
        if QMin['template']['intacc'] > 0.:
            string += '''%%method
    intacc %3.1f\nend\n\n''' % (QMin['template']['intacc'])


        # Gaussian point charge scheme
        if 'cpcm' in QMin['template']['keys'].lower():
            string += '''%cpcm
    surfacetype vdw_gaussian\nend\n\n'''


        # excited states
        if ncalc > 0 and 'AOoverlap' not in QMin:
            string += '%tddft\n'
            if not QMin['template']['no_tda']:
                string += 'tda true\n'
            else:
                string += 'tda false\n'
            if QMin['template']['gridxc']:
                string += 'gridxc %s\n' % (QMin['template']['gridxc'])
            if 'theodore' in QMin:
                string += 'tprint 0.0001\n'
            if restr and trip:
                string += 'triplets true\n'
            string += 'nroots %i\n' % (ncalc)
            if restr and 'soc' in QMin:
                string += 'dosoc true\n'
                string += 'printlevel 3\n'
            # string+="dotrans all\n" #TODO
            if dograd:
                if multigrad:
                    if singgrad:
                        string += 'sgradlist '
                        string += ','.join([str(i) for i in sorted(singgrad)])
                        string += '\n'
                    if tripgrad:
                        string += 'tgradlist '
                        string += ','.join([str(i) for i in sorted(tripgrad)])
                        string += '\n'
                elif egrad:
                    string += 'iroot %i\n' % (egrad[1] - (gsmult == egrad[0]))
            string += 'end\n\n'

        # output
        string += '%output\n'
        if 'AOoverlap' in QMin or 'ion' in QMin or 'theodore' in QMin:
            string += 'Print[ P_Overlap ] 1\n'
        if 'master' in QMin or 'theodore' in QMin:
            string += 'Print[ P_MOs ] 1\n'
        string += 'end\n\n'

        # scf
        string += '%scf\n'
        if 'AOoverlap' in QMin:
            string += 'maxiter 0\n'
        else:
            string += 'maxiter %i\n' % (QMin['template']['maxiter'])
        string += 'end\n\n'

        # rel
        if QMin['template']['picture_change']:
            string += '%rel\nPictureChange true\nend\n\n'


        # TODO: workaround
        # if 'soc' in QMin and 'grad' in QMin:
            # string+='%rel\nonecenter true\nend\n\n'

        # charge mult geom
        string += '%coords\nCtyp xyz\nunits bohrs\n'
        if 'AOoverlap' in QMin:
            string += 'charge %i\n' % (2. * charge)
        else:
            string += 'charge %i\n' % (charge)
        string += 'mult %i\n' % (gsmult)
        string += 'coords\n'
        for iatom, atom in enumerate(QMin['geo']):
            label = atom[0]
            string += '%4s %16.9f %16.9f %16.9f' % (label, atom[1], atom[2], atom[3])
            if iatom in QMin['template']['basis_per_atom']:
                string += ' newgto "%s" end' % (QMin['template']['basis_per_atom'][iatom])
            string += '\n'
        string += 'end\nend\n\n'

        # point charges
        if QMin['qmmm']:
            string += '%pointcharges "ORCA.pc"\n\n'
        elif QMin['template']['cobramm']:
            string += '%pointcharges "charge.dat"\n\n'
        if QMin['template']['paste_input_file']:
            string += '\n'
            for line in QMin['template']['paste_input_file']:
                string += line
            string += '\n'
        return string

    def run(self):

        QMin = self._QMin
        # TODO: specific logic checks!!!
        if 'nacdt' in QMin or 'nacdr' in QMin:
            raise Error('Within the SHARC-ORCA interface couplings can only be calculated via the overlap method. "nacdr" and "nacdt" are not supported.', 44)

        schedule = self.generate_joblist()
        self.printQMin()
        pprint.pprint(schedule, depth=3)
        self.runjobs(schedule)

    def get_QMout(self):
        raise NotImplementedError

    def main(self):
        raise NotImplementedError

    @staticmethod
    def _getOrcaVersion(path):
        # run orca with nonexisting file
        string = os.path.join(path, 'orca') + ' nonexisting'
        try:
            proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        except OSError:
            ty, val, tb = sys.exc_info()
            raise Error(f'Orca call has had some serious problems:\n {ty.__name__}: {val}', 25).with_traceback(tb)
        comm = proc.communicate()[0].decode()
        # find version string
        for line in comm.split('\n'):
            if 'Program Version' in line:
                s = line.split('-')[0].split()[2].split('.')
                s = tuple(int(i) for i in s)
                return s
        raise Error('Could not find Orca version!', 26)
