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
from os import stat
import pprint
import sys
import math
import time
import datetime
import struct
from functools import reduce
from multiprocessing import Pool
from copy import deepcopy
from socket import gethostname

# internal
from SHARC_INTERFACE import INTERFACE
from globals import DEBUG, PRINT
from utils import *
from constants import IToMult, rcm_to_Eh

authors = 'Sebastian Mai, Lea Ibele, Moritz Heindl and Severin Polonius'
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
                 'nooverlap': False,
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
        if QMin['nooverlap']:
            basenames.append('mos')
        step = QMin['step']
        for job in QMin['joblist']:
            for base in basenames:
                fromfile = os.path.join(QMin['savedir'], f'{base}.{job}')
                tofile = os.path.join(QMin['savedir'], f'{base}_{step}.{job}')
                move(fromfile, tofile)
        if QMin['nooverlap']:
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
            print(str(e), e.__traceback__)
            print('*' * 50 + '\n')
            return
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
                errorcodes[job] = pool.apply_async(ORCA.runORCA, [WORKDIR, QMin1], error_callback=lambda e: error_handler(e, WORKDIR)).get()
                time.sleep(QMin['delay'])
        string = 'Error Codes:\n'
        success = True
        for j, job in enumerate(errorcodes):
            code = errorcodes[job]
            if code != 0:
                success = False
            string += '\t{}\t{}'.format(job + ' ' * (10 - len(job)), code)
            if (j + 1) % 4 == 0:
                string += '\n'
        print(string)
        if not success:
            raise Error('Some subprocesses did not finish successfully!\n\
                See {}:{} for error messages in ORCA output.'.format(gethostname(), QMin['scratchdir']), 75)
        if PRINT:
            print('>>>>>>>>>>>>> Saving files')
            starttime = datetime.datetime.now()
        for ijobset, jobset in enumerate(schedule):
            if not jobset:
                continue
            for ijob, job in enumerate(jobset):
                if 'master' in job:
                    WORKDIR = os.path.join(QMin['scratchdir'], job)
                    # if not 'samestep' in QMin or 'molden' in QMin:
                    # if 'molden' in QMin or not 'nooverlap' in QMin:
                    # saveMolden(WORKDIR,jobset[job])
                    if 'samestep' not in QMin:
                        ORCA.saveFiles(WORKDIR, jobset[job])
                    if 'ion' in QMin and ijobset == 0 and ijob == 0:
                        ORCA.saveAOmatrix(WORKDIR, QMin)
        # saveGeometry(QMin)
        if PRINT:
            endtime = datetime.datetime.now()
            print('Saving Runtime: %s' % (endtime - starttime))
        return errorcodes

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
    # --------------------------------------------- File setup ----------------------------------

        # check for initial orbitals
        initorbs = {}
        if 'always_guess' in QMin:
            QMin['initorbs'] = {}
        elif 'init' in QMin or 'always_orb_init' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['pwd'], 'ORCA.gbw.init')
                if os.path.isfile(filename):
                    initorbs[job] = filename
            for job in QMin['joblist']:
                filename = os.path.join(QMin['pwd'], 'ORCA.gbw.%i.init' % (job))
                if os.path.isfile(filename):
                    initorbs[job] = filename
            if 'always_orb_init' in QMin and len(initorbs) < QMin['njobs']:
                print('Initial orbitals missing for some jobs!')
                sys.exit(70)
            QMin['initorbs'] = initorbs
        elif 'newstep' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['savedir'], f'ORCA.gbw.{job}')
                if os.path.isfile(filename):
                    initorbs[job] = filename + '.old'     # file will be moved to .old
                else:
                    print(f'File {filename} missing in savedir!')
                    sys.exit(71)
            QMin['initorbs'] = initorbs
        elif 'samestep' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['savedir'], f'ORCA.gbw.{job}')
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    print(f'File {filename} missing in savedir!')
                    sys.exit(72)
            QMin['initorbs'] = initorbs
        elif 'restart' in QMin:
            for job in QMin['joblist']:
                filename = os.path.join(QMin['savedir'], 'ORCA.gbw.{job}.old' % (job))
                if os.path.isfile(filename):
                    initorbs[job] = filename
                else:
                    print(f'File {filename} missing in savedir!')
                    sys.exit(73)
            QMin['initorbs'] = initorbs


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
                string += '\n\n$new_job\n\n%%base "ORCA"\n\n'
            QMin_copy = deepcopy(QMin)
            QMin_copy['gradmap'] = chunk
            string += ORCA.ORCAinput_string(QMin_copy)
        if not gradmaps:
            string += ORCA.ORCAinput_string(QMin)
        return string


    # ======================================================================= #


    @staticmethod
    def ORCAinput_string(QMin):

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
            string += f'%pal\n  nprocs {QMin["resources"]["ncpu"]}\nend\n\n'
        string += f'%maxcore {QMin["resources"]["memory"]}\n\n'

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
            string += '%%method\nfrozencore FC_NONE\nend\n\n'

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
        for iatom, xyz in enumerate(QMin['coords']):
            label = QMin['elements'][iatom]
            string += '%4s %16.9f %16.9f %16.9f' % (label, xyz[0], xyz[1], xyz[2])
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
        # self.printQMin()
        # pprint.pprint(schedule, depth=3)
        errorcodes = self.runjobs(schedule)
        print('ERRORCODES', errorcodes)
        errorcodes = self.run_wfoverlap(errorcodes)
        print('ERRORCODES', errorcodes)
        errorcodes = self.run_theodore(errorcodes)
        print('ERRORCODES', errorcodes)

        self._QMout = self.getQMout()

        if 'backup' in QMin:
            self.backupdata(QMin['backup'])

        self._QMout['runtime'] = self.clock.measuretime()
        # Remove Scratchfiles from SCRATCHDIR
        if not self._DEBUG:
            cleandir(QMin['scratchdir'])
            if 'cleanup' in QMin:
                cleandir(QMin['savedir'])

    def run_theodore(self, errorcodes):
        QMin = self._QMin
        if 'theodore' in QMin:
            print('>>>>>>>>>>>>> Starting the TheoDORE job execution')

            for ijob in QMin['jobs']:
                if not QMin['jobs'][ijob]['restr']:
                    if DEBUG:
                        print('Skipping Job %s because it is unrestricted.' % (ijob))
                    continue
                else:
                    mults = QMin['jobs'][ijob]['mults']
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += QMin['states'][i - 1] - (i == gsmult)
                    if ns == 0:
                        if DEBUG:
                            print('Skipping Job %s because it contains no excited states.' % (ijob))
                        continue
                WORKDIR = os.path.join(QMin['scratchdir'], 'master_%i' % ijob)
                self.setupWORKDIR_TH(WORKDIR)
                os.environ
                errorcodes['theodore_%i' % ijob] = ORCA.runTHEODORE(WORKDIR, QMin['theodir'])

            # Error code handling
            j = 0
            string = 'Error Codes:\n'
            for i in errorcodes:
                if 'theodore' in i:
                    string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
                    j += 1
                    if j == 4:
                        j = 0
                        string += '\n'
            print(string)
            if any((i != 0 for i in errorcodes.values())):
                print('Some subprocesses did not finish successfully!')
                sys.exit(98)

            print('')

        return errorcodes


    def setupWORKDIR_TH(self, WORKDIR):
        # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
        QMin = self._QMin
        # write dens_ana.in
        inputstring = '''rtype='cclib'
    rfile='ORCA.log'
    read_binary=True
    jmol_orbitals=False
    molden_orbitals=False
    Om_formula=2
    eh_pop=1
    comp_ntos=True
    print_OmFrag=True
    output_file='tden_summ.txt'
    prop_list=%s
    at_lists=%s
    ''' % (str(QMin['resources']['theodore_prop']), str(QMin['resources']['theodore_fragment']))

        filename = os.path.join(WORKDIR, 'dens_ana.in')
        writefile(filename, inputstring)
        fromfile = os.path.join(WORKDIR, 'ORCA.cis')
        tofile = os.path.join(WORKDIR, 'orca.cis')
        link(fromfile, tofile)
        if DEBUG:
            print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
            print(inputstring)
            print('TheoDORE input written to: %s' % (filename))
            print('====================================================================')

        return

    # ======================================================================= #

    @staticmethod
    def runTHEODORE(WORKDIR, THEODIR):
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = os.path.join(THEODIR, 'bin', 'analyze_tden.py')
        stdoutfile = open(os.path.join(WORKDIR, 'theodore.out'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'theodore.err'), 'w')
        if PRINT or DEBUG:
            starttime = datetime.datetime.now()
            sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
            sys.stdout.flush()
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(99)
        stdoutfile.close()
        stderrfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
            sys.stdout.flush()
        os.chdir(prevdir)
        return runerror


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

    @staticmethod
    def saveFiles(WORKDIR, QMin):

        # copy the gbw files from master directories
        job = QMin['IJOB']
        fromfile = os.path.join(WORKDIR, 'ORCA.gbw')
        tofile = os.path.join(QMin['savedir'], 'ORCA.gbw.%i' % (job))
        shutil.copy(fromfile, tofile)
        if PRINT:
            print(shorten_DIR(tofile))

        # make Molden files and copy to savedir
        ORCA.saveMolden(WORKDIR, QMin)

        # if necessary, extract the MOs and write them to savedir
        if 'ion' in QMin or QMin['nooverlap']:
            mofile = os.path.join(QMin['savedir'], 'mos.%i' % job)
            f = os.path.join(WORKDIR, 'ORCA.gbw')
            string = ORCA.get_MO_from_gbw(f, QMin)
            writefile(mofile, string)
            if PRINT:
                print(shorten_DIR(mofile))

        # if necessary, extract the TDDFT coefficients and write them to savedir
        if 'ion' in QMin or QMin['nooverlap']:
            f = os.path.join(WORKDIR, 'ORCA.cis')
            strings = ORCA.get_dets_from_cis(f, QMin)
            for f in strings:
                writefile(f, strings[f])
                if PRINT:
                    print(shorten_DIR(f))
    # ======================================================================= #

    @staticmethod
    def saveAOmatrix(WORKDIR, QMin):
        # filename=os.path.join(WORKDIR,'ORCA.log')
        # NAO,Smat=get_smat(filename)
        # filename=os.path.join(WORKDIR,'ORCA.molden.input')
        # NAO,Smat=get_smat_from_Molden(filename)
        filename = os.path.join(WORKDIR, 'ORCA.gbw')
        NAO, Smat = ORCA.get_smat_from_gbw(filename)

        string = '%i %i\n' % (NAO, NAO)
        for irow in range(NAO):
            for icol in range(NAO):
                string += '% .7e ' % (Smat[icol][irow])
            string += '\n'
        filename = os.path.join(QMin['savedir'], 'AO_overl')
        writefile(filename, string)
        if PRINT:
            print(shorten_DIR(filename))

    # ======================================================================= #
    @staticmethod
    def saveMolden(WORKDIR, QMin):

        # run orca_2mkl
        prevdir = os.getcwd()
        os.chdir(WORKDIR)
        string = 'orca_2mkl ORCA -molden'
        stdoutfile = open(os.path.join(WORKDIR, 'orca_2mkl.out'), 'w')
        stderrfile = open(os.path.join(WORKDIR, 'orca_2mkl.err'), 'w')
        try:
            runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(79)
        stdoutfile.close()
        stderrfile.close()
        if PRINT or DEBUG:
            endtime = datetime.datetime.now()
            # sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR),endtime,endtime-starttime,runerror))
            sys.stdout.flush()
        os.chdir(prevdir)

        job = QMin['IJOB']
        fromfile = os.path.join(WORKDIR, 'ORCA.molden.input')
        tofile = os.path.join(QMin['savedir'], 'ORCA.molden.%i' % (job))
        shutil.copy(fromfile, tofile)
        if PRINT:
            print(shorten_DIR(tofile))
    # ======================================================================= #


    @staticmethod
    def get_smat_from_gbw(file1, file2=''):

        if not file2:
            file2 = file1

        # run orca_fragovl
        string = 'orca_fragovl %s %s' % (file1, file2)
        try:
            proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(89)
        comm = proc.communicate()[0].decode()
        out = comm.split('\n')

        # get size of matrix
        for line in reversed(out):
            # print line
            s = line.split()
            if len(s) >= 1:
                NAO = int(line.split()[0]) + 1
                break

        # read matrix
        nblock = 6
        ao_ovl = [[0. for i in range(NAO)] for j in range(NAO)]
        for x in range(NAO):
            for y in range(NAO):
                block = x // nblock
                xoffset = x % nblock + 1
                yoffset = block * (NAO + 1) + y + 10
                ao_ovl[x][y] = float(out[yoffset].split()[xoffset])

        return NAO, ao_ovl

    # ======================================================================= #
    @staticmethod
    def get_MO_from_gbw(filename, QMin):

        # run orca_fragovl
        string = 'orca_fragovl %s %s' % (filename, filename)
        try:
            proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        except OSError:
            print('Call have had some serious problems:', OSError)
            sys.exit(80)
        comm = proc.communicate()[0].decode()
        data = comm.split('\n')
        # get size of matrix
        for line in reversed(data):
            # print line
            s = line.split()
            if len(s) >= 1:
                NAO = int(line.split()[0]) + 1
                break

        job = QMin['IJOB']
        restr = QMin['jobs'][job]['restr']

        # find MO block
        iline = -1
        while True:
            iline += 1
            if len(data) <= iline:
                print('MOs not found!')
                sys.exit(81)
            line = data[iline]
            if 'FRAGMENT A MOs MATRIX' in line:
                break
        iline += 3

        # formatting
        nblock = 6
        npre = 11
        ndigits = 16
        # default_pos=[14,30,46,62,78,94]
        default_pos = [npre + 3 + ndigits * i for i in range(nblock)]  # does not include shift

        # get coefficients for alpha
        NMO_A = NAO
        MO_A = [[0. for i in range(NAO)] for j in range(NMO_A)]
        for imo in range(NMO_A):
            jblock = imo // nblock
            jcol = imo % nblock
            for iao in range(NAO):
                shift = max(0, len(str(iao)) - 3)
                jline = iline + jblock * (NAO + 1) + iao
                line = data[jline]
                # fix too long floats in strings
                dots = [idx for idx, item in enumerate(line.lower()) if '.' in item]
                diff = [dots[i] - default_pos[i] - shift for i in range(len(dots))]
                if jcol == 0:
                    pre = 0
                else:
                    pre = diff[jcol - 1]
                post = diff[jcol]
                # fixed
                val = float(line[npre + shift + jcol * ndigits + pre: npre + shift + ndigits + jcol * ndigits + post])
                MO_A[imo][iao] = val
        iline += ((NAO - 1) // nblock + 1) * (NAO + 1)

        # coefficients for beta
        if not restr:
            NMO_B = NAO
            MO_B = [[0. for i in range(NAO)] for j in range(NMO_B)]
            for imo in range(NMO_B):
                jblock = imo // nblock
                jcol = imo % nblock
                for iao in range(NAO):
                    shift = max(0, len(str(iao)) - 3)
                    jline = iline + jblock * (NAO + 1) + iao
                    line = data[jline]
                    # fix too long floats in strings
                    dots = [idx for idx, item in enumerate(line.lower()) if '.' in item]
                    diff = [dots[i] - default_pos[i] - shift for i in range(len(dots))]
                    if jcol == 0:
                        pre = 0
                    else:
                        pre = diff[jcol - 1]
                    post = diff[jcol]
                    # fixed
                    val = float(line[npre + shift + jcol * ndigits + pre: npre + shift + ndigits + jcol * ndigits + post])
                    MO_B[imo][iao] = val


        NMO = NMO_A - QMin['frozcore']
        if restr:
            NMO = NMO_A - QMin['frozcore']
        else:
            NMO = NMO_A + NMO_B - 2 * QMin['frozcore']

        # make string
        string = '''2mocoef
    header
    1
    MO-coefficients from Orca
    1
    %i   %i
    a
    mocoef
    (*)
    ''' % (NAO, NMO)
        x = 0
        for imo, mo in enumerate(MO_A):
            if imo < QMin['frozcore']:
                continue
            for c in mo:
                if x >= 3:
                    string += '\n'
                    x = 0
                string += '% 6.12e ' % c
                x += 1
            if x > 0:
                string += '\n'
                x = 0
        if not restr:
            x = 0
            for imo, mo in enumerate(MO_B):
                if imo < QMin['frozcore']:
                    continue
                for c in mo:
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

        return string

    # ======================================================================= #

    @staticmethod
    def get_dets_from_cis(filename, QMin):

        # get general infos
        job = QMin['IJOB']
        restr = QMin['jobs'][job]['restr']
        mults = QMin['jobs'][job]['mults']
        gsmult = QMin['multmap'][-job][0]
        nstates_to_extract = deepcopy(QMin['states'])
        nstates_to_skip = [QMin['states_to_do'][i] - QMin['states'][i] for i in range(len(QMin['states']))]
        for i in range(len(nstates_to_extract)):
            if not i + 1 in mults:
                nstates_to_extract[i] = 0
                nstates_to_skip[i] = 0
            elif i + 1 == gsmult:
                nstates_to_extract[i] -= 1
        # print job,restr,mults,gsmult,nstates_to_extract

        # get infos from logfile
        logfile = os.path.join(os.path.dirname(filename), 'ORCA.log')
        data = readfile(logfile)
        infos = {}
        for iline, line in enumerate(data):
            if '# of contracted basis functions' in line:
                infos['nbsuse'] = int(line.split()[-1])
            if 'Orbital ranges used for CIS calculation:' in line:
                s = data[iline + 1].replace('.', ' ').split()
                infos['NFC'] = int(s[3])
                infos['NOA'] = int(s[4]) - int(s[3]) + 1
                infos['NVA'] = int(s[7]) - int(s[6]) + 1
                if restr:
                    infos['NOB'] = infos['NOA']
                    infos['NVB'] = infos['NVA']
                else:
                    s = data[iline + 2].replace('.', ' ').split()
                    infos['NOB'] = int(s[4]) - int(s[3]) + 1
                    infos['NVB'] = int(s[7]) - int(s[6]) + 1

        if 'NOA' not in infos:
            nstates_onfile = 0
            charge = QMin['chargemap'][gsmult]
            nelec = float(QMin['Atomcharge'] - charge)
            infos['NOA'] = int(nelec / 2. + float(gsmult - 1) / 2.)
            infos['NOB'] = int(nelec / 2. - float(gsmult - 1) / 2.)
            infos['NVA'] = infos['nbsuse'] - infos['NOA']
            infos['NVB'] = infos['nbsuse'] - infos['NOB']
            infos['NFC'] = 0
        else:
            # get all info from cis file
            CCfile = open(filename, 'rb')
            nvec = struct.unpack('i', CCfile.read(4))[0]
            header = [struct.unpack('i', CCfile.read(4))[0] for i in range(8)]
            # print infos
            # print header
            if infos['NOA'] != header[1] - header[0] + 1:
                print('Number of orbitals in %s not consistent' % filename)
                sys.exit(82)
            if infos['NVA'] != header[3] - header[2] + 1:
                print('Number of orbitals in %s not consistent' % filename)
                sys.exit(83)
            if not restr:
                if infos['NOB'] != header[5] - header[4] + 1:
                    print('Number of orbitals in %s not consistent' % filename)
                    sys.exit(84)
                if infos['NVB'] != header[7] - header[6] + 1:
                    print('Number of orbitals in %s not consistent' % filename)
                    sys.exit(85)
            if QMin['template']['no_tda']:
                nstates_onfile = nvec // 2
            else:
                nstates_onfile = nvec


        # get ground state configuration
        # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
        if restr:
            occ_A = [3 for i in range(infos['NFC'] + infos['NOA'])] + [0 for i in range(infos['NVA'])]
        if not restr:
            occ_A = [1 for i in range(infos['NFC'] + infos['NOA'])] + [0 for i in range(infos['NVA'])]
            occ_B = [2 for i in range(infos['NFC'] + infos['NOB'])] + [0 for i in range(infos['NVB'])]
        occ_A = tuple(occ_A)
        if not restr:
            occ_B = tuple(occ_B)

        # get infos
        nocc_A = infos['NOA']
        nvir_A = infos['NVA']
        nocc_B = infos['NOB']
        nvir_B = infos['NVB']

        # get eigenvectors
        eigenvectors = {}
        for imult, mult in enumerate(mults):
            eigenvectors[mult] = []
            if mult == gsmult:
                # add ground state
                if restr:
                    key = tuple(occ_A[QMin['frozcore']:])
                else:
                    key = tuple(occ_A[QMin['frozcore']:] + occ_B[QMin['frozcore']:])
                eigenvectors[mult].append({key: 1.0})
            for istate in range(nstates_to_extract[mult - 1]):
                CCfile.read(40)
                dets = {}
                for iocc in range(header[0], header[1] + 1):
                    for ivirt in range(header[2], header[3] + 1):
                        dets[(iocc, ivirt, 1)] = struct.unpack('d', CCfile.read(8))[0]
                if not restr:
                    for iocc in range(header[4], header[5] + 1):
                        for ivirt in range(header[6], header[7] + 1):
                            dets[(iocc, ivirt, 2)] = struct.unpack('d', CCfile.read(8))[0]
                if QMin['template']['no_tda']:
                    CCfile.read(40)
                    for iocc in range(header[0], header[1] + 1):
                        for ivirt in range(header[2], header[3] + 1):
                            dets[(iocc, ivirt, 1)] += struct.unpack('d', CCfile.read(8))[0]
                            dets[(iocc, ivirt, 1)] /= 2.
                    if not restr:
                        for iocc in range(header[4], header[5] + 1):
                            for ivirt in range(header[6], header[7] + 1):
                                dets[(iocc, ivirt, 2)] += struct.unpack('d', CCfile.read(8))[0]
                                dets[(iocc, ivirt, 2)] /= 2.

                # pprint.pprint(dets)
                # truncate vectors
                norm = 0.
                for k in sorted(dets, key=lambda x: dets[x]**2, reverse=True):
                    factor = 1.
                    if norm > factor * QMin['wfthres']:
                        del dets[k]
                        continue
                    norm += dets[k]**2
                # pprint.pprint(dets)
                # create strings and expand singlets
                dets2 = {}
                if restr:
                    for iocc, ivirt, dummy in dets:
                        # singlet
                        if mult == 1:
                            # alpha excitation
                            key = list(occ_A)
                            key[iocc] = 2
                            key[ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] * math.sqrt(0.5)
                            # beta excitation
                            key[iocc] = 1
                            key[ivirt] = 2
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] * math.sqrt(0.5)
                        # triplet
                        elif mult == 3:
                            key = list(occ_A)
                            key[iocc] = 1
                            key[ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                else:
                    for iocc, ivirt, dummy in dets:
                        if dummy == 1:
                            key = list(occ_A + occ_B)
                            key[iocc] = 0
                            key[ivirt] = 1
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                        elif dummy == 2:
                            key = list(occ_A + occ_B)
                            key[infos['NFC'] + nocc_A + nvir_A + iocc] = 0
                            key[infos['NFC'] + nocc_A + nvir_A + ivirt] = 2
                            dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                # pprint.pprint(dets2)
                # remove frozen core
                dets3 = {}
                for key in dets2:
                    problem = False
                    if restr:
                        if any([key[i] != 3 for i in range(QMin['frozcore'])]):
                            problem = True
                    else:
                        if any([key[i] != 1 for i in range(QMin['frozcore'])]):
                            problem = True
                        if any([key[i] != 2 for i in range(nocc_A + nvir_A + QMin['frozcore'], nocc_A + nvir_A + 2 * QMin['frozcore'])]):
                            problem = True
                    if problem:
                        print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                        continue
                        # sys.exit(86)
                    if restr:
                        key2 = key[QMin['frozcore']:]
                    else:
                        key2 = key[QMin['frozcore']:QMin['frozcore'] + nocc_A + nvir_A] + key[nocc_A + nvir_A + 2 * QMin['frozcore']:]
                    dets3[key2] = dets2[key]
                # pprint.pprint(dets3)
                # append
                eigenvectors[mult].append(dets3)
            # skip extra roots
            for istate in range(nstates_to_skip[mult - 1]):
                CCfile.read(40)
                for iocc in range(header[0], header[1] + 1):
                    for ivirt in range(header[2], header[3] + 1):
                        CCfile.read(8)
                if not restr:
                    for iocc in range(header[4], header[5] + 1):
                        for ivirt in range(header[6], header[7] + 1):
                            CCfile.read(8)
                if QMin['template']['no_tda']:
                    CCfile.read(40)
                    for iocc in range(header[0], header[1] + 1):
                        for ivirt in range(header[2], header[3] + 1):
                            CCfile.read(8)
                    if not restr:
                        for iocc in range(header[4], header[5] + 1):
                            for ivirt in range(header[6], header[7] + 1):
                                CCfile.read(8)


        strings = {}
        for imult, mult in enumerate(mults):
            filename = os.path.join(QMin['savedir'], 'dets.%i' % mult)
            strings[filename] = ORCA.format_ci_vectors(eigenvectors[mult])

        return strings

    @staticmethod
    def format_ci_vectors(ci_vectors):

        # get nstates, norb and ndets
        alldets = set()
        for dets in ci_vectors:
            for key in dets:
                alldets.add(key)
        ndets = len(alldets)
        nstates = len(ci_vectors)
        norb = len(next(iter(alldets)))

        string = '%i %i %i\n' % (nstates, norb, ndets)
        for det in sorted(alldets, reverse=True):
            for o in det:
                if o == 0:
                    string += 'e'
                elif o == 1:
                    string += 'a'
                elif o == 2:
                    string += 'b'
                elif o == 3:
                    string += 'd'
            for istate in range(len(ci_vectors)):
                if det in ci_vectors[istate]:
                    string += ' %11.7f ' % ci_vectors[istate][det]
                else:
                    string += ' %11.7f ' % 0.
            string += '\n'
        return string

    # ======================================================================= #
    def get_Double_AOovl(self):
        QMin = self._QMin
        # get geometries
        job = sorted(QMin['jobs'].keys())[0]
        filename1 = os.path.join(QMin['savedir'], 'ORCA.gbw.%i.old' % job)
        filename2 = os.path.join(QMin['savedir'], 'ORCA.gbw.%i' % job)

        # NAO,Smat=get_smat_from_Molden(filename1,filename2)
        NAO, Smat = ORCA.get_smat_from_gbw(filename1, filename2)

        # Smat is already off-diagonal block matrix NAO*NAO
        # we want the lower left quarter, but transposed
        string = '%i %i\n' % (NAO, NAO)
        for irow in range(0, NAO):
            for icol in range(0, NAO):
                string += '% .15e ' % (Smat[irow][icol])          # note the exchanged indices => transposition
            string += '\n'
        filename = os.path.join(QMin['savedir'], 'AO_overl.mixed')
        writefile(filename, string)
        return

    # =============================================================================================== #
    # =============================================================================================== #
    # ====================================== ORCA output parsing ================================ #
    # =============================================================================================== #
    # =============================================================================================== #

    def getQMout(self):
        QMin = self._QMin

        if PRINT:
            print('>>>>>>>>>>>>> Reading output files')
        starttime = datetime.datetime.now()

        QMout = {}
        nmstates = QMin['nmstates']
        natom = QMin['natom']
        joblist = QMin['joblist']



        # Hamiltonian
        if 'h' in QMin or 'soc' in QMin:
            # make Hamiltonian
            if 'h' not in QMout:
                QMout['h'] = makecmatrix(nmstates, nmstates)
            # go through all jobs
            for job in joblist:
                # first get energies from TAPE21
                logfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.log' % (job))
                energies = self.getenergy(logfile, job)
                # print energies
                # also get SO matrix and mapping
                if 'soc' in QMin and QMin['jobs'][job]['restr']:
                    submatrix, invstatemap = ORCA.getsocm(logfile)
                mults = QMin['multmap'][-job]
                if 3 in mults and QMin['OrcaVersion'] < (4, 1):
                    mults = [3]
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                        m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                        if m1 not in mults or m2 not in mults:
                            continue
                        if i == j:
                            QMout['h'][i][j] = energies[(m1, s1)]
                        elif 'soc' in QMin and QMin['jobs'][job]['restr']:
                            if m1 == m2 == 1:
                                continue
                            x = invstatemap[(m1, s1, ms1)]
                            y = invstatemap[(m2, s2, ms2)]
                            QMout['h'][i][j] = submatrix[x - 1][y - 1]

        # Dipole Moments
        if 'dm' in QMin:
            # make matrix
            if 'dm' not in QMout:
                QMout['dm'] = [makecmatrix(nmstates, nmstates) for i in range(3)]
            # go through all jobs
            for job in joblist:
                logfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.log' % (job))
                dipoles = self.gettdm(logfile, job)
                mults = QMin['multmap'][-job]
                if 3 in mults and QMin['OrcaVersion'] < (4, 1):
                    mults = [3]
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    if m1 not in QMin['jobs'][job]['mults']:
                        continue
                    for j in range(nmstates):
                        m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                        if m2 not in QMin['jobs'][job]['mults']:
                            continue
                        if i == j:
                            # TODO: does not work with restricted triplet
                            #isgs= (s1==1)
                            isgs = (QMin['gsmap'][i + 1] == i + 1)
                            if isgs:
                                logfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.log' % (job))
                            elif (m1, s1) in QMin['gradmap']:
                                path, isgs = QMin['jobgrad'][(m1, s1)]
                                logfile = os.path.join(QMin['scratchdir'], path, 'ORCA.log')
                            else:
                                continue
                            dm = ORCA.getdm(logfile, isgs)
                            for ixyz in range(3):
                                QMout['dm'][ixyz][i][j] = dm[ixyz]
                            continue
                        if not m1 == m2 == mults[0] or not ms1 == ms2:
                            continue
                        if s1 == 1:
                            for ixyz in range(3):
                                QMout['dm'][ixyz][i][j] = dipoles[(m2, s2)][ixyz]
                        elif s2 == 1:
                            for ixyz in range(3):
                                QMout['dm'][ixyz][i][j] = dipoles[(m1, s1)][ixyz]

        # Gradients
        if 'grad' in QMin:
            if 'grad' not in QMout:
                QMout['grad'] = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
            if QMin['qmmm'] and 'pcgrad' not in QMout:
                QMout['pcgrad'] = [[[0. for i in range(3)] for j in QMin['pointcharges']] for k in range(nmstates)]
            if QMin['template']['cobramm']:
                ncharges = len(readfile("charge.dat")) - 1
                QMout['pcgrad'] = [[[0. for i in range(3)] for j in range(ncharges)] for k in range(nmstates)]
            for grad in QMin['gradmap']:
                path, isgs = QMin['jobgrad'][grad]
                gsmult = QMin['jobs'][int(path.split('_')[1])]['mults'][0]
                restr = QMin['jobs'][int(path.split('_')[1])]['restr']
                if isgs:
                    fname = '.ground'
                    if QMin['states'][gsmult - 1] == 1:
                        fname = ''
                else:
                    if restr:
                        fname = '.' + IToMult[grad[0]].lower() + '.root%i' % (grad[1] - (grad[0] == gsmult))
                    else:
                        fname = '.singlet.root%i' % (grad[1] - (grad[0] == gsmult))
                logfile = os.path.join(QMin['scratchdir'], path, 'ORCA.engrad' + fname)
                g = ORCA.getgrad(logfile, natom)
                # print g
                if QMin['qmmm']:
                    logfile = os.path.join(QMin['scratchdir'], path, 'ORCA.pcgrad' + fname)
                    gpc = ORCA.getpcgrad(logfile)
                for istate in QMin['statemap']:
                    state = QMin['statemap'][istate]
                    # print grad,istate,state
                    if (state[0], state[1]) == grad:
                        QMout['grad'][istate - 1] = g
                        if QMin['qmmm']:
                            QMout['pcgrad'][istate - 1] = gpc
                if QMin['template']['cobramm']:
                    logfile = os.path.join(QMin['scratchdir'], path, 'ORCA.pcgrad' + fname)
                    gpc = ORCA.getpcgrad(logfile)
                for istate in QMin['statemap']:
                    state = QMin['statemap'][istate]
                    # print grad,istate,state
                    if (state[0], state[1]) == grad:
                        QMout['grad'][istate - 1] = g
                        if QMin['template']['cobramm']:
                            QMout['pcgrad'][istate - 1] = gpc
            if QMin['neglected_gradient'] != 'zero':
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    if not (m1, s1) in QMin['gradmap']:
                        if QMin['neglected_gradient'] == 'gs':
                            j = QMin['gsmap'][i + 1] - 1
                        elif QMin['neglected_gradient'] == 'closest':
                            e1 = QMout['h'][i][i]
                            de = 999.
                            for grad in QMin['gradmap']:
                                for k in range(nmstates):
                                    m2, s2, ms2 = tuple(QMin['statemap'][k + 1])
                                    if grad == (m2, s2):
                                        break
                                e2 = QMout['h'][k][k]
                                if de > abs(e1 - e2):
                                    de = abs(e1 - e2)
                                    j = k
                        QMout['grad'][i] = QMout['grad'][j]
                        if QMin['qmmm']:
                            QMout['pcgrad'][i] = QMout['pcgrad'][j]

        # Regular Overlaps
        if 'overlap' in QMin:
            if 'overlap' not in QMout:
                QMout['overlap'] = makecmatrix(nmstates, nmstates)
            for mult in itmult(QMin['states']):
                job = QMin['multmap'][mult]
                outfile = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
                out = readfile(outfile)
                if PRINT:
                    print('Overlaps: ' + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                        m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                        if not m1 == m2 == mult:
                            continue
                        if not ms1 == ms2:
                            continue
                        QMout['overlap'][i][j] = ORCA.getsmate(out, s1, s2)

        # Phases from overlaps
        if 'phases' in QMin:
            if 'phases' not in QMout:
                QMout['phases'] = [complex(1., 0.) for i in range(nmstates)]
            if 'overlap' in QMout:
                for i in range(nmstates):
                    if QMout['overlap'][i][i].real < 0.:
                        QMout['phases'][i] = complex(-1., 0.)

        # Dyson norms
        if 'ion' in QMin:
            if 'prop' not in QMout:
                QMout['prop'] = makecmatrix(nmstates, nmstates)
            for ion in QMin['ionmap']:
                outfile = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
                out = readfile(outfile)
                if PRINT:
                    print('Dyson:    ' + shorten_DIR(outfile))
                for i in range(nmstates):
                    for j in range(nmstates):
                        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                        m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                        if not (ion[0], ion[2]) == (m1, m2) and not (ion[0], ion[2]) == (m2, m1):
                            continue
                        if not abs(ms1 - ms2) == 0.5:
                            continue
                        # switch multiplicities such that m1 is smaller mult
                        if m1 > m2:
                            s1, s2 = s2, s1
                            m1, m2 = m2, m1
                            ms1, ms2 = ms2, ms1
                        # compute M_S overlap factor
                        if ms1 < ms2:
                            factor = (ms1 + 1. + (m1 - 1.) / 2.) / m1
                        else:
                            factor = (-ms1 + 1. + (m1 - 1.) / 2.) / m1
                        QMout['prop'][i][j] = ORCA.getDyson(out, s1, s2) * factor

        # TheoDORE
        if 'theodore' in QMin:
            if 'theodore' not in QMout:
                QMout['theodore'] = makecmatrix(QMin['template']['theodore_n'], nmstates)
            for job in joblist:
                if not QMin['jobs'][job]['restr']:
                    continue
                else:
                    mults = QMin['jobs'][job]['mults']
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += QMin['states'][i - 1] - (i == gsmult)
                    if ns == 0:
                        continue
                sumfile = os.path.join(QMin['scratchdir'], 'master_%i/tden_summ.txt' % job)
                omffile = os.path.join(QMin['scratchdir'], 'master_%i/OmFrag.txt' % job)
                props = ORCA.get_theodore(sumfile, omffile)
                for i in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    if (m1, s1) in props:
                        for j in range(QMin['template']['theodore_n']):
                            QMout['theodore'][i][j] = props[(m1, s1)][j]

        # QM/MM energy terms

        # transform back from QM to QM/MM
        # if QMin['template']['qmmm']:
        #     QMin, QMout = transform_QM_QMMM(QMin, QMout)

        endtime = datetime.datetime.now()
        if PRINT:
            print("Readout Runtime: %s" % (endtime - starttime))

        if DEBUG:
            # pprint.pprint(QMout)
            copydir = os.path.join(QMin['savedir'], 'debug_ORCA_stdout')
            if not os.path.isdir(copydir):
                mkdir(copydir)
            for job in joblist:
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.log' % (job))
                shutil.copy(outfile, os.path.join(copydir, "ORCA_%i.log" % job))
                if QMin['jobs'][job]['restr'] and 'theodore' in QMin:
                    outfile = os.path.join(QMin['scratchdir'], 'master_%i/tden_summ.txt' % job)
                    try:
                        shutil.copy(outfile, os.path.join(copydir, 'THEO_%i.out' % (job)))
                    except IOError:
                        pass
                    outfile = os.path.join(QMin['scratchdir'], 'master_%i/OmFrag.txt' % job)
                    try:
                        shutil.copy(outfile, os.path.join(copydir, 'THEO_OMF_%i.out' % (job)))
                    except IOError:
                        pass
            if 'grad' in QMin:
                for grad in QMin['gradmap']:
                    path, isgs = QMin['jobgrad'][grad]
                    outfile = os.path.join(QMin['scratchdir'], path, 'ORCA.log')
                    shutil.copy(outfile, os.path.join(copydir, "ORCA_GRAD_%i_%i.log" % grad))
            if 'overlap' in QMin:
                for mult in itmult(QMin['states']):
                    job = QMin['multmap'][mult]
                    outfile = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i/wfovl.out' % (mult, job))
                    shutil.copy(outfile, os.path.join(copydir, 'WFOVL_%i_%i.out' % (mult, job)))
            if 'ion' in QMin:
                for ion in QMin['ionmap']:
                    outfile = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i/wfovl.out' % ion)
                    shutil.copy(outfile, os.path.join(copydir, 'Dyson_%i_%i_%i_%i.out' % ion))

        if QMin['save_stuff']:
            copydir = os.path.join(QMin['savedir'], 'save_stuff')
            if not os.path.isdir(copydir):
                mkdir(copydir)
            for job in joblist:
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.log' % (job))
                shutil.copy(outfile, os.path.join(copydir, "ORCA_%i.log" % job))
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.gbw' % (job))
                shutil.copy(outfile, os.path.join(copydir, "ORCA_%i.gbw" % job))
                outfile = os.path.join(QMin['scratchdir'], 'master_%i/ORCA.cis' % (job))
                shutil.copy(outfile, os.path.join(copydir, "ORCA_%i.cis" % job))

        return QMout

    # ======================================================================= #


    def getenergy(self, logfile, ijob):

        QMin = self._QMin

        # open file
        f = readfile(logfile)
        if PRINT:
            print('Energy:   ' + shorten_DIR(logfile))

        # read ground state
        for iline, line in enumerate(f):
            if 'TOTAL SCF ENERGY' in line:
                gsenergy = float(f[iline + 3].split()[3])
            if 'Dispersion correction' in line:
                gsenergy += float(line.split()[-1])
                break

        # figure out the excited state settings
        mults = QMin['jobs'][ijob]['mults']
        restr = QMin['jobs'][ijob]['restr']
        gsmult = mults[0]
        estates_to_extract = deepcopy(QMin['states'])
        estates_to_extract[gsmult - 1] -= 1
        for imult in range(len(estates_to_extract)):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0
        for imult in range(len(estates_to_extract)):
            if imult + 1 in mults:
                estates_to_extract[imult] = max(estates_to_extract)
        # TODO: restricted triplet energies

        # extract excitation energies
        # loop also works if no energies should be extracted
        energies = {(gsmult, 1): gsenergy}
        for imult in mults:
            nstates = estates_to_extract[imult - 1]
            # print nstates
            if nstates > 0:
                strings = [['TD-DFT/TDA',
                            'TD-DFT',
                            'RPA',
                            'CIS'],
                           ['EXCITED STATES']
                           ]
                if QMin['OrcaVersion'] >= (4, 1):
                    if restr:
                        if imult == 1:
                            strings.append(['SINGLETS'])
                        if imult == 3:
                            strings.append(['TRIPLETS'])
                for iline, line in enumerate(f):
                    if all([any([i in line for i in st]) for st in strings]):
                        # print line
                        # if 'TD-DFT/TDA EXCITED STATES' in line or 'TD-DFT EXCITED STATES' in line or 'RPA EXCITED STATES' in line or 'CIS-EXCITED STATES' in line:
                        # if QMin['OrcaVersion']>=(4,1):
                        break
                finalstring = ['Entering ', '-EXCITATION SPECTRA']
                while True:
                    iline += 1
                    if iline >= len(f):
                        print('Error in parsing excitation energies')
                        sys.exit(102)
                    line = f[iline]
                    if any([i in line for i in finalstring]):
                        break
                    if 'STATE' in line:
                        # print line
                        s = line.replace(':', ' ').split()
                        e = gsenergy + float(s[-2]) * rcm_to_Eh
                        i = int(s[1])
                        if i > nstates:
                            break
                        energies[(imult, i + (gsmult == imult))] = e
        return energies

    ## ======================================================================= #

    @staticmethod
    def getsocm(outfile):

        # read the standard out into memory
        out = readfile(outfile)
        if PRINT:
            print('SOC:      ' + shorten_DIR(outfile))

        # get number of states (nsing=ntrip in Orca)
        for line in out:
            if 'Number of roots to be determined' in line:
                nst = int(line.split()[-1])
                break
        nrS = nst
        nrT = nst

        # make statemap for the state ordering of the SO matrix
        inv_statemap = {}
        inv_statemap[(1, 1, 0.0)] = 1
        i = 1
        for x in range(nrS):
            i += 1
            inv_statemap[(1, x + 2, 0.0)] = i
        spin = [0.0, -1.0, +1.0]
        for y in range(3):
            for x in range(nrT):
                i += 1
                inv_statemap[(3, x + 1, spin[y])] = i

        # get matrix
        iline = -1
        while True:
            iline += 1
            line = out[iline]
            if 'The full SOC matrix' in line:
                break
        iline += 5
        ncol = 6
        real = [[0 + 0j for i in range(4 * nst + 1)] for j in range(4 * nst + 1)]
        for x in range(len(real)):
            for y in range(len(real[0])):
                block = x // ncol
                xoffset = 1 + x % ncol
                yoffset = block * (4 * nst + 2) + y
                # print iline,x,y,block,xoffset,yoffset
                val = float(out[iline + yoffset].split()[xoffset])
                if abs(val) > 1e-16:
                    real[y][x] = val

        iline += ((4 * nst) // ncol + 1) * (4 * nst + 2) + 2
        for x in range(len(real)):
            for y in range(len(real[0])):
                block = x // ncol
                xoffset = 1 + x % ncol
                yoffset = block * (4 * nst + 2) + y
                val = float(out[iline + yoffset].split()[xoffset])
                if abs(val) > 1e-16:
                    real[y][x] += (0 + 1j) * val


        return real, inv_statemap

    # ======================================================================= #


    def gettdm(self, logfile, ijob):

        QMin = self._QMin

        # open file
        f = readfile(logfile)
        if PRINT:
            print('Dipoles:  ' + shorten_DIR(logfile))

        # figure out the excited state settings
        mults = QMin['jobs'][ijob]['mults']
        if 3 in mults and QMin['OrcaVersion'] < (4, 1):
            mults = [3]
        restr = QMin['jobs'][ijob]['restr']
        gsmult = mults[0]
        estates_to_extract = deepcopy(QMin['states'])
        estates_to_extract[gsmult - 1] -= 1
        for imult in range(len(estates_to_extract)):
            if not imult + 1 in mults:
                estates_to_extract[imult] = 0

        # print "getting cool dipoles"
        # extract transition dipole moments
        dipoles = {}
        for imult in mults:
            if not imult == gsmult:
                continue
            nstates = estates_to_extract[imult - 1]
            if nstates > 0:
                for iline, line in enumerate(f):
                    if '  ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
                        # print line
                        for istate in range(nstates):
                            shift = 5 + istate
                            s = f[iline + shift].split()
                            dm = [float(i) for i in s[5:8]]
                            dipoles[(imult, istate + 1 + (gsmult == imult))] = dm
        # print dipoles
        return dipoles

    # ======================================================================= #


    @staticmethod
    def getdm(logfile, isgs):

        # open file
        f = readfile(logfile)
        if PRINT:
            print('Dipoles:  ' + shorten_DIR(logfile))

        if isgs:
            findstring = 'ORCA ELECTRIC PROPERTIES CALCULATION'
        else:
            findstring = '*** CIS RELAXED DENSITY ***'
        for iline, line in enumerate(f):
            if findstring in line:
                break
        while True:
            iline += 1
            line = f[iline]
            if 'Total Dipole Moment' in line:
                s = line.split()
                dmx = float(s[4])
                dmy = float(s[5])
                dmz = float(s[6])
                dm = [dmx, dmy, dmz]
                return dm

    # ======================================================================= #


    @staticmethod
    def getgrad(logfile, natom):

        # initialize
        g = [[0. for i in range(3)] for j in range(natom)]

        # read file
        if os.path.isfile(logfile):
            out = readfile(logfile)
            if PRINT:
                print('Gradient: ' + shorten_DIR(logfile))

            # get gradient
            string = 'The current gradient in Eh/bohr'
            shift = 2
            for iline, line in enumerate(out):
                if string in line:
                    for iatom in range(natom):
                        for ixyz in range(3):
                            s = out[iline + shift + 3 * iatom + ixyz]
                            g[iatom][ixyz] = float(s)

        # read binary file otherwise
        else:
            logfile += '.grad.tmp'
            Gfile = open(logfile, 'rb')
            if PRINT:
                print('Gradient: ' + shorten_DIR(logfile))

            # get gradient
            Gfile.read(8 + 28 * natom)    # skip header
            for iatom in range(natom):
                for ixyz in range(3):
                    f = struct.unpack('d', Gfile.read(8))[0]
                    g[iatom][ixyz] = f

        return g


    # ======================================================================= #


    @staticmethod
    def getpcgrad(logfile):

        # read file
        out = readfile(logfile)
        if PRINT:
            print('Gradient: ' + shorten_DIR(logfile))

        g = []
        for iatom in range(len(out) - 1):
            atom_grad = [0. for i in range(3)]
            s = out[iatom + 1].split()
            for ixyz in range(3):
                atom_grad[ixyz] = float(s[ixyz])
            g.append(atom_grad)
        return g

    # ======================================================================= #


    @staticmethod
    def getsmate(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                print('Overlap of states %i - %i not found!' % (s1, s2))
                sys.exit(103)
            if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    # ======================================================================= #

    @staticmethod
    def getDyson(out, s1, s2):
        ilines = -1
        while True:
            ilines += 1
            if ilines == len(out):
                print('Dyson norm of states %i - %i not found!' % (s1, s2))
                sys.exit(104)
            if containsstring('Dyson norm matrix <PsiA_i|PsiB_j>', out[ilines]):
                break
        ilines += 1 + s1
        f = out[ilines].split()
        return float(f[s2 + 1])

    # ======================================================================= #


if __name__ == '__main__':
    orca = ORCA(DEBUG, PRINT)
    orca.main()
