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

# This script calculates QC results for a system described by the LVC model
#
# Reads QM.in
# Calculates SOC matrix, dipole moments, gradients, nacs and overlaps
# Writes these back to QM.out

# IMPORTS
# external
import numpy as np
from openmm.app import Simulation, AmberPrmtopFile
from openmm import System, State, NonbondedForce, Platform, CustomIntegrator
import datetime
from typing import Optional
from io import TextIOWrapper
import os
from qmin import QMinBase

# internal
from SHARC_FAST import SHARC_FAST
from utils import question
from constants import au2a, kJpermol_to_Eh

date = datetime.date

__author__ = 'Severin Polonius'
__version__ = '4.0'
versiondate = datetime.datetime(2023, 9, 6)

changelogstring = '''
'''


class SHARC_OpenMM(SHARC_FAST):
    """
    Interface for the [OpenMM program](https://openmm.org/)

    ---
    Evaluates energis and gradients for the ground state
    Works on CPU and CUDA platforms
    """

    @staticmethod
    def version():
        return "4.0"

    @staticmethod
    def versiondate() -> date:
        return versiondate

    @staticmethod
    def changelogstring() -> str:
        return changelogstring

    @staticmethod
    def authors() -> str:
        return __author__

    @staticmethod
    def name() -> str:
        return "OpenMM"

    @staticmethod
    def description() -> str:
        return "Interface for the OpenMM program for MM calculations"

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set:
        return {'h', 'grad', 'overlap', 'dm'}

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'OpenMM interface setup':^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        while True:
            self.template_file = question("Specify path to OpenMM.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
            try:
                self.read_template(self.template_file)
            except (RuntimeError, OSError, ValueError) as e:
                self.log.info(f"There is a problem with '{self.template_file}':\n{e}")
                self.QMin.template = QMinBase()
            else:
                break
        self.extra_files = [self.QMin.template['prmtop']]

        if question("Do you have a resources file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            self.resources_file = question("Specify path to OpenMM.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
        return INFOS

    def create_restart_files(self):
        pass

    def run(self):
        self.simulation.context.setPositions(self.QMin.coords['coords'] * (au2a / 10))
        state: State = self.simulation.context.getState(getEnergy=self.QMin.requests['h'], getForces=self.QMin.requests['grad'])

        energy: float = state.getPotentialEnergy()._value / kJpermol_to_Eh    # kJ/mol -> Hartree
        self.QMout['h'] = np.full((1, 1), energy, dtype=float)

        if self.QMin.requests['grad']:
            gradients: np.ndarray = -state.getForces(
                asNumpy=True
            ) / kJpermol_to_Eh * 0.1 * au2a    # kJ/(mol*nm) -> Hartree/bohr
            self.QMout['grad'] = gradients._value[np.newaxis, ...]

        if self.QMin.requests['multipolar_fit']:
            self.QMout['multipolar_fit'] = self._charges[np.newaxis, np.newaxis, ...]

        if self.QMin.requests['dm']:
            chrg = np.array(self._charges)
            self.QMout['dm'] = np.einsum('ix,i->x', self.QMin.coords['coords'], chrg).reshape((3, 1, 1))

        self.QMout['overlap'] = np.ones((1, 1), dtype=float)


    def getQMout(self):
        self.QMout.states = self.QMin.molecule['states']
        self.QMout.nstates = self.QMin.molecule['nstates']
        self.QMout.nmstates = self.QMin.molecule['nmstates']
        self.QMout.natom = self.QMin.molecule['natom']
        self.QMout.npc = self.QMin.molecule['npc']
        self.QMout.point_charges = self.QMin.molecule['npc'] > 0
        return self.QMout

    def read_template(self, template_filename='OpenMM.template'):
        super().read_template(template_filename)
        self._read_template = False
        if 'prmtop' not in self.QMin.template:
            self.log.error(f"'prmtop' not set in '{template_filename}'! Exiting")
            raise RuntimeError(f"'prmtop' not set in '{template_filename}'! Exiting")
        if not os.path.isfile(self.QMin.template['prmtop']):
            self.log.error(f"'prmtop' with {self.QMin.template['prmtop']} cannot be opened!")
            raise ValueError(f"'prmtop' with {self.QMin.template['prmtop']} cannot be opened!")

        self.log.debug(f"{self.QMin.template}")
        self._read_template = True

    def read_resources(self, resources_filename='OpenMM.resources'):
        self.QMin.resources.update({'ncpu': 1, 'cuda': False})
        if not os.path.isfile(resources_filename):
            self.log.warning(f"{resources_filename} not found! Continueing without further settings.")
            self._read_resources = True
            return

        super().read_resources(resources_filename)
        self.QMin.resources.update({'ncpu': 1, 'cuda': False})
        if 'ncpu' not in self.QMin.resources:
            self.log.warning("Number of CPUs not set in resources with 'ncpu' -> set to 1")
        if 'cuda' in self.QMin.resources:
            self.log.info("CUDA platform will be used for the calculation")

        self.log.debug(f"{self.QMin.resources}")
        self._read_resources = True

    def setup_interface(self):
        QMin = self.QMin
        prmtop = AmberPrmtopFile(QMin.template['prmtop'])

        # standard params http://docs.openmm.org/latest/api-python/generated/openmm.app.amberprmtopfile.AmberPrmtopFile.html#openmm.app.amberprmtopfile.AmberPrmtopFile.createSystem
        # nonbondedMethod=ff.NoCutoff, nonbondedCutoff=1.0*u.nanometer, constraints=None, rigidWater=True, implicitSolvent=None, implicitSolventSaltConc=0.0*(u.moles/u.liter), implicitSolventKappa=None, temperature=298.15*u.kelvin, soluteDielectric=1.0, solventDielectric=78.5, removeCMMotion=True, hydrogenMass=None, ewaldErrorTolerance=0.0005, switchDistance=0.0*u.nanometer, gbsaModel='ACE'
        system: System = prmtop.createSystem(nonbondedCutoff='NoCutoff', rigidWater=False, removeCMMotion=False)

        # 'Do nothing' integrator
        integrator = CustomIntegrator(0.0005)
        integrator.addComputePerDof("x", "x")
        integrator.addComputePerDof("v", "v")

        platform = Platform.getPlatformByName('CPU')
        properties = {'Threads': str(QMin.resources['ncpu'])}
        if QMin.resources['cuda']:
            platform = Platform.getPlatformByName('CUDA')
            properties.update({'DeviceIndex': '0', 'Precision': 'double'})

        self.simulation = Simulation(prmtop.topology, system, integrator, platform, platformProperties=properties)
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        npart = system.getNumParticles()
        self.log.debug(f"System size: {npart}")
        self._charges = np.fromiter(map(lambda x: nonbonded.getParticleParameters(x)[0]._value, range(npart)), dtype=float, count=npart)

    def _request_logic(self):
        super()._request_logic()

        if self.QMin.molecule['states'] != [1]:
            self.log.error('MM calculation only implemented for the S0 state! "states" has to be "1"')
            raise ValueError('MM calculation only implemented for the S0 state! "states" has to be "1"')

        # prepare savedir
        possibletasks = self.get_features()
        requests = {k for k in self.QMin.requests.keys() if self.QMin.requests[k]}
        tasks = possibletasks & requests

        if len(tasks) == 0:
            self.log.error(f'No tasks found! Tasks are {possibletasks}.')
            raise RuntimeError(f'No tasks found! Tasks are {possibletasks}.')

        not_allowed = {'soc', 'dmdr', 'socdr', 'ion', 'nacdr', 'theodore'}
        if not requests.isdisjoint(not_allowed):
            self.log.error('Cannot perform tasks: {}'.format(' '.join(requests & not_allowed)))
            raise RuntimeError('Cannot perform tasks: {}'.format(' '.join(requests & not_allowed)))


if __name__ == "__main__":
    from logger import loglevel
    omm = SHARC_OpenMM(loglevel=loglevel)
    omm.main()
