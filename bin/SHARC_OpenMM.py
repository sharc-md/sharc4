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
from openmm.unit import Quantity, bohr, picosecond, nanometer

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import au2a, kJpermol_to_Eh

__author__ = 'Sebastian Mai and Severin Polonius'
__version__ = '3.0'
versiondate = datetime.datetime(2022, 4, 8)

changelogstring = '''
'''


class OpenMM(INTERFACE):

    _version = __version__
    _versiondate = versiondate
    _authors = __author__
    _changelogstring = changelogstring
    _read_resources = True

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
        self.simulation.context.setPositions(Quantity(self._QMin['coords'], unit=bohr))
        self.simulation.context.setVelocities(
            Quantity(np.zeros(self._QMin['coords'].shape), unit=nanometer / picosecond)
        )
        state: State = self.simulation.context.getState(getForces=True, getEnergy=True)
        gradients: np.ndarray = -state.getForces(
            asNumpy=True
        ) / kJpermol_to_Eh * 0.1 * au2a    # kJ/(mol*nm) -> Hartree/bohr
        energy: float = state.getPotentialEnergy()._value / kJpermol_to_Eh    # kJ/mol -> Hartree

        self._QMout['h'] = [[energy]]
        self._QMout['grad'] = [gradients.tolist()]
        self._QMout['multipolar_fit'] = self._charges
        if 'dm' in self._QMin:
            chrg = np.array(self._charges)
            self._QMout['dm'] = np.einsum('ix,i->x', self._QMin['coords'], chrg).reshape((3, 1, 1)).tolist()
        self._QMout['overlap'] = [[1.]]
        self._QMout['phases'] = [complex(1., 0.)]

    def getQMout(self):
        return self._QMout

    def read_template(self, template_filename='OpenMM.template'):
        QMin = self._QMin
        paths = {'prmtop': ''}
        lines = readfile(template_filename)
        special = {}
        QMin['template'] = {**paths, **self.parse_keywords(lines, paths=paths, special=special)}
        QMin.update(QMin['template'])
        self._read_template = True

    def read_resources(self, resources_filename='OpenMM.resources'):
        super().read_resources(resources_filename)
        self._read_resources = True

    def setup_run(self):
        QMin = self._QMin
        prmtop = AmberPrmtopFile(QMin['prmtop'])

        # standard params http://docs.openmm.org/latest/api-python/generated/openmm.app.amberprmtopfile.AmberPrmtopFile.html#openmm.app.amberprmtopfile.AmberPrmtopFile.createSystem
        # nonbondedMethod=ff.NoCutoff, nonbondedCutoff=1.0*u.nanometer, constraints=None, rigidWater=True, implicitSolvent=None, implicitSolventSaltConc=0.0*(u.moles/u.liter), implicitSolventKappa=None, temperature=298.15*u.kelvin, soluteDielectric=1.0, solventDielectric=78.5, removeCMMotion=True, hydrogenMass=None, ewaldErrorTolerance=0.0005, switchDistance=0.0*u.nanometer, gbsaModel='ACE'
        system: System = prmtop.createSystem(nonbondedCutoff='NoCutoff', rigidWater=False, removeCMMotion=False)

        # 'Do nothing' integrator
        integrator = CustomIntegrator(0.0005)
        integrator.addComputePerDof("x", "x")
        integrator.addComputePerDof("v", "v")

        platform = Platform.getPlatformByName('CPU')
        properties = {'Threads': str(QMin['ncpu'])}

        self.simulation = Simulation(prmtop.topology, system, integrator, platform, platformProperties=properties)
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        self._charges = []
        for i in range(system.getNumParticles()):
            charge, _, _ = nonbonded.getParticleParameters(i)
            self._charges.append(charge._value)

    def _request_logic(self):
        QMin = self._QMin

        if QMin['states'] != [1]:
            raise Error('MM calculation only implemented for the S0 state! "states" has to be "1"')

        # prepare savedir
        if not os.path.isdir(QMin['savedir']):
            mkdir(QMin['savedir'])
        possibletasks = {'h', 'grad', 'overlap'}
        tasks = possibletasks & QMin.keys()

        if len(tasks) == 0:
            raise Error(f'No tasks found! Tasks are {possibletasks}.', 39)

        not_allowed = {'soc', 'dmdr', 'socdr', 'ion', 'nacdr', 'theodore'}
        if not QMin.keys().isdisjoint(not_allowed):
            raise Error('Cannot perform tasks: {}'.format(' '.join(QMin.keys() & not_allowed)), 13)

    def create_restart_files(self):
        pass

if __name__ == "__main__":
    omm = OpenMM()
    omm.main()
