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
from matplotlib.pyplot import getp
import numpy as np
from openmm.app import Simulation, AmberPrmtopFile
from openmm import VerletIntegrator, System, State, NonbondedForce, Platform, CustomIntegrator
from openmm.unit import Quantity, bohr, kilocalories_per_mole, picosecond, angstrom, nanometer

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import kcal_to_Eh as kcal_per_mole_to_Eh, au2a, au2eV

__author__ = 'Sebastian Mai and Severin Polonius'
__version__ = '3.0'
versiondate = datetime.datetime(2022, 4, 8)

changelogstring = '''
'''
np.set_printoptions(linewidth=400, formatter={'all': lambda x: str(x)})


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

    # TODO: warnings for unsupported settings, i.e. nacs, socs etc
    def run(self):
        # crd = [Quantity(Vec3(*x), unit=angstrom) for x in self._QMin['coords'] * au2a]
        # print(crd[0])
        self.simulation.context.setPositions(Quantity(self._QMin['coords'], unit=bohr))
        self.simulation.context.setVelocities(Quantity(np.zeros(self._QMin['coords'].shape), unit=nanometer/picosecond))
        # print(self.simulation.context.getSystem())
        # for _ in range(5):
        #     results: State = self.simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        #     pos = results.getPositions(asNumpy=True)._value
        #     forces: Quantity = results.getForces()._value
        #     energy: Quantity = results.getPotentialEnergy()._value
        #     self.simulation.step(1)
        #     print(energy)
        #     print(pos[:10, :])
        #     print(forces[0])
        #     print()
        # self.simulation.step(1)
        state: State = self.simulation.context.getState(getForces=True, getEnergy=True)
        # pos = state.getPositions(asNumpy=True)._value
        # forces: Quantity = state.getForces()._value
        # energy: Quantity = state.getPotentialEnergy()._value
        # print(energy)
        # print(pos[:10, :])
        # print(forces[0])
        # print()
        # # gradients: np.ndarray = -np.array(state.getForces(asNumpy=False).value_as_unit(kilocalories_per_mole/angstrom)) / 627.5094740631  # kJ/(mol*nm) -> Hartree/bohr 2625.4996394799 2625.499624258563
        gradients: np.ndarray = -state.getForces(asNumpy=True) / 2625.4996394799 * 0.1 * au2a # kJ/(mol*nm) -> Hartree/bohr
        # print(np.sqrt(np.sum(gradients**2)/3/gradients.shape[0])/au2a * au2eV )
        energy: float = state.getPotentialEnergy()._value / 2625.4996394799  # kJ/mol -> Hartree

        self._QMout['h'] = [[energy]]
        self._QMout['grad'] = [gradients.tolist()]
        self._QMout['multipolar_fit'] = self._charges
        if 'dm' in self._QMin:
            chrg = np.array(self._charges)
            self._QMout['dm'] = np.einsum('ix,i->x', self._QMin['coords'], chrg).reshape((3, 1, 1)).tolist()
        self._QMout['overlap'] = [[1.]]
        self._QMout['phases'] = [complex(1., 0.)]

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
        QMin = self._QMin
        self._read_resources = True

    def setup_run(self):
        QMin = self._QMin
        prmtop = AmberPrmtopFile(QMin['prmtop'])
        # standard params
        # nonbondedMethod=ff.NoCutoff, nonbondedCutoff=1.0*u.nanometer, constraints=None, rigidWater=True, implicitSolvent=None, implicitSolventSaltConc=0.0*(u.moles/u.liter), implicitSolventKappa=None, temperature=298.15*u.kelvin, soluteDielectric=1.0, solventDielectric=78.5, removeCMMotion=True, hydrogenMass=None, ewaldErrorTolerance=0.0005, switchDistance=0.0*u.nanometer, gbsaModel='ACE'
        system: System = prmtop.createSystem(nonbondedCutoff='NoCutoff', rigidWater=False, removeCMMotion=False)
        # integrator = VerletIntegrator(0.0005)
        integrator = CustomIntegrator(0.0005)
        # integrator.addPerDofVariable("a", 0)
        # integrator.addUpdateContextState()
        # integrator.addComputePerDof("x", "x+dt*v+0.5*a*dt*dt")
        # integrator.addComputePerDof("v", "v+0.5*dt*(f/m+a)")
        # integrator.addComputePerDof("a", "f/m")
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

        # if 'grad' in QMin:
        #     print('Warning: gradient only calculated for first singlet!')


if __name__ == "__main__":
    omm = OpenMM()
    omm.main()