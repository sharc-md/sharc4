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
from openmm.app import Simulation, AmberInpcrdFile, AmberPrmtopFile
from openmm import VerletIntegrator, System, State, NonbondedForce
from openmm.unit import Quantity, bohr, kilocalories_per_mole

# internal
from SHARC_INTERFACE import INTERFACE
from utils import *
from constants import kcal_to_Eh as kcal_per_mole_to_Eh

__author__ = 'Sebastian Mai and Severin Polonius'
__version__ = '3.0'
versiondate = datetime.datetime(2022, 4, 8)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)


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
        self.simulation.context.setPositions(Quantity(self.coords, unit=bohr))
        self.simulation.step(1)
        state: State = self.simulation.context.getState(getForces=True, getEnergy=True, getParameters=True)
        self.simulation.topology
        gradients: np.ndarray = -state.getForces(asNumpy=True
                                                 ).value_in_unit(kilocalories_per_mole / bohr) / kcal_per_mole_to_Eh
        energy: float = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole) / kcal_per_mole_to_Eh
        state.getParameters()

        self._QMout['h'] = [[energy]]
        self._QMout['grad'] = [gradients]
        self._QMout['multipolar_fit'] = self._charges

    def read_template(self, template_filename):
        QMin = self._QMin
        paths = {'prmtop': '',
                 'inpcrd': ''}
        lines = readfile(template_filename)
        special = {}
        QMin['template'] = {**paths, **self.parse_keywords(lines, paths=paths, special=special)}

    def setup_run(self):
        QMin = self._QMin
        prmtop = AmberPrmtopFile(QMin['prmtop'])
        self.coords = AmberInpcrdFile(QMin['inpcrd']).getPositions(asNumpy=True).value_in_unit(bohr)
        # standard params
        # nonbondedMethod=ff.NoCutoff, nonbondedCutoff=1.0*u.nanometer, constraints=None, rigidWater=True, implicitSolvent=None, implicitSolventSaltConc=0.0*(u.moles/u.liter), implicitSolventKappa=None, temperature=298.15*u.kelvin, soluteDielectric=1.0, solventDielectric=78.5, removeCMMotion=True, hydrogenMass=None, ewaldErrorTolerance=0.0005, switchDistance=0.0*u.nanometer, gbsaModel='ACE'
        system: System = prmtop.createSystem(nonbondedCutoff=1 * 1e9)
        integrator = VerletIntegrator(0.0)
        self.simulation = Simulation(prmtop.topology, system, integrator, platformProperties={'Threads': QMin['ncpu']})
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        self._charges = []
        for i in range(system.getNumParticles()):
            charge, _, _ = nonbonded.getParticleParameters(i)
            self._charges.append(charge._value)

    def _request_logic(self):
        pass