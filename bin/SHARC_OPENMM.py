#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
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
from openmm import System, State, NonbondedForce, CustomNonbondedForce ,Platform, CustomIntegrator, unit
from openmm.app.forcefield import NoCutoff
import datetime
from typing import Optional
from io import TextIOWrapper
import os
from qmin import QMinBase
import logging

# internal
from SHARC_FAST import SHARC_FAST
from utils import question
from constants import au2a, kJpermol_per_Eh, coulomb_kJpermol

date = datetime.date

__author__ = "Severin Polonius"
__version__ = "4.0"
versiondate = datetime.datetime(2023, 9, 6)

changelogstring = """
"""


class SHARC_OPENMM(SHARC_FAST):
    """
    Interface for the [OpenMM program](https://openmm.org/)

    ---
    Evaluates energis and gradients for the ground state
    Works on CPU and CUDA platforms
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "prmtop": None,
                "coulomb_constant": "default"   # options: default, openmm, amber, si
            }
        )
        self.QMin.template.types.update(
            {
                "prmtop": str,
                "coulomb_constant": str
            }
        )
        self.QMin.resources.update({"ncpu": 1, "cuda_device": None})
        self.QMin.resources.types.update({"ncpu": int, "cuda_device": int})

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
        return "OPENMM"

    @staticmethod
    def description() -> str:
        return "     FAST interface for OpenMM for MM force fields (based on Amber prmtop)"

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set:
        return {"h", "grad", "overlap", "dm", "multipolar_fit"}
    
    def _check_charge(self):
        """
        Do not check charge for total system
        """

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'OpenMM interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        while True:
            self.template_file = question("Specify path to OPENMM.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
            try:
                self.read_template(self.template_file)
            except (RuntimeError, OSError, ValueError) as e:
                self.log.info(f"There is a problem with '{self.template_file}':\n{e}")
                self.QMin.template = QMinBase()
            else:
                break
        self.extra_files = [self.QMin.template["prmtop"]]

        if question("Do you have a resources file?", bool, KEYSTROKES=KEYSTROKES, default=True):
            self.resources_file = question("Specify path to OPENMM.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
        return INFOS

    def run(self):
        self.simulation.context.setPositions(self.QMin.coords["coords"] * (au2a / 10))
        state: State = self.simulation.context.getState(getEnergy=self.QMin.requests["h"], getForces=self.QMin.requests["grad"])

        # Debug: print energy contribution from each OpenMM force
        if self.log.isEnabledFor(logging.DEBUG):
            for i, force in enumerate(self.simulation.system.getForces()):
                debug_state = self.simulation.context.getState(
                    getEnergy=True,
                    groups={i}
                )
                energy_kjmol = debug_state.getPotentialEnergy()._value
                energy_Eh = energy_kjmol / kJpermol_per_Eh

                self.log.debug(
                    f"Energy {i:2d} {force.__class__.__name__:25s}: "
                    f"{energy_kjmol:16.8f} kJ/mol = {energy_Eh:16.8f} Hartree"
                )

        energy: float = state.getPotentialEnergy()._value / kJpermol_per_Eh  # kJ/mol -> Hartree
        self.QMout["h"] = np.full((1, 1), energy, dtype=float)

        if self.QMin.requests["grad"]:
            gradients: np.ndarray = -state.getForces(asNumpy=True) / kJpermol_per_Eh * 0.1 * au2a  # kJ/(mol*nm) -> Hartree/bohr
            self.QMout["grad"] = gradients._value[np.newaxis, ...]

        if self.QMin.requests["multipolar_fit"]:
            self.QMout.multipolar_fit_settings = f" order: 0, grid: default, firstlayer: default, density: default, layers: default"
            self.QMout["multipolar_fit"] = {}
            self.QMout["multipolar_fit"][(self.states[0], self.states[0])] = self._charges.reshape(-1,1)
        if self.QMin.requests["dm"]:
            chrg = np.array(self._charges)
            self.QMout["dm"] = np.einsum("ix,i->x", self.QMin.coords["coords"], chrg).reshape((3, 1, 1))

        self.QMout["overlap"] = np.ones((1, 1), dtype=float)

    def getQMout(self):
        self.QMout.states = self.QMin.molecule["states"]
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = self.QMin.molecule["nmstates"]
        self.QMout.natom = self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = self.QMin.molecule["npc"] > 0
        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout

    def read_template(self, template_filename="OPENMM.template"):
        super().read_template(template_filename)
        self._read_template = False
        if not self.QMin.template["prmtop"]:
            self.log.error(f"'prmtop' not set in '{template_filename}'! Exiting")
            raise RuntimeError(f"'prmtop' not set in '{template_filename}'! Exiting")
        if os.path.dirname(template_filename) and not os.path.isabs(self.QMin.template["prmtop"]):
            self.QMin.template["prmtop"] = os.path.join(os.path.dirname(template_filename), self.QMin.template["prmtop"])

        if not os.path.isfile(self.QMin.template["prmtop"]):
            self.log.error(f"'prmtop' with {self.QMin.template['prmtop']} cannot be opened!")
            raise ValueError(f"'prmtop' with {self.QMin.template['prmtop']} cannot be opened!")

        self.log.debug(f"{self.QMin.template}")
        self._read_template = True

    def read_resources(self, resources_filename="OPENMM.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning(f"{resources_filename} not found! Continueing without further settings.")
            self._read_resources = True
            return

        super().read_resources(resources_filename)
        if self.QMin.resources["cuda_device"] is not None:
            self.log.info("CUDA platform will be used for the calculation! Double precision might be slow!")

        self.log.debug(f"{self.QMin.resources}")
        self._read_resources = True

    def setup_interface(self):
        super().setup_interface()
        QMin = self.QMin
        prmtop = AmberPrmtopFile(QMin.template["prmtop"])

        # standard params http://docs.openmm.org/latest/api-python/generated/openmm.app.amberprmtopfile.AmberPrmtopFile.html#openmm.app.amberprmtopfile.AmberPrmtopFile.createSystem
        # nonbondedMethod=ff.NoCutoff, nonbondedCutoff=1.0*u.nanometer, constraints=None, rigidWater=True, implicitSolvent=None, implicitSolventSaltConc=0.0*(u.moles/u.liter), implicitSolventKappa=None, temperature=298.15*u.kelvin, soluteDielectric=1.0, solventDielectric=78.5, removeCMMotion=True, hydrogenMass=None, ewaldErrorTolerance=0.0005, switchDistance=0.0*u.nanometer, gbsaModel='ACE'
        system: System = prmtop.createSystem(nonbondedMethod=NoCutoff, rigidWater=False, removeCMMotion=False)

        if QMin.template["coulomb_constant"].lower() != "default":
            # ---------------------------------------------------------------------
            # START COULOMB OVERWRITE MODIFICATION
            # ---------------------------------------------------------------------
            # define custom Coulomb function for separate debug printout and to match Amber units
            
            # 1. Locate the native NonbondedForce
            nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
            
            # 2. Define the Custom Nonbonded expression using Amber's exact constant in kJ*nm/mol/e^2
            match QMin.template["coulomb_constant"].lower():
                case "amber":
                    coulomb_constant = 138.9306477184  
                case "openmm":
                    coulomb_constant = 138.935457532
                case "si":
                    coulomb_constant = coulomb_kJpermol # 138.9354576444
            coulomb_expr = f"{coulomb_constant} * q1 * q2 / r"
            
            custom_electrostatics = CustomNonbondedForce(coulomb_expr)
            custom_electrostatics.addPerParticleParameter("q")
            custom_electrostatics.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
            
            # 3. Transfer charges to custom force and neutralize original force charges
            # (This leaves the original NonbondedForce to compute ONLY Lennard-Jones vdW)
            npart = system.getNumParticles()
            for idx in range(npart):
                charge, sigma, epsilon = nonbonded.getParticleParameters(idx)
                
                # Add charge to our new custom constant engine
                custom_electrostatics.addParticle([charge])
                
                # Clear charge from standard engine to avoid double-counting
                # Keeping sigma and epsilon intact preserves standard vdW math
                nonbonded.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, epsilon)
                
            # 4. Handle exclusion pairs (like 1-2, 1-3, and water intramolecular pairs)
            # CustomNonbondedForce ignores exclusions by default unless added here
            for idx in range(nonbonded.getNumExceptions()):
                p1, p2, q_prod, vdw_sig, vdw_eps = nonbonded.getExceptionParameters(idx)
                # Add an exclusion to the custom electrostatic engine for these specific pairs
                custom_electrostatics.addExclusion(p1, p2)
                
            # 5. Push the new custom force back into the system topology
            system.addForce(custom_electrostatics)
            # ---------------------------------------------------------------------
            # END COULOMB OVERWRITE MODIFICATION
            # ---------------------------------------------------------------------

        # Debug print of force groups
        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)
            self.log.debug(
                f"OpenMM force group {i}: {force.__class__.__name__}"
            )

        # 'Do nothing' integrator
        integrator = CustomIntegrator(0.0005)
        integrator.addComputePerDof("x", "x")
        integrator.addComputePerDof("v", "v")

        platform = Platform.getPlatformByName("CPU")
        properties = {"Threads": str(QMin.resources["ncpu"])}
        if QMin.resources["cuda_device"] is not None:
            platform = Platform.getPlatformByName("CUDA")
            properties = {"DeviceIndex": str(QMin.resources["cuda_device"]), "Precision": "single"}

        self.simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        npart = system.getNumParticles()
        self.log.debug(f"System size: {npart}")
        self._charges = np.fromiter(
            map(lambda x: nonbonded.getParticleParameters(x)[0]._value, range(npart)), dtype=float, count=npart
        )

    def _request_logic(self):
        super()._request_logic()

        if self.QMin.molecule["states"] != [1]:
            self.log.error('MM calculation only implemented for the S0 state! "states" has to be "1"')
            raise ValueError('MM calculation only implemented for the S0 state! "states" has to be "1"')

        # prepare savedir
        possibletasks = self.get_features()
        requests = {k for k in self.QMin.requests.keys() if self.QMin.requests[k]}
        tasks = possibletasks & requests

        if len(tasks) == 0:
            self.log.error(f"No tasks found! Tasks are {possibletasks}.")
            raise RuntimeError(f"No tasks found! Tasks are {possibletasks}.")

        not_allowed = {"soc", "dmdr", "socdr", "ion", "nacdr", "theodore"}
        if not requests.isdisjoint(not_allowed):
            self.log.error("Cannot perform tasks: {}".format(" ".join(requests & not_allowed)))
            raise RuntimeError("Cannot perform tasks: {}".format(" ".join(requests & not_allowed)))


if __name__ == "__main__":
    from logger import loglevel

    omm = SHARC_OPENMM(loglevel=loglevel)
    omm.main()
