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
import datetime
import os
import shutil
from io import TextIOWrapper
import re
import copy

try:
    import torch
    pytorch_available = True
except ImportError:
    pytorch_available = False

import numpy as np
# internal
from constants import ATOMCHARGE, FROZENS, ANG_TO_BOHR, rcm_to_Eh, kcal_to_Eh, EV_TO_EH, kJpermol_to_Eh
from factory import factory
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import (ATOM, InDir, expand_path, itnmstates, mkdir, question,
                   readfile)

VERSION = "4.0"
VERSIONDATE = datetime.datetime(2023, 8, 24)

CHANGELOGSTRING = """
"""
np.set_printoptions(linewidth=400)


class SHARC_UMBRELLA(SHARC_HYBRID):
    _version = VERSION
    _versiondate = VERSIONDATE
    _changelogstring = CHANGELOGSTRING
    _step = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update(
            {
                "restraint_file": "UMBRELLA.restraints",
                "child-program": None,
                "child-dir": None,
            }
        )
        self.QMin.template.types.update(
            {
                "restraint_file": str,
                "child-program": str,
                "child-dir": str,
            }
        )

        self.child_interface = None
        self.restraints = None

        self.pytorch = pytorch_available


    @staticmethod
    def description():
        return "Hybrid interface for adding umbrella-sampling-style restraints"

    @staticmethod
    def version():
        return SHARC_UMBRELLA._version

    @staticmethod
    def name() -> str:
        return "UMBRELLA"
    
    @staticmethod
    def about():
        pass

    @staticmethod
    def versiondate():
        return SHARC_UMBRELLA._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_UMBRELLA._changelogstring

    @staticmethod
    def authors() -> str:
        return "Sebastian Mai"





    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set:

        if (not self._read_template):
            self.template_file = question(
                "Please specify the path to your UMBRELLA.template file", str, KEYSTROKES=KEYSTROKES, default="UMBRELLA.template"
            )
            
            self.read_template(self.template_file)

        child_features = self.child_interface.get_features(KEYSTROKES=KEYSTROKES)
        self.log.debug(child_features)
        return set(child_features)
    


    def get_infos(self, INFOS, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'UMBRELLA interface setup':=^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        ## template already read in get_features
        # self.template_file = question("Specify path to QMMM.template", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
        # self.read_template(self.template_file)

        if question("Do you have an UMBRELLA.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question("Specify path to UMBRELLA.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True, 
                                           default='UMBRELLA.resources')

        self.log.info(f"\n{' Setting up child interface ':=^80s}\n")
        self.child_interface.QMin.molecule["states"] = INFOS["states"]
        self.child_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        return INFOS


    def prepare(self, INFOS, dir_path) -> None:
        QMin = self.QMin

        if "link_files" in INFOS:
            os.symlink(expand_path(self.template_file), os.path.join(dir_path, self.name() + ".template"))
            os.symlink(
                expand_path(self.QMin.template["restraint_file"]),
                os.path.join(dir_path, os.path.split(self.QMin.template["restraint_file"])[1]),
            )
            if "resources_file" in self.__dict__:
                os.symlink(expand_path(self.resources_file), os.path.join(dir_path, self.name() + ".resources"))
        else:
            shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))
            shutil.copy(self.QMin.template["restraint_file"], os.path.join(dir_path, os.path.split(self.QMin.template["restraint_file"])[1]))
            if "resources_file" in self.__dict__:
                shutil.copy(self.resources_file, os.path.join(dir_path, self.name() + ".resources"))

        if not QMin.save["savedir"]:
            self.log.warning("savedir not specified, setting savedir to current directory!")
            QMin.save["savedir"] = os.getcwd()

        qmdir = dir_path + f"/{QMin.template['child-dir']}"
        mkdir(qmdir)
        
        # folder setup and savedir
        qm_savedir = os.path.join(dir_path, QMin.save["savedir"], "QM_" + QMin.template["child-program"].upper())
        self.log.debug(f"qm_savedir {qm_savedir}")
        if not os.path.isdir(qm_savedir):
            mkdir(qm_savedir)
        self.child_interface.QMin.save["savedir"] = qm_savedir
        self.child_interface.QMin.resources["scratchdir"] = os.path.join(
            QMin.resources["scratchdir"], "QM_" + QMin.template["child-program"].upper()
        )
        self.child_interface.prepare(INFOS, qmdir)







    def read_template(self, template_file="UMBRELLA.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        required: set = {
            "child-program",
        }

        if not required.issubset(self.QMin.template.keys()):
            self.log.error(
                '"{}" not specified in {}'.format(
                    '", "'.join(filter(lambda x: x not in self.QMin.template, required)), template_file
                )
            )
            raise RuntimeError()

        # make the child
        self.child_interface: SHARC_INTERFACE = factory(self.QMin.template["child-program"])(
            persistent=self.persistent, logname=f"QM {self.QMin.template['child-program']}", loglevel=self.log.level
        )
        self.child_interface.QMin.molecule['states'] = self.QMin.molecule['states']
        

        # check directory
        if not self.QMin.template["child-dir"]:
            self.QMin.template["child-dir"] = self.child_interface.name()
            self.log.info(f"'child-dir not set in template setting to name of program: {self.QMin.template['child-dir']}")

        # check if restraint_file is relative or absolute path
        if not os.path.isabs(self.QMin.template["restraint_file"]):
            #  path from location of template
            self.QMin.template["restraint_file"] = os.path.join(os.path.dirname(template_file), self.QMin.template["restraint_file"])
        if not os.path.isfile(self.QMin.template["restraint_file"]):
            self.log.error(f"{self.QMin.template['restraint_file']} not found! Specify 'restraint_file' in template!")
            raise RuntimeError()
        data = readfile(self.QMin.template["restraint_file"])

        ## example restraint file:
        # r 100. kcal/mol bohr    1.4 angstrom 1 2     # counting starts from 1
        # a 200. kcal/mol radians 90. degree   2 1 3   # middle atom is apex
        # d 300. kcal/mol radians 10. degree   1 2 3 4 # angles will always be between 0 and 180°
        # first is type
        # 2-4 is force constant, value is multiplied by first unit and divided by second unit
        # 5-6 is equilibrium value, multiplied by unit
        # 7-10 are atom indices to define the bond/angle/dihedral
    
        # process restraint file
        self.restraints = []
        factors = {'kcal/mol': kcal_to_Eh,
                   'kj/mol': 1./kJpermol_to_Eh,
                   'eh': 1.,
                   'cm-1': rcm_to_Eh,
                   'ev': EV_TO_EH,
                   'angstrom': ANG_TO_BOHR,
                   'bohr': 1.,
                   'degree': np.pi/180.,
                   'radian': 1.,
                   'one': 1.,
                   'per': 1.,
                   }
        types = {'r': 2,
                 'a': 3,
                 'd': 4,
                 'de': 2}
        for line in data:
            line = re.sub('#.*$', '', line)
            s = line.split()
            if not s:
                continue
            if s[0] not in types:
                continue
            if len(s) < 5+types[s[0]]:
                continue
            restraint = (s[0].lower(), 
                         float(s[1]) * factors[s[2].lower()] / factors[s[3].lower()], 
                         float(s[4]) * factors[s[5].lower()], 
                         [int(i)-1 for i in s[6:]] 
                         )
            self.log.info(str(restraint))
            self.restraints.append(restraint)



    def read_resources(self, resources_filename="UMBRELLA.resources"):
        if not os.path.isfile(resources_filename):
            self.log.warning(f"File '{resources_filename}' not found! Continuuing without appling any settings")
            self._read_resources = True
            return
        super().read_resources(resources_filename)
        self._read_resources = True



    def setup_interface(self):
        # prepare info for child interface
        el = self.QMin.molecule["elements"]
        # setup mol for qm
        self.child_interface.setup_mol(self.QMin)

        qm_savedir = os.path.join(self.QMin.save["savedir"], "QM_" + self.QMin.template["child-program"].upper())
        if not os.path.isdir(qm_savedir):
            mkdir(qm_savedir)
        # read template and resources
        with InDir(self.QMin.template["child-dir"]) as _:
            self.child_interface.read_resources()
            self.child_interface.QMin.save["savedir"] = qm_savedir  # overwrite savedir
            self.child_interface.read_template()
            self.child_interface.setup_interface()

        return




    def run(self):
        self.child_interface.QMin.coords["coords"] = self.QMin.coords["coords"].copy()
        if self.QMin.coords["pccoords"]:
            self.child_interface.QMin.coords["pccoords"] = self.QMin.coords["pccoords"].copy()

        for key, value in self.QMin.requests.items():
            if value is not None:
                self.child_interface.QMin.requests[key] = value
        self.child_interface._request_logic()

        # add h request to child if needed for "de" restraints:
        if not self.QMin.requests["h"] and not self.QMin.requests["soc"]:
            self.child_interface.QMin.requests["h"] = True

        # if there are "de" restraints, we need to make sure to compute all necessary gradients
        if self.QMin.requests["grad"]:
            self.original_gradient_request = copy.deepcopy(self.QMin.requests["grad"])
            gradrequests = set(self.QMin.requests["grad"])
            for i, restraint in enumerate(self.restraints):
                t, _, _, indices = restraint
                if t == "de":
                    gradrequests.add(indices[0]+1)
                    gradrequests.add(indices[1]+1)
            self.child_interface.QMin.requests["grad"] = sorted(gradrequests)
            

        with InDir(self.QMin.template["child-dir"]) as _:
            self.child_interface.run()
            self.child_interface.getQMout()



    def getQMout(self):
        self.QMout = self.child_interface.QMout

        # compute restraint energy and gradient
        E = []
        if self.QMin.requests["grad"]:
            Grad = []
        coords = self.QMin.coords['coords']
        for i, restraint in enumerate(self.restraints):
            t, k, v0, indices = restraint
            e = 0.
            if self.QMin.requests["grad"]:
                grad = np.zeros_like(self.QMout['grad'][0])
            match t:
                case 'r':
                    if self.pytorch:
                        with torch.enable_grad():
                            coords_p = torch.from_numpy(coords)
                            coords_p.requires_grad = bool(self.QMin.requests["grad"])
                            Ri = coords_p[indices[0]]
                            Rj = coords_p[indices[1]]
                            Rij = torch.norm(Ri - Rj)
                            e = k / 2. * (Rij - v0) ** 2
                            if self.QMin.requests["grad"]:
                                e.backward()
                                grad = coords_p.grad.detach().clone().numpy()
                            e = e.detach().numpy()
                    else:
                        Ri = coords[indices[0]]
                        Rj = coords[indices[1]]
                        Rij = np.linalg.norm(Ri-Rj)
                        e = k/2. * (Rij - v0)**2
                        if self.QMin.requests["grad"]:
                            grad[indices[0],:] = +k*(Rij - v0)/Rij*(Ri-Rj)
                            grad[indices[1],:] = -k*(Rij - v0)/Rij*(Ri-Rj)
                case 'a':
                    if self.pytorch:
                        with torch.enable_grad():
                            coords_p = torch.from_numpy(coords)
                            coords_p.requires_grad = bool(self.QMin.requests["grad"])
                            Ri = coords_p[indices[0]]
                            Rj = coords_p[indices[1]]
                            Rk = coords_p[indices[2]]
                            R1 = Ri - Rj
                            R2 = Rk - Rj
                            aijk = torch.arccos(torch.dot(R1,R2)/torch.norm(R1)/torch.norm(R2))
                            e = k/2. * (aijk - v0)**2
                            if self.QMin.requests["grad"]:
                                e.backward()
                                grad = coords_p.grad.detach().clone().numpy()
                            e = e.detach().numpy()
                    else:
                        Ri = coords[indices[0]]
                        Rj = coords[indices[1]]
                        Rk = coords[indices[2]]
                        R1 = Ri - Rj
                        R2 = Rk - Rj
                        aijk = np.arccos(np.dot(R1,R2)/np.linalg.norm(R1)/np.linalg.norm(R2))
                        e = k/2. * (aijk - v0)**2
                        if self.QMin.requests["grad"]:
                            # https://math.stackexchange.com/questions/1165532/gradient-of-an-angle-in-terms-of-the-vertices (answer by nben, Apr 17, 2015)
                            dcosa_i = ( R2 / np.linalg.norm(R2) - R1 / np.linalg.norm(R1) * np.cos(aijk) ) / np.linalg.norm(R1)
                            dcosa_k = ( R1 / np.linalg.norm(R1) - R2 / np.linalg.norm(R2) * np.cos(aijk) ) / np.linalg.norm(R2)
                            g1 = -k*(aijk - v0) / np.sin(aijk) 
                            grad[indices[0],:] =  g1 * dcosa_i
                            grad[indices[2],:] =  g1 * dcosa_k
                            grad[indices[1],:] = -g1 * dcosa_i -g1 * dcosa_k
                case 'd':
                    if self.pytorch:
                        with torch.enable_grad():
                            coords_p = torch.from_numpy(coords)
                            coords_p.requires_grad = bool(self.QMin.requests["grad"])
                            Ri = coords_p[indices[0]]
                            Rj = coords_p[indices[1]]
                            Rk = coords_p[indices[2]]
                            Rl = coords_p[indices[3]]
                            R1 = Ri - Rj
                            R2 = Rk - Rj
                            R3 = Rk - Rl
                            ## https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
                            # dijkl = torch.atan2(
                            #     torch.dot(R2,torch.cross(torch.cross(R1,R2),torch.cross(R2,R3))),
                            #     torch.norm(R2)*torch.dot(torch.cross(R1,R2),torch.cross(R2,R3))
                            # )
                            ## https://github.com/ochsenfeld-lab/adaptive_sampling/blob/227efd3d3c218a8f7e50c4bc1d0983767a112a9c/adaptive_sampling/colvars/colvars.py#L179
                            R2u = R2 / torch.norm(R2)
                            n1 = -R1 - torch.dot(-R1,R2u) * R2u
                            n2 =  R3 - torch.dot( R3,R2u) * R2u
                            dijkl = torch.atan2(torch.dot(torch.cross(R2u,n1), n2), torch.dot(n1,n2))
                            self.log.info(dijkl)
                            e = k/2. * (dijkl - v0)**2
                            if self.QMin.requests["grad"]:
                                e.backward()
                                grad = coords_p.grad.detach().clone().numpy()
                            e = e.detach().numpy()
                    else:
                        Ri = coords[indices[0]]
                        Rj = coords[indices[1]]
                        Rk = coords[indices[2]]
                        Rl = coords[indices[3]]
                        R1 = Rj - Ri
                        R2 = Rk - Rj
                        R3 = Rl - Rk
                        m = np.cross(R1,R2)/np.linalg.norm(np.cross(R1,R2))
                        n = np.cross(R2,R3)/np.linalg.norm(np.cross(R2,R3))
                        dijkl = np.arccos(np.dot(m,n)) 
                        e = k/2. * (dijkl - v0)**2
                        if self.QMin.requests["grad"]:
                            # https://nosarthur.github.io/free%20energy%20perturbation/2017/02/01/dihedral-force.html
                            dcosphi_i = np.cross(np.cross(np.cross(m,n),m),R2)/np.linalg.norm(np.cross(R1,R2))
                            dcosphi_j = np.cross(np.cross(np.cross(n,m),n),R3)/np.linalg.norm(np.cross(R2,R3)) - np.cross(np.cross(np.cross(m,n),m),R1+R2)/np.linalg.norm(np.cross(R1,R2))
                            dcosphi_k = np.cross(np.cross(np.cross(m,n),m),R1)/np.linalg.norm(np.cross(R1,R2)) - np.cross(np.cross(np.cross(n,m),n),R3+R2)/np.linalg.norm(np.cross(R2,R3))
                            dcosphi_l = np.cross(np.cross(np.cross(n,m),n),R2)/np.linalg.norm(np.cross(R2,R3))
                            g1 = -k*(dijkl - v0) / np.sin(dijkl)
                            grad[indices[0],:] = g1 * dcosphi_i
                            grad[indices[1],:] = g1 * dcosphi_j
                            grad[indices[2],:] = g1 * dcosphi_k
                            grad[indices[3],:] = g1 * dcosphi_l
                case 'de':
                    Ei = self.child_interface.QMout.h[indices[0],indices[0]].real
                    Ej = self.child_interface.QMout.h[indices[1],indices[1]].real
                    dE = Ei - Ej
                    e = k/2. * (dE - v0)**2
                    if self.QMin.requests["grad"]:
                        gi = self.child_interface.QMout.grad[indices[0]]
                        gj = self.child_interface.QMout.grad[indices[1]]
                        g1 = k*(dE - v0)
                        grad = g1 * (gi - gj)

            # save the computed results
            E.append(e)
            if self.QMin.requests["grad"]:
                Grad.append(grad)
        # compound all restraints
        Etot = sum(E)
        if self.QMin.requests["grad"]:
            Gradtot = np.zeros_like(self.QMout['grad'][0])
            for grad in Grad:
                Gradtot += grad

        # apply to QMout
        if self.QMin.requests["h"]:
            self.QMout.h = self.child_interface.QMout.h.copy()
            self.QMout.h += np.eye(self.QMout.h.shape[0], dtype=float) * Etot

        if self.QMin.requests["grad"]:
            self.QMout.grad = self.child_interface.QMout.grad.copy()
            for i,_ in enumerate(self.QMout.grad):
                self.QMout.grad[i] += Gradtot
                # TODO: set gradients to zero that were not originally requested, but keep M_S sublevels

        self.QMout.runtime = self.clock.measuretime()
        return self.QMout





    # savedir handling

    def _step_logic(self):
        super()._step_logic()
        self.child_interface._step_logic()

    def write_step_file(self):
        super().write_step_file()
        self.child_interface.write_step_file()
        
    def update_step(self, step: int = None):
        super().update_step(step)
        self.child_interface.update_step(step)

    def create_restart_files(self):
        self.child_interface.create_restart_files()

    def clean_savedir(self):
        super().clean_savedir()
        self.child_interface.clean_savedir()





if __name__ == "__main__":
    from logger import loglevel

    try:
        umbrella = SHARC_UMBRELLA(loglevel=loglevel)
        umbrella.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
